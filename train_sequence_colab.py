#!/usr/bin/env python3
"""Train a Colab/server Phase 2 MyDream sequence model.

This script is intended for Google Colab or a server environment with
TensorFlow installed. It reads the sequence dataset produced by
``build_sequence_dataset.py`` and can score an alarm-window sequence dataset so
the output remains compatible with ``analyze_alarm_failures.py``.

Install in Colab:

    !pip install tensorflow pandas scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


TARGET = "label_deep_soon"
DEFAULT_THRESHOLD = 0.5
MODEL_TYPES = [
    "gru",
    "lstm",
    "bigru",
    "bigru_attention",
    "cnn",
    "cnn_gru",
    "tcn",
    "tcn_attention",
    "inception_time",
    "transformer",
    "transformer_tcn",
    "patchtst_lite",
]
OUTPUT_COLUMNS = [
    "session_id",
    "candidate_time",
    "deadline_time",
    "stage_at_candidate",
    "stage_at_deadline",
    "next_deep_start",
    "target",
    "actual",
    "probability",
    "prediction",
]
NUMERIC_CONTEXT_COLUMNS = [
    "elapsed_sleep_minutes",
    "minutes_before_deadline",
    "time_of_day_sin",
    "time_of_day_cos",
    "target_wake_hour_sin",
    "target_wake_hour_cos",
    "minutes_since_stage_start",
    "minutes_since_last_deep",
    "deep_cycle_position",
    "recent_30m_awake_minutes",
    "recent_30m_light_minutes",
    "recent_30m_deep_minutes",
    "recent_30m_rem_minutes",
    "recent_30m_unknown_minutes",
    "sequence_awake_ratio",
    "sequence_light_ratio",
    "sequence_deep_ratio",
    "sequence_rem_ratio",
    "sequence_unknown_ratio",
    "sequence_stage_transition_count",
    "sequence_known_stage_transition_count",
    "sequence_known_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyDream Colab sequence model.")
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--predict-sequence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="gru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--context-units", type=int, default=16)
    parser.add_argument("--conv-filters", type=int, default=16)
    parser.add_argument("--conv-kernel-size", type=int, default=5)
    parser.add_argument(
        "--tcn-dilations",
        default="1,2,4,8",
        help="Comma-separated dilation rates for TCN residual blocks.",
    )
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_stage_count(sequence_dir: Path) -> int:
    with (sequence_dir / "stage_vocab.json").open("r", encoding="utf-8") as handle:
        stage_to_id = json.load(handle)
    return max(int(value) for value in stage_to_id.values()) + 1


def load_dataset(
    sequence_dir: Path,
    stage_count: int | None = None,
    context_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(sequence_dir / "sequence_metadata.csv")
    stages = np.load(sequence_dir / "sequence_stage_ids.npy").astype(np.int32)
    if len(metadata) != len(stages):
        raise ValueError(f"metadata rows ({len(metadata)}) do not match sequences ({len(stages)})")
    if stage_count is None:
        stage_count = load_stage_count(sequence_dir)
    if context_columns is None:
        context_columns = [column for column in NUMERIC_CONTEXT_COLUMNS if column in metadata.columns]
    missing_context_columns = sorted(set(context_columns) - set(metadata.columns))
    if missing_context_columns:
        print(f"Missing context columns; filling with zero: {missing_context_columns}")
    context = pd.DataFrame(index=metadata.index)
    for column in context_columns:
        if column in metadata.columns:
            context[column] = pd.to_numeric(metadata[column], errors="coerce").fillna(0.0)
        else:
            context[column] = 0.0
    context_values = context.to_numpy(dtype=np.float32)
    return metadata, stages, context_values


def standardize_context(
    train_context: np.ndarray,
    columns: list[str],
    *others: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    mean = train_context.mean(axis=0)
    std = train_context.std(axis=0)
    std[std == 0] = 1.0
    scaled_train = (train_context - mean) / std
    scaled_others = [(context - mean) / std for context in others]
    stats = {
        "columns": columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return scaled_train, scaled_others, stats


def build_model(
    model_type: str,
    window_minutes: int,
    stage_count: int,
    context_dim: int,
    embedding_dim: int,
    hidden_units: int,
    dense_units: int,
    context_units: int,
    conv_filters: int,
    conv_kernel_size: int,
    tcn_dilations: tuple[int, ...],
    transformer_heads: int,
    transformer_layers: int,
    transformer_ff_dim: int,
    dropout: float,
    learning_rate: float,
):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    def attention_pool(sequence, name: str):
        weights = layers.Dense(1, name=f"{name}_score")(sequence)
        weights = layers.Softmax(axis=1, name=f"{name}_weights")(weights)
        weighted = layers.Multiply(name=f"{name}_weighted")([sequence, weights])
        return layers.Lambda(lambda values: tf.reduce_sum(values, axis=1), name=f"{name}_pool")(weighted)

    def tcn_stack(sequence, name_prefix: str):
        output = layers.Conv1D(hidden_units, kernel_size=1, padding="same", name=f"{name_prefix}_projection")(sequence)
        for block_index, dilation in enumerate(tcn_dilations):
            residual = output
            y = layers.Conv1D(
                hidden_units,
                kernel_size=conv_kernel_size,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
                name=f"{name_prefix}_block_{block_index}_conv_1",
            )(output)
            y = layers.Dropout(dropout, name=f"{name_prefix}_block_{block_index}_dropout_1")(y)
            y = layers.Conv1D(
                hidden_units,
                kernel_size=conv_kernel_size,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
                name=f"{name_prefix}_block_{block_index}_conv_2",
            )(y)
            y = layers.Dropout(dropout, name=f"{name_prefix}_block_{block_index}_dropout_2")(y)
            output = layers.Add(name=f"{name_prefix}_block_{block_index}_residual")([residual, y])
            output = layers.Activation("relu", name=f"{name_prefix}_block_{block_index}_activation")(output)
        return output

    def transformer_stack(sequence, name_prefix: str):
        if transformer_heads < 1:
            raise ValueError("--transformer-heads must be at least 1")
        if transformer_layers < 1:
            raise ValueError("--transformer-layers must be at least 1")
        key_dim = max(1, hidden_units // transformer_heads)
        output = layers.Dense(hidden_units, name=f"{name_prefix}_projection")(sequence)
        position_indices = np.arange(int(output.shape[1]), dtype=np.int32)[None, :]
        position_embedding = layers.Embedding(
            input_dim=int(output.shape[1]),
            output_dim=hidden_units,
            name=f"{name_prefix}_position_embedding",
        )(position_indices)
        output = layers.Add(name=f"{name_prefix}_position_add")([output, position_embedding])
        for block_index in range(transformer_layers):
            attention_input = layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{name_prefix}_block_{block_index}_attention_norm",
            )(output)
            attention = layers.MultiHeadAttention(
                num_heads=transformer_heads,
                key_dim=key_dim,
                dropout=dropout,
                name=f"{name_prefix}_block_{block_index}_attention",
            )(attention_input, attention_input)
            attention = layers.Dropout(dropout, name=f"{name_prefix}_block_{block_index}_attention_dropout")(
                attention
            )
            output = layers.Add(name=f"{name_prefix}_block_{block_index}_attention_residual")([output, attention])
            feed_forward_input = layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{name_prefix}_block_{block_index}_ff_norm",
            )(output)
            feed_forward = layers.Dense(
                transformer_ff_dim,
                activation="relu",
                name=f"{name_prefix}_block_{block_index}_ff_expand",
            )(feed_forward_input)
            feed_forward = layers.Dropout(dropout, name=f"{name_prefix}_block_{block_index}_ff_dropout")(
                feed_forward
            )
            feed_forward = layers.Dense(hidden_units, name=f"{name_prefix}_block_{block_index}_ff_project")(
                feed_forward
            )
            output = layers.Add(name=f"{name_prefix}_block_{block_index}_ff_residual")([output, feed_forward])
        return layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_output_norm")(output)

    stage_input = keras.Input(shape=(window_minutes,), dtype="int32", name="stage_sequence")
    context_input = keras.Input(shape=(context_dim,), dtype="float32", name="context")
    x = layers.Embedding(input_dim=stage_count, output_dim=embedding_dim, name="stage_embedding")(stage_input)
    if model_type == "gru":
        x = layers.GRU(hidden_units, dropout=dropout, name="stage_gru")(x)
    elif model_type == "lstm":
        x = layers.LSTM(hidden_units, dropout=dropout, name="stage_lstm")(x)
    elif model_type == "bigru":
        x = layers.Bidirectional(
            layers.GRU(hidden_units, dropout=dropout),
            name="stage_bigru",
        )(x)
    elif model_type == "bigru_attention":
        x = layers.Bidirectional(
            layers.GRU(hidden_units, dropout=dropout, return_sequences=True),
            name="stage_bigru",
        )(x)
        x = attention_pool(x, "bigru_attention")
    elif model_type == "cnn":
        x = layers.Conv1D(
            hidden_units,
            kernel_size=conv_kernel_size,
            padding="same",
            activation="relu",
            name="stage_conv",
        )(x)
        x = layers.GlobalMaxPooling1D(name="stage_pool")(x)
        x = layers.Dropout(dropout)(x)
    elif model_type == "cnn_gru":
        x = layers.Conv1D(
            conv_filters,
            kernel_size=conv_kernel_size,
            padding="same",
            activation="relu",
            name="stage_conv",
        )(x)
        x = layers.GRU(hidden_units, dropout=dropout, name="stage_gru")(x)
    elif model_type == "tcn":
        x = tcn_stack(x, "tcn")
        x = layers.GlobalAveragePooling1D(name="tcn_pool")(x)
    elif model_type == "tcn_attention":
        x = tcn_stack(x, "tcn")
        x = attention_pool(x, "tcn_attention")
    elif model_type == "inception_time":
        branch_filters = max(8, conv_filters)
        branches = [
            layers.Conv1D(
                branch_filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=f"inception_conv_{kernel_size}",
            )(x)
            for kernel_size in (3, 5, 9)
        ]
        pooled = layers.MaxPooling1D(pool_size=3, strides=1, padding="same", name="inception_pool_source")(x)
        pooled = layers.Conv1D(
            branch_filters,
            kernel_size=1,
            padding="same",
            activation="relu",
            name="inception_pool_projection",
        )(pooled)
        x = layers.Concatenate(name="inception_concat")(branches + [pooled])
        x = layers.Dropout(dropout, name="inception_dropout")(x)
        x = layers.GlobalAveragePooling1D(name="inception_pool")(x)
    elif model_type == "transformer":
        x = transformer_stack(x, "transformer")
        x = layers.GlobalAveragePooling1D(name="transformer_pool")(x)
    elif model_type == "transformer_tcn":
        x = transformer_stack(x, "transformer")
        x = tcn_stack(x, "post_transformer_tcn")
        x = layers.GlobalAveragePooling1D(name="transformer_tcn_pool")(x)
    elif model_type == "patchtst_lite":
        x = layers.Dense(hidden_units, name="patch_projection_input")(x)
        x = layers.Conv1D(
            hidden_units,
            kernel_size=min(10, window_minutes),
            strides=5,
            padding="same",
            activation="relu",
            name="patch_embedding",
        )(x)
        x = transformer_stack(x, "patch_transformer")
        x = layers.GlobalAveragePooling1D(name="patch_transformer_pool")(x)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    context_branch = layers.Dense(context_units, activation="relu", name="context_dense")(context_input)
    merged = layers.Concatenate(name="merge")([x, context_branch])
    merged = layers.Dense(dense_units, activation="relu", name="dense")(merged)
    merged = layers.Dropout(dropout)(merged)
    output = layers.Dense(1, activation="sigmoid", name="p_next_deep_10m")(merged)
    model = keras.Model(inputs=[stage_input, context_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def metrics_for_split(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    rows = int(len(y_true))
    positive_count = int(y_true.sum())
    result: dict[str, Any] = {
        "rows": rows,
        "positive_count": positive_count,
        "positive_rate": round(float(positive_count / rows), 6) if rows else 0.0,
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    if y_true.nunique() > 1:
        result["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 6)
        result["pr_auc"] = round(float(average_precision_score(y_true, probabilities)), 6)
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def threshold_report(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
                "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
                "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
                "predicted_positive_count": int(predictions.sum()),
                "predicted_positive_rate": round(float(predictions.mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def prediction_rows(metadata: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = metadata.copy()
    rows["target"] = TARGET
    rows["actual"] = rows[TARGET].astype(int)
    rows["probability"] = probabilities
    rows["prediction"] = (probabilities >= threshold).astype(int)
    return rows.loc[:, [column for column in OUTPUT_COLUMNS if column in rows.columns]]


def available_context_columns(metadata: pd.DataFrame) -> list[str]:
    return [column for column in NUMERIC_CONTEXT_COLUMNS if column in metadata.columns]


def parse_tcn_dilations(value: str) -> tuple[int, ...]:
    try:
        dilations = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--tcn-dilations must contain comma-separated positive integers") from error
    if not dilations or any(dilation < 1 for dilation in dilations):
        raise ValueError("--tcn-dilations must contain comma-separated positive integers")
    return dilations


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.random_state)
    sequence_dir = args.sequence_dir
    output_dir = args.output_dir or sequence_dir / f"model_sequence_{args.model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_count = load_stage_count(sequence_dir)
    metadata, stages, context = load_dataset(sequence_dir, stage_count)
    context_columns = available_context_columns(metadata)
    train_mask = metadata["split"] == "train"
    validation_mask = metadata["split"] == "validation"
    test_mask = metadata["split"] == "test"
    train_context, scaled, context_stats = standardize_context(
        context[train_mask],
        context_columns,
        context[validation_mask],
        context[test_mask],
    )
    validation_context, test_context = scaled

    y_train = metadata.loc[train_mask, TARGET].astype(int)
    class_values = np.array([0, 1])
    class_weights = compute_class_weight(class_weight="balanced", classes=class_values, y=y_train.to_numpy())
    class_weight = {int(label): float(weight) for label, weight in zip(class_values, class_weights)}
    tcn_dilations = parse_tcn_dilations(args.tcn_dilations)

    model = build_model(
        args.model_type,
        window_minutes=stages.shape[1],
        stage_count=stage_count,
        context_dim=context.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_units=args.hidden_units,
        dense_units=args.dense_units,
        context_units=args.context_units,
        conv_filters=args.conv_filters,
        conv_kernel_size=args.conv_kernel_size,
        tcn_dilations=tcn_dilations,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        {"stage_sequence": stages[train_mask], "context": train_context},
        y_train,
        validation_data=(
            {"stage_sequence": stages[validation_mask], "context": validation_context},
            metadata.loc[validation_mask, TARGET].astype(int),
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    metrics: dict[str, Any] = {
        "sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "target": TARGET,
        "model": f"{args.model_type}_sequence_model",
        "threshold": args.threshold,
        "architecture": {
            "model_type": args.model_type,
            "embedding_dim": args.embedding_dim,
            "hidden_units": args.hidden_units,
            "dense_units": args.dense_units,
            "context_units": args.context_units,
            "conv_filters": args.conv_filters,
            "conv_kernel_size": args.conv_kernel_size,
            "tcn_dilations": list(tcn_dilations),
            "transformer_heads": args.transformer_heads,
            "transformer_layers": args.transformer_layers,
            "transformer_ff_dim": args.transformer_ff_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
        },
        "class_weight": class_weight,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    prediction_frames = []
    threshold_frames = []
    split_inputs = {
        "train": (train_mask, train_context),
        "validation": (validation_mask, validation_context),
        "test": (test_mask, test_context),
    }
    for split, (mask, split_context) in split_inputs.items():
        probabilities = model.predict(
            {"stage_sequence": stages[mask], "context": split_context},
            batch_size=args.batch_size,
            verbose=0,
        ).reshape(-1)
        y_true = metadata.loc[mask, TARGET].astype(int)
        metrics[split] = metrics_for_split(y_true, probabilities, args.threshold)
        rows = prediction_rows(metadata.loc[mask].copy(), probabilities, args.threshold)
        rows.insert(0, "split", split)
        prediction_frames.append(rows)
        thresholds = threshold_report(y_true, probabilities)
        thresholds.insert(0, "split", split)
        threshold_frames.append(thresholds)

    if args.predict_sequence_dir:
        scoring_metadata, scoring_stages, scoring_context = load_dataset(
            args.predict_sequence_dir,
            stage_count,
            context_columns,
        )
        _, scaled_scoring_list, _ = standardize_context(context[train_mask], context_columns, scoring_context)
        scoring_context = scaled_scoring_list[0]
        scoring_probabilities = model.predict(
            {"stage_sequence": scoring_stages, "context": scoring_context},
            batch_size=args.batch_size,
            verbose=0,
        ).reshape(-1)
        alarm_predictions = prediction_rows(scoring_metadata, scoring_probabilities, args.threshold)
        alarm_predictions.to_csv(output_dir / "alarm_predictions_long.csv", index=False)
        metrics["alarm_predictions"] = {
            "sequence_dir": str(args.predict_sequence_dir),
            "file": str(output_dir / "alarm_predictions_long.csv"),
            "rows": int(len(alarm_predictions)),
            "positive_count": int(alarm_predictions["actual"].sum()),
            "positive_ratio": round(float(alarm_predictions["actual"].mean()), 6)
            if len(alarm_predictions)
            else 0.0,
        }
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "sequence_predictions.csv", index=False)
    pd.concat(threshold_frames, ignore_index=True).to_csv(output_dir / "sequence_threshold_report.csv", index=False)
    model.save(output_dir / "sequence_model.keras")
    with (output_dir / "context_scaler.json").open("w", encoding="utf-8") as handle:
        json.dump(context_stats, handle, indent=2)
    with (output_dir / "sequence_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"Wrote {output_dir / 'sequence_model.keras'}")
    print(f"Wrote {output_dir / 'sequence_metrics.json'}")
    print(f"Wrote {output_dir / 'sequence_threshold_report.csv'}")
    if args.predict_sequence_dir:
        print(f"Wrote {output_dir / 'alarm_predictions_long.csv'}")


if __name__ == "__main__":
    main()
