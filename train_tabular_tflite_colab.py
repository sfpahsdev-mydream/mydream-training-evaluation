#!/usr/bin/env python3
"""Train a TFLite-friendly MyDream tabular model.

This is the Android/Wear OS deployment-oriented alternative to the current
LightGBM Phase 1 model. It uses the same leakage-safe tabular feature set, but
trains a small Keras MLP so the output can be converted directly to TensorFlow
Lite and combined with the GRU score on device.

Install in Colab/server:

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
DEFAULT_INPUT_DIR = Path("out/verify_week_period_profile")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "model_tabular_tflite"
DEFAULT_THRESHOLD = 0.5
NUMERIC_FEATURES = (
    "minutes_before_deadline",
    "elapsed_sleep_minutes",
    "target_wake_hour_sin",
    "target_wake_hour_cos",
    "time_of_day_sin",
    "time_of_day_cos",
    "minutes_since_stage_start",
    "minutes_since_last_deep",
    "deep_cycle_position",
    "recent_30m_awake_minutes",
    "recent_30m_light_minutes",
    "recent_30m_deep_minutes",
    "recent_30m_rem_minutes",
)
CATEGORICAL_FEATURES = ("stage_at_candidate", "previous_stage", "day_of_week")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyDream tabular TFLite model.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--training-candidates-file", default="training_candidates_1min.csv")
    parser.add_argument("--alarm-candidates-file", default="alarm_candidates_1min.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--dense-units", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def chronological_session_split(
    candidates: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    session_dates = (
        candidates.assign(candidate_dt=pd.to_datetime(candidates["candidate_time"], utc=True))
        .groupby("session_id", as_index=False)["candidate_dt"]
        .min()
        .sort_values("candidate_dt")
    )
    if validation_ratio + test_ratio >= 1:
        raise ValueError("--validation-ratio + --test-ratio must be less than 1")

    test_count = max(1, round(len(session_dates) * test_ratio))
    validation_count = max(1, round(len(session_dates) * validation_ratio))
    train_sessions = session_dates.iloc[: -(validation_count + test_count)]["session_id"]
    validation_sessions = session_dates.iloc[-(validation_count + test_count) : -test_count]["session_id"]
    test_sessions = session_dates.iloc[-test_count:]["session_id"]

    return (
        candidates.index[candidates["session_id"].isin(train_sessions)],
        candidates.index[candidates["session_id"].isin(validation_sessions)],
        candidates.index[candidates["session_id"].isin(test_sessions)],
    )


def build_feature_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = candidates.loc[:, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Unknown").astype("category")
    return pd.get_dummies(features, columns=list(CATEGORICAL_FEATURES), dummy_na=False)


def align_feature_matrix(features: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    return features.reindex(columns=columns, fill_value=0)


def standardize_features(
    train_features: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    mean = train_features.mean(axis=0).to_numpy(dtype=np.float32)
    std = train_features.std(axis=0).to_numpy(dtype=np.float32)
    std[std == 0] = 1.0
    train_values = train_features.to_numpy(dtype=np.float32)
    scaled_train = (train_values - mean) / std
    scaled_others = [
        (other.to_numpy(dtype=np.float32) - mean) / std
        for other in others
    ]
    return scaled_train, scaled_others, {
        "columns": list(train_features.columns),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target": TARGET,
    }


def build_model(
    input_dim: int,
    hidden_units: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
) -> Any:
    import tensorflow as tf

    keras = tf.keras
    layers = keras.layers

    tabular_input = keras.Input(shape=(input_dim,), dtype="float32", name="tabular_features")
    x = layers.Dense(hidden_units, activation="relu", name="tabular_dense_1")(tabular_input)
    x = layers.Dropout(dropout, name="tabular_dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="tabular_dense_2")(x)
    output = layers.Dense(1, activation="sigmoid", name="p_deep_soon_tabular")(x)
    model = keras.Model(inputs=tabular_input, outputs=output)
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
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    if y_true.nunique() > 1:
        result["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 6)
        result["pr_auc"] = round(float(average_precision_score(y_true, probabilities)), 6)
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def prediction_rows(candidates: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = candidates.copy()
    rows["target"] = TARGET
    rows["actual"] = rows[TARGET].astype(int)
    rows["probability"] = probabilities
    rows["prediction"] = (probabilities >= threshold).astype(int)
    return rows.loc[:, [column for column in OUTPUT_COLUMNS if column in rows.columns]]


def threshold_report(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8):
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


def convert_float32(tf: Any, model: Any, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    output_path.write_bytes(converter.convert())


def convert_float16(tf: Any, model: Any, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output_path.write_bytes(converter.convert())


def run_tflite(tf: Any, tflite_path: Path, features: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    outputs: list[float] = []
    for row in features:
        interpreter.set_tensor(input_detail["index"], row.reshape(input_detail["shape"]).astype(input_detail["dtype"]))
        interpreter.invoke()
        outputs.append(float(interpreter.get_tensor(output_detail["index"]).reshape(-1)[0]))
    return np.asarray(outputs, dtype=np.float32)


def verify_tflite(tf: Any, model: Any, tflite_path: Path, features: np.ndarray, sample_rows: int = 512) -> dict[str, Any]:
    row_count = min(sample_rows, len(features))
    sample = features[:row_count]
    keras_output = model.predict(sample, batch_size=min(512, row_count), verbose=0).reshape(-1)
    tflite_output = run_tflite(tf, tflite_path, sample)
    diff = np.abs(keras_output - tflite_output)
    return {
        "sample_rows": int(row_count),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "keras_min": float(keras_output.min()) if len(keras_output) else 0.0,
        "keras_max": float(keras_output.max()) if len(keras_output) else 0.0,
        "tflite_min": float(tflite_output.min()) if len(tflite_output) else 0.0,
        "tflite_max": float(tflite_output.max()) if len(tflite_output) else 0.0,
    }


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.input_dir / args.training_candidates_file)
    alarm_candidates = pd.read_csv(args.input_dir / args.alarm_candidates_file)
    if TARGET not in candidates.columns or TARGET not in alarm_candidates.columns:
        raise ValueError(f"Missing required target column: {TARGET}")

    features = build_feature_matrix(candidates)
    alarm_features = align_feature_matrix(build_feature_matrix(alarm_candidates), features.columns)
    train_index, validation_index, test_index = chronological_session_split(
        candidates,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    train_x, scaled, scaler = standardize_features(
        features.loc[train_index],
        features.loc[validation_index],
        features.loc[test_index],
        alarm_features,
    )
    validation_x, test_x, alarm_x = scaled

    y_train = candidates.loc[train_index, TARGET].astype(int)
    class_values = np.array([0, 1])
    class_weights = compute_class_weight(class_weight="balanced", classes=class_values, y=y_train.to_numpy())
    class_weight = {int(label): float(weight) for label, weight in zip(class_values, class_weights)}

    model = build_model(
        input_dim=train_x.shape[1],
        hidden_units=args.hidden_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train_x,
        y_train,
        validation_data=(validation_x, candidates.loc[validation_index, TARGET].astype(int)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    metrics: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target": TARGET,
        "threshold": args.threshold,
        "class_weight": class_weight,
        "architecture": {
            "model": "tabular_mlp",
            "hidden_units": args.hidden_units,
            "dense_units": args.dense_units,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "encoded_feature_count": int(train_x.shape[1]),
        },
        "features": scaler,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    prediction_frames = []
    threshold_frames = []
    split_data = {
        "train": (train_index, train_x),
        "validation": (validation_index, validation_x),
        "test": (test_index, test_x),
    }
    for split, (index, split_x) in split_data.items():
        probabilities = model.predict(split_x, batch_size=args.batch_size, verbose=0).reshape(-1)
        y_true = candidates.loc[index, TARGET].astype(int)
        metrics[split] = metrics_for_split(y_true, probabilities, args.threshold)
        rows = prediction_rows(candidates.loc[index].copy(), probabilities, args.threshold)
        rows.insert(0, "split", split)
        prediction_frames.append(rows)
        thresholds = threshold_report(y_true, probabilities)
        thresholds.insert(0, "split", split)
        threshold_frames.append(thresholds)

    alarm_probabilities = model.predict(alarm_x, batch_size=args.batch_size, verbose=0).reshape(-1)
    alarm_predictions = prediction_rows(alarm_candidates, alarm_probabilities, args.threshold)
    alarm_predictions.to_csv(args.output_dir / "alarm_predictions_long.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "tabular_predictions.csv", index=False)
    pd.concat(threshold_frames, ignore_index=True).to_csv(args.output_dir / "tabular_threshold_report.csv", index=False)

    model.save(args.output_dir / "tabular_model.keras")
    with (args.output_dir / "tabular_feature_scaler.json").open("w", encoding="utf-8") as handle:
        json.dump(scaler, handle, indent=2)

    float32_path = args.output_dir / "tabular_model_float32.tflite"
    convert_float32(tf, model, float32_path)
    manifest: dict[str, Any] = {
        "model": "tabular_mlp",
        "keras_model": str(args.output_dir / "tabular_model.keras"),
        "float32_tflite": str(float32_path),
        "float32_size_bytes": float32_path.stat().st_size,
        "feature_scaler": str(args.output_dir / "tabular_feature_scaler.json"),
        "input_tensor": "tabular_features",
        "output_tensor": "p_deep_soon_tabular",
        "threshold": args.threshold,
        "alarm_predictions": str(args.output_dir / "alarm_predictions_long.csv"),
    }
    if args.float16:
        float16_path = args.output_dir / "tabular_model_float16.tflite"
        convert_float16(tf, model, float16_path)
        manifest["float16_tflite"] = str(float16_path)
        manifest["float16_size_bytes"] = float16_path.stat().st_size
    if not args.skip_verify:
        manifest["float32_verification"] = verify_tflite(tf, model, float32_path, alarm_x)

    with (args.output_dir / "tabular_tflite_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with (args.output_dir / "tabular_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"Wrote {args.output_dir / 'tabular_model.keras'}")
    print(f"Wrote {float32_path}")
    print(f"Wrote {args.output_dir / 'alarm_predictions_long.csv'}")
    print(f"Wrote {args.output_dir / 'tabular_tflite_manifest.json'}")


if __name__ == "__main__":
    main()
