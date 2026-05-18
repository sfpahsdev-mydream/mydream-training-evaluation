#!/usr/bin/env python3
"""Train the first MyDream Phase 2 sequence baseline.

This is a lightweight local baseline that uses flattened one-hot stage
sequences plus candidate context metadata. It is not the final GRU/CNN model,
but it validates that the Phase 2 sequence dataset carries useful signal before
moving heavier training to Colab or a server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "label_deep_soon"
DEFAULT_THRESHOLD = 0.5
STAGE_COUNT = 5
NUMERIC_METADATA_FEATURES = [
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
    "sequence_known_ratio",
]
CATEGORICAL_METADATA_FEATURES = [
    "stage_at_candidate",
    "day_of_week",
]
OUTPUT_COLUMNS = [
    "sequence_id",
    "split",
    "session_id",
    "candidate_time",
    "deadline_time",
    "stage_at_candidate",
    "stage_at_deadline",
    "minutes_before_deadline",
    "sequence_known_ratio",
    TARGET,
    "probability",
    "prediction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyDream sequence baseline model.")
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--predict-sequence-dir",
        type=Path,
        help="Optional second sequence dataset to score, such as alarm-window candidates.",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_stage_count(sequence_dir: Path) -> int:
    vocab_path = sequence_dir / "stage_vocab.json"
    if not vocab_path.exists():
        return STAGE_COUNT
    with vocab_path.open("r", encoding="utf-8") as handle:
        stage_to_id = json.load(handle)
    return max(int(value) for value in stage_to_id.values()) + 1


def one_hot_sequences(sequences: np.ndarray, stage_count: int) -> np.ndarray:
    encoded = np.eye(stage_count, dtype=np.float32)[sequences.astype(np.int64)]
    return encoded.reshape(sequences.shape[0], sequences.shape[1] * stage_count)


def build_features(sequence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(sequence_dir / "sequence_metadata.csv")
    sequences = np.load(sequence_dir / "sequence_stage_ids.npy")
    if len(metadata) != len(sequences):
        raise ValueError(f"metadata rows ({len(metadata)}) do not match sequences ({len(sequences)})")
    stage_count = load_stage_count(sequence_dir)
    sequence_features = one_hot_sequences(sequences, stage_count)
    sequence_columns = [
        f"stage_t{t:02d}_{stage_id}"
        for t in range(sequences.shape[1])
        for stage_id in range(stage_count)
    ]
    sequence_frame = pd.DataFrame(sequence_features, columns=sequence_columns)

    missing_numeric = sorted(set(NUMERIC_METADATA_FEATURES) - set(metadata.columns))
    missing_categorical = sorted(set(CATEGORICAL_METADATA_FEATURES) - set(metadata.columns))
    if missing_numeric or missing_categorical:
        raise ValueError(
            f"Missing metadata columns: numeric={missing_numeric}, categorical={missing_categorical}"
        )

    context = metadata.loc[:, NUMERIC_METADATA_FEATURES + CATEGORICAL_METADATA_FEATURES].copy()
    for column in NUMERIC_METADATA_FEATURES:
        context[column] = pd.to_numeric(context[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_METADATA_FEATURES:
        context[column] = context[column].fillna("Unknown").astype(str)

    features = pd.concat([sequence_frame, context], axis=1)
    return metadata, features


def build_scoring_features(sequence_dir: Path, expected_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata, features = build_features(sequence_dir)
    missing = sorted(set(expected_columns) - set(features.columns))
    if missing:
        raise ValueError(f"Scoring dataset is missing expected feature columns: {missing}")
    return metadata, features.loc[:, expected_columns]


def build_pipeline(sequence_feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("sequence", "passthrough", sequence_feature_columns),
            ("numeric", StandardScaler(), NUMERIC_METADATA_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_METADATA_FEATURES,
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        solver="lbfgs",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def metrics_for_split(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    positive_count = int(y_true.sum())
    rows = int(len(y_true))
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


def prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    split: str,
    threshold: float,
) -> pd.DataFrame:
    mask = metadata["split"] == split
    rows = metadata.loc[mask, [column for column in OUTPUT_COLUMNS if column in metadata.columns]].copy()
    rows["probability"] = probabilities
    rows["prediction"] = (probabilities >= threshold).astype(int)
    return rows


def alarm_prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = metadata.copy()
    rows["target"] = TARGET
    rows["actual"] = rows[TARGET].astype(int)
    rows["probability"] = probabilities
    rows["prediction"] = (probabilities >= threshold).astype(int)
    columns = [
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
    return rows.loc[:, [column for column in columns if column in rows.columns]]


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir
    output_dir = args.output_dir or sequence_dir / "model_sequence_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, features = build_features(sequence_dir)
    training_feature_columns = list(features.columns)
    sequence_feature_columns = [column for column in features.columns if column.startswith("stage_t")]
    train_mask = metadata["split"] == "train"
    validation_mask = metadata["split"] == "validation"
    test_mask = metadata["split"] == "test"

    pipeline = build_pipeline(sequence_feature_columns)
    pipeline.set_params(model__max_iter=args.max_iter)
    pipeline.fit(features.loc[train_mask], metadata.loc[train_mask, TARGET])

    metrics: dict[str, Any] = {
        "sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "target": TARGET,
        "model": "logistic_regression_flattened_sequence_baseline",
        "threshold": args.threshold,
        "features": {
            "window_stage_one_hot_features": len(sequence_feature_columns),
            "numeric_metadata_features": NUMERIC_METADATA_FEATURES,
            "categorical_metadata_features": CATEGORICAL_METADATA_FEATURES,
        },
    }
    prediction_frames = []
    threshold_frames = []
    for split, mask in (("train", train_mask), ("validation", validation_mask), ("test", test_mask)):
        probabilities = pipeline.predict_proba(features.loc[mask])[:, 1]
        y_true = metadata.loc[mask, TARGET]
        metrics[split] = metrics_for_split(y_true, probabilities, args.threshold)
        split_predictions = prediction_frame(metadata, probabilities, split, args.threshold)
        prediction_frames.append(split_predictions)
        split_thresholds = threshold_report(y_true, probabilities)
        split_thresholds.insert(0, "split", split)
        threshold_frames.append(split_thresholds)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True)

    joblib.dump(pipeline, output_dir / "sequence_baseline_logistic.joblib")
    predictions.to_csv(output_dir / "sequence_predictions.csv", index=False)
    thresholds.to_csv(output_dir / "sequence_threshold_report.csv", index=False)

    if args.predict_sequence_dir:
        scoring_metadata, scoring_features = build_scoring_features(
            args.predict_sequence_dir,
            training_feature_columns,
        )
        scoring_probabilities = pipeline.predict_proba(scoring_features)[:, 1]
        alarm_predictions = alarm_prediction_frame(scoring_metadata, scoring_probabilities, args.threshold)
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

    with (output_dir / "sequence_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"Wrote {output_dir / 'sequence_baseline_logistic.joblib'}")
    print(f"Wrote {output_dir / 'sequence_metrics.json'}")
    print(f"Wrote {output_dir / 'sequence_threshold_report.csv'}")
    print(f"Wrote {output_dir / 'sequence_predictions.csv'}")
    if args.predict_sequence_dir:
        print(f"Wrote {output_dir / 'alarm_predictions_long.csv'}")


if __name__ == "__main__":
    main()
