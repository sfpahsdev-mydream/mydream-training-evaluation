#!/usr/bin/env python3
"""Train and evaluate first MyDream LightGBM candidate models.

This script is intended for Google Colab or a server environment.

Expected input files:
- candidates_5min.csv
- training_sessions.csv
- summary.json

Install dependencies in Colab first:

    !pip install lightgbm pandas scikit-learn

Example:

    python train_lightgbm_colab.py \
      --input-dir out/mydream_sleep_2026-05-17 \
      --output-dir out/mydream_sleep_2026-05-17/model_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGETS = ("label_wakeable_at_candidate", "label_deep_soon")

# Keep this feature list leakage-safe for the first model.
# Do not add next_stage here: it describes future sleep-stage information.
NUMERIC_FEATURES = (
    "minutes_before_deadline",
    "elapsed_sleep_minutes",
    "minutes_since_stage_start",
    "recent_30m_awake_minutes",
    "recent_30m_light_minutes",
    "recent_30m_deep_minutes",
    "recent_30m_rem_minutes",
    "recent_30m_unknown_minutes",
)
CATEGORICAL_FEATURES = ("stage_at_candidate", "previous_stage")
EXCLUDED_LEAKAGE_COLUMNS = (
    "next_stage",
    "candidate_time",
    "deadline_time",
    "session_id",
    "label_wakeable",
    "label_wakeable_window",
    "label_wakeable_at_candidate",
    "label_deep_soon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyDream LightGBM models.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def chronological_session_split(
    candidates: pd.DataFrame,
    test_ratio: float,
) -> tuple[pd.Index, pd.Index]:
    session_dates = (
        candidates.assign(candidate_dt=pd.to_datetime(candidates["candidate_time"], utc=True))
        .groupby("session_id", as_index=False)["candidate_dt"]
        .min()
        .sort_values("candidate_dt")
    )
    test_count = max(1, round(len(session_dates) * test_ratio))
    train_sessions = session_dates.iloc[:-test_count]["session_id"]
    test_sessions = session_dates.iloc[-test_count:]["session_id"]

    train_index = candidates.index[candidates["session_id"].isin(train_sessions)]
    test_index = candidates.index[candidates["session_id"].isin(test_sessions)]
    return train_index, test_index


def build_feature_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = candidates.loc[:, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return pd.get_dummies(features, columns=list(CATEGORICAL_FEATURES), dummy_na=False)


def target_baseline(target: str, row_count: int) -> list[int]:
    if target == "label_deep_soon":
        return [0] * row_count
    return [1] * row_count


def metrics_dict(y_true: pd.Series, y_pred: list[int], y_prob: list[float] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
    return metrics


def train_target(
    target: str,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = candidates[target].astype(int)
    train_x = features.loc[train_index]
    test_x = features.loc[test_index]
    train_y = y.loc[train_index]
    test_y = y.loc[test_index]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=2,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(test_x, test_y)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    baseline_predictions = target_baseline(target, len(test_y))

    importances = (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance": model.feature_importances_,
            },
        )
        .sort_values("importance", ascending=False)
        .head(25)
    )

    summary = {
        "target": target,
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "positive_train": int(train_y.sum()),
        "positive_test": int(test_y.sum()),
        "lightgbm": metrics_dict(test_y, predictions.tolist(), probabilities.tolist()),
        "baseline": metrics_dict(test_y, baseline_predictions),
        "feature_importance_top25": importances.to_dict(orient="records"),
    }

    prediction_rows = candidates.loc[test_index, ["session_id", "candidate_time", "deadline_time"]].copy()
    prediction_rows["target"] = target
    prediction_rows["actual"] = test_y.values
    prediction_rows["probability"] = probabilities
    prediction_rows["prediction"] = predictions
    return summary, prediction_rows


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "model_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(input_dir / "candidates_5min.csv")
    features = build_feature_matrix(candidates)
    train_index, test_index = chronological_session_split(candidates, args.test_ratio)

    summaries = []
    prediction_frames = []
    for target in TARGETS:
        summary, predictions = train_target(
            target=target,
            candidates=candidates,
            features=features,
            train_index=train_index,
            test_index=test_index,
            random_state=args.random_state,
        )
        summaries.append(summary)
        prediction_frames.append(predictions)

    result = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split": {
            "method": "chronological_session_split",
            "test_ratio": args.test_ratio,
            "train_sessions": int(candidates.loc[train_index, "session_id"].nunique()),
            "test_sessions": int(candidates.loc[test_index, "session_id"].nunique()),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
        },
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "excluded_leakage_columns": list(EXCLUDED_LEAKAGE_COLUMNS),
            "encoded_feature_count": int(features.shape[1]),
        },
        "targets": summaries,
    }

    (output_dir / "model_eval_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(result["split"], indent=2))
    print(f"Wrote {output_dir / 'model_eval_summary.json'}")
    print(f"Wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
