#!/usr/bin/env python3
"""Train and evaluate first MyDream LightGBM candidate models.

This script is intended for Google Colab or a server environment.

Expected input files:
- training_candidates_1min.csv
- alarm_candidates_1min.csv

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

import joblib
import lightgbm as lgb
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


TARGETS = ("label_deep_soon",)
DEEP_SOON_SOFT_RISK_THRESHOLD = 0.10
DEEP_SOON_HARD_RISK_THRESHOLD = 0.20
SMART_DEEP_SOON_THRESHOLD = 0.50
TARGET_WINDOW_MINUTES = 10.0
TARGET_WINDOW_BONUS = 0.10
MAX_EARLY_WINDOW_PENALTY = 0.10

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
FUTURE_CATEGORICAL_FEATURES = ("next_stage",)
EXCLUDED_LEAKAGE_COLUMNS = (
    "next_stage",
    "candidate_time",
    "deadline_time",
    "wake_time_source",
    "stage_at_deadline",
    "next_deep_start",
    "session_id",
    "label_wakeable",
    "label_wakeable_window",
    "label_wakeable_at_candidate",
    "label_deep_soon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyDream LightGBM models.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--training-candidates-file", default="training_candidates_1min.csv")
    parser.add_argument("--alarm-candidates-file", default="alarm_candidates_1min.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-sample-rows", type=int, default=300)
    parser.add_argument("--include-next-stage", action="store_true")
    parser.add_argument("--smart-deep-soon-threshold", type=float, default=SMART_DEEP_SOON_THRESHOLD)
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

    train_index = candidates.index[candidates["session_id"].isin(train_sessions)]
    validation_index = candidates.index[candidates["session_id"].isin(validation_sessions)]
    test_index = candidates.index[candidates["session_id"].isin(test_sessions)]
    return train_index, validation_index, test_index


def feature_columns(include_next_stage: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    categorical = CATEGORICAL_FEATURES + (FUTURE_CATEGORICAL_FEATURES if include_next_stage else ())
    return NUMERIC_FEATURES, categorical


def build_feature_matrix(candidates: pd.DataFrame, include_next_stage: bool) -> pd.DataFrame:
    numeric_features, categorical_features = feature_columns(include_next_stage)
    required = set(numeric_features + categorical_features)
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = candidates.loc[:, list(numeric_features + categorical_features)].copy()
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    for column in categorical_features:
        features[column] = features[column].fillna("Unknown").astype("category")
    return pd.get_dummies(features, columns=list(categorical_features), dummy_na=False)


def align_feature_matrix(features: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    return features.reindex(columns=columns, fill_value=0)


def target_baseline(target: str, row_count: int) -> list[int]:
    return [0] * row_count


def metrics_dict(y_true: pd.Series, y_pred: list[int], y_prob: list[float] | None = None) -> dict[str, Any]:
    positive_count = int(y_true.sum())
    negative_count = int(len(y_true) - positive_count)
    has_both_classes = len(set(y_true)) > 1
    metrics: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": round(float(positive_count / len(y_true)), 6) if len(y_true) else 0.0,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": (
            round(float(balanced_accuracy_score(y_true, y_pred)), 6)
            if has_both_classes
            else round(float(accuracy_score(y_true, y_pred)), 6)
        ),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "positive_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_prob is not None and has_both_classes:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 6)
    return metrics


def threshold_metrics(target: str, y_true: pd.Series, y_prob: list[float]) -> list[dict[str, Any]]:
    rows = []
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        y_pred = [1 if probability >= threshold else 0 for probability in y_prob]
        predicted_positive = sum(y_pred)
        rows.append(
            {
                "target": target,
                "threshold": threshold,
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
                "predicted_positive_count": int(predicted_positive),
                "predicted_positive_rate": round(float(predicted_positive / len(y_true)), 6),
            }
        )
    return rows


def train_target(
    target: str,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    train_index: pd.Index,
    validation_index: pd.Index,
    test_index: pd.Index,
    random_state: int,
    output_dir: Path,
) -> tuple[lgb.LGBMClassifier, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = candidates[target].astype(int)
    train_x = features.loc[train_index]
    validation_x = features.loc[validation_index]
    test_x = features.loc[test_index]
    train_y = y.loc[train_index]
    validation_y = y.loc[validation_index]
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
        eval_set=[(validation_x, validation_y)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    validation_probabilities = model.predict_proba(validation_x)[:, 1]
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    baseline_predictions = target_baseline(target, len(test_y))
    model_path = output_dir / f"{target}.joblib"
    booster_path = output_dir / f"{target}.lightgbm.txt"
    joblib.dump(model, model_path)
    model.booster_.save_model(booster_path)

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
        "validation_rows": int(len(validation_index)),
        "test_rows": int(len(test_index)),
        "positive_train": int(train_y.sum()),
        "positive_validation": int(validation_y.sum()),
        "positive_test": int(test_y.sum()),
        "positive_train_rate": round(float(train_y.mean()), 6),
        "positive_validation_rate": round(float(validation_y.mean()), 6),
        "positive_test_rate": round(float(test_y.mean()), 6),
        "model_file": str(model_path),
        "booster_file": str(booster_path),
        "validation_thresholds": threshold_metrics(
            target,
            validation_y,
            validation_probabilities.tolist(),
        ),
        "lightgbm": metrics_dict(test_y, predictions.tolist(), probabilities.tolist()),
        "baseline": metrics_dict(test_y, baseline_predictions),
        "feature_importance_top25": importances.to_dict(orient="records"),
    }

    prediction_columns = [
        "session_id",
        "candidate_time",
        "deadline_time",
        "stage_at_candidate",
        "stage_at_deadline",
        "next_deep_start",
    ]
    prediction_rows = candidates.loc[test_index, [column for column in prediction_columns if column in candidates]].copy()
    prediction_rows["target"] = target
    prediction_rows["actual"] = test_y.values
    prediction_rows["probability"] = probabilities
    prediction_rows["prediction"] = predictions
    threshold_rows = pd.DataFrame(threshold_metrics(target, test_y, probabilities.tolist()))
    return model, summary, prediction_rows, threshold_rows


def predict_target(
    target: str,
    model: lgb.LGBMClassifier,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    prediction_columns = [
        "session_id",
        "candidate_time",
        "deadline_time",
        "stage_at_candidate",
        "stage_at_deadline",
        "next_deep_start",
    ]
    rows = candidates.loc[:, [column for column in prediction_columns if column in candidates]].copy()
    rows["target"] = target
    rows["actual"] = candidates[target].astype(int).values if target in candidates else 0
    rows["probability"] = probabilities
    rows["prediction"] = predictions
    return rows


def prediction_index_columns(predictions: pd.DataFrame) -> list[str]:
    columns = ["session_id", "candidate_time", "deadline_time"]
    for column in ("stage_at_candidate", "stage_at_deadline", "next_deep_start"):
        if column in predictions.columns:
            columns.append(column)
    return columns


def build_prediction_sample(predictions: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    index_columns = prediction_index_columns(predictions)
    pivot_source = predictions.copy()
    for column in index_columns:
        pivot_source[column] = pivot_source[column].fillna("")
    wide = pivot_source.pivot_table(
        index=index_columns,
        columns="target",
        values=["actual", "probability", "prediction"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{target}" for value, target in wide.columns]
    sample = wide.reset_index().sort_values(["session_id", "candidate_time"]).copy()
    sample["deadline_urgency"] = (
        1
        - (
            pd.to_datetime(sample["deadline_time"], utc=True)
            - pd.to_datetime(sample["candidate_time"], utc=True)
        ).dt.total_seconds()
        / (30 * 60)
    ).clip(0, 1)
    sample["candidate_score_draft"] = sample["probability_label_deep_soon"] * (0.5 + 0.5 * sample["deadline_urgency"])
    sample = add_candidate_scoring_columns(sample)
    return sample.sort_values("score_weighted_after_risk", ascending=False).head(max_rows)


def add_candidate_scoring_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["minutes_before_deadline"] = (
        pd.to_datetime(candidates["deadline_time"], utc=True)
        - pd.to_datetime(candidates["candidate_time"], utc=True)
    ).dt.total_seconds() / 60
    candidates["deep_soon_soft_risk"] = (
        candidates["probability_label_deep_soon"] >= DEEP_SOON_SOFT_RISK_THRESHOLD
    )
    candidates["deep_soon_hard_risk"] = (
        candidates["probability_label_deep_soon"] >= DEEP_SOON_HARD_RISK_THRESHOLD
    )
    candidates["deep_soon_risk_level"] = "none"
    candidates.loc[candidates["deep_soon_soft_risk"], "deep_soon_risk_level"] = "soft"
    candidates.loc[candidates["deep_soon_hard_risk"], "deep_soon_risk_level"] = "hard"
    candidates["score_weighted"] = (
        0.75 * candidates["probability_label_deep_soon"]
        + 0.25 * candidates["deadline_urgency"]
    )
    candidates["score_multiply"] = candidates["probability_label_deep_soon"] * candidates["deadline_urgency"]
    candidates["risk_penalty"] = 0.0
    candidates["target_window_bonus"] = 0.0
    candidates.loc[
        candidates["minutes_before_deadline"] <= TARGET_WINDOW_MINUTES,
        "target_window_bonus",
    ] = TARGET_WINDOW_BONUS
    candidates["early_window_penalty"] = (
        ((candidates["minutes_before_deadline"] - TARGET_WINDOW_MINUTES).clip(lower=0) / 20)
        * MAX_EARLY_WINDOW_PENALTY
    ).clip(upper=MAX_EARLY_WINDOW_PENALTY)
    candidates["score_weighted_after_risk"] = (
        candidates["score_weighted"]
        + candidates["target_window_bonus"]
        - candidates["early_window_penalty"]
    ).clip(lower=0)
    return candidates


def build_session_top_candidates(predictions: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    index_columns = prediction_index_columns(predictions)
    pivot_source = predictions.copy()
    for column in index_columns:
        pivot_source[column] = pivot_source[column].fillna("")
    wide = pivot_source.pivot_table(
        index=index_columns,
        columns="target",
        values=["actual", "probability", "prediction"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{target}" for value, target in wide.columns]
    candidates = wide.reset_index().copy()
    candidates["deadline_urgency"] = (
        1
        - (
            pd.to_datetime(candidates["deadline_time"], utc=True)
            - pd.to_datetime(candidates["candidate_time"], utc=True)
        ).dt.total_seconds()
        / (30 * 60)
    ).clip(0, 1)
    candidates = add_candidate_scoring_columns(candidates)
    return (
        candidates.sort_values(["session_id", "score_weighted_after_risk"], ascending=[True, False])
        .groupby("session_id", as_index=False)
        .head(top_n)
        .sort_values(["session_id", "score_weighted_after_risk"], ascending=[True, False])
    )


def build_session_recommendation_summary(session_top_candidates: pd.DataFrame) -> pd.DataFrame:
    summary = (
        session_top_candidates.sort_values(
            ["session_id", "score_weighted_after_risk"],
            ascending=[True, False],
        )
        .groupby("session_id", as_index=False)
        .head(1)
        .copy()
    )
    summary["minutes_before_deadline"] = (
        pd.to_datetime(summary["deadline_time"], utc=True)
        - pd.to_datetime(summary["candidate_time"], utc=True)
    ).dt.total_seconds() / 60
    summary["selected_within_target_window"] = (
        summary["minutes_before_deadline"] <= TARGET_WINDOW_MINUTES
    )
    summary["selected_was_deep_soon"] = summary["actual_label_deep_soon"].astype(bool)
    summary["selected_rank"] = 1
    return summary[
        [
            "session_id",
            "candidate_time",
            "deadline_time",
            "minutes_before_deadline",
            "probability_label_deep_soon",
            "deep_soon_risk_level",
            "score_weighted",
            "risk_penalty",
            "target_window_bonus",
            "early_window_penalty",
            "score_weighted_after_risk",
            "selected_within_target_window",
            "selected_was_deep_soon",
            "selected_rank",
        ]
    ].rename(
        columns={
            "candidate_time": "selected_candidate_time",
            "probability_label_deep_soon": "p_deep_soon",
        }
    )


def recommendation_policy_metrics(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {
            "sessions": 0,
            "selected_deep_soon_rate": 0.0,
            "selected_within_target_window_rate": 0.0,
            "average_minutes_before_deadline": 0.0,
            "median_minutes_before_deadline": 0.0,
            "soft_or_hard_risk_selection_rate": 0.0,
            "hard_risk_selection_rate": 0.0,
        }

    risk_counts = summary["deep_soon_risk_level"].value_counts().to_dict()
    return {
        "sessions": int(len(summary)),
        "selected_deep_soon_count": int(summary["selected_was_deep_soon"].sum()),
        "selected_deep_soon_rate": round(float(summary["selected_was_deep_soon"].mean()), 6),
        "selected_within_target_window_count": int(summary["selected_within_target_window"].sum()),
        "selected_within_target_window_rate": round(
            float(summary["selected_within_target_window"].mean()),
            6,
        ),
        "average_minutes_before_deadline": round(float(summary["minutes_before_deadline"].mean()), 3),
        "median_minutes_before_deadline": round(float(summary["minutes_before_deadline"].median()), 3),
        "soft_or_hard_risk_selection_count": int((summary["deep_soon_risk_level"] != "none").sum()),
        "soft_or_hard_risk_selection_rate": round(
            float((summary["deep_soon_risk_level"] != "none").mean()),
            6,
        ),
        "hard_risk_selection_count": int((summary["deep_soon_risk_level"] == "hard").sum()),
        "hard_risk_selection_rate": round(
            float((summary["deep_soon_risk_level"] == "hard").mean()),
            6,
        ),
        "risk_level_counts": {str(key): int(value) for key, value in risk_counts.items()},
    }


def too_early_penalty(minutes_before_next_deep: float | None) -> str:
    if minutes_before_next_deep is None:
        return "fail_strong"
    if 0 <= minutes_before_next_deep <= 10:
        return "none"
    if minutes_before_next_deep <= 20:
        return "weak"
    if minutes_before_next_deep <= 30:
        return "medium"
    return "fail_strong"


def build_alarm_backtest_summary(predictions: pd.DataFrame, smart_threshold: float) -> pd.DataFrame:
    index_columns = prediction_index_columns(predictions)
    pivot_source = predictions.copy()
    for column in index_columns:
        pivot_source[column] = pivot_source[column].fillna("")
    wide = pivot_source.pivot_table(
        index=index_columns,
        columns="target",
        values=["actual", "probability", "prediction"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{target}" for value, target in wide.columns]
    candidates = wide.reset_index().copy()
    candidates["candidate_dt"] = pd.to_datetime(candidates["candidate_time"], utc=True)
    candidates["deadline_dt"] = pd.to_datetime(candidates["deadline_time"], utc=True)
    candidates["next_deep_dt"] = pd.to_datetime(candidates.get("next_deep_start"), utc=True, errors="coerce")
    candidates["minutes_before_deadline"] = (
        candidates["deadline_dt"] - candidates["candidate_dt"]
    ).dt.total_seconds() / 60

    smart_candidates = candidates[
        (candidates["candidate_dt"] < candidates["deadline_dt"])
        & (candidates["probability_label_deep_soon"] >= smart_threshold)
        & (~candidates["stage_at_candidate"].isin(["Deep", "Unknown"]))
    ].copy()
    smart_selected = (
        smart_candidates.sort_values(
            ["session_id", "candidate_dt", "probability_label_deep_soon"],
            ascending=[True, False, False],
        )
        .groupby("session_id", as_index=False)
        .head(1)
    )

    fallback_base = (
        candidates.sort_values(["session_id", "deadline_dt"])
        .groupby("session_id", as_index=False)
        .tail(1)
    )
    fallback_base = fallback_base[~fallback_base["session_id"].isin(smart_selected["session_id"])].copy()
    fallback_base["candidate_dt"] = fallback_base["deadline_dt"]
    fallback_base["candidate_time"] = fallback_base["deadline_time"]
    fallback_base["stage_at_candidate"] = fallback_base.get("stage_at_deadline", "Unknown")
    fallback_base["next_deep_dt"] = pd.NaT

    smart_selected["alarm_type"] = "smart"
    fallback_base["alarm_type"] = "fallback"
    selected = pd.concat([smart_selected, fallback_base], ignore_index=True, sort=False)
    selected["candidate_dt"] = pd.to_datetime(selected["candidate_dt"], utc=True)
    selected["deadline_dt"] = pd.to_datetime(selected["deadline_dt"], utc=True)
    selected["next_deep_dt"] = pd.to_datetime(selected["next_deep_dt"], utc=True, errors="coerce")
    selected["selected_alarm_time"] = selected["candidate_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    selected["stage_at_alarm"] = selected["stage_at_candidate"].fillna("Unknown")
    selected["next_deep_start"] = selected["next_deep_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    selected.loc[selected["next_deep_dt"].isna(), "next_deep_start"] = ""
    selected["minutes_before_next_deep"] = (
        selected["next_deep_dt"] - selected["candidate_dt"]
    ).dt.total_seconds() / 60
    selected.loc[selected["next_deep_dt"].isna(), "minutes_before_next_deep"] = pd.NA
    selected["deep_prevention_success"] = (
        (selected["alarm_type"] == "smart")
        & selected["minutes_before_next_deep"].between(0, 10, inclusive="both")
    )
    selected["too_early_penalty"] = [
        too_early_penalty(None if pd.isna(value) else float(value))
        for value in selected["minutes_before_next_deep"]
    ]
    selected.loc[selected["alarm_type"] == "fallback", "too_early_penalty"] = "fallback"
    selected["time_success"] = selected["candidate_dt"] <= selected["deadline_dt"]
    selected["sleep_quality_success"] = selected["stage_at_alarm"].isin(["Awake", "Light", "Rem"]).astype(int)

    return selected[
        [
            "session_id",
            "selected_alarm_time",
            "alarm_type",
            "stage_at_alarm",
            "next_deep_start",
            "minutes_before_next_deep",
            "deep_prevention_success",
            "too_early_penalty",
            "time_success",
            "sleep_quality_success",
            "deadline_time",
            "probability_label_deep_soon",
        ]
    ].sort_values(["session_id"])


def alarm_backtest_metrics(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"sessions": 0}
    smart = summary["alarm_type"] == "smart"
    return {
        "sessions": int(len(summary)),
        "smart_alarm_count": int(smart.sum()),
        "fallback_alarm_count": int((~smart).sum()),
        "smart_alarm_rate": round(float(smart.mean()), 6),
        "deep_prevention_success_count": int(summary["deep_prevention_success"].sum()),
        "deep_prevention_success_rate": round(float(summary["deep_prevention_success"].mean()), 6),
        "time_success_rate": round(float(summary["time_success"].mean()), 6),
        "sleep_quality_success_rate": round(float(summary["sleep_quality_success"].mean()), 6),
        "too_early_penalty_counts": {
            str(key): int(value) for key, value in summary["too_early_penalty"].value_counts().to_dict().items()
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "model_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_candidates_path = input_dir / args.training_candidates_file
    alarm_candidates_path = input_dir / args.alarm_candidates_file
    candidates = pd.read_csv(training_candidates_path)
    alarm_candidates = pd.read_csv(alarm_candidates_path)
    missing_targets = sorted(set(TARGETS) - set(candidates.columns))
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")
    features = build_feature_matrix(candidates, args.include_next_stage)
    alarm_features = align_feature_matrix(
        build_feature_matrix(alarm_candidates, args.include_next_stage),
        features.columns,
    )
    train_index, validation_index, test_index = chronological_session_split(
        candidates,
        args.validation_ratio,
        args.test_ratio,
    )

    summaries = []
    prediction_frames = []
    alarm_prediction_frames = []
    threshold_frames = []
    for target in TARGETS:
        model, summary, predictions, threshold_report = train_target(
            target=target,
            candidates=candidates,
            features=features,
            train_index=train_index,
            validation_index=validation_index,
            test_index=test_index,
            random_state=args.random_state,
            output_dir=output_dir,
        )
        summaries.append(summary)
        prediction_frames.append(predictions)
        alarm_prediction_frames.append(predict_target(target, model, alarm_candidates, alarm_features))
        threshold_frames.append(threshold_report)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    alarm_predictions = pd.concat(alarm_prediction_frames, ignore_index=True)
    threshold_report = pd.concat(threshold_frames, ignore_index=True)
    prediction_sample = build_prediction_sample(predictions, args.max_sample_rows)
    session_top_candidates = build_session_top_candidates(predictions)
    session_recommendation_summary = build_session_recommendation_summary(session_top_candidates)
    alarm_backtest_summary = build_alarm_backtest_summary(alarm_predictions, args.smart_deep_soon_threshold)
    numeric_features, categorical_features = feature_columns(args.include_next_stage)

    result = {
        "input_dir": str(input_dir),
        "training_candidates_file": str(training_candidates_path),
        "alarm_candidates_file": str(alarm_candidates_path),
        "output_dir": str(output_dir),
        "split": {
            "method": "chronological_session_train_validation_test_split",
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "train_sessions": int(candidates.loc[train_index, "session_id"].nunique()),
            "validation_sessions": int(candidates.loc[validation_index, "session_id"].nunique()),
            "test_sessions": int(candidates.loc[test_index, "session_id"].nunique()),
            "train_rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "test_rows": int(len(test_index)),
        },
        "features": {
            "numeric": list(numeric_features),
            "categorical": list(categorical_features),
            "include_next_stage": bool(args.include_next_stage),
            "excluded_leakage_columns": list(EXCLUDED_LEAKAGE_COLUMNS),
            "encoded_feature_count": int(features.shape[1]),
        },
        "prediction_sample": {
            "file": str(output_dir / "prediction_sample.csv"),
            "rows": int(len(prediction_sample)),
            "score_draft": "P_deep_soon * (0.5 + 0.5 * deadline_urgency)",
        },
        "threshold_report": {
            "file": str(output_dir / "threshold_report.csv"),
            "rows": int(len(threshold_report)),
        },
        "session_top_candidates": {
            "file": str(output_dir / "session_top_candidates.csv"),
            "rows": int(len(session_top_candidates)),
            "score_weighted": "0.75 * P_deep_soon + 0.25 * deadline_urgency",
            "score_multiply": "P_deep_soon * deadline_urgency",
            "deep_soon_soft_risk_threshold": DEEP_SOON_SOFT_RISK_THRESHOLD,
            "deep_soon_hard_risk_threshold": DEEP_SOON_HARD_RISK_THRESHOLD,
            "risk_penalty": {
                "soft": 0.10,
                "hard": 0.25,
            },
            "target_window_minutes": TARGET_WINDOW_MINUTES,
            "target_window_bonus": TARGET_WINDOW_BONUS,
            "max_early_window_penalty": MAX_EARLY_WINDOW_PENALTY,
            "ranking_score": "score_weighted_after_risk",
        },
        "alarm_backtest_summary": {
            "file": str(output_dir / "alarm_backtest_summary.csv"),
            "rows": int(len(alarm_backtest_summary)),
            "smart_deep_soon_threshold": args.smart_deep_soon_threshold,
            "source": "alarm_candidates_file",
            "policy_metrics": alarm_backtest_metrics(alarm_backtest_summary),
        },
        "session_recommendation_summary": {
            "file": str(output_dir / "session_recommendation_summary.csv"),
            "rows": int(len(session_recommendation_summary)),
            "policy_metrics": recommendation_policy_metrics(session_recommendation_summary),
        },
        "targets": summaries,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(output_dir / "test_predictions_long.csv", index=False)
    alarm_predictions.to_csv(output_dir / "alarm_predictions_long.csv", index=False)
    threshold_report.to_csv(output_dir / "threshold_report.csv", index=False)
    prediction_sample.to_csv(output_dir / "prediction_sample.csv", index=False)
    session_top_candidates.to_csv(output_dir / "session_top_candidates.csv", index=False)
    session_recommendation_summary.to_csv(
        output_dir / "session_recommendation_summary.csv",
        index=False,
    )
    alarm_backtest_summary.to_csv(output_dir / "alarm_backtest_summary.csv", index=False)
    print(json.dumps(result["split"], indent=2))
    for target_summary in summaries:
        metrics = target_summary["lightgbm"]
        print(
            f"{target_summary['target']}: "
            f"precision={metrics['precision']} recall={metrics['recall']} "
            f"roc_auc={metrics.get('roc_auc', 'n/a')} positive_rate={metrics['positive_rate']}"
        )
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'threshold_report.csv'}")
    print(f"Wrote {output_dir / 'test_predictions_long.csv'}")
    print(f"Wrote {output_dir / 'alarm_predictions_long.csv'}")
    print(f"Wrote {output_dir / 'prediction_sample.csv'}")
    print(f"Wrote {output_dir / 'session_top_candidates.csv'}")
    print(f"Wrote {output_dir / 'session_recommendation_summary.csv'}")
    print(f"Wrote {output_dir / 'alarm_backtest_summary.csv'}")


if __name__ == "__main__":
    main()
