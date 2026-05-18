#!/usr/bin/env python3
"""Analyze MyDream smart-alarm backtest failure cases.

The script reads one or more model evaluation directories that contain
``alarm_predictions_long.csv`` and produces threshold-level summaries plus
session-level failure cases. It intentionally recomputes alarm selection per
threshold so different model runs can be compared with the same policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
TARGET = "label_deep_soon"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze smart-alarm failure cases.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        required=True,
        help="Model output directory containing alarm_predictions_long.csv. Repeat for comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for combined outputs. Defaults to <first-model-dir>/failure_analysis.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        help="Threshold to analyze. Repeatable. Defaults to 0.2 through 0.8.",
    )
    parser.add_argument(
        "--focus-threshold",
        type=float,
        action="append",
        default=[0.4, 0.6],
        help="Thresholds that should get detailed case CSVs. Defaults to 0.4 and 0.6.",
    )
    return parser.parse_args()


def read_alarm_predictions(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "alarm_predictions_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    predictions = pd.read_csv(path)
    predictions = predictions[predictions["target"] == TARGET].copy()
    if predictions.empty:
        raise ValueError(f"No {TARGET} rows found in {path}")

    predictions["candidate_dt"] = pd.to_datetime(predictions["candidate_time"], utc=True)
    predictions["deadline_dt"] = pd.to_datetime(predictions["deadline_time"], utc=True)
    predictions["next_deep_dt"] = pd.to_datetime(predictions["next_deep_start"], utc=True, errors="coerce")
    predictions["minutes_before_deadline"] = (
        predictions["deadline_dt"] - predictions["candidate_dt"]
    ).dt.total_seconds() / 60
    predictions["minutes_before_next_deep"] = (
        predictions["next_deep_dt"] - predictions["candidate_dt"]
    ).dt.total_seconds() / 60
    return predictions


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


def select_alarm(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = predictions.copy()
    candidates["is_valid_smart_candidate"] = (
        (candidates["candidate_dt"] < candidates["deadline_dt"])
        & (candidates["probability"] >= threshold)
        & (~candidates["stage_at_candidate"].isin(["Deep", "Unknown"]))
    )

    smart_selected = (
        candidates[candidates["is_valid_smart_candidate"]]
        .sort_values(["session_id", "candidate_dt", "probability"], ascending=[True, False, False])
        .groupby("session_id", as_index=False)
        .head(1)
        .copy()
    )
    smart_selected["alarm_type"] = "smart"

    fallback_base = (
        candidates.sort_values(["session_id", "deadline_dt"])
        .groupby("session_id", as_index=False)
        .tail(1)
        .copy()
    )
    fallback_base = fallback_base[~fallback_base["session_id"].isin(smart_selected["session_id"])].copy()
    fallback_base["candidate_dt"] = fallback_base["deadline_dt"]
    fallback_base["candidate_time"] = fallback_base["deadline_time"]
    fallback_base["stage_at_candidate"] = fallback_base["stage_at_deadline"].fillna("Unknown")
    fallback_base["next_deep_dt"] = pd.NaT
    fallback_base["next_deep_start"] = ""
    fallback_base["minutes_before_next_deep"] = pd.NA
    fallback_base["alarm_type"] = "fallback"

    selected = pd.concat([smart_selected, fallback_base], ignore_index=True, sort=False)
    selected["threshold"] = threshold
    selected["selected_alarm_time"] = selected["candidate_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    selected["stage_at_alarm"] = selected["stage_at_candidate"].fillna("Unknown")
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
    return selected


def opportunity_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    candidates = predictions[
        (predictions["candidate_dt"] < predictions["deadline_dt"])
        & (~predictions["stage_at_candidate"].isin(["Deep", "Unknown"]))
    ].copy()
    rows: list[dict[str, Any]] = []
    for session_id, group in candidates.groupby("session_id"):
        actual_positive = group[group["actual"] == 1]
        high_probability = group.sort_values("probability", ascending=False).head(1)
        latest_actual_positive = (
            actual_positive.sort_values("candidate_dt").tail(1) if not actual_positive.empty else pd.DataFrame()
        )
        row: dict[str, Any] = {
            "session_id": session_id,
            "has_actual_deep_soon_opportunity": not actual_positive.empty,
            "actual_deep_soon_candidate_count": int(len(actual_positive)),
            "max_probability": float(high_probability["probability"].iloc[0]) if not high_probability.empty else 0.0,
        }
        if not latest_actual_positive.empty:
            latest = latest_actual_positive.iloc[0]
            row.update(
                {
                    "latest_actual_deep_soon_time": latest["candidate_time"],
                    "latest_actual_deep_soon_minutes_before_deadline": latest["minutes_before_deadline"],
                    "latest_actual_deep_soon_probability": latest["probability"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_failure_classification(selected: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    merged = selected.merge(opportunities, on="session_id", how="left")
    merged["has_actual_deep_soon_opportunity"] = merged["has_actual_deep_soon_opportunity"].fillna(False)
    merged["actual_deep_soon_candidate_count"] = merged["actual_deep_soon_candidate_count"].fillna(0).astype(int)

    merged["failure_category"] = "unknown"
    merged.loc[merged["deep_prevention_success"], "failure_category"] = "smart_success"
    merged.loc[
        (merged["alarm_type"] == "smart")
        & (~merged["deep_prevention_success"])
        & (merged["too_early_penalty"] == "fail_strong"),
        "failure_category",
    ] = "strong_fail"
    merged.loc[
        (merged["alarm_type"] == "smart")
        & (~merged["deep_prevention_success"])
        & (merged["too_early_penalty"].isin(["weak", "medium"])),
        "failure_category",
    ] = "smart_near_miss"
    merged.loc[
        (merged["alarm_type"] == "smart")
        & (~merged["deep_prevention_success"])
        & (merged["failure_category"] == "unknown"),
        "failure_category",
    ] = "smart_false_alarm"
    merged.loc[
        (merged["alarm_type"] == "fallback") & merged["has_actual_deep_soon_opportunity"],
        "failure_category",
    ] = "fallback_missed_opportunity"
    merged.loc[
        (merged["alarm_type"] == "fallback") & (~merged["has_actual_deep_soon_opportunity"]),
        "failure_category",
    ] = "fallback_no_signal"
    merged.loc[
        (merged["alarm_type"] == "fallback")
        & (merged["stage_at_alarm"] == "Unknown")
        & (~merged["has_actual_deep_soon_opportunity"]),
        "failure_category",
    ] = "fallback_unknown_no_signal"
    return merged


def summarize_cases(cases: pd.DataFrame, model_name: str, threshold: float) -> dict[str, Any]:
    smart = cases["alarm_type"] == "smart"
    fallback = ~smart
    category_counts = cases["failure_category"].value_counts().to_dict()
    penalty_counts = cases["too_early_penalty"].value_counts().to_dict()
    stage_counts = cases["stage_at_alarm"].value_counts().to_dict()
    return {
        "model": model_name,
        "threshold": threshold,
        "sessions": int(len(cases)),
        "smart_alarm": int(smart.sum()),
        "fallback": int(fallback.sum()),
        "smart_rate": round(float(smart.mean()), 6),
        "deep_success": int(cases["deep_prevention_success"].sum()),
        "success_per_smart": round(
            float(cases.loc[smart, "deep_prevention_success"].mean()) if smart.any() else 0.0,
            6,
        ),
        "success_total": round(float(cases["deep_prevention_success"].mean()), 6),
        "sleep_quality_success": round(float(cases["sleep_quality_success"].mean()), 6),
        "strong_fail": int(category_counts.get("strong_fail", 0)),
        "smart_near_miss": int(category_counts.get("smart_near_miss", 0)),
        "smart_false_alarm": int(category_counts.get("smart_false_alarm", 0)),
        "fallback_missed_opportunity": int(category_counts.get("fallback_missed_opportunity", 0)),
        "fallback_no_signal": int(category_counts.get("fallback_no_signal", 0)),
        "fallback_unknown_no_signal": int(category_counts.get("fallback_unknown_no_signal", 0)),
        "penalty_counts": penalty_counts,
        "stage_counts": stage_counts,
    }


def main() -> None:
    args = parse_args()
    thresholds = args.threshold or list(DEFAULT_THRESHOLDS)
    focus_thresholds = set(args.focus_threshold or [])
    output_dir = args.output_dir or args.model_dir[0] / "failure_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict[str, Any]] = []
    for model_dir in args.model_dir:
        model_name = model_dir.name
        predictions = read_alarm_predictions(model_dir)
        opportunities = opportunity_summary(predictions)

        model_case_frames: list[pd.DataFrame] = []
        for threshold in thresholds:
            selected = select_alarm(predictions, threshold)
            cases = add_failure_classification(selected, opportunities)
            cases.insert(0, "model", model_name)
            all_summary_rows.append(summarize_cases(cases, model_name, threshold))
            model_case_frames.append(cases)

            if threshold in focus_thresholds:
                safe_threshold = str(threshold).replace(".", "_")
                cases.to_csv(
                    output_dir / f"{model_name}_alarm_failure_cases_t{safe_threshold}.csv",
                    index=False,
                )

        pd.concat(model_case_frames, ignore_index=True).to_csv(
            output_dir / f"{model_name}_alarm_failure_cases_all_thresholds.csv",
            index=False,
        )

    summary = pd.DataFrame(all_summary_rows)
    summary.to_csv(output_dir / "threshold_comparison.csv", index=False)

    compact_columns = [
        "model",
        "threshold",
        "sessions",
        "smart_alarm",
        "fallback",
        "smart_rate",
        "deep_success",
        "success_per_smart",
        "success_total",
        "strong_fail",
        "smart_near_miss",
        "fallback_missed_opportunity",
        "fallback_unknown_no_signal",
        "sleep_quality_success",
    ]
    summary.loc[:, compact_columns].to_csv(output_dir / "model_comparison_summary.csv", index=False)
    print(f"Wrote {output_dir / 'threshold_comparison.csv'}")
    print(f"Wrote {output_dir / 'model_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
