#!/usr/bin/env python3
"""Evaluate Android-compatible MyDream alarm decision policies offline.

This script compares candidate-level policy decisions from one or more saved
sequence-model prediction files and one tabular prediction file. Unlike the
on-device Lab screen, it is intended for repeatable offline model selection,
threshold sweeps, and CSV export.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET = "label_deep_soon"
KEY_COLUMNS = ["session_id", "candidate_time", "deadline_time"]
TARGET_HORIZON_MINUTES = 10
SEARCH_WINDOW_MINUTES = 30.0
STRICT_DEADLINE_GATE_MINUTES = 20.0
UNKNOWN_RATIO_LIMIT = 0.3
GRU_WEIGHT = 0.8
DEADLINE_WEIGHT = 0.2
DEFAULT_THRESHOLDS = (0.55,)
SWEEP_THRESHOLDS = tuple(round(value / 100, 2) for value in range(30, 76, 5))


@dataclass(frozen=True)
class Policy:
    variant: str


POLICIES = (
    Policy("only"),
    Policy("deadline"),
    Policy("strict deadline gate"),
    Policy("unknown coverage gate"),
    Policy("deadline + unknown gate"),
    Policy("tabular"),
    Policy("tabular + deadline"),
    Policy("tabular + deadline + unknown gate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Android-compatible wake decision policies.")
    parser.add_argument(
        "--profile-root",
        type=Path,
        help="Dataset/result root. Resolves candidate, metadata, stage, and default model paths underneath it.",
    )
    parser.add_argument(
        "--gru-predictions",
        type=Path,
        help="Backward-compatible single GRU prediction path. Ignored when --sequence-prediction is supplied.",
    )
    parser.add_argument(
        "--sequence-prediction",
        action="append",
        dest="sequence_predictions",
        metavar="NAME=PATH",
        help="Sequence-model alarm prediction file to evaluate. Repeat for comparable models.",
    )
    parser.add_argument(
        "--tabular-predictions",
        type=Path,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="Candidate features used for unknown-coverage gate inputs.",
    )
    parser.add_argument(
        "--sequence-metadata",
        type=Path,
        help="Sequence metadata containing the Android-compatible 60-minute unknown ratio.",
    )
    parser.add_argument(
        "--stages",
        type=Path,
        help="Stage intervals used to recompute Android-compatible target coverage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--threshold", type=float, action="append", help="Decision threshold. Repeatable.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate thresholds 0.30 through 0.75 in increments of 0.05.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        help="Only evaluate sessions in the last N deadline dates of the supplied candidate set.",
    )
    args = parser.parse_args()
    root = args.profile_root or Path("out/verify_week_period_profile")
    args.gru_predictions = args.gru_predictions or (
        root / "sequence_experiments" / "gru" / "gru64_dense32_dropout00" / "alarm_predictions_long.csv"
    )
    args.tabular_predictions = args.tabular_predictions or root / "model_tabular_tflite" / "alarm_predictions_long.csv"
    args.candidates = args.candidates or root / "alarm_candidates_1min.csv"
    args.sequence_metadata = args.sequence_metadata or root / "sequence_60m_alarm" / "sequence_metadata.csv"
    args.stages = args.stages or root / "stages.csv"
    args.output_dir = args.output_dir or root / "policy_evaluation"
    return args


def read_predictions(path: Path, score_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["target", "probability"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {missing}")
    frame = frame[frame["target"] == TARGET].copy()
    if frame.empty:
        raise ValueError(f"No {TARGET} rows found in {path}")
    frame[score_column] = pd.to_numeric(frame["probability"], errors="raise")
    return frame.loc[:, KEY_COLUMNS + [score_column]]


def read_candidates(path: Path, sequence_metadata_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates file: {path}")
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["stage_at_candidate"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing candidate columns in {path}: {missing}")
    if not sequence_metadata_path.exists():
        raise FileNotFoundError(f"Missing sequence metadata file: {sequence_metadata_path}")
    metadata = pd.read_csv(sequence_metadata_path)
    metadata_required = set(KEY_COLUMNS + ["sequence_unknown_ratio"])
    metadata_missing = sorted(metadata_required - set(metadata.columns))
    if metadata_missing:
        raise ValueError(f"Missing sequence metadata columns in {sequence_metadata_path}: {metadata_missing}")
    metadata["sequence_unknown_ratio"] = pd.to_numeric(metadata["sequence_unknown_ratio"], errors="raise")
    frame = frame.merge(
        metadata.loc[:, KEY_COLUMNS + ["sequence_unknown_ratio"]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    frame["sequence_unknown_ratio"] = frame["sequence_unknown_ratio"].clip(lower=0.0, upper=1.0)
    return frame


def read_stages(path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stages file: {path}")
    stages = pd.read_csv(path)
    required = {"session_id", "type", "start", "end"}
    missing = sorted(required - set(stages.columns))
    if missing:
        raise ValueError(f"Missing stage columns in {path}: {missing}")
    stages["start_dt"] = pd.to_datetime(stages["start"], utc=True)
    stages["end_dt"] = pd.to_datetime(stages["end"], utc=True)
    by_session: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    for session_id, group in stages.groupby("session_id"):
        by_session[str(session_id)] = [
            (row.start_dt, row.end_dt, str(row.type))
            for row in group.sort_values("start_dt").itertuples()
        ]
    return by_session


def parse_sequence_predictions(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if not args.sequence_predictions:
        return [("GRU", args.gru_predictions)]
    predictions = []
    for item in args.sequence_predictions:
        if "=" not in item:
            raise ValueError(f"--sequence-prediction must use NAME=PATH format: {item}")
        name, path_text = item.split("=", 1)
        if not name.strip() or not path_text.strip():
            raise ValueError(f"--sequence-prediction must use NAME=PATH format: {item}")
        predictions.append((name.strip(), Path(path_text.strip())))
    return predictions


def merge_inputs(args: argparse.Namespace, sequence_prediction: Path) -> pd.DataFrame:
    sequence = read_predictions(sequence_prediction, "sequence_score")
    tabular = read_predictions(args.tabular_predictions, "tabular_score")
    candidates = read_candidates(args.candidates, args.sequence_metadata)
    merged = candidates.merge(sequence, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    merged = merged.merge(tabular, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(sequence) or len(merged) != len(tabular):
        raise ValueError(
            "Prediction files and candidates do not align exactly after merge: "
            f"candidates={len(candidates)}, sequence={len(sequence)}, tabular={len(tabular)}, merged={len(merged)}"
        )
    merged["candidate_dt"] = pd.to_datetime(merged["candidate_time"], utc=True)
    merged["deadline_dt"] = pd.to_datetime(merged["deadline_time"], utc=True)
    merged["minutes_before_deadline"] = (
        merged["deadline_dt"] - merged["candidate_dt"]
    ).dt.total_seconds() / 60
    merged["deadline_closeness"] = (
        1.0 - merged["minutes_before_deadline"] / SEARCH_WINDOW_MINUTES
    ).clip(lower=0.0, upper=1.0)
    if args.recent_days:
        if args.recent_days <= 0:
            raise ValueError("--recent-days must be greater than zero")
        local_deadline_date = merged["deadline_dt"].dt.tz_convert("Asia/Seoul").dt.date
        latest_date = local_deadline_date.max()
        first_date = latest_date - pd.Timedelta(days=args.recent_days - 1)
        merged = merged[local_deadline_date.between(first_date, latest_date)].copy()
    if merged.empty:
        raise ValueError("No rows remain after input merge/filtering.")
    return merged


def stage_at(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]],
    timestamp: pd.Timestamp,
) -> str | None:
    for start, end, stage_type in intervals:
        if start <= timestamp < end:
            return stage_type
    return None


def actual_label(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]],
    candidate_dt: pd.Timestamp,
) -> tuple[str, bool | None]:
    current = stage_at(intervals, candidate_dt)
    if current == "Deep":
        return "excluded_already_deep", None
    if current in (None, "Unknown"):
        return "target_unknown", None
    horizon = [
        stage_at(intervals, candidate_dt + pd.Timedelta(minutes=offset))
        for offset in range(TARGET_HORIZON_MINUTES + 1)
    ]
    if any(stage in (None, "Unknown") for stage in horizon):
        return "target_unknown", None
    horizon_end = candidate_dt + pd.Timedelta(minutes=TARGET_HORIZON_MINUTES)
    deep_soon = any(
        stage_type == "Deep" and candidate_dt < start <= horizon_end
        for start, _, stage_type in intervals
    )
    return "labeled", deep_soon


def add_actual_labels(
    frame: pd.DataFrame,
    stages_by_session: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]],
) -> pd.DataFrame:
    output = frame.copy()
    labels = []
    for row in output.itertuples():
        intervals = stages_by_session.get(str(row.session_id), [])
        labels.append(actual_label(intervals, row.candidate_dt))
    output["target_status"] = [status for status, _ in labels]
    output["actual_deep_soon"] = [actual for _, actual in labels]
    return output


def policy_decision(frame: pd.DataFrame, sequence_model: str, policy: Policy, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    if policy.variant == "only":
        output["score"] = output["sequence_score"]
        output["decision"] = "SMART_WAKE"
    elif policy.variant == "deadline":
        output["score"] = GRU_WEIGHT * output["sequence_score"] + DEADLINE_WEIGHT * output["deadline_closeness"]
        output["decision"] = "SMART_WAKE"
    elif policy.variant == "strict deadline gate":
        output["score"] = output["sequence_score"]
        output["decision"] = output["minutes_before_deadline"].gt(STRICT_DEADLINE_GATE_MINUTES).map(
            {True: "SKIP_TOO_EARLY", False: "SMART_WAKE"}
        )
    elif policy.variant == "unknown coverage gate":
        output["score"] = output["sequence_score"]
        output["decision"] = output["sequence_unknown_ratio"].gt(UNKNOWN_RATIO_LIMIT).map(
            {True: "SKIP_UNKNOWN_TOO_HIGH", False: "SMART_WAKE"}
        )
    elif policy.variant == "deadline + unknown gate":
        output["score"] = GRU_WEIGHT * output["sequence_score"] + DEADLINE_WEIGHT * output["deadline_closeness"]
        output["decision"] = output["sequence_unknown_ratio"].gt(UNKNOWN_RATIO_LIMIT).map(
            {True: "SKIP_UNKNOWN_TOO_HIGH", False: "SMART_WAKE"}
        )
    elif policy.variant == "tabular":
        output["score"] = 0.5 * output["sequence_score"] + 0.5 * output["tabular_score"]
        output["decision"] = "SMART_WAKE"
    elif policy.variant == "tabular + deadline":
        output["score"] = (
            0.4 * output["sequence_score"]
            + 0.4 * output["tabular_score"]
            + DEADLINE_WEIGHT * output["deadline_closeness"]
        )
        output["decision"] = "SMART_WAKE"
    elif policy.variant == "tabular + deadline + unknown gate":
        output["score"] = (
            0.4 * output["sequence_score"]
            + 0.4 * output["tabular_score"]
            + DEADLINE_WEIGHT * output["deadline_closeness"]
        )
        output["decision"] = output["sequence_unknown_ratio"].gt(UNKNOWN_RATIO_LIMIT).map(
            {True: "SKIP_UNKNOWN_TOO_HIGH", False: "SMART_WAKE"}
        )
    else:
        raise ValueError(f"Unknown policy: {policy.variant}")
    score_below = output["score"] < threshold
    output.loc[output["decision"].eq("SMART_WAKE") & score_below, "decision"] = "WAIT"
    output["sequence_model"] = sequence_model
    output["policy_variant"] = policy.variant
    output["policy"] = f"{sequence_model}-{policy.variant}" if policy.variant == "only" else f"{sequence_model} + {policy.variant}"
    output["threshold"] = threshold
    labeled = output["target_status"].eq("labeled")
    actual = output["actual_deep_soon"].eq(True)
    smart = output["decision"].eq("SMART_WAKE")
    output["true_smart"] = labeled & actual & smart
    output["false_smart"] = labeled & ~actual & smart
    output["missed_smart"] = labeled & actual & ~smart
    output["utility"] = 0
    output.loc[output["true_smart"], "utility"] = 1
    output.loc[output["false_smart"] | output["missed_smart"], "utility"] = -1
    return output


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def summarize_policy(frame: pd.DataFrame) -> dict[str, Any]:
    sample_count = len(frame)
    labeled = frame[frame["target_status"] == "labeled"]
    smart_count = int((frame["decision"] == "SMART_WAKE").sum())
    wait_count = int((frame["decision"] == "WAIT").sum())
    covered_count = smart_count + wait_count
    true_smart = int(frame["true_smart"].sum())
    false_smart = int(frame["false_smart"].sum())
    missed_smart = int(frame["missed_smart"].sum())
    actual_positive = int(labeled["actual_deep_soon"].eq(True).sum())
    utility_sum = int(frame["utility"].sum())
    per_session = labeled.groupby("session_id")["utility"].mean()
    input_columns = ["sequence_score", "tabular_score", "sequence_unknown_ratio", "score"]
    invalid_inputs = int((~np.isfinite(frame[input_columns].to_numpy(dtype=float))).any(axis=1).sum())
    return {
        "policy": frame["policy"].iat[0],
        "sequence_model": frame["sequence_model"].iat[0],
        "policy_variant": frame["policy_variant"].iat[0],
        "threshold": float(frame["threshold"].iat[0]),
        "samples": sample_count,
        "sessions": int(frame["session_id"].nunique()),
        "invalid_inputs": invalid_inputs,
        "max_unknown_ratio": float(frame["sequence_unknown_ratio"].max()),
        "mean_score": float(frame["score"].mean()),
        "smart": smart_count,
        "wait": wait_count,
        "skip_early": int((frame["decision"] == "SKIP_TOO_EARLY").sum()),
        "skip_unknown": int((frame["decision"] == "SKIP_UNKNOWN_TOO_HIGH").sum()),
        "count_valid": (
            smart_count
            + wait_count
            + int((frame["decision"] == "SKIP_TOO_EARLY").sum())
            + int((frame["decision"] == "SKIP_UNKNOWN_TOO_HIGH").sum())
        ) == sample_count,
        "covered": covered_count,
        "coverage": ratio(covered_count, sample_count),
        "labeled": int(len(labeled)),
        "target_unknown": int((frame["target_status"] == "target_unknown").sum()),
        "excluded_already_deep": int((frame["target_status"] == "excluded_already_deep").sum()),
        "actual_deep_soon": actual_positive,
        "true_smart": true_smart,
        "false_smart": false_smart,
        "missed_smart": missed_smart,
        "precision": ratio(true_smart, true_smart + false_smart),
        "recall": ratio(true_smart, actual_positive),
        "utility_labeled": ratio(utility_sum, len(labeled)),
        "utility_all": ratio(utility_sum, sample_count),
        "session_utility_mean": float(per_session.mean()) if not per_session.empty else None,
        "session_utility_min": float(per_session.min()) if not per_session.empty else None,
        "session_utility_max": float(per_session.max()) if not per_session.empty else None,
    }


def session_summary(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame[frame["target_status"] == "labeled"].copy()
    if labeled.empty:
        return pd.DataFrame()
    grouped = labeled.groupby(["sequence_model", "policy_variant", "policy", "threshold", "session_id"], as_index=False)
    summary = grouped.agg(
        labeled=("actual_deep_soon", "size"),
        actual_deep_soon=("actual_deep_soon", lambda series: int(series.eq(True).sum())),
        true_smart=("true_smart", "sum"),
        false_smart=("false_smart", "sum"),
        missed_smart=("missed_smart", "sum"),
        utility=("utility", "mean"),
    )
    return summary


def main() -> None:
    args = parse_args()
    thresholds = list(SWEEP_THRESHOLDS if args.sweep else (args.threshold or DEFAULT_THRESHOLDS))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = read_stages(args.stages)
    policy_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for sequence_model, sequence_prediction in parse_sequence_predictions(args):
        candidates = merge_inputs(args, sequence_prediction)
        labeled_candidates = add_actual_labels(candidates, stages)
        for threshold in thresholds:
            for policy in POLICIES:
                evaluated = policy_decision(labeled_candidates, sequence_model, policy, threshold)
                policy_frames.append(evaluated)
                summary_rows.append(summarize_policy(evaluated))

    all_results = pd.concat(policy_frames, ignore_index=True)
    details_columns = KEY_COLUMNS + [
        "policy",
        "sequence_model",
        "policy_variant",
        "threshold",
        "sequence_score",
        "tabular_score",
        "deadline_closeness",
        "sequence_unknown_ratio",
        "score",
        "decision",
        "target_status",
        "actual_deep_soon",
        "true_smart",
        "false_smart",
        "missed_smart",
        "utility",
    ]
    all_results.loc[:, details_columns].to_csv(output_dir / "policy_candidate_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "policy_summary.csv", index=False)
    session_summary(all_results).to_csv(output_dir / "session_policy_summary.csv", index=False)
    if args.sweep or len(thresholds) > 1:
        pd.DataFrame(summary_rows).to_csv(output_dir / "threshold_sweep.csv", index=False)

    print(f"Wrote {output_dir / 'policy_candidate_results.csv'}")
    print(f"Wrote {output_dir / 'policy_summary.csv'}")
    print(f"Wrote {output_dir / 'session_policy_summary.csv'}")
    if args.sweep or len(thresholds) > 1:
        print(f"Wrote {output_dir / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
