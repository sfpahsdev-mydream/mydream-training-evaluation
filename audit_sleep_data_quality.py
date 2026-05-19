#!/usr/bin/env python3
"""Audit MyDream sleep data quality before model tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, time, timedelta, timezone
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("out/verify_week_period_profile")
DEFAULT_OUTPUT_DIR_NAME = "data_quality"
LABEL_WINDOWS = (5, 10, 15)
PRE_ENTRY_WINDOWS = ((0, 5), (5, 10), (10, 15), (5, 15))
LOCAL_TZ = timezone(timedelta(hours=9))
FIXED_WAKE_TIMES = ("05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MyDream sleep data quality.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--high-unknown-ratio", type=float, default=0.5)
    parser.add_argument("--deadline-gap-minutes", type=float, default=30.0)
    parser.add_argument(
        "--fixed-wake-time",
        action="append",
        help="HH:MM fixed wake time to coverage-test. Repeatable. Defaults to 05:00 through 11:00.",
    )
    return parser.parse_args()


def read_csv(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def ratio(value: float, total: float) -> float:
    return round(float(value / total), 6) if total else 0.0


def add_session_times(sessions: pd.DataFrame) -> pd.DataFrame:
    result = sessions.copy()
    result["session_start_dt"] = pd.to_datetime(result["start"], utc=True)
    result["session_end_dt"] = pd.to_datetime(result["end"], utc=True)
    return result


def parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def fixed_deadline_for_session(session_end_dt: pd.Timestamp, session_start_dt: pd.Timestamp, wake_time: time) -> pd.Timestamp:
    wake_local = session_end_dt.to_pydatetime().astimezone(LOCAL_TZ)
    local_deadline = datetime.combine(wake_local.date(), wake_time, tzinfo=LOCAL_TZ)
    deadline = pd.Timestamp(local_deadline.astimezone(timezone.utc))
    if deadline <= session_start_dt:
        deadline += pd.Timedelta(days=1)
    return deadline


def stage_at_instant(stage_intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]], instant: pd.Timestamp) -> str:
    for start, end, stage_type in stage_intervals:
        if start <= instant < end:
            return stage_type if stage_type in {"Awake", "Light", "Deep", "Rem"} else "Unknown"
    return "Unknown"


def has_deep_soon(stage_intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]], instant: pd.Timestamp, window_minutes: int = 10) -> bool:
    if stage_at_instant(stage_intervals, instant) == "Deep":
        return False
    window_end = instant + pd.Timedelta(minutes=window_minutes)
    return any(stage_type == "Deep" and instant < start <= window_end for start, _, stage_type in stage_intervals)


def audit_stage_continuity(stages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    stage_rows = stages.copy()
    stage_rows["stage_start_dt"] = pd.to_datetime(stage_rows["start"], utc=True)
    stage_rows["stage_end_dt"] = pd.to_datetime(stage_rows["end"], utc=True)
    stage_rows["computed_duration_min"] = (
        stage_rows["stage_end_dt"] - stage_rows["stage_start_dt"]
    ).dt.total_seconds() / 60
    stage_rows["duration_delta_min"] = stage_rows["computed_duration_min"] - stage_rows["duration_min"]

    for session_id, group in stage_rows.sort_values(["session_id", "stage_start_dt"]).groupby("session_id"):
        previous_end = group["stage_end_dt"].shift(1)
        gaps = (group["stage_start_dt"] - previous_end).dt.total_seconds() / 60
        gap_count = int((gaps > 0.01).sum())
        overlap_count = int((gaps < -0.01).sum())
        max_gap = float(gaps[gaps > 0.01].max()) if (gaps > 0.01).any() else 0.0
        max_overlap = float((-gaps[gaps < -0.01]).max()) if (gaps < -0.01).any() else 0.0
        rows.append(
            {
                "session_id": session_id,
                "stage_count": int(len(group)),
                "stage_total_duration_min": round(float(group["computed_duration_min"].sum()), 3),
                "gap_count": gap_count,
                "overlap_count": overlap_count,
                "max_gap_min": round(max_gap, 3),
                "max_overlap_min": round(max_overlap, 3),
                "duration_mismatch_count": int((group["duration_delta_min"].abs() > 0.01).sum()),
            }
        )
    return pd.DataFrame(rows), stage_rows


def audit_alarm_window(
    sessions: pd.DataFrame,
    alarm_candidates: pd.DataFrame,
    high_unknown_ratio: float,
    deadline_gap_minutes: float,
) -> pd.DataFrame:
    alarm = alarm_candidates.copy()
    alarm["candidate_dt"] = pd.to_datetime(alarm["candidate_time"], utc=True)
    alarm["deadline_dt"] = pd.to_datetime(alarm["deadline_time"], utc=True)

    grouped = alarm.groupby("session_id")
    rows = grouped.agg(
        alarm_candidate_count=("candidate_time", "size"),
        alarm_unknown_count=("stage_at_candidate", lambda values: int((values == "Unknown").sum())),
        alarm_deep_count=("stage_at_candidate", lambda values: int((values == "Deep").sum())),
        alarm_deep_soon_positive_count=("label_deep_soon", "sum"),
        alarm_wakeable_count=("label_wakeable_at_candidate", "sum"),
        first_alarm_candidate_dt=("candidate_dt", "min"),
        deadline_dt=("deadline_dt", "max"),
        stage_at_deadline=("stage_at_deadline", "last"),
    ).reset_index()
    rows["alarm_unknown_ratio"] = rows["alarm_unknown_count"] / rows["alarm_candidate_count"]
    rows["alarm_deep_soon_positive_ratio"] = (
        rows["alarm_deep_soon_positive_count"] / rows["alarm_candidate_count"]
    )
    rows = rows.merge(
        sessions.loc[:, ["session_id", "duration_min", "stage_count", "session_start_dt", "session_end_dt"]],
        on="session_id",
        how="left",
    )
    rows["deadline_minus_session_end_min"] = (
        rows["deadline_dt"] - rows["session_end_dt"]
    ).dt.total_seconds() / 60
    rows["alarm_window_after_session_end_min"] = (
        rows["first_alarm_candidate_dt"] - rows["session_end_dt"]
    ).dt.total_seconds() / 60
    rows["high_unknown_alarm_window"] = rows["alarm_unknown_ratio"] >= high_unknown_ratio
    rows["deadline_far_after_session_end"] = rows["deadline_minus_session_end_min"] > deadline_gap_minutes
    return rows.sort_values(
        ["high_unknown_alarm_window", "alarm_unknown_ratio", "deadline_minus_session_end_min"],
        ascending=[False, False, False],
    )


def audit_label_windows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    rows["candidate_dt"] = pd.to_datetime(rows["candidate_time"], utc=True)
    rows["next_deep_dt"] = pd.to_datetime(rows["next_deep_start"], utc=True, errors="coerce")
    minutes_to_deep = (rows["next_deep_dt"] - rows["candidate_dt"]).dt.total_seconds() / 60
    valid = ~rows["stage_at_candidate"].isin(["Deep", "Unknown"])
    result_rows = []
    for window in LABEL_WINDOWS:
        label = valid & minutes_to_deep.gt(0) & minutes_to_deep.le(window)
        by_stage = rows.loc[valid].assign(label_variant=label[valid].astype(int)).groupby("stage_at_candidate")
        for stage, group in by_stage:
            result_rows.append(
                {
                    "window_minutes": window,
                    "stage_at_candidate": stage,
                    "candidate_count": int(len(group)),
                    "positive_count": int(group["label_variant"].sum()),
                    "positive_ratio": ratio(group["label_variant"].sum(), len(group)),
                }
            )
        result_rows.append(
            {
                "window_minutes": window,
                "stage_at_candidate": "__all_non_deep_known__",
                "candidate_count": int(valid.sum()),
                "positive_count": int(label.sum()),
                "positive_ratio": ratio(label.sum(), valid.sum()),
            }
        )
    return pd.DataFrame(result_rows)


def append_label_summary(
    result_rows: list[dict[str, Any]],
    rows: pd.DataFrame,
    valid: pd.Series,
    label: pd.Series,
    label_name: str,
    label_family: str,
    window_start_minutes: int | None,
    window_end_minutes: int,
) -> None:
    by_stage = rows.loc[valid].assign(label_variant=label[valid].astype(int)).groupby("stage_at_candidate")
    for stage, group in by_stage:
        result_rows.append(
            {
                "label_family": label_family,
                "label_name": label_name,
                "window_start_minutes": window_start_minutes,
                "window_end_minutes": window_end_minutes,
                "stage_at_candidate": stage,
                "candidate_count": int(len(group)),
                "positive_count": int(group["label_variant"].sum()),
                "positive_ratio": ratio(group["label_variant"].sum(), len(group)),
            }
        )
    result_rows.append(
        {
            "label_family": label_family,
            "label_name": label_name,
            "window_start_minutes": window_start_minutes,
            "window_end_minutes": window_end_minutes,
            "stage_at_candidate": "__all_non_deep_known__",
            "candidate_count": int(valid.sum()),
            "positive_count": int(label.sum()),
            "positive_ratio": ratio(label.sum(), valid.sum()),
        }
    )


def audit_label_design(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    rows["candidate_dt"] = pd.to_datetime(rows["candidate_time"], utc=True)
    rows["next_deep_dt"] = pd.to_datetime(rows["next_deep_start"], utc=True, errors="coerce")
    minutes_to_deep = (rows["next_deep_dt"] - rows["candidate_dt"]).dt.total_seconds() / 60
    valid = ~rows["stage_at_candidate"].isin(["Deep", "Unknown"])
    result_rows: list[dict[str, Any]] = []
    for window in LABEL_WINDOWS:
        label = valid & minutes_to_deep.gt(0) & minutes_to_deep.le(window)
        append_label_summary(
            result_rows,
            rows,
            valid,
            label,
            label_name=f"deep_within_{window}m",
            label_family="deep_soon",
            window_start_minutes=0,
            window_end_minutes=window,
        )
    for start, end in PRE_ENTRY_WINDOWS:
        label = valid & minutes_to_deep.gt(start) & minutes_to_deep.le(end)
        append_label_summary(
            result_rows,
            rows,
            valid,
            label,
            label_name=f"deep_pre_entry_{start}_{end}m",
            label_family="deep_pre_entry",
            window_start_minutes=start,
            window_end_minutes=end,
        )
    return pd.DataFrame(result_rows)


def audit_fixed_wake_coverage(
    sessions: pd.DataFrame,
    stage_rows: pd.DataFrame,
    fixed_wake_times: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_groups = {}
    for session_id, group in stage_rows.sort_values(["session_id", "stage_start_dt"]).groupby("session_id"):
        stage_groups[session_id] = [
            (row.stage_start_dt, row.stage_end_dt, row.type)
            for row in group.itertuples(index=False)
        ]
    session_rows: list[dict[str, Any]] = []
    for session in sessions.itertuples(index=False):
        stages = stage_groups.get(session.session_id)
        if not stages:
            continue
        for wake_time_text in fixed_wake_times:
            wake_time = parse_hhmm(wake_time_text)
            deadline = fixed_deadline_for_session(
                session.session_end_dt,
                session.session_start_dt,
                wake_time,
            )
            candidates = [deadline - pd.Timedelta(minutes=30 - offset) for offset in range(31)]
            candidate_stages = [stage_at_instant(stages, candidate) for candidate in candidates]
            unknown_count = sum(stage == "Unknown" for stage in candidate_stages)
            deep_count = sum(stage == "Deep" for stage in candidate_stages)
            known_non_deep_count = sum(stage not in {"Unknown", "Deep"} for stage in candidate_stages)
            deep_soon_count = sum(has_deep_soon(stages, candidate) for candidate in candidates)
            unknown_ratio = unknown_count / len(candidates)
            if unknown_ratio <= 0.3:
                coverage_class = "evaluable"
            elif unknown_ratio <= 0.7:
                coverage_class = "partial"
            else:
                coverage_class = "not_evaluable"
            session_rows.append(
                {
                    "wake_time": wake_time_text,
                    "session_id": session.session_id,
                    "candidate_count": len(candidates),
                    "unknown_count": unknown_count,
                    "unknown_ratio": round(float(unknown_ratio), 6),
                    "deep_count": deep_count,
                    "known_non_deep_count": known_non_deep_count,
                    "deep_soon_positive_count": deep_soon_count,
                    "coverage_class": coverage_class,
                    "deadline_time": deadline.isoformat(),
                    "deadline_minus_session_end_min": round(
                        float((deadline - session.session_end_dt).total_seconds() / 60),
                        3,
                    ),
                    "window_start_minus_session_end_min": round(
                        float((candidates[0] - session.session_end_dt).total_seconds() / 60),
                        3,
                    ),
                }
            )
    by_session = pd.DataFrame(session_rows)
    if by_session.empty:
        return by_session, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for wake_time_text, group in by_session.groupby("wake_time"):
        class_counts = group["coverage_class"].value_counts().to_dict()
        sessions_total = len(group)
        summary_rows.append(
            {
                "wake_time": wake_time_text,
                "sessions": sessions_total,
                "evaluable_sessions": int(class_counts.get("evaluable", 0)),
                "partial_sessions": int(class_counts.get("partial", 0)),
                "not_evaluable_sessions": int(class_counts.get("not_evaluable", 0)),
                "evaluable_rate": ratio(class_counts.get("evaluable", 0), sessions_total),
                "evaluable_or_partial_rate": ratio(
                    class_counts.get("evaluable", 0) + class_counts.get("partial", 0),
                    sessions_total,
                ),
                "average_unknown_ratio": round(float(group["unknown_ratio"].mean()), 6),
                "median_unknown_ratio": round(float(group["unknown_ratio"].median()), 6),
                "sessions_with_deep_soon_opportunity": int((group["deep_soon_positive_count"] > 0).sum()),
                "deep_soon_opportunity_rate": ratio((group["deep_soon_positive_count"] > 0).sum(), sessions_total),
                "average_deadline_minus_session_end_min": round(
                    float(group["deadline_minus_session_end_min"].mean()),
                    3,
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("wake_time")
    return by_session.sort_values(["wake_time", "unknown_ratio"], ascending=[True, False]), summary


def stage_distribution(stages: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        stages.groupby("type", as_index=False)
        .agg(stage_count=("type", "size"), duration_min=("duration_min", "sum"))
        .sort_values("duration_min", ascending=False)
    )
    total_duration = distribution["duration_min"].sum()
    distribution["duration_ratio"] = distribution["duration_min"].apply(lambda value: ratio(value, total_duration))
    return distribution


def summarize(
    sessions: pd.DataFrame,
    training_sessions: pd.DataFrame,
    stages: pd.DataFrame,
    training_candidates: pd.DataFrame,
    alarm_candidates: pd.DataFrame,
    continuity: pd.DataFrame,
    alarm_quality: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "session_count": int(len(sessions)),
        "training_session_count": int(len(training_sessions)),
        "stage_count": int(len(stages)),
        "training_candidate_count": int(len(training_candidates)),
        "alarm_candidate_count": int(len(alarm_candidates)),
        "sessions_without_stages": int((sessions["stage_count"] == 0).sum()),
        "sessions_shorter_than_180m": int((sessions["duration_min"] < 180).sum()),
        "sessions_longer_than_720m": int((sessions["duration_min"] > 720).sum()),
        "sessions_with_stage_gap": int((continuity["gap_count"] > 0).sum()),
        "sessions_with_stage_overlap": int((continuity["overlap_count"] > 0).sum()),
        "max_stage_gap_min": round(float(continuity["max_gap_min"].max()), 3) if not continuity.empty else 0.0,
        "max_stage_overlap_min": round(float(continuity["max_overlap_min"].max()), 3) if not continuity.empty else 0.0,
        "alarm_unknown_candidate_count": int((alarm_candidates["stage_at_candidate"] == "Unknown").sum()),
        "alarm_unknown_candidate_ratio": ratio(
            (alarm_candidates["stage_at_candidate"] == "Unknown").sum(),
            len(alarm_candidates),
        ),
        "alarm_sessions_high_unknown_ratio_ge_0_5": int(alarm_quality["high_unknown_alarm_window"].sum()),
        "alarm_sessions_deadline_more_than_30m_after_session_end": int(
            alarm_quality["deadline_far_after_session_end"].sum()
        ),
        "alarm_sessions_with_deep_soon_positive": int(
            (alarm_quality["alarm_deep_soon_positive_count"] > 0).sum()
        ),
        "alarm_sessions_without_deep_soon_positive": int(
            (alarm_quality["alarm_deep_soon_positive_count"] == 0).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = add_session_times(read_csv(input_dir, "sessions.csv"))
    training_sessions = read_csv(input_dir, "training_sessions.csv")
    stages = read_csv(input_dir, "stages.csv")
    training_candidates = read_csv(input_dir, "training_candidates_1min.csv")
    alarm_candidates = read_csv(input_dir, "alarm_candidates_1min.csv")

    continuity, stage_rows = audit_stage_continuity(stages)
    alarm_quality = audit_alarm_window(
        sessions,
        alarm_candidates,
        high_unknown_ratio=args.high_unknown_ratio,
        deadline_gap_minutes=args.deadline_gap_minutes,
    )
    training_label_windows = audit_label_windows(training_candidates)
    alarm_label_windows = audit_label_windows(alarm_candidates)
    training_label_design = audit_label_design(training_candidates)
    alarm_label_design = audit_label_design(alarm_candidates)
    fixed_wake_times = args.fixed_wake_time or list(FIXED_WAKE_TIMES)
    fixed_coverage_by_session, fixed_coverage_summary = audit_fixed_wake_coverage(
        sessions.merge(training_sessions[["session_id"]], on="session_id", how="inner"),
        stage_rows,
        fixed_wake_times,
    )
    stage_dist = stage_distribution(stages)
    summary = summarize(
        sessions,
        training_sessions,
        stages,
        training_candidates,
        alarm_candidates,
        continuity,
        alarm_quality,
    )

    continuity.to_csv(output_dir / "stage_continuity_by_session.csv", index=False)
    alarm_quality.to_csv(output_dir / "alarm_window_quality_by_session.csv", index=False)
    training_label_windows.to_csv(output_dir / "training_label_window_sensitivity.csv", index=False)
    alarm_label_windows.to_csv(output_dir / "alarm_label_window_sensitivity.csv", index=False)
    training_label_design.to_csv(output_dir / "training_label_design_comparison.csv", index=False)
    alarm_label_design.to_csv(output_dir / "alarm_label_design_comparison.csv", index=False)
    fixed_coverage_by_session.to_csv(output_dir / "fixed_wake_time_coverage_by_session.csv", index=False)
    fixed_coverage_summary.to_csv(output_dir / "fixed_wake_time_coverage_summary.csv", index=False)
    stage_dist.to_csv(output_dir / "stage_distribution.csv", index=False)
    stage_rows.loc[stage_rows["duration_delta_min"].abs() > 0.01].to_csv(
        output_dir / "stage_duration_mismatches.csv",
        index=False,
    )
    with (output_dir / "data_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote {output_dir / 'data_quality_summary.json'}")
    print(f"Wrote {output_dir / 'alarm_window_quality_by_session.csv'}")
    print(f"Wrote {output_dir / 'training_label_window_sensitivity.csv'}")
    print(f"Wrote {output_dir / 'alarm_label_window_sensitivity.csv'}")
    print(f"Wrote {output_dir / 'training_label_design_comparison.csv'}")
    print(f"Wrote {output_dir / 'alarm_label_design_comparison.csv'}")
    print(f"Wrote {output_dir / 'fixed_wake_time_coverage_summary.csv'}")


if __name__ == "__main__":
    main()
