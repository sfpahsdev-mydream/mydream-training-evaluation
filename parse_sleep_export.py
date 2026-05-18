#!/usr/bin/env python3
"""Parse MyDream sleep JSONL exports into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


WAKEABLE_TYPES = {"Awake", "Light"}
DEEP_TYPE = "Deep"
STAGE_TYPES = ("Awake", "Light", "Deep", "Rem", "Unknown")
LOCAL_TZ = timezone(timedelta(hours=9))
MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class Stage:
    session_id: str
    stage_type: str
    start: datetime
    end: datetime

    @property
    def duration_min(self) -> float:
        return (self.end - self.start).total_seconds() / 60


@dataclass(frozen=True)
class Session:
    session_id: str
    start: datetime
    end: datetime
    stages: list[Stage]

    @property
    def duration_min(self) -> float:
        return (self.end - self.start).total_seconds() / 60


@dataclass(frozen=True)
class WakeProfile:
    month_dow_minutes: dict[tuple[str, int], float]
    month_dow_counts: dict[tuple[str, int], int]
    week_period_minutes: dict[tuple[int, int, str], float]
    week_period_counts: dict[tuple[int, int, str], int]
    weekday_weekend_minutes: dict[str, float]
    weekday_weekend_counts: dict[str, int]
    min_samples: int


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_sessions(path: Path) -> tuple[list[Session], list[str]]:
    sessions: list[Session] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                session = parse_session(raw)
                errors.extend(validate_session(session, line_number))
                sessions.append(session)
            except Exception as error:  # noqa: BLE001 - report and continue parsing.
                errors.append(f"line {line_number}: {error}")

    return sessions, errors


def parse_session(raw: dict[str, Any]) -> Session:
    session_id = str(raw["session_id"])
    start = parse_datetime(str(raw["start"]))
    end = parse_datetime(str(raw["end"]))
    stages = [
        Stage(
            session_id=session_id,
            stage_type=str(stage.get("type", "Unknown")),
            start=parse_datetime(str(stage["start"])),
            end=parse_datetime(str(stage["end"])),
        )
        for stage in raw.get("stages", [])
    ]
    return Session(session_id=session_id, start=start, end=end, stages=stages)


def validate_session(session: Session, line_number: int) -> list[str]:
    errors: list[str] = []
    if session.end <= session.start:
        errors.append(f"line {line_number}: session end is not after start: {session.session_id}")
    if not session.stages:
        errors.append(f"line {line_number}: session has no stages: {session.session_id}")

    sorted_stages = sorted(session.stages, key=lambda stage: stage.start)
    for stage in sorted_stages:
        if stage.end <= stage.start:
            errors.append(
                f"line {line_number}: stage end is not after start: "
                f"{session.session_id} {stage.stage_type}",
            )
    for previous, current in zip(sorted_stages, sorted_stages[1:]):
        if current.start < previous.end:
            errors.append(
                f"line {line_number}: overlapping stages in {session.session_id}: "
                f"{previous.stage_type} -> {current.stage_type}",
            )
    return errors


def candidate_times(
    deadline: datetime,
    search_window_minutes: int,
    candidate_interval_minutes: int,
) -> Iterable[datetime]:
    start = deadline - timedelta(minutes=search_window_minutes)
    current = start
    while current <= deadline:
        yield current
        current += timedelta(minutes=candidate_interval_minutes)


def session_candidate_times(session: Session, candidate_interval_minutes: int) -> Iterable[datetime]:
    if not session.stages:
        return
    start = min(stage.start for stage in session.stages)
    end = max(stage.end for stage in session.stages)
    current = start
    while current < end:
        yield current
        current += timedelta(minutes=candidate_interval_minutes)


def parse_hhmm(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected HH:MM wake time, got {value!r}") from error


def actual_wake_time(session: Session) -> datetime:
    return max(stage.end for stage in session.stages)


def minute_of_day(instant: datetime, local_tz: timezone = LOCAL_TZ) -> float:
    local = instant.astimezone(local_tz)
    return local.hour * 60 + local.minute + local.second / 60


def weekday_weekend_key(day_of_week: int) -> str:
    return "weekend" if day_of_week >= 5 else "weekday"


def week_period_key(local_datetime: datetime) -> tuple[int, int, str]:
    iso_year, iso_week, _ = local_datetime.isocalendar()
    return (iso_year, iso_week, weekday_weekend_key(local_datetime.weekday()))


def median_by_key(values: dict[Any, list[float]]) -> tuple[dict[Any, float], dict[Any, int]]:
    medians = {key: float(median(items)) for key, items in values.items() if items}
    counts = {key: len(items) for key, items in values.items() if items}
    return medians, counts


def build_wake_profile(
    sessions: list[Session],
    min_samples: int,
    local_tz: timezone = LOCAL_TZ,
) -> WakeProfile:
    by_month_dow: dict[tuple[str, int], list[float]] = {}
    by_week_period: dict[tuple[int, int, str], list[float]] = {}
    by_weekday_weekend: dict[str, list[float]] = {}

    for session in sessions:
        if not session.stages:
            continue
        wake_local = actual_wake_time(session).astimezone(local_tz)
        month = wake_local.strftime("%Y-%m")
        dow = wake_local.weekday()
        minute = wake_local.hour * 60 + wake_local.minute + wake_local.second / 60
        by_month_dow.setdefault((month, dow), []).append(minute)
        by_week_period.setdefault(week_period_key(wake_local), []).append(minute)
        by_weekday_weekend.setdefault(weekday_weekend_key(dow), []).append(minute)

    month_dow_minutes, month_dow_counts = median_by_key(by_month_dow)
    week_period_minutes, week_period_counts = median_by_key(by_week_period)
    weekday_weekend_minutes, weekday_weekend_counts = median_by_key(by_weekday_weekend)
    return WakeProfile(
        month_dow_minutes=month_dow_minutes,
        month_dow_counts=month_dow_counts,
        week_period_minutes=week_period_minutes,
        week_period_counts=week_period_counts,
        weekday_weekend_minutes=weekday_weekend_minutes,
        weekday_weekend_counts=weekday_weekend_counts,
        min_samples=min_samples,
    )


def profile_target_wake_minute(
    profile: WakeProfile,
    wake_local: datetime,
) -> tuple[float, str, int]:
    month = wake_local.strftime("%Y-%m")
    dow = wake_local.weekday()
    month_dow_key = (month, dow)
    if profile.month_dow_counts.get(month_dow_key, 0) >= profile.min_samples:
        return profile.month_dow_minutes[month_dow_key], "month_day_of_week_median", profile.month_dow_counts[month_dow_key]
    weekly_key = week_period_key(wake_local)
    if profile.week_period_counts.get(weekly_key, 0) >= profile.min_samples:
        return profile.week_period_minutes[weekly_key], "week_period_median", profile.week_period_counts[weekly_key]
    fallback_key = weekday_weekend_key(dow)
    return (
        profile.weekday_weekend_minutes[fallback_key],
        "weekday_weekend_median",
        profile.weekday_weekend_counts[fallback_key],
    )


def local_datetime_from_minute(local_date: datetime, minute: float, local_tz: timezone = LOCAL_TZ) -> datetime:
    whole_minutes = int(minute)
    seconds = int(round((minute - whole_minutes) * 60))
    if seconds == 60:
        whole_minutes += 1
        seconds = 0
    hour = whole_minutes // 60
    minute_part = whole_minutes % 60
    return datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        hour,
        minute_part,
        seconds,
        tzinfo=local_tz,
    )


def fixed_wake_time_for_session(
    session: Session,
    weekday_wake_time: time,
    weekend_wake_time: time,
    local_tz: timezone = LOCAL_TZ,
) -> tuple[datetime, str, int]:
    wake_local = actual_wake_time(session).astimezone(local_tz)
    wake_clock = weekend_wake_time if wake_local.weekday() >= 5 else weekday_wake_time
    local_deadline = datetime.combine(wake_local.date(), wake_clock, tzinfo=local_tz)
    deadline = local_deadline.astimezone(timezone.utc)
    if deadline <= session.start:
        deadline += timedelta(days=1)
    return deadline, weekday_weekend_key(wake_local.weekday()), 0


def wake_time_for_session(
    session: Session,
    weekday_wake_time: time,
    weekend_wake_time: time,
    wake_time_policy: str,
    wake_profile: WakeProfile | None,
    local_tz: timezone = LOCAL_TZ,
) -> tuple[datetime, str, int]:
    if wake_time_policy == "fixed_weekday_weekend":
        return fixed_wake_time_for_session(session, weekday_wake_time, weekend_wake_time, local_tz)
    if wake_time_policy != "month_day_of_week_profile":
        raise ValueError(f"Unknown wake_time_policy: {wake_time_policy}")
    if wake_profile is None:
        raise ValueError("wake_profile is required for month_day_of_week_profile policy")

    wake_local = actual_wake_time(session).astimezone(local_tz)
    target_minute, source, sample_count = profile_target_wake_minute(wake_profile, wake_local)
    local_deadline = local_datetime_from_minute(wake_local, target_minute, local_tz)
    deadline = local_deadline.astimezone(timezone.utc)
    if deadline <= session.start:
        deadline += timedelta(days=1)
    return deadline, source, sample_count


def stage_at_time(stages: list[Stage], instant: datetime) -> str:
    for stage in stages:
        if stage.start <= instant < stage.end:
            return stage.stage_type
    return "Unknown"


def stage_index_at_time(stages: list[Stage], instant: datetime) -> int | None:
    for index, stage in enumerate(stages):
        if stage.start <= instant < stage.end:
            return index
    return None


def stage_context(stages: list[Stage], instant: datetime) -> dict[str, Any]:
    sorted_stages = sorted(stages, key=lambda stage: stage.start)
    index = stage_index_at_time(sorted_stages, instant)
    if index is None:
        return {
            "previous_stage": "Unknown",
            "next_stage": "Unknown",
            "minutes_since_stage_start": "",
            "current_stage_duration_so_far": "",
        }

    current = sorted_stages[index]
    previous_stage = sorted_stages[index - 1].stage_type if index > 0 else "Unknown"
    next_stage = sorted_stages[index + 1].stage_type if index + 1 < len(sorted_stages) else "Unknown"
    minutes_since_start = (instant - current.start).total_seconds() / 60

    return {
        "previous_stage": previous_stage,
        "next_stage": next_stage,
        "minutes_since_stage_start": round(minutes_since_start, 3),
        "current_stage_duration_so_far": round(minutes_since_start, 3),
    }


def overlaps_window(stage: Stage, start: datetime, end: datetime) -> bool:
    return stage.start < end and stage.end > start


def overlap_minutes(stage: Stage, start: datetime, end: datetime) -> float:
    overlap_start = max(stage.start, start)
    overlap_end = min(stage.end, end)
    if overlap_end <= overlap_start:
        return 0
    return (overlap_end - overlap_start).total_seconds() / 60


def recent_stage_minutes(
    stages: list[Stage],
    candidate: datetime,
    window_minutes: int,
) -> dict[str, float]:
    window_start = candidate - timedelta(minutes=window_minutes)
    minutes_by_stage = {stage_type: 0.0 for stage_type in STAGE_TYPES}
    for stage in stages:
        stage_type = stage.stage_type if stage.stage_type in minutes_by_stage else "Unknown"
        minutes_by_stage[stage_type] += overlap_minutes(stage, window_start, candidate)
    return {stage_type: round(minutes, 3) for stage_type, minutes in minutes_by_stage.items()}


def label_wakeable(stages: list[Stage], candidate: datetime, window_minutes: int) -> int:
    window_start = candidate - timedelta(minutes=window_minutes)
    window_end = candidate + timedelta(minutes=window_minutes)
    return int(
        any(
            stage.stage_type in WAKEABLE_TYPES and overlaps_window(stage, window_start, window_end)
            for stage in stages
        ),
    )


def label_wakeable_at_candidate(stages: list[Stage], candidate: datetime) -> int:
    return int(stage_at_time(stages, candidate) in WAKEABLE_TYPES)


def label_deep_soon(stages: list[Stage], candidate: datetime, window_minutes: int) -> int:
    if stage_at_time(stages, candidate) == DEEP_TYPE:
        return 0
    window_end = candidate + timedelta(minutes=window_minutes)
    return int(any(candidate < stage.start <= window_end and stage.stage_type == DEEP_TYPE for stage in stages))


def next_deep_start(stages: list[Stage], candidate: datetime) -> datetime | None:
    deep_starts = [stage.start for stage in stages if stage.stage_type == DEEP_TYPE and stage.start > candidate]
    return min(deep_starts) if deep_starts else None


def previous_deep_end(stages: list[Stage], candidate: datetime) -> datetime | None:
    deep_ends = [stage.end for stage in stages if stage.stage_type == DEEP_TYPE and stage.end <= candidate]
    return max(deep_ends) if deep_ends else None


def minutes_since_last_deep(stages: list[Stage], candidate: datetime) -> float:
    previous_end = previous_deep_end(stages, candidate)
    if previous_end is None:
        return -1.0
    return round((candidate - previous_end).total_seconds() / 60, 3)


def deep_cycle_position(stages: list[Stage], candidate: datetime) -> int:
    return sum(1 for stage in stages if stage.stage_type == DEEP_TYPE and stage.start < candidate)


def cyclic_time_features(minute: float, prefix: str) -> dict[str, float]:
    radians = 2 * math.pi * minute / MINUTES_PER_DAY
    return {
        f"{prefix}_sin": round(math.sin(radians), 6),
        f"{prefix}_cos": round(math.cos(radians), 6),
    }


def time_of_day_features(candidate: datetime, local_tz: timezone = LOCAL_TZ) -> dict[str, float | int]:
    local = candidate.astimezone(local_tz)
    minute = local.hour * 60 + local.minute + local.second / 60
    return {
        **cyclic_time_features(minute, "time_of_day"),
        "day_of_week": local.weekday(),
    }


def write_sessions_csv(path: Path, sessions: list[Session]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["session_id", "start", "end", "duration_min", "stage_count", "has_stage_overlap"],
        )
        writer.writeheader()
        for session in sessions:
            writer.writerow(
                {
                    "session_id": session.session_id,
                    "start": session.start.isoformat(),
                    "end": session.end.isoformat(),
                    "duration_min": round(session.duration_min, 3),
                    "stage_count": len(session.stages),
                    "has_stage_overlap": has_stage_overlap(session.stages),
                },
            )


def write_stages_csv(path: Path, sessions: list[Session]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["session_id", "type", "start", "end", "duration_min"])
        writer.writeheader()
        for session in sessions:
            for stage in sorted(session.stages, key=lambda item: item.start):
                writer.writerow(
                    {
                        "session_id": stage.session_id,
                        "type": stage.stage_type,
                        "start": stage.start.isoformat(),
                        "end": stage.end.isoformat(),
                        "duration_min": round(stage.duration_min, 3),
                    },
                )


def write_candidates_csv(
    path: Path,
    sessions: list[Session],
    candidate_interval_minutes: int,
    wakeable_window_minutes: int,
    deep_soon_window_minutes: int,
    weekday_wake_time: time,
    weekend_wake_time: time,
    wake_time_policy: str,
    wake_profile: WakeProfile | None,
    mode: str,
    search_window_minutes: int | None = None,
    exclude_deep: bool = True,
    exclude_unknown: bool = True,
) -> tuple[int, int, int]:
    count = 0
    excluded_current_deep = 0
    excluded_unknown_stage = 0
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "session_id",
                "candidate_time",
                "deadline_time",
                "wake_time_source",
                "wake_profile_sample_count",
                "stage_at_deadline",
                "minutes_before_deadline",
                "target_wake_hour_sin",
                "target_wake_hour_cos",
                "elapsed_sleep_minutes",
                "time_of_day_sin",
                "time_of_day_cos",
                "day_of_week",
                "stage_at_candidate",
                "next_deep_start",
                "previous_stage",
                "next_stage",
                "minutes_since_stage_start",
                "current_stage_duration_so_far",
                "minutes_since_last_deep",
                "deep_cycle_position",
                "recent_30m_awake_minutes",
                "recent_30m_light_minutes",
                "recent_30m_deep_minutes",
                "recent_30m_rem_minutes",
                "recent_30m_unknown_minutes",
                "label_wakeable",
                "label_wakeable_window",
                "label_wakeable_at_candidate",
                "label_deep_soon",
            ],
        )
        writer.writeheader()
        for session in sessions:
            deadline, wake_time_source, wake_profile_sample_count = wake_time_for_session(
                session,
                weekday_wake_time,
                weekend_wake_time,
                wake_time_policy,
                wake_profile,
            )
            stage_at_deadline = stage_at_time(session.stages, deadline)
            target_wake_features = cyclic_time_features(minute_of_day(deadline), "target_wake_hour")
            if mode == "training":
                candidates = session_candidate_times(session, candidate_interval_minutes)
            elif mode == "alarm":
                if search_window_minutes is None:
                    raise ValueError("search_window_minutes is required for alarm candidates")
                candidates = candidate_times(deadline, search_window_minutes, candidate_interval_minutes)
            else:
                raise ValueError(f"Unknown candidate generation mode: {mode}")

            for candidate in candidates:
                stage_at_candidate = stage_at_time(session.stages, candidate)
                if exclude_deep and stage_at_candidate == DEEP_TYPE:
                    excluded_current_deep += 1
                    continue
                if exclude_unknown and stage_at_candidate == "Unknown":
                    excluded_unknown_stage += 1
                    continue
                wakeable_window = label_wakeable(session.stages, candidate, wakeable_window_minutes)
                wakeable_at_candidate = label_wakeable_at_candidate(session.stages, candidate)
                context = stage_context(session.stages, candidate)
                recent_30m = recent_stage_minutes(session.stages, candidate, 30)
                next_deep = next_deep_start(session.stages, candidate)
                tod = time_of_day_features(candidate)
                writer.writerow(
                    {
                        "session_id": session.session_id,
                        "candidate_time": candidate.isoformat(),
                        "deadline_time": deadline.isoformat(),
                        "wake_time_source": wake_time_source,
                        "wake_profile_sample_count": wake_profile_sample_count,
                        "stage_at_deadline": stage_at_deadline,
                        "minutes_before_deadline": round((deadline - candidate).total_seconds() / 60, 3),
                        "target_wake_hour_sin": target_wake_features["target_wake_hour_sin"],
                        "target_wake_hour_cos": target_wake_features["target_wake_hour_cos"],
                        "elapsed_sleep_minutes": round((candidate - session.start).total_seconds() / 60, 3),
                        "time_of_day_sin": tod["time_of_day_sin"],
                        "time_of_day_cos": tod["time_of_day_cos"],
                        "day_of_week": tod["day_of_week"],
                        "stage_at_candidate": stage_at_candidate,
                        "next_deep_start": next_deep.isoformat() if next_deep else "",
                        "previous_stage": context["previous_stage"],
                        "next_stage": context["next_stage"],
                        "minutes_since_stage_start": context["minutes_since_stage_start"],
                        "current_stage_duration_so_far": context["current_stage_duration_so_far"],
                        "minutes_since_last_deep": minutes_since_last_deep(session.stages, candidate),
                        "deep_cycle_position": deep_cycle_position(session.stages, candidate),
                        "recent_30m_awake_minutes": recent_30m["Awake"],
                        "recent_30m_light_minutes": recent_30m["Light"],
                        "recent_30m_deep_minutes": recent_30m["Deep"],
                        "recent_30m_rem_minutes": recent_30m["Rem"],
                        "recent_30m_unknown_minutes": recent_30m["Unknown"],
                        "label_wakeable": wakeable_window,
                        "label_wakeable_window": wakeable_window,
                        "label_wakeable_at_candidate": wakeable_at_candidate,
                        "label_deep_soon": label_deep_soon(session.stages, candidate, deep_soon_window_minutes),
                    },
                )
                count += 1
    return count, excluded_current_deep, excluded_unknown_stage


def summarize_candidates(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    candidate_count = len(rows)
    stage_counts: dict[str, int] = {}
    wakeable_positive = 0
    wakeable_window_positive = 0
    wakeable_at_candidate_positive = 0
    deep_soon_positive = 0
    both_positive = 0
    both_negative = 0
    deep_stage_with_deep_soon_positive = 0

    for row in rows:
        stage = row["stage_at_candidate"]
        label_wakeable_value = int(row["label_wakeable"])
        label_wakeable_window_value = int(row.get("label_wakeable_window", row["label_wakeable"]))
        label_wakeable_at_candidate_value = int(row.get("label_wakeable_at_candidate", 0))
        label_deep_soon_value = int(row["label_deep_soon"])

        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        wakeable_positive += label_wakeable_value
        wakeable_window_positive += label_wakeable_window_value
        wakeable_at_candidate_positive += label_wakeable_at_candidate_value
        deep_soon_positive += label_deep_soon_value
        if label_wakeable_value == 1 and label_deep_soon_value == 1:
            both_positive += 1
        if label_wakeable_value == 0 and label_deep_soon_value == 0:
            both_negative += 1
        if stage == DEEP_TYPE and label_deep_soon_value == 1:
            deep_stage_with_deep_soon_positive += 1

    return {
        "candidate_stage_counts": stage_counts,
        "label_wakeable_positive_count": wakeable_positive,
        "label_wakeable_positive_ratio": round(wakeable_positive / candidate_count, 4) if candidate_count else 0,
        "label_wakeable_window_positive_count": wakeable_window_positive,
        "label_wakeable_window_positive_ratio": (
            round(wakeable_window_positive / candidate_count, 4) if candidate_count else 0
        ),
        "label_wakeable_at_candidate_positive_count": wakeable_at_candidate_positive,
        "label_wakeable_at_candidate_positive_ratio": (
            round(wakeable_at_candidate_positive / candidate_count, 4) if candidate_count else 0
        ),
        "label_deep_soon_positive_count": deep_soon_positive,
        "label_deep_soon_positive_ratio": round(deep_soon_positive / candidate_count, 4) if candidate_count else 0,
        "both_positive_count": both_positive,
        "both_negative_count": both_negative,
        "deep_stage_with_deep_soon_positive_count": deep_stage_with_deep_soon_positive,
    }


def summarize_candidate_session_coverage(path: Path, training_sessions: list[Session]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    candidate_counts_by_session: dict[str, int] = {}
    deep_soon_sessions: set[str] = set()
    for row in rows:
        session_id = row["session_id"]
        candidate_counts_by_session[session_id] = candidate_counts_by_session.get(session_id, 0) + 1
        if int(row["label_deep_soon"]) == 1:
            deep_soon_sessions.add(session_id)

    counts = list(candidate_counts_by_session.values())
    bucket_counts = {
        "1": 0,
        "2-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-31": 0,
    }

    for count in counts:
        if count == 1:
            bucket_counts["1"] += 1
        elif count <= 5:
            bucket_counts["2-5"] += 1
        elif count <= 10:
            bucket_counts["6-10"] += 1
        elif count <= 20:
            bucket_counts["11-20"] += 1
        else:
            bucket_counts["21-31"] += 1

    sorted_counts = sorted(counts)
    middle = len(sorted_counts) // 2
    if not sorted_counts:
        median = 0.0
    elif len(sorted_counts) % 2:
        median = float(sorted_counts[middle])
    else:
        median = (sorted_counts[middle - 1] + sorted_counts[middle]) / 2

    training_session_count = len(training_sessions)
    sessions_with_candidates = len(candidate_counts_by_session)
    return {
        "sessions_with_candidates": sessions_with_candidates,
        "sessions_without_candidates": training_session_count - sessions_with_candidates,
        "sessions_with_candidates_ratio": (
            round(sessions_with_candidates / training_session_count, 4) if training_session_count else 0
        ),
        "average_candidates_per_covered_session": round(sum(counts) / len(counts), 3) if counts else 0,
        "median_candidates_per_covered_session": round(median, 3),
        "min_candidates_per_covered_session": min(counts) if counts else 0,
        "max_candidates_per_covered_session": max(counts) if counts else 0,
        "sessions_with_deep_soon_positive": len(deep_soon_sessions),
        "sessions_without_deep_soon_positive_among_covered": sessions_with_candidates - len(deep_soon_sessions),
        "candidate_count_buckets_by_session": bucket_counts,
    }


def prefix_keys(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def filter_training_sessions(
    sessions: list[Session],
    min_session_minutes: int,
    max_session_minutes: int,
) -> tuple[list[Session], dict[str, int]]:
    included: list[Session] = []
    excluded = {
        "excluded_no_stages": 0,
        "excluded_too_short": 0,
        "excluded_too_long": 0,
    }

    for session in sessions:
        if not session.stages:
            excluded["excluded_no_stages"] += 1
        elif session.duration_min < min_session_minutes:
            excluded["excluded_too_short"] += 1
        elif session.duration_min > max_session_minutes:
            excluded["excluded_too_long"] += 1
        else:
            included.append(session)

    return included, excluded


def has_stage_overlap(stages: list[Stage]) -> bool:
    sorted_stages = sorted(stages, key=lambda stage: stage.start)
    return any(current.start < previous.end for previous, current in zip(sorted_stages, sorted_stages[1:]))


def build_summary(
    sessions: list[Session],
    training_sessions: list[Session],
    errors: list[str],
    training_candidate_count: int,
    alarm_candidate_count: int,
    exclusions: dict[str, int],
    training_candidate_summary: dict[str, Any],
    training_candidate_session_coverage: dict[str, Any],
    alarm_candidate_summary: dict[str, Any],
    alarm_candidate_session_coverage: dict[str, Any],
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    stages = [stage for session in sessions for stage in session.stages]
    training_stages = [stage for session in training_sessions for stage in session.stages]
    stage_type_counts: dict[str, int] = {}
    for stage in stages:
        stage_type_counts[stage.stage_type] = stage_type_counts.get(stage.stage_type, 0) + 1
    training_stage_type_counts: dict[str, int] = {}
    for stage in training_stages:
        training_stage_type_counts[stage.stage_type] = training_stage_type_counts.get(stage.stage_type, 0) + 1

    durations = [session.duration_min for session in sessions]
    training_durations = [session.duration_min for session in training_sessions]
    return {
        "session_count": len(sessions),
        "training_session_count": len(training_sessions),
        "stage_count": len(stages),
        "training_stage_count": len(training_stages),
        "training_candidate_count": training_candidate_count,
        "alarm_candidate_count": alarm_candidate_count,
        "candidate_config": candidate_config,
        "sessions_without_stages": sum(1 for session in sessions if not session.stages),
        "sessions_with_stage_overlap": sum(1 for session in sessions if has_stage_overlap(session.stages)),
        "average_session_duration_min": round(sum(durations) / len(durations), 3) if durations else 0,
        "average_training_session_duration_min": (
            round(sum(training_durations) / len(training_durations), 3) if training_durations else 0
        ),
        "stage_type_counts": stage_type_counts,
        "training_stage_type_counts": training_stage_type_counts,
        **exclusions,
        **prefix_keys("training", training_candidate_summary),
        **prefix_keys("training", training_candidate_session_coverage),
        **prefix_keys("alarm", alarm_candidate_summary),
        **prefix_keys("alarm", alarm_candidate_session_coverage),
        "error_count": len(errors),
        "errors": errors[:100],
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse MyDream sleep JSONL export.")
    parser.add_argument("--input", type=Path, help="Path to MyDream JSONL export.")
    parser.add_argument("--out-dir", type=Path, help="Directory for generated CSV/JSON outputs.")
    parser.add_argument("--search-window-minutes", type=int, default=30)
    parser.add_argument("--candidate-interval-minutes", type=int, default=1)
    parser.add_argument(
        "--wake-time-policy",
        choices=("month_day_of_week_profile", "fixed_weekday_weekend"),
        default="month_day_of_week_profile",
    )
    parser.add_argument("--wake-profile-min-samples", type=int, default=2)
    parser.add_argument("--weekday-wake-time", type=parse_hhmm, default=parse_hhmm("06:20"))
    parser.add_argument("--weekend-wake-time", type=parse_hhmm, default=parse_hhmm("09:00"))
    parser.add_argument("--wakeable-window-minutes", type=int, default=5)
    parser.add_argument("--deep-soon-window-minutes", type=int, default=10)
    parser.add_argument("--min-session-minutes", type=int, default=180)
    parser.add_argument("--max-session-minutes", type=int, default=720)
    args = parser.parse_args()

    if args.input is None:
        args.input = choose_input_file()
    if args.out_dir is None:
        args.out_dir = default_output_dir(args.input)

    return args


def choose_input_file() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as error:  # noqa: BLE001 - tkinter may be unavailable in some runtimes.
        raise SystemExit(
            "No --input was provided and a file picker could not be opened. "
            "Run again with --input C:\\path\\to\\export.jsonl.",
        ) from error

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Select MyDream sleep JSONL export",
        filetypes=[
            ("JSONL files", "*.jsonl"),
            ("NDJSON files", "*.ndjson"),
            ("JSON files", "*.json"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not selected:
        raise SystemExit("No input file selected.")
    return Path(selected)


def default_output_dir(input_path: Path) -> Path:
    return Path(__file__).resolve().parent / "out" / input_path.stem


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions, errors = load_sessions(args.input)
    training_sessions, exclusions = filter_training_sessions(
        sessions,
        min_session_minutes=args.min_session_minutes,
        max_session_minutes=args.max_session_minutes,
    )
    wake_profile = (
        build_wake_profile(training_sessions, args.wake_profile_min_samples)
        if args.wake_time_policy == "month_day_of_week_profile"
        else None
    )
    write_sessions_csv(args.out_dir / "sessions.csv", sessions)
    write_stages_csv(args.out_dir / "stages.csv", sessions)
    write_sessions_csv(args.out_dir / "training_sessions.csv", training_sessions)
    training_candidates_path = args.out_dir / f"training_candidates_{args.candidate_interval_minutes}min.csv"
    alarm_candidates_path = args.out_dir / f"alarm_candidates_{args.candidate_interval_minutes}min.csv"
    training_candidate_count, training_excluded_current_deep, training_excluded_unknown_stage = write_candidates_csv(
        training_candidates_path,
        training_sessions,
        candidate_interval_minutes=args.candidate_interval_minutes,
        wakeable_window_minutes=args.wakeable_window_minutes,
        deep_soon_window_minutes=args.deep_soon_window_minutes,
        weekday_wake_time=args.weekday_wake_time,
        weekend_wake_time=args.weekend_wake_time,
        wake_time_policy=args.wake_time_policy,
        wake_profile=wake_profile,
        mode="training",
        exclude_deep=True,
        exclude_unknown=True,
    )
    alarm_candidate_count, alarm_excluded_current_deep, alarm_excluded_unknown_stage = write_candidates_csv(
        alarm_candidates_path,
        training_sessions,
        candidate_interval_minutes=args.candidate_interval_minutes,
        wakeable_window_minutes=args.wakeable_window_minutes,
        deep_soon_window_minutes=args.deep_soon_window_minutes,
        weekday_wake_time=args.weekday_wake_time,
        weekend_wake_time=args.weekend_wake_time,
        wake_time_policy=args.wake_time_policy,
        wake_profile=wake_profile,
        mode="alarm",
        search_window_minutes=args.search_window_minutes,
        exclude_deep=False,
        exclude_unknown=False,
    )
    training_candidate_summary = summarize_candidates(training_candidates_path)
    training_candidate_session_coverage = summarize_candidate_session_coverage(training_candidates_path, training_sessions)
    alarm_candidate_summary = summarize_candidates(alarm_candidates_path)
    alarm_candidate_session_coverage = summarize_candidate_session_coverage(alarm_candidates_path, training_sessions)
    candidate_config = {
        "deadline_source": "historical_wake_profile" if wake_profile else "fixed_user_defined_wake_time",
        "wake_time_policy": args.wake_time_policy,
        "wake_profile_min_samples": args.wake_profile_min_samples,
        "fixed_weekday_wake_time": args.weekday_wake_time.strftime("%H:%M"),
        "fixed_weekend_wake_time": args.weekend_wake_time.strftime("%H:%M"),
        "wake_profile_month_day_of_week_cells": len(wake_profile.month_dow_minutes) if wake_profile else 0,
        "wake_profile_week_period_cells": len(wake_profile.week_period_minutes) if wake_profile else 0,
        "wake_profile_weekday_weekend_cells": len(wake_profile.weekday_weekend_minutes) if wake_profile else 0,
        "search_window_minutes": args.search_window_minutes,
        "candidate_interval_minutes": args.candidate_interval_minutes,
        "deep_soon_window_minutes": args.deep_soon_window_minutes,
        "training_generation_scope": "full_recorded_sleep_stage_coverage",
        "alarm_generation_scope": "wake_time_minus_30_minutes_to_wake_time",
        "training_excluded_current_deep_candidates": training_excluded_current_deep,
        "training_excluded_unknown_stage_candidates": training_excluded_unknown_stage,
        "alarm_excluded_current_deep_candidates": alarm_excluded_current_deep,
        "alarm_excluded_unknown_stage_candidates": alarm_excluded_unknown_stage,
    }
    write_summary(
        args.out_dir / "summary.json",
        build_summary(
            sessions,
            training_sessions,
            errors,
            training_candidate_count,
            alarm_candidate_count,
            exclusions,
            training_candidate_summary,
            training_candidate_session_coverage,
            alarm_candidate_summary,
            alarm_candidate_session_coverage,
            candidate_config,
        ),
    )

    print(f"Parsed sessions: {len(sessions)}")
    print(f"Training sessions: {len(training_sessions)}")
    print(f"Generated training candidates: {training_candidate_count}")
    print(f"Generated alarm candidates: {alarm_candidate_count}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
