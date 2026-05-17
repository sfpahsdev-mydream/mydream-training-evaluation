#!/usr/bin/env python3
"""Parse MyDream sleep JSONL exports into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


WAKEABLE_TYPES = {"Awake", "Light"}
DEEP_TYPE = "Deep"
STAGE_TYPES = ("Awake", "Light", "Deep", "Rem", "Unknown")


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
    include_deadline_candidate: bool,
) -> Iterable[datetime]:
    start = deadline - timedelta(minutes=search_window_minutes)
    current = start
    while current <= deadline:
        if current == deadline and not include_deadline_candidate:
            break
        yield current
        current += timedelta(minutes=candidate_interval_minutes)


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
    search_window_minutes: int,
    candidate_interval_minutes: int,
    wakeable_window_minutes: int,
    deep_soon_window_minutes: int,
    include_deadline_candidate: bool,
) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "session_id",
                "candidate_time",
                "deadline_time",
                "minutes_before_deadline",
                "elapsed_sleep_minutes",
                "stage_at_candidate",
                "previous_stage",
                "next_stage",
                "minutes_since_stage_start",
                "current_stage_duration_so_far",
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
            deadline = session.end
            for candidate in candidate_times(
                deadline,
                search_window_minutes,
                candidate_interval_minutes,
                include_deadline_candidate,
            ):
                wakeable_window = label_wakeable(session.stages, candidate, wakeable_window_minutes)
                wakeable_at_candidate = label_wakeable_at_candidate(session.stages, candidate)
                context = stage_context(session.stages, candidate)
                recent_30m = recent_stage_minutes(session.stages, candidate, 30)
                writer.writerow(
                    {
                        "session_id": session.session_id,
                        "candidate_time": candidate.isoformat(),
                        "deadline_time": deadline.isoformat(),
                        "minutes_before_deadline": round((deadline - candidate).total_seconds() / 60, 3),
                        "elapsed_sleep_minutes": round((candidate - session.start).total_seconds() / 60, 3),
                        "stage_at_candidate": stage_at_time(session.stages, candidate),
                        "previous_stage": context["previous_stage"],
                        "next_stage": context["next_stage"],
                        "minutes_since_stage_start": context["minutes_since_stage_start"],
                        "current_stage_duration_so_far": context["current_stage_duration_so_far"],
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
    return count


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
    candidate_count: int,
    exclusions: dict[str, int],
    candidate_summary: dict[str, Any],
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
        "candidate_count": candidate_count,
        "sessions_without_stages": sum(1 for session in sessions if not session.stages),
        "sessions_with_stage_overlap": sum(1 for session in sessions if has_stage_overlap(session.stages)),
        "average_session_duration_min": round(sum(durations) / len(durations), 3) if durations else 0,
        "average_training_session_duration_min": (
            round(sum(training_durations) / len(training_durations), 3) if training_durations else 0
        ),
        "stage_type_counts": stage_type_counts,
        "training_stage_type_counts": training_stage_type_counts,
        **exclusions,
        **candidate_summary,
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
    parser.add_argument("--candidate-interval-minutes", type=int, default=5)
    parser.add_argument("--wakeable-window-minutes", type=int, default=5)
    parser.add_argument("--deep-soon-window-minutes", type=int, default=10)
    parser.add_argument("--min-session-minutes", type=int, default=180)
    parser.add_argument("--max-session-minutes", type=int, default=720)
    parser.add_argument(
        "--include-deadline-candidate",
        action="store_true",
        help="Include candidate_time == deadline_time. Disabled by default to avoid session.end Unknown labels.",
    )
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
    write_sessions_csv(args.out_dir / "sessions.csv", sessions)
    write_stages_csv(args.out_dir / "stages.csv", sessions)
    write_sessions_csv(args.out_dir / "training_sessions.csv", training_sessions)
    candidates_path = args.out_dir / f"candidates_{args.candidate_interval_minutes}min.csv"
    candidate_count = write_candidates_csv(
        candidates_path,
        training_sessions,
        search_window_minutes=args.search_window_minutes,
        candidate_interval_minutes=args.candidate_interval_minutes,
        wakeable_window_minutes=args.wakeable_window_minutes,
        deep_soon_window_minutes=args.deep_soon_window_minutes,
        include_deadline_candidate=args.include_deadline_candidate,
    )
    candidate_summary = summarize_candidates(candidates_path)
    write_summary(
        args.out_dir / "summary.json",
        build_summary(sessions, training_sessions, errors, candidate_count, exclusions, candidate_summary),
    )

    print(f"Parsed sessions: {len(sessions)}")
    print(f"Training sessions: {len(training_sessions)}")
    print(f"Generated candidates: {candidate_count}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
