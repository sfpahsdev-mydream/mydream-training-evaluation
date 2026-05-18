#!/usr/bin/env python3
"""Build a Phase 2 sleep-pattern sequence dataset for MyDream.

The dataset uses candidate rows as labels/metadata and reconstructs the recent
sleep-stage sequence from ``stages.csv``. This keeps Deep-stage history in the
sequence even though Deep candidates are excluded from tabular training rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_TO_ID = {
    "Unknown": 0,
    "Awake": 1,
    "Light": 2,
    "Deep": 3,
    "Rem": 4,
}
ID_TO_STAGE = {value: key for key, value in STAGE_TO_ID.items()}
DEFAULT_WINDOW_MINUTES = 60
MINUTE_SECONDS = 60


METADATA_COLUMNS = [
    "sequence_id",
    "split",
    "session_id",
    "candidate_time",
    "deadline_time",
    "wake_time_source",
    "stage_at_candidate",
    "stage_at_deadline",
    "next_deep_start",
    "minutes_before_deadline",
    "elapsed_sleep_minutes",
    "time_of_day_sin",
    "time_of_day_cos",
    "target_wake_hour_sin",
    "target_wake_hour_cos",
    "day_of_week",
    "minutes_since_stage_start",
    "minutes_since_last_deep",
    "deep_cycle_position",
    "recent_30m_awake_minutes",
    "recent_30m_light_minutes",
    "recent_30m_deep_minutes",
    "recent_30m_rem_minutes",
    "recent_30m_unknown_minutes",
    "sequence_known_timestep_count",
    "sequence_unknown_timestep_count",
    "sequence_known_ratio",
    "label_wakeable_window",
    "label_wakeable_at_candidate",
    "label_deep_soon",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MyDream sequence dataset.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--candidates-file", default="training_candidates_1min.csv")
    parser.add_argument("--stages-file", default="stages.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, help="Optional debug limit after sorting by time.")
    return parser.parse_args()


def chronological_session_splits(
    candidates: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if validation_ratio + test_ratio >= 1:
        raise ValueError("--validation-ratio + --test-ratio must be less than 1")

    session_dates = (
        candidates.groupby("session_id", as_index=False)["candidate_dt"]
        .min()
        .sort_values("candidate_dt")
    )
    test_count = max(1, round(len(session_dates) * test_ratio))
    validation_count = max(1, round(len(session_dates) * validation_ratio))
    train_sessions = session_dates.iloc[: -(validation_count + test_count)]["session_id"]
    validation_sessions = session_dates.iloc[-(validation_count + test_count) : -test_count]["session_id"]
    test_sessions = session_dates.iloc[-test_count:]["session_id"]

    split_by_session = {session_id: "train" for session_id in train_sessions}
    split_by_session.update({session_id: "validation" for session_id in validation_sessions})
    split_by_session.update({session_id: "test" for session_id in test_sessions})
    return split_by_session


def build_session_stage_grid(
    session_candidates: pd.DataFrame,
    session_stages: pd.DataFrame,
    window_minutes: int,
) -> tuple[np.ndarray, int]:
    candidate_seconds = session_candidates["candidate_seconds"].to_numpy(dtype=np.int64)
    min_query_seconds = int(candidate_seconds.min() - (window_minutes - 1) * MINUTE_SECONDS)
    max_query_seconds = int(candidate_seconds.max())
    grid_seconds = np.arange(
        min_query_seconds,
        max_query_seconds + MINUTE_SECONDS,
        MINUTE_SECONDS,
        dtype=np.int64,
    )
    stage_grid = np.zeros(len(grid_seconds), dtype=np.uint8)

    for stage in session_stages.itertuples(index=False):
        stage_type = getattr(stage, "type")
        stage_id = STAGE_TO_ID.get(stage_type, STAGE_TO_ID["Unknown"])
        start_seconds = int(getattr(stage, "start_seconds"))
        end_seconds = int(getattr(stage, "end_seconds"))
        start_index = max(0, int(np.searchsorted(grid_seconds, start_seconds, side="left")))
        end_index = min(len(grid_seconds), int(np.searchsorted(grid_seconds, end_seconds, side="left")))
        if end_index > start_index:
            stage_grid[start_index:end_index] = stage_id

    return stage_grid, min_query_seconds


def build_sequences(
    candidates: pd.DataFrame,
    stages: pd.DataFrame,
    window_minutes: int,
) -> np.ndarray:
    sequences = np.zeros((len(candidates), window_minutes), dtype=np.uint8)
    offsets = np.arange(window_minutes - 1, -1, -1, dtype=np.int64) * MINUTE_SECONDS
    stages_by_session = {session_id: group for session_id, group in stages.groupby("session_id", sort=False)}

    for session_id, session_candidates in candidates.groupby("session_id", sort=False):
        session_stages = stages_by_session.get(session_id)
        if session_stages is None:
            continue
        stage_grid, base_seconds = build_session_stage_grid(session_candidates, session_stages, window_minutes)
        candidate_seconds = session_candidates["candidate_seconds"].to_numpy(dtype=np.int64)
        indices = ((candidate_seconds[:, None] - offsets[None, :] - base_seconds) // MINUTE_SECONDS).astype(np.int64)
        valid = (indices >= 0) & (indices < len(stage_grid))
        session_sequences = np.zeros(indices.shape, dtype=np.uint8)
        session_sequences[valid] = stage_grid[indices[valid]]
        sequences[session_candidates.index.to_numpy()] = session_sequences

    return sequences


def sequence_summary(candidates: pd.DataFrame, sequences: np.ndarray, window_minutes: int) -> dict[str, Any]:
    stage_counts = {
        ID_TO_STAGE[stage_id]: int((sequences == stage_id).sum())
        for stage_id in sorted(ID_TO_STAGE)
    }
    split_counts = {str(key): int(value) for key, value in candidates["split"].value_counts().to_dict().items()}
    label_positive = int(candidates["label_deep_soon"].sum())
    unknown_timestep_count = int((sequences == STAGE_TO_ID["Unknown"]).sum())
    total_timesteps = int(sequences.size)
    return {
        "sequence_count": int(len(candidates)),
        "window_minutes": int(window_minutes),
        "shape": [int(sequences.shape[0]), int(sequences.shape[1])],
        "split_counts": split_counts,
        "label_deep_soon_positive_count": label_positive,
        "label_deep_soon_positive_ratio": round(label_positive / len(candidates), 6) if len(candidates) else 0.0,
        "stage_timestep_counts": stage_counts,
        "unknown_timestep_count": unknown_timestep_count,
        "unknown_timestep_ratio": round(unknown_timestep_count / total_timesteps, 6) if total_timesteps else 0.0,
        "full_known_sequence_count": int((candidates["sequence_unknown_timestep_count"] == 0).sum()),
        "full_known_sequence_ratio": round(
            float((candidates["sequence_unknown_timestep_count"] == 0).mean()) if len(candidates) else 0.0,
            6,
        ),
        "average_known_ratio": round(float(candidates["sequence_known_ratio"].mean()) if len(candidates) else 0.0, 6),
        "candidate_stage_counts": {
            str(key): int(value) for key, value in candidates["stage_at_candidate"].value_counts().to_dict().items()
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / f"sequence_{args.window_minutes}m"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = input_dir / args.candidates_file
    stages_path = input_dir / args.stages_file
    candidates = pd.read_csv(candidates_path)
    stages = pd.read_csv(stages_path)

    required_candidate_columns = {"session_id", "candidate_time", "label_deep_soon"}
    required_stage_columns = {"session_id", "type", "start", "end"}
    missing_candidates = sorted(required_candidate_columns - set(candidates.columns))
    missing_stages = sorted(required_stage_columns - set(stages.columns))
    if missing_candidates:
        raise ValueError(f"Missing candidate columns: {missing_candidates}")
    if missing_stages:
        raise ValueError(f"Missing stage columns: {missing_stages}")

    candidates["candidate_dt"] = pd.to_datetime(candidates["candidate_time"], utc=True)
    candidates = candidates.sort_values(["session_id", "candidate_dt"]).reset_index(drop=True)
    if args.max_rows:
        candidates = candidates.head(args.max_rows).copy()
    candidates["candidate_seconds"] = candidates["candidate_dt"].map(lambda value: int(value.timestamp()))

    stages["start_dt"] = pd.to_datetime(stages["start"], utc=True)
    stages["end_dt"] = pd.to_datetime(stages["end"], utc=True)
    stages["start_seconds"] = stages["start_dt"].map(lambda value: int(value.timestamp()))
    stages["end_seconds"] = stages["end_dt"].map(lambda value: int(value.timestamp()))

    split_by_session = chronological_session_splits(candidates, args.validation_ratio, args.test_ratio)
    candidates["split"] = candidates["session_id"].map(split_by_session).fillna("train")

    sequences = build_sequences(candidates, stages, args.window_minutes)
    unknown_counts = (sequences == STAGE_TO_ID["Unknown"]).sum(axis=1)
    candidates["sequence_id"] = np.arange(len(candidates), dtype=np.int64)
    candidates["sequence_unknown_timestep_count"] = unknown_counts.astype(int)
    candidates["sequence_known_timestep_count"] = args.window_minutes - candidates["sequence_unknown_timestep_count"]
    candidates["sequence_known_ratio"] = candidates["sequence_known_timestep_count"] / args.window_minutes

    metadata_columns = [column for column in METADATA_COLUMNS if column in candidates.columns]
    metadata = candidates.loc[:, metadata_columns].copy()

    np.save(output_dir / "sequence_stage_ids.npy", sequences)
    metadata.to_csv(output_dir / "sequence_metadata.csv", index=False)
    with (output_dir / "stage_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(STAGE_TO_ID, handle, indent=2, sort_keys=True)
    summary = sequence_summary(candidates, sequences, args.window_minutes)
    with (output_dir / "sequence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote {output_dir / 'sequence_stage_ids.npy'}")
    print(f"Wrote {output_dir / 'sequence_metadata.csv'}")
    print(f"Wrote {output_dir / 'sequence_summary.json'}")


if __name__ == "__main__":
    main()
