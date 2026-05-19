#!/usr/bin/env python3
"""Create fixed Android TFLite parity sample assets from saved sequence outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_COLUMNS = [
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
    "sequence_awake_ratio",
    "sequence_light_ratio",
    "sequence_deep_ratio",
    "sequence_rem_ratio",
    "sequence_unknown_ratio",
    "sequence_stage_transition_count",
    "sequence_known_stage_transition_count",
    "sequence_known_ratio",
]

DEFAULT_TABULAR_COLUMNS = [
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
    "stage_at_candidate_Awake",
    "stage_at_candidate_Light",
    "stage_at_candidate_Rem",
    "previous_stage_Awake",
    "previous_stage_Deep",
    "previous_stage_Light",
    "previous_stage_Rem",
    "previous_stage_Unknown",
    "day_of_week_0",
    "day_of_week_1",
    "day_of_week_2",
    "day_of_week_3",
    "day_of_week_4",
    "day_of_week_5",
    "day_of_week_6",
]


def numeric_or_zero(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Android parity sample JSON assets.")
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--tabular-prediction-csv", type=Path)
    parser.add_argument("--tabular-candidates-csv", type=Path)
    parser.add_argument("--tabular-scaler-json", type=Path)
    parser.add_argument("--scaler-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--multi-output", type=Path)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    return parser.parse_args()


def build_sample(
    metadata: pd.DataFrame,
    stages: np.ndarray,
    predictions: pd.DataFrame,
    tabular_predictions: pd.DataFrame | None,
    tabular_candidates: pd.DataFrame | None,
    tabular_scaler: dict[str, object] | None,
    scaler: dict[str, object],
    row_index: int,
    source_prediction_csv: Path,
) -> dict[str, object]:
    context_columns = scaler.get("columns", DEFAULT_CONTEXT_COLUMNS)
    row = metadata.iloc[row_index]
    prediction = predictions.iloc[row_index]
    tabular_prediction = find_matching_prediction(row, tabular_predictions)
    tabular_candidate = find_matching_prediction(row, tabular_candidates)
    raw_context = []
    for column in context_columns:
        raw_context.append(numeric_or_zero(row.get(column, 0.0)))

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    raw = np.asarray(raw_context, dtype=np.float32)
    scaled = ((raw - mean) / std).astype(float).tolist()
    tabular_raw, tabular_scaled = build_tabular_features(tabular_candidate, tabular_scaler)

    sample = {
        "sample_id": f"alarm_row_{row_index}",
        "session_id": str(row["session_id"]),
        "candidate_time": str(row["candidate_time"]),
        "deadline_time": str(row["deadline_time"]),
        "stage_sequence_60m": stages[row_index].astype(int).tolist(),
        "context_columns": list(context_columns),
        "context_raw_22": raw.astype(float).tolist(),
        "context_scaled_22": scaled,
        "expected_gru_score": numeric_or_zero(prediction["probability"]),
        "expected_tabular_score": (
            numeric_or_zero(tabular_prediction["probability"])
            if tabular_prediction is not None
            else None
        ),
        "tabular_columns": list(tabular_scaler.get("columns", DEFAULT_TABULAR_COLUMNS))
        if tabular_scaler is not None
        else None,
        "tabular_raw_28": tabular_raw,
        "tabular_scaled_28": tabular_scaled,
        "model_version": "gru64_dense32_dropout00",
        "model_file": "mydream_sequence_gru64_dense32_dropout00/sequence_model_float32.tflite",
        "source_prediction_csv": str(source_prediction_csv),
    }
    return sample


def find_matching_prediction(row: pd.Series, predictions: pd.DataFrame | None) -> pd.Series | None:
    if predictions is None:
        return None
    matches = predictions[
        (predictions["session_id"].astype(str) == str(row["session_id"]))
        & (predictions["candidate_time"].astype(str) == str(row["candidate_time"]))
        & (predictions["deadline_time"].astype(str) == str(row["deadline_time"]))
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one matching tabular prediction for "
            f"session_id={row['session_id']} candidate_time={row['candidate_time']} "
            f"deadline_time={row['deadline_time']}, found {len(matches)}."
        )
    return matches.iloc[0]


def build_tabular_features(
    candidate: pd.Series | None,
    scaler: dict[str, object] | None,
) -> tuple[list[float] | None, list[float] | None]:
    if candidate is None or scaler is None:
        return None, None
    columns = list(scaler.get("columns", DEFAULT_TABULAR_COLUMNS))
    if columns != DEFAULT_TABULAR_COLUMNS:
        raise ValueError("Unexpected tabular column order.")

    raw_values: list[float] = []
    for column in columns:
        if column.startswith("stage_at_candidate_"):
            expected = column.removeprefix("stage_at_candidate_")
            raw_values.append(1.0 if str(candidate.get("stage_at_candidate", "Unknown")) == expected else 0.0)
        elif column.startswith("previous_stage_"):
            expected = column.removeprefix("previous_stage_")
            raw_values.append(1.0 if str(candidate.get("previous_stage", "Unknown")) == expected else 0.0)
        elif column.startswith("day_of_week_"):
            expected = int(column.removeprefix("day_of_week_"))
            raw_values.append(1.0 if int(numeric_or_zero(candidate.get("day_of_week", 0))) == expected else 0.0)
        else:
            raw_values.append(numeric_or_zero(candidate.get(column, 0.0)))

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    std[std == 0] = 1.0
    raw = np.asarray(raw_values, dtype=np.float32)
    scaled = ((raw - mean) / std).astype(float).tolist()
    return raw.astype(float).tolist(), scaled


def select_multi_sample_indices(metadata: pd.DataFrame, predictions: pd.DataFrame, samples_per_bucket: int) -> list[int]:
    work = metadata.copy()
    work["probability"] = pd.to_numeric(predictions["probability"], errors="coerce")
    work["row_index"] = np.arange(len(work))

    buckets: list[tuple[str, pd.DataFrame]] = [
        ("low", work.sort_values("probability", ascending=True)),
        ("mid", work.assign(distance=(work["probability"] - 0.5).abs()).sort_values("distance")),
        ("threshold", work.assign(distance=(work["probability"] - 0.55).abs()).sort_values("distance")),
        ("high", work.sort_values("probability", ascending=False)),
        (
            "unknown",
            work[work["sequence_unknown_ratio"] > 0].sort_values("sequence_unknown_ratio", ascending=False),
        ),
        (
            "deep_soon",
            work[pd.to_numeric(work["label_deep_soon"], errors="coerce") == 1].sort_values(
                "probability",
                ascending=False,
            ),
        ),
    ]

    selected: list[int] = []
    seen: set[int] = set()
    for _, bucket in buckets:
        for row_index in bucket["row_index"].astype(int).head(samples_per_bucket * 4):
            if row_index not in seen:
                selected.append(row_index)
                seen.add(row_index)
            if len([index for index in selected if index in set(bucket["row_index"].astype(int))]) >= samples_per_bucket:
                break
    return selected


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.sequence_dir / "sequence_metadata.csv")
    stages = np.load(args.sequence_dir / "sequence_stage_ids.npy")
    predictions = pd.read_csv(args.prediction_csv)
    tabular_predictions = (
        pd.read_csv(args.tabular_prediction_csv)
        if args.tabular_prediction_csv
        else None
    )
    tabular_candidates = (
        pd.read_csv(args.tabular_candidates_csv)
        if args.tabular_candidates_csv
        else None
    )
    tabular_scaler = (
        json.loads(args.tabular_scaler_json.read_text(encoding="utf-8"))
        if args.tabular_scaler_json
        else None
    )
    scaler = json.loads(args.scaler_json.read_text(encoding="utf-8"))
    context_columns = scaler.get("columns", DEFAULT_CONTEXT_COLUMNS)

    if metadata.empty:
        raise ValueError("Sequence metadata is empty.")
    if len(metadata) != len(stages):
        raise ValueError("Sequence metadata and stage array lengths differ.")
    if len(metadata) != len(predictions):
        raise ValueError("Sequence metadata and prediction rows differ.")
    if list(context_columns) != DEFAULT_CONTEXT_COLUMNS:
        raise ValueError("Unexpected context column order.")
    if args.row_index < 0 or args.row_index >= len(metadata):
        raise ValueError(f"row-index out of range: {args.row_index}")

    sample = build_sample(
        metadata,
        stages,
        predictions,
        tabular_predictions,
        tabular_candidates,
        tabular_scaler,
        scaler,
        args.row_index,
        args.prediction_csv,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"expected_gru_score={sample['expected_gru_score']}")

    if args.multi_output:
        indices = select_multi_sample_indices(metadata, predictions, args.samples_per_bucket)
        samples = [
            build_sample(
                metadata,
                stages,
                predictions,
                tabular_predictions,
                tabular_candidates,
                tabular_scaler,
                scaler,
                index,
                args.prediction_csv,
            )
            for index in indices
        ]
        payload = {
            "sample_count": len(samples),
            "threshold": 0.55,
            "model_version": "gru64_dense32_dropout00",
            "samples": samples,
        }
        args.multi_output.parent.mkdir(parents=True, exist_ok=True)
        args.multi_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.multi_output}")
        print(f"sample_count={len(samples)}")


if __name__ == "__main__":
    main()
