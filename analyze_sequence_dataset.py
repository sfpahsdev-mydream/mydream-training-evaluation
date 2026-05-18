#!/usr/bin/env python3
"""Analyze a MyDream Phase 2 sequence dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KNOWN_RATIO_BINS = [-0.001, 0.25, 0.5, 0.75, 0.999, 1.0]
KNOWN_RATIO_LABELS = ["0-25%", "25-50%", "50-75%", "75-99%", "100%"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MyDream sequence dataset quality.")
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        stage_to_id = json.load(handle)
    id_to_stage = {int(value): key for key, value in stage_to_id.items()}
    return stage_to_id, id_to_stage


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def split_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in metadata.groupby("split", sort=False):
        positive = int(group["label_deep_soon"].sum())
        full_known = int((group["sequence_unknown_timestep_count"] == 0).sum())
        rows.append(
            {
                "split": split,
                "sequence_count": int(len(group)),
                "label_deep_soon_positive": positive,
                "label_deep_soon_positive_ratio": ratio(positive, len(group)),
                "average_known_ratio": round(float(group["sequence_known_ratio"].mean()), 6),
                "full_known_sequence_count": full_known,
                "full_known_sequence_ratio": ratio(full_known, len(group)),
                "avg_minutes_before_deadline": round(float(group["minutes_before_deadline"].mean()), 3),
                "median_minutes_before_deadline": round(float(group["minutes_before_deadline"].median()), 3),
            }
        )
    return pd.DataFrame(rows)


def candidate_stage_label_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, group in metadata.groupby("stage_at_candidate", sort=True):
        positive = int(group["label_deep_soon"].sum())
        rows.append(
            {
                "stage_at_candidate": stage,
                "sequence_count": int(len(group)),
                "label_deep_soon_positive": positive,
                "label_deep_soon_positive_ratio": ratio(positive, len(group)),
                "average_known_ratio": round(float(group["sequence_known_ratio"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def known_ratio_buckets(metadata: pd.DataFrame) -> pd.DataFrame:
    buckets = pd.cut(
        metadata["sequence_known_ratio"],
        bins=KNOWN_RATIO_BINS,
        labels=KNOWN_RATIO_LABELS,
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for bucket, group in metadata.groupby(buckets, observed=False):
        if group.empty:
            count = 0
            positive = 0
        else:
            count = int(len(group))
            positive = int(group["label_deep_soon"].sum())
        rows.append(
            {
                "known_ratio_bucket": str(bucket),
                "sequence_count": count,
                "label_deep_soon_positive": positive,
                "label_deep_soon_positive_ratio": ratio(positive, count),
            }
        )
    return pd.DataFrame(rows)


def position_stage_ratios(sequences: np.ndarray, id_to_stage: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(sequences)
    window_minutes = sequences.shape[1]
    for position in range(window_minutes):
        values = sequences[:, position]
        row: dict[str, Any] = {
            "position": position,
            "minutes_before_candidate": window_minutes - 1 - position,
        }
        for stage_id, stage_name in sorted(id_to_stage.items()):
            count = int((values == stage_id).sum())
            row[f"{stage_name}_count"] = count
            row[f"{stage_name}_ratio"] = ratio(count, total)
        rows.append(row)
    return pd.DataFrame(rows)


def transition_counts(sequences: np.ndarray, id_to_stage: dict[int, str]) -> pd.DataFrame:
    left = sequences[:, :-1].reshape(-1)
    right = sequences[:, 1:].reshape(-1)
    pairs = pd.DataFrame({"from_id": left, "to_id": right})
    counts = pairs.value_counts(["from_id", "to_id"]).reset_index(name="count")
    total = int(counts["count"].sum())
    counts["from_stage"] = counts["from_id"].map(id_to_stage)
    counts["to_stage"] = counts["to_id"].map(id_to_stage)
    counts["ratio"] = counts["count"].map(lambda value: ratio(int(value), total))
    return counts.loc[:, ["from_stage", "to_stage", "count", "ratio"]].sort_values("count", ascending=False)


def end_stage_match_summary(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    stage_to_id: dict[str, int],
) -> dict[str, Any]:
    expected = metadata["stage_at_candidate"].map(stage_to_id).fillna(stage_to_id["Unknown"]).to_numpy(dtype=np.uint8)
    actual = sequences[:, -1]
    matches = actual == expected
    mismatch_count = int((~matches).sum())
    by_stage: dict[str, Any] = {}
    for stage, stage_id in sorted(stage_to_id.items()):
        mask = expected == stage_id
        by_stage[stage] = {
            "count": int(mask.sum()),
            "mismatch_count": int((mask & (~matches)).sum()),
            "mismatch_ratio": ratio(int((mask & (~matches)).sum()), int(mask.sum())),
        }
    return {
        "candidate_end_stage_match_count": int(matches.sum()),
        "candidate_end_stage_mismatch_count": mismatch_count,
        "candidate_end_stage_mismatch_ratio": ratio(mismatch_count, len(matches)),
        "by_stage": by_stage,
    }


def build_report(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    stage_to_id: dict[str, int],
    id_to_stage: dict[int, str],
) -> dict[str, Any]:
    label_positive = int(metadata["label_deep_soon"].sum())
    unknown_id = stage_to_id["Unknown"]
    unknown_timestep_count = int((sequences == unknown_id).sum())
    total_timesteps = int(sequences.size)
    return {
        "sequence_count": int(len(metadata)),
        "window_minutes": int(sequences.shape[1]),
        "label_deep_soon_positive_count": label_positive,
        "label_deep_soon_positive_ratio": ratio(label_positive, len(metadata)),
        "unknown_timestep_count": unknown_timestep_count,
        "unknown_timestep_ratio": ratio(unknown_timestep_count, total_timesteps),
        "average_known_ratio": round(float(metadata["sequence_known_ratio"].mean()), 6),
        "full_known_sequence_count": int((metadata["sequence_unknown_timestep_count"] == 0).sum()),
        "full_known_sequence_ratio": ratio(int((metadata["sequence_unknown_timestep_count"] == 0).sum()), len(metadata)),
        "end_stage_match": end_stage_match_summary(metadata, sequences, stage_to_id),
    }


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir
    output_dir = args.output_dir or sequence_dir / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(sequence_dir / "sequence_metadata.csv")
    sequences = np.load(sequence_dir / "sequence_stage_ids.npy")
    stage_to_id, id_to_stage = load_vocab(sequence_dir / "stage_vocab.json")
    if len(metadata) != len(sequences):
        raise ValueError(
            f"metadata rows ({len(metadata)}) do not match sequence rows ({len(sequences)})"
        )

    report = build_report(metadata, sequences, stage_to_id, id_to_stage)
    with (output_dir / "sequence_quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    split_summary(metadata).to_csv(output_dir / "sequence_split_summary.csv", index=False)
    candidate_stage_label_summary(metadata).to_csv(
        output_dir / "sequence_candidate_stage_label_summary.csv",
        index=False,
    )
    known_ratio_buckets(metadata).to_csv(output_dir / "sequence_known_ratio_buckets.csv", index=False)
    position_stage_ratios(sequences, id_to_stage).to_csv(
        output_dir / "sequence_position_stage_ratios.csv",
        index=False,
    )
    transition_counts(sequences, id_to_stage).to_csv(output_dir / "sequence_transition_counts.csv", index=False)

    print(f"Wrote {output_dir / 'sequence_quality_report.json'}")
    print(f"Wrote {output_dir / 'sequence_split_summary.csv'}")


if __name__ == "__main__":
    main()
