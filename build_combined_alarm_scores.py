#!/usr/bin/env python3
"""Build combined MyDream alarm-window prediction sets.

The output directories are compatible with ``analyze_alarm_failures.py``. Each
recipe writes an ``alarm_predictions_long.csv`` whose ``probability`` column is
the candidate gate score for that recipe.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TARGET = "label_deep_soon"
KEY_COLUMNS = ["session_id", "candidate_time", "deadline_time"]
BASE_COLUMNS = [
    "session_id",
    "candidate_time",
    "deadline_time",
    "stage_at_candidate",
    "stage_at_deadline",
    "next_deep_start",
    "target",
    "actual",
    "probability",
    "prediction",
]


@dataclass(frozen=True)
class ScoreRecipe:
    name: str
    tabular_weight: float
    sequence_weight: float
    deadline_weight: float
    tabular_gate: float | None = None
    description: str = ""


DEFAULT_RECIPES = (
    ScoreRecipe(
        name="combined_gru_50_tab_50",
        tabular_weight=0.5,
        sequence_weight=0.5,
        deadline_weight=0.0,
        description="Equal tabular and GRU risk score.",
    ),
    ScoreRecipe(
        name="combined_gru_60_tab_30_deadline_10",
        tabular_weight=0.3,
        sequence_weight=0.6,
        deadline_weight=0.1,
        description="GRU-led score with a small deadline-closeness boost.",
    ),
    ScoreRecipe(
        name="combined_gru_60_tab_20_deadline_20",
        tabular_weight=0.2,
        sequence_weight=0.6,
        deadline_weight=0.2,
        description="GRU-led score with stronger deadline-closeness boost.",
    ),
    ScoreRecipe(
        name="combined_gru_70_tab_20_deadline_10",
        tabular_weight=0.2,
        sequence_weight=0.7,
        deadline_weight=0.1,
        description="Heavier GRU score with light tabular and deadline support.",
    ),
    ScoreRecipe(
        name="combined_gru_60_tab_gate_0_4_deadline_10",
        tabular_weight=0.3,
        sequence_weight=0.6,
        deadline_weight=0.1,
        tabular_gate=0.4,
        description="GRU-led score, but only candidates passing the current tabular gate stay eligible.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined alarm-window score outputs.")
    parser.add_argument(
        "--tabular-predictions",
        type=Path,
        default=Path("out/verify_week_period_profile/model_eval/alarm_predictions_long.csv"),
    )
    parser.add_argument(
        "--sequence-predictions",
        type=Path,
        default=Path("colab_result/alarm_predictions_long.csv"),
    )
    parser.add_argument(
        "--sequence-name",
        default="gru",
        help="Sequence model label used in metadata. Defaults to gru.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/verify_week_period_profile/combined_alarm_scores"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction column threshold for the generated CSVs. Analysis can still sweep other thresholds.",
    )
    return parser.parse_args()


def read_predictions(path: Path, probability_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    predictions = pd.read_csv(path)
    predictions = predictions[predictions["target"] == TARGET].copy()
    if predictions.empty:
        raise ValueError(f"No {TARGET} rows found in {path}")
    predictions[probability_name] = pd.to_numeric(predictions["probability"], errors="coerce").fillna(0.0)
    return predictions


def deadline_closeness(frame: pd.DataFrame) -> pd.Series:
    candidate_dt = pd.to_datetime(frame["candidate_time"], utc=True)
    deadline_dt = pd.to_datetime(frame["deadline_time"], utc=True)
    minutes_before_deadline = (deadline_dt - candidate_dt).dt.total_seconds() / 60
    return (1 - (minutes_before_deadline / 30)).clip(lower=0, upper=1)


def merge_predictions(tabular: pd.DataFrame, sequence: pd.DataFrame) -> pd.DataFrame:
    sequence_columns = KEY_COLUMNS + ["probability_sequence"]
    merged = tabular.merge(sequence.loc[:, sequence_columns], on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(tabular) or len(merged) != len(sequence):
        raise ValueError(
            "Tabular and sequence predictions do not align exactly: "
            f"tabular={len(tabular)} sequence={len(sequence)} merged={len(merged)}"
        )
    merged["deadline_closeness"] = deadline_closeness(merged)
    return merged


def apply_recipe(frame: pd.DataFrame, recipe: ScoreRecipe, threshold: float) -> pd.DataFrame:
    scored = frame.copy()
    score = (
        recipe.tabular_weight * scored["probability_tabular"]
        + recipe.sequence_weight * scored["probability_sequence"]
        + recipe.deadline_weight * scored["deadline_closeness"]
    )
    if recipe.tabular_gate is not None:
        score = score.where(scored["probability_tabular"] >= recipe.tabular_gate, 0.0)
    scored["probability"] = score.clip(lower=0, upper=1)
    scored["prediction"] = (scored["probability"] >= threshold).astype(int)
    return scored.loc[:, BASE_COLUMNS]


def recipe_metadata(recipe: ScoreRecipe, sequence_name: str) -> dict[str, Any]:
    return {
        "name": recipe.name,
        "sequence_name": sequence_name,
        "target": TARGET,
        "tabular_weight": recipe.tabular_weight,
        "sequence_weight": recipe.sequence_weight,
        "deadline_weight": recipe.deadline_weight,
        "tabular_gate": recipe.tabular_gate,
        "description": recipe.description,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tabular = read_predictions(args.tabular_predictions, "probability_tabular")
    sequence = read_predictions(args.sequence_predictions, "probability_sequence")
    merged = merge_predictions(tabular, sequence)

    manifest: list[dict[str, Any]] = []
    for recipe in DEFAULT_RECIPES:
        recipe_dir = output_dir / recipe.name
        recipe_dir.mkdir(parents=True, exist_ok=True)
        predictions = apply_recipe(merged, recipe, args.threshold)
        predictions.to_csv(recipe_dir / "alarm_predictions_long.csv", index=False)
        metadata = recipe_metadata(recipe, args.sequence_name)
        metadata.update(
            {
                "tabular_predictions": str(args.tabular_predictions),
                "sequence_predictions": str(args.sequence_predictions),
                "rows": int(len(predictions)),
                "sessions": int(predictions["session_id"].nunique()),
                "default_prediction_threshold": args.threshold,
                "score_min": float(predictions["probability"].min()),
                "score_max": float(predictions["probability"].max()),
                "score_mean": float(predictions["probability"].mean()),
            }
        )
        with (recipe_dir / "score_recipe.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        manifest.append({"model_dir": str(recipe_dir), **metadata})

    pd.DataFrame(manifest).to_csv(output_dir / "combined_score_manifest.csv", index=False)
    print(f"Wrote {output_dir / 'combined_score_manifest.csv'}")
    for row in manifest:
        print(f"Wrote {row['model_dir']}/alarm_predictions_long.csv")


if __name__ == "__main__":
    main()
