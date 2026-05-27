#!/usr/bin/env python3
"""Repeat selected sequence-model training across random seeds and aggregate results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class Experiment:
    name: str
    model_type: str
    hidden_units: int
    dense_units: int
    dropout: float
    conv_filters: int = 16


EXPERIMENTS = (
    Experiment("gru64_dense32_dropout00", "gru", hidden_units=64, dense_units=32, dropout=0.0),
    Experiment("transformer64_dense32_dropout10", "transformer", hidden_units=64, dense_units=32, dropout=0.1),
    Experiment(
        "cnn32_gru64_dense32_dropout00",
        "cnn_gru",
        hidden_units=64,
        dense_units=32,
        dropout=0.0,
        conv_filters=32,
    ),
)
METRICS = ("deep_success", "strong_fail", "success_per_smart", "sleep_quality_success", "smart_alarm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat MyDream sequence model evaluation across random seeds.")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seeds = args.seeds or list(DEFAULT_SEEDS)
    args.output_root = args.output_root or args.profile_root / "sequence_experiments" / "repeated_evaluation"
    return args


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def train_command(
    project_root: Path,
    args: argparse.Namespace,
    experiment: Experiment,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(project_root / "train_sequence_colab.py"),
        "--sequence-dir",
        str(args.profile_root / "sequence_60m"),
        "--predict-sequence-dir",
        str(args.profile_root / "sequence_60m_alarm"),
        "--output-dir",
        str(output_dir),
        "--model-type",
        experiment.model_type,
        "--hidden-units",
        str(experiment.hidden_units),
        "--dense-units",
        str(experiment.dense_units),
        "--dropout",
        str(experiment.dropout),
        "--conv-filters",
        str(experiment.conv_filters),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--random-state",
        str(seed),
    ]


def analyze_command(project_root: Path, args: argparse.Namespace, model_dirs: list[Path], output_dir: Path) -> list[str]:
    command = [sys.executable, str(project_root / "analyze_alarm_failures.py")]
    for model_dir in model_dirs:
        command.extend(["--model-dir", str(model_dir)])
    command.extend(
        [
            "--threshold",
            str(args.threshold),
            "--focus-threshold",
            str(args.threshold),
            "--output-dir",
            str(output_dir),
        ]
    )
    return command


def aggregate_results(args: argparse.Namespace, summary_frames: list[pd.DataFrame]) -> None:
    per_seed = pd.concat(summary_frames, ignore_index=True)
    per_seed.to_csv(args.output_root / "per_seed_summary.csv", index=False)

    numeric_metrics = [metric for metric in METRICS if metric in per_seed.columns]
    aggregate = (
        per_seed.groupby("model", as_index=False)[numeric_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate.columns
    ]
    aggregate.to_csv(args.output_root / "aggregate_summary.csv", index=False)

    baseline = per_seed[per_seed["model"] == "gru64_dense32_dropout00"].set_index("seed")
    delta_rows: list[dict[str, float | int | str]] = []
    for model, model_rows in per_seed.groupby("model"):
        if model == "gru64_dense32_dropout00":
            continue
        for row in model_rows.itertuples(index=False):
            reference = baseline.loc[row.seed]
            delta_rows.append(
                {
                    "seed": int(row.seed),
                    "model": model,
                    "deep_success_delta_vs_gru": int(row.deep_success - reference["deep_success"]),
                    "strong_fail_delta_vs_gru": int(row.strong_fail - reference["strong_fail"]),
                    "success_per_smart_delta_vs_gru": float(
                        row.success_per_smart - reference["success_per_smart"]
                    ),
                    "sleep_quality_success_delta_vs_gru": float(
                        row.sleep_quality_success - reference["sleep_quality_success"]
                    ),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(args.output_root / "delta_vs_gru_per_seed.csv", index=False)

    delta_summary = (
        deltas.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "count"),
            deep_success_delta_mean=("deep_success_delta_vs_gru", "mean"),
            deep_success_delta_min=("deep_success_delta_vs_gru", "min"),
            strong_fail_delta_mean=("strong_fail_delta_vs_gru", "mean"),
            strong_fail_delta_max=("strong_fail_delta_vs_gru", "max"),
            success_per_smart_delta_mean=("success_per_smart_delta_vs_gru", "mean"),
            deep_success_not_worse_count=("deep_success_delta_vs_gru", lambda rows: int((rows >= 0).sum())),
            strong_fail_improved_count=("strong_fail_delta_vs_gru", lambda rows: int((rows < 0).sum())),
        )
    )
    delta_summary.to_csv(args.output_root / "delta_vs_gru_summary.csv", index=False)
    print(f"Wrote {args.output_root / 'per_seed_summary.csv'}")
    print(f"Wrote {args.output_root / 'aggregate_summary.csv'}")
    print(f"Wrote {args.output_root / 'delta_vs_gru_per_seed.csv'}")
    print(f"Wrote {args.output_root / 'delta_vs_gru_summary.csv'}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        seed_root = args.output_root / f"seed_{seed}"
        model_dirs: list[Path] = []
        for experiment in EXPERIMENTS:
            model_dir = seed_root / experiment.name
            model_dirs.append(model_dir)
            predictions_path = model_dir / "alarm_predictions_long.csv"
            if args.skip_existing and predictions_path.exists():
                print(f"Skipping existing result: {predictions_path}")
            else:
                run(train_command(project_root, args, experiment, seed, model_dir), args.dry_run)

        comparison_dir = seed_root / "alarm_failure_comparison"
        run(analyze_command(project_root, args, model_dirs, comparison_dir), args.dry_run)
        if not args.dry_run:
            seed_summary = pd.read_csv(comparison_dir / "model_comparison_summary.csv")
            seed_summary.insert(0, "seed", seed)
            summary_frames.append(seed_summary)

    if not args.dry_run:
        aggregate_results(args, summary_frames)


if __name__ == "__main__":
    main()
