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
    context_units: int = 16
    conv_filters: int = 16
    conv_kernel_size: int = 5
    tcn_dilations: str = "1,2,4,8"
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 128


PACKAGED_EXPERIMENTS = (
    Experiment("gru64_dense32_dropout00", "gru", hidden_units=64, dense_units=32, dropout=0.0),
    Experiment("tcn64_dense32_dropout00", "tcn", hidden_units=64, dense_units=32, dropout=0.0),
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
LARGE_CAPACITY_EXPERIMENTS = (
    Experiment(
        "gru128_dense64_dropout10",
        "gru",
        hidden_units=128,
        dense_units=64,
        dropout=0.1,
        context_units=32,
    ),
    Experiment(
        "gru256_dense128_dropout10",
        "gru",
        hidden_units=256,
        dense_units=128,
        dropout=0.1,
        context_units=64,
    ),
    Experiment(
        "cnn64_gru128_dense128_dropout10",
        "cnn_gru",
        hidden_units=128,
        dense_units=128,
        dropout=0.1,
        context_units=64,
        conv_filters=64,
    ),
    Experiment(
        "cnn64_gru256_dense128_dropout10",
        "cnn_gru",
        hidden_units=256,
        dense_units=128,
        dropout=0.1,
        context_units=64,
        conv_filters=64,
    ),
    Experiment(
        "tcn128_dense128_dropout10",
        "tcn",
        hidden_units=128,
        dense_units=128,
        dropout=0.1,
        context_units=64,
        conv_filters=128,
    ),
    Experiment(
        "tcn256_dense128_dropout10",
        "tcn",
        hidden_units=256,
        dense_units=128,
        dropout=0.1,
        context_units=64,
        conv_filters=256,
    ),
    Experiment(
        "transformer128_dense128_dropout10",
        "transformer",
        hidden_units=128,
        dense_units=128,
        dropout=0.1,
        context_units=64,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=256,
    ),
    Experiment(
        "transformer256_2layer_dense128_dropout10",
        "transformer",
        hidden_units=256,
        dense_units=128,
        dropout=0.1,
        context_units=64,
        transformer_heads=8,
        transformer_layers=2,
        transformer_ff_dim=512,
    ),
)
METRICS = ("deep_success", "strong_fail", "success_per_smart", "sleep_quality_success", "smart_alarm")
POLICY_METRICS = (
    "precision",
    "recall",
    "utility_labeled",
    "utility_all",
    "true_smart",
    "false_smart",
    "missed_smart",
    "coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat MyDream sequence model evaluation across random seeds.")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    parser.add_argument(
        "--experiment-set",
        choices=["packaged", "large", "all"],
        default="packaged",
        help="Candidate set to repeat. Defaults to the already packaged Android candidates.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--evaluate-policies",
        action="store_true",
        help="Run coverage-aware candidate policy evaluation for every repeated sequence model.",
    )
    parser.add_argument(
        "--tabular-predictions",
        type=Path,
        help="Tabular alarm predictions used by --evaluate-policies. Defaults under --profile-root.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seeds = args.seeds or list(DEFAULT_SEEDS)
    args.thresholds = args.thresholds or [0.55]
    default_output_name = (
        "repeated_evaluation"
        if args.experiment_set == "packaged"
        else f"repeated_evaluation_{args.experiment_set}"
    )
    args.output_root = args.output_root or args.profile_root / "sequence_experiments" / default_output_name
    args.tabular_predictions = (
        args.tabular_predictions or args.profile_root / "model_tabular_tflite" / "alarm_predictions_long.csv"
    )
    return args


def selected_experiments(name: str) -> tuple[Experiment, ...]:
    if name == "packaged":
        return PACKAGED_EXPERIMENTS
    if name == "large":
        return (PACKAGED_EXPERIMENTS[0],) + LARGE_CAPACITY_EXPERIMENTS
    return PACKAGED_EXPERIMENTS + LARGE_CAPACITY_EXPERIMENTS


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
        "--context-units",
        str(experiment.context_units),
        "--dropout",
        str(experiment.dropout),
        "--conv-filters",
        str(experiment.conv_filters),
        "--conv-kernel-size",
        str(experiment.conv_kernel_size),
        "--tcn-dilations",
        experiment.tcn_dilations,
        "--transformer-heads",
        str(experiment.transformer_heads),
        "--transformer-layers",
        str(experiment.transformer_layers),
        "--transformer-ff-dim",
        str(experiment.transformer_ff_dim),
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
    for threshold in args.thresholds:
        command.extend(["--threshold", str(threshold), "--focus-threshold", str(threshold)])
    command.extend(["--output-dir", str(output_dir)])
    return command


def policy_command(project_root: Path, args: argparse.Namespace, model_dirs: list[Path], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "evaluate_decision_policies.py"),
        "--profile-root",
        str(args.profile_root),
        "--tabular-predictions",
        str(args.tabular_predictions),
        "--output-dir",
        str(output_dir),
    ]
    for model_dir in model_dirs:
        command.extend(
            [
                "--sequence-prediction",
                f"{model_dir.name}={model_dir / 'alarm_predictions_long.csv'}",
            ]
        )
    for threshold in args.thresholds:
        command.extend(["--threshold", str(threshold)])
    return command


def aggregate_results(args: argparse.Namespace, summary_frames: list[pd.DataFrame]) -> None:
    per_seed = pd.concat(summary_frames, ignore_index=True)
    per_seed.to_csv(args.output_root / "per_seed_summary.csv", index=False)

    numeric_metrics = [metric for metric in METRICS if metric in per_seed.columns]
    aggregate = (
        per_seed.groupby(["model", "threshold"], as_index=False)[numeric_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate.columns
    ]
    aggregate.to_csv(args.output_root / "aggregate_summary.csv", index=False)

    baseline_name = "gru64_dense32_dropout00"
    baseline = per_seed[per_seed["model"] == baseline_name].loc[
        :, ["seed", "threshold", "deep_success", "strong_fail", "success_per_smart", "sleep_quality_success"]
    ]
    baseline = baseline.rename(
        columns={
            "deep_success": "deep_success_gru",
            "strong_fail": "strong_fail_gru",
            "success_per_smart": "success_per_smart_gru",
            "sleep_quality_success": "sleep_quality_success_gru",
        }
    )
    deltas = per_seed[per_seed["model"] != baseline_name].merge(
        baseline,
        on=["seed", "threshold"],
        how="left",
        validate="many_to_one",
    )
    deltas["deep_success_delta_vs_gru"] = deltas["deep_success"] - deltas["deep_success_gru"]
    deltas["strong_fail_delta_vs_gru"] = deltas["strong_fail"] - deltas["strong_fail_gru"]
    deltas["success_per_smart_delta_vs_gru"] = deltas["success_per_smart"] - deltas["success_per_smart_gru"]
    deltas["sleep_quality_success_delta_vs_gru"] = (
        deltas["sleep_quality_success"] - deltas["sleep_quality_success_gru"]
    )
    deltas = deltas.loc[
        :,
        [
            "seed",
            "threshold",
            "model",
            "deep_success_delta_vs_gru",
            "strong_fail_delta_vs_gru",
            "success_per_smart_delta_vs_gru",
            "sleep_quality_success_delta_vs_gru",
        ],
    ]
    deltas.to_csv(args.output_root / "delta_vs_gru_per_seed.csv", index=False)

    delta_summary = (
        deltas.groupby(["model", "threshold"], as_index=False)
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


def aggregate_policy_results(args: argparse.Namespace, policy_frames: list[pd.DataFrame]) -> None:
    per_seed = pd.concat(policy_frames, ignore_index=True)
    per_seed.to_csv(args.output_root / "policy_per_seed_summary.csv", index=False)
    numeric_metrics = [metric for metric in POLICY_METRICS if metric in per_seed.columns]
    aggregate = (
        per_seed.groupby(["sequence_model", "policy_variant", "threshold"], as_index=False)[numeric_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate.columns
    ]
    aggregate.to_csv(args.output_root / "policy_aggregate_summary.csv", index=False)
    print(f"Wrote {args.output_root / 'policy_per_seed_summary.csv'}")
    print(f"Wrote {args.output_root / 'policy_aggregate_summary.csv'}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    policy_frames: list[pd.DataFrame] = []
    experiments = selected_experiments(args.experiment_set)
    for seed in args.seeds:
        seed_root = args.output_root / f"seed_{seed}"
        model_dirs: list[Path] = []
        for experiment in experiments:
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
        if args.evaluate_policies:
            policy_dir = seed_root / "policy_evaluation"
            run(policy_command(project_root, args, model_dirs, policy_dir), args.dry_run)
            if not args.dry_run:
                policy_summary = pd.read_csv(policy_dir / "policy_summary.csv")
                policy_summary.insert(0, "seed", seed)
                policy_frames.append(policy_summary)

    if not args.dry_run:
        aggregate_results(args, summary_frames)
        if args.evaluate_policies:
            aggregate_policy_results(args, policy_frames)


if __name__ == "__main__":
    main()
