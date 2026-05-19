#!/usr/bin/env python3
"""Run MyDream sequence-model experiment matrices.

This helper is intended for Colab or a GPU/CPU server with TensorFlow
installed. It runs configured sequence-model variants and then compares their
alarm-window behavior with ``analyze_alarm_failures.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEQUENCE_DIR = Path("out/verify_week_period_profile/sequence_60m")
DEFAULT_ALARM_SEQUENCE_DIR = Path("out/verify_week_period_profile/sequence_60m_alarm")
DEFAULT_TABULAR_MODEL_DIR = Path("out/verify_week_period_profile/model_eval")
DEFAULT_OUTPUT_ROOT = Path("out/verify_week_period_profile/sequence_experiments")


@dataclass(frozen=True)
class Experiment:
    name: str
    model_type: str
    hidden_units: int
    dense_units: int
    dropout: float
    conv_filters: int = 16
    conv_kernel_size: int = 5


GRU_TUNING_EXPERIMENTS = (
    Experiment("gru32_dense16_dropout02", "gru", hidden_units=32, dense_units=16, dropout=0.2),
    Experiment("gru64_dense16_dropout02", "gru", hidden_units=64, dense_units=16, dropout=0.2),
    Experiment("gru32_dense32_dropout02", "gru", hidden_units=32, dense_units=32, dropout=0.2),
    Experiment("gru64_dense32_dropout02", "gru", hidden_units=64, dense_units=32, dropout=0.2),
    Experiment("gru64_dense32_dropout00", "gru", hidden_units=64, dense_units=32, dropout=0.0),
)

CNN_GRU_EXPERIMENTS = (
    Experiment("cnn16_gru32_dense16_dropout02", "cnn_gru", hidden_units=32, dense_units=16, dropout=0.2),
    Experiment("cnn32_gru32_dense16_dropout02", "cnn_gru", hidden_units=32, dense_units=16, dropout=0.2, conv_filters=32),
    Experiment("cnn16_gru64_dense16_dropout02", "cnn_gru", hidden_units=64, dense_units=16, dropout=0.2),
    Experiment("cnn32_gru64_dense32_dropout02", "cnn_gru", hidden_units=64, dense_units=32, dropout=0.2, conv_filters=32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MyDream sequence experiment matrix.")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--predict-sequence-dir", type=Path, default=DEFAULT_ALARM_SEQUENCE_DIR)
    parser.add_argument("--tabular-model-dir", type=Path, default=DEFAULT_TABULAR_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-set", choices=["gru", "cnn_gru", "all"], default="gru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def selected_experiments(name: str) -> tuple[Experiment, ...]:
    if name == "gru":
        return GRU_TUNING_EXPERIMENTS
    if name == "cnn_gru":
        return CNN_GRU_EXPERIMENTS
    return GRU_TUNING_EXPERIMENTS + CNN_GRU_EXPERIMENTS


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def train_command(args: argparse.Namespace, experiment: Experiment, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "train_sequence_colab.py",
        "--sequence-dir",
        str(args.sequence_dir),
        "--predict-sequence-dir",
        str(args.predict_sequence_dir),
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
        "--conv-kernel-size",
        str(experiment.conv_kernel_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--random-state",
        str(args.random_state),
    ]


def analyze_command(args: argparse.Namespace, model_dirs: list[Path], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "analyze_alarm_failures.py",
        "--model-dir",
        str(args.tabular_model_dir),
    ]
    for model_dir in model_dirs:
        command.extend(["--model-dir", str(model_dir)])
    command.extend(["--output-dir", str(output_dir)])
    return command


def main() -> None:
    args = parse_args()
    experiments = selected_experiments(args.experiment_set)
    output_root = args.output_root / args.experiment_set
    output_root.mkdir(parents=True, exist_ok=True)

    model_dirs: list[Path] = []
    for experiment in experiments:
        model_dir = output_root / experiment.name
        model_dirs.append(model_dir)
        run_command(train_command(args, experiment, model_dir), args.dry_run)

    comparison_dir = output_root / "alarm_failure_comparison"
    run_command(analyze_command(args, model_dirs, comparison_dir), args.dry_run)


if __name__ == "__main__":
    main()
