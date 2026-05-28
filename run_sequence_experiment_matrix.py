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
    context_units: int = 16
    conv_filters: int = 16
    conv_kernel_size: int = 5
    tcn_dilations: str = "1,2,4,8"
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 128


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

EXPANDED_ARCHITECTURE_EXPERIMENTS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MyDream sequence experiment matrix.")
    parser.add_argument(
        "--profile-root",
        type=Path,
        help=(
            "Dataset/result root with sequence_60m, sequence_60m_alarm, sequence_experiments/gru, "
            "and optional model_tabular_tflite children. Overrides default input/output roots."
        ),
    )
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--predict-sequence-dir", type=Path, default=DEFAULT_ALARM_SEQUENCE_DIR)
    parser.add_argument("--tabular-model-dir", type=Path, default=DEFAULT_TABULAR_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-set", choices=["gru", "cnn_gru", "expanded", "large", "all"], default="gru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--comparison-model-dir",
        type=Path,
        action="append",
        default=[],
        help="Existing model output directory to include in final comparison without retraining. Repeatable.",
    )
    parser.add_argument(
        "--no-tabular-model",
        action="store_true",
        help="Do not include --tabular-model-dir in final comparison.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training candidates whose output already contains alarm_predictions_long.csv.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()
    if args.profile_root:
        args.sequence_dir = args.profile_root / "sequence_60m"
        args.predict_sequence_dir = args.profile_root / "sequence_60m_alarm"
        args.tabular_model_dir = args.profile_root / "model_tabular_tflite"
        args.output_root = args.profile_root / "sequence_experiments"
        if not args.comparison_model_dir:
            args.comparison_model_dir = [
                args.profile_root / "sequence_experiments" / "gru" / "gru64_dense32_dropout00"
            ]
    return args


def selected_experiments(name: str) -> tuple[Experiment, ...]:
    if name == "gru":
        return GRU_TUNING_EXPERIMENTS
    if name == "cnn_gru":
        return CNN_GRU_EXPERIMENTS
    if name == "expanded":
        return EXPANDED_ARCHITECTURE_EXPERIMENTS
    if name == "large":
        return LARGE_CAPACITY_EXPERIMENTS
    return (
        GRU_TUNING_EXPERIMENTS
        + CNN_GRU_EXPERIMENTS
        + EXPANDED_ARCHITECTURE_EXPERIMENTS
        + LARGE_CAPACITY_EXPERIMENTS
    )


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
        str(args.random_state),
    ]


def analyze_command(args: argparse.Namespace, model_dirs: list[Path], output_dir: Path) -> list[str]:
    command = [sys.executable, "analyze_alarm_failures.py"]
    if not args.no_tabular_model:
        command.extend(["--model-dir", str(args.tabular_model_dir)])
    for model_dir in args.comparison_model_dir:
        command.extend(["--model-dir", str(model_dir)])
    for model_dir in model_dirs:
        command.extend(["--model-dir", str(model_dir)])
    command.extend(["--output-dir", str(output_dir)])
    return command


def main() -> None:
    args = parse_args()
    experiments = selected_experiments(args.experiment_set)
    output_root = args.output_root / args.experiment_set
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    model_dirs: list[Path] = []
    for experiment in experiments:
        model_dir = output_root / experiment.name
        model_dirs.append(model_dir)
        predictions_path = model_dir / "alarm_predictions_long.csv"
        if args.skip_existing and predictions_path.exists():
            print(f"Skipping existing result: {predictions_path}")
            continue
        run_command(train_command(args, experiment, model_dir), args.dry_run)

    comparison_dir = output_root / "alarm_failure_comparison"
    run_command(analyze_command(args, model_dirs, comparison_dir), args.dry_run)


if __name__ == "__main__":
    main()
