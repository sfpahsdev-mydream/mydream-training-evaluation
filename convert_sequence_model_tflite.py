#!/usr/bin/env python3
"""Convert a trained MyDream sequence Keras model to TensorFlow Lite.

Run this in Colab or a TensorFlow-enabled server. The script can also compare
Keras and TFLite outputs on a small sample from a sequence dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL_DIR = Path("out/verify_week_period_profile/sequence_experiments/gru/gru64_dense32_dropout00")
DEFAULT_OUTPUT_DIR = Path("out/verify_week_period_profile/tflite/gru64_dense32_dropout00")
DEFAULT_SEQUENCE_DIR = Path("out/verify_week_period_profile/sequence_60m_alarm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MyDream sequence model to TFLite.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--sample-rows", type=int, default=256)
    parser.add_argument("--float16", action="store_true", help="Also write a float16-quantized TFLite model.")
    parser.add_argument(
        "--no-rebuild-unrolled-gru",
        action="store_true",
        help="Convert the saved model directly instead of rebuilding GRU with unroll=True.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def load_context_scaler(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "context_scaler.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing context scaler: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sequence_metrics(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "sequence_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence metrics: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sample(sequence_dir: Path, scaler: dict[str, Any], sample_rows: int) -> tuple[np.ndarray, np.ndarray]:
    metadata = pd.read_csv(sequence_dir / "sequence_metadata.csv")
    stages = np.load(sequence_dir / "sequence_stage_ids.npy").astype(np.int32)
    if len(metadata) != len(stages):
        raise ValueError("sequence_metadata.csv and sequence_stage_ids.npy row counts do not match")

    row_count = min(sample_rows, len(metadata))
    stage_sample = stages[:row_count]

    columns = scaler["columns"]
    context = pd.DataFrame(index=metadata.index[:row_count])
    for column in columns:
        if column in metadata.columns:
            context[column] = pd.to_numeric(metadata.loc[: row_count - 1, column], errors="coerce").fillna(0.0)
        else:
            context[column] = 0.0
    values = context.to_numpy(dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    std[std == 0] = 1.0
    context_sample = (values - mean) / std
    return stage_sample, context_sample.astype(np.float32)


def convert_float32(tf: Any, model: Any, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def convert_float16(tf: Any, model: Any, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def layer_by_name(model: Any, name: str) -> Any:
    try:
        return model.get_layer(name)
    except ValueError as error:
        raise ValueError(f"Saved model is missing expected layer {name!r}") from error


def rebuild_tflite_friendly_model(tf: Any, saved_model: Any, context_dim: int, model_type: str) -> Any:
    """Rebuild recurrent architectures with static unrolled recurrence.

    Colab GPU-trained GRU layers can save as CudnnRNN/TensorList graphs that
    are awkward for TFLite. Rebuilding the same Keras layers with
    ``unroll=True`` keeps the learned weights while producing a simpler
    conversion graph. Non-recurrent architectures should use
    ``--no-rebuild-unrolled-gru``.
    """
    keras = tf.keras
    layers = keras.layers

    if model_type not in {"gru", "cnn_gru"}:
        raise ValueError(
            f"Model type {model_type!r} is not supported by recurrent rebuild. "
            "Use --no-rebuild-unrolled-gru for non-recurrent architectures.",
        )

    embedding_layer = layer_by_name(saved_model, "stage_embedding")
    gru_layer = layer_by_name(saved_model, "stage_gru")
    context_dense_layer = layer_by_name(saved_model, "context_dense")
    dense_layer = layer_by_name(saved_model, "dense")
    output_layer = layer_by_name(saved_model, "p_next_deep_10m")

    stage_input = keras.Input(shape=(60,), dtype="int32", name="stage_sequence")
    context_input = keras.Input(shape=(context_dim,), dtype="float32", name="context")
    x = layers.Embedding(
        input_dim=embedding_layer.input_dim,
        output_dim=embedding_layer.output_dim,
        name="stage_embedding",
    )(stage_input)
    if model_type == "cnn_gru":
        conv_layer = layer_by_name(saved_model, "stage_conv")
        x = layers.Conv1D(
            conv_layer.filters,
            kernel_size=conv_layer.kernel_size[0],
            padding=conv_layer.padding,
            activation=conv_layer.activation,
            name="stage_conv",
        )(x)
    x = layers.GRU(
        gru_layer.units,
        dropout=0.0,
        recurrent_dropout=0.0,
        reset_after=gru_layer.reset_after,
        unroll=True,
        name="stage_gru",
    )(x)
    context_branch = layers.Dense(context_dense_layer.units, activation="relu", name="context_dense")(context_input)
    merged = layers.Concatenate(name="merge")([x, context_branch])
    merged = layers.Dense(dense_layer.units, activation="relu", name="dense")(merged)
    output = layers.Dense(output_layer.units, activation="sigmoid", name="p_next_deep_10m")(merged)
    rebuilt = keras.Model(inputs=[stage_input, context_input], outputs=output)

    layer_names = ["stage_embedding", "stage_gru", "context_dense", "dense", "p_next_deep_10m"]
    if model_type == "cnn_gru":
        layer_names.insert(1, "stage_conv")
    for name in layer_names:
        rebuilt.get_layer(name).set_weights(saved_model.get_layer(name).get_weights())
    return rebuilt


def run_tflite(tf: Any, tflite_path: Path, stage_sample: np.ndarray, context_sample: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    outputs: list[float] = []
    for stage_row, context_row in zip(stage_sample, context_sample):
        for detail in input_details:
            dtype = detail["dtype"]
            shape = detail["shape"]
            if np.issubdtype(dtype, np.integer):
                value = stage_row.reshape(shape).astype(dtype)
            else:
                value = context_row.reshape(shape).astype(dtype)
            interpreter.set_tensor(detail["index"], value)
        interpreter.invoke()
        outputs.append(float(interpreter.get_tensor(output_details[0]["index"]).reshape(-1)[0]))
    return np.asarray(outputs, dtype=np.float32)


def verify_model(tf: Any, model: Any, tflite_path: Path, sequence_dir: Path, scaler: dict[str, Any], sample_rows: int) -> dict[str, Any]:
    stage_sample, context_sample = load_sample(sequence_dir, scaler, sample_rows)
    keras_output = model.predict(
        {"stage_sequence": stage_sample, "context": context_sample},
        batch_size=min(256, len(stage_sample)),
        verbose=0,
    ).reshape(-1)
    tflite_output = run_tflite(tf, tflite_path, stage_sample, context_sample)
    diff = np.abs(keras_output - tflite_output)
    return {
        "sample_rows": int(len(stage_sample)),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "keras_min": float(keras_output.min()) if len(keras_output) else 0.0,
        "keras_max": float(keras_output.max()) if len(keras_output) else 0.0,
        "tflite_min": float(tflite_output.min()) if len(tflite_output) else 0.0,
        "tflite_max": float(tflite_output.max()) if len(tflite_output) else 0.0,
    }


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    model_path = args.model_dir / "sequence_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Keras model: {model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_model = tf.keras.models.load_model(model_path)
    scaler = load_context_scaler(args.model_dir)
    metrics = load_sequence_metrics(args.model_dir)
    model_type = metrics.get("architecture", {}).get("model_type", "gru")
    model = (
        saved_model
        if args.no_rebuild_unrolled_gru
        else rebuild_tflite_friendly_model(tf, saved_model, context_dim=len(scaler["columns"]), model_type=model_type)
    )

    float32_path = args.output_dir / "sequence_model_float32.tflite"
    convert_float32(tf, model, float32_path)
    manifest: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "sequence_dir": str(args.sequence_dir),
        "float32_tflite": str(float32_path),
        "float32_size_bytes": float32_path.stat().st_size,
        "context_columns": scaler["columns"],
        "model_type": model_type,
        "conversion_model": "saved_model_direct" if args.no_rebuild_unrolled_gru else "rebuilt_tflite_friendly",
    }

    if args.float16:
        float16_path = args.output_dir / "sequence_model_float16.tflite"
        convert_float16(tf, model, float16_path)
        manifest["float16_tflite"] = str(float16_path)
        manifest["float16_size_bytes"] = float16_path.stat().st_size

    if not args.skip_verify:
        source_model = saved_model if not args.no_rebuild_unrolled_gru else model
        if not args.no_rebuild_unrolled_gru:
            stage_sample, context_sample = load_sample(args.sequence_dir, scaler, args.sample_rows)
            saved_output = saved_model.predict(
                {"stage_sequence": stage_sample, "context": context_sample},
                batch_size=min(256, len(stage_sample)),
                verbose=0,
            ).reshape(-1)
            rebuilt_output = model.predict(
                {"stage_sequence": stage_sample, "context": context_sample},
                batch_size=min(256, len(stage_sample)),
                verbose=0,
            ).reshape(-1)
            rebuild_diff = np.abs(saved_output - rebuilt_output)
            manifest["rebuild_verification"] = {
                "sample_rows": int(len(stage_sample)),
                "max_abs_diff": float(rebuild_diff.max()) if len(rebuild_diff) else 0.0,
                "mean_abs_diff": float(rebuild_diff.mean()) if len(rebuild_diff) else 0.0,
            }
        manifest["float32_verification"] = verify_model(
            tf,
            source_model,
            float32_path,
            args.sequence_dir,
            scaler,
            args.sample_rows,
        )

    for filename in ("context_scaler.json", "sequence_metrics.json"):
        source = args.model_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)

    with (args.output_dir / "tflite_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Wrote {float32_path}")
    if args.float16:
        print(f"Wrote {args.output_dir / 'sequence_model_float16.tflite'}")
    print(f"Wrote {args.output_dir / 'tflite_manifest.json'}")


if __name__ == "__main__":
    main()
