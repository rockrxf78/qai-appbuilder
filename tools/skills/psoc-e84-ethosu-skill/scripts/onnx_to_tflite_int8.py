#!/usr/bin/env python3
"""
Convert ONNX -> SavedModel -> fully integer TFLite (int8).
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _load_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TensorFlow is required for TFLite conversion (`pip install tensorflow`)."
        ) from exc
    return tf


def _convert_onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    try:
        mod = importlib.import_module("onnx2tf")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "onnx2tf is required for ONNX->SavedModel conversion (`pip install onnx2tf`)."
        ) from exc

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(mod, "convert"):
        try:
            # Preferred path for recent onnx2tf versions.
            mod.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
                non_verbose=True,
            )
            return
        except TypeError:
            # Fallback for versions with slightly different argument names.
            mod.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
            )
            return

    cli = ["onnx2tf", "-i", str(onnx_path), "-o", str(saved_model_dir)]
    completed = subprocess.run(cli, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "onnx2tf conversion failed.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def _iter_representative_samples(
    rep_data_path: Path, input_layout: str, max_samples: int
):
    if rep_data_path.is_file():
        data = np.load(rep_data_path)
    else:
        arrays = []
        for item in sorted(rep_data_path.glob("*.npy")):
            arrays.append(np.load(item))
        if not arrays:
            raise RuntimeError(f"No .npy files found under {rep_data_path}")
        data = np.concatenate(arrays, axis=0)

    if data.ndim < 2:
        raise RuntimeError(f"Representative data must be batched. Got shape {data.shape}")

    count = min(data.shape[0], max_samples)
    for idx in range(count):
        sample = data[idx : idx + 1].astype(np.float32)
        if input_layout == "nchw" and sample.ndim == 4:
            sample = np.transpose(sample, (0, 2, 3, 1))
        yield [sample]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX to fully-int8 TFLite")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument(
        "--saved-model-dir",
        default="",
        help="Reuse an existing SavedModel directory instead of running onnx2tf",
    )
    parser.add_argument(
        "--work-dir",
        default="build/onnx2tf_saved_model",
        help="Working directory for generated SavedModel",
    )
    parser.add_argument("--tflite-out", required=True, help="Output int8 TFLite file")
    parser.add_argument(
        "--representative-data",
        required=True,
        help="Path to representative dataset (.npy file or directory of .npy files)",
    )
    parser.add_argument(
        "--input-layout",
        default="nhwc",
        choices=["nhwc", "nchw"],
        help="Layout of representative data before feeding converter (default: nhwc)",
    )
    parser.add_argument(
        "--max-representative-samples",
        type=int,
        default=100,
        help="Max calibration samples",
    )
    parser.add_argument(
        "--force-float-io",
        action="store_true",
        help="Keep TFLite model float32 I/O while internal ops remain int8",
    )
    args = parser.parse_args()

    tf = _load_tensorflow()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX file not found: {onnx_path}")

    rep_path = Path(args.representative_data).resolve()
    if not rep_path.exists():
        raise RuntimeError(f"representative data path not found: {rep_path}")

    if args.saved_model_dir:
        saved_model_dir = Path(args.saved_model_dir).resolve()
    else:
        saved_model_dir = Path(args.work_dir).resolve()
        _convert_onnx_to_saved_model(onnx_path, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _iter_representative_samples(
        rep_path, args.input_layout, args.max_representative_samples
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if not args.force_float_io:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path = Path(args.tflite_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print(f"[OK] int8 TFLite written: {out_path}")

    meta = {
        "onnx": str(onnx_path),
        "saved_model_dir": str(saved_model_dir),
        "tflite_out": str(out_path),
        "representative_data": str(rep_path),
        "input_layout": args.input_layout,
        "force_float_io": args.force_float_io,
        "max_representative_samples": args.max_representative_samples,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] metadata written: {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
