#!/usr/bin/env python3
"""
Compare model outputs across PyTorch, ONNX Runtime, and TFLite against golden.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _metrics(candidate: np.ndarray, golden: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float64) - golden.astype(np.float64)
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(np.mean(np.square(diff)))),
        "cosine_similarity": _cosine_similarity(candidate, golden),
    }


def _run_onnx(onnx_path: Path, model_input: np.ndarray) -> np.ndarray:
    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("onnxruntime missing (`pip install onnxruntime`)") from exc
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: model_input.astype(np.float32)})[0]
    return np.asarray(output)


def _run_tflite(tflite_path: Path, model_input: np.ndarray, input_layout: str) -> np.ndarray:
    try:
        tf = importlib.import_module("tensorflow")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("tensorflow missing (`pip install tensorflow`)") from exc

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    prepared = model_input
    if input_layout == "nhwc" and prepared.ndim == 4:
        prepared = np.transpose(prepared, (0, 2, 3, 1))

    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            raise RuntimeError("invalid input quantization scale=0")
        prepared = np.round(prepared / scale + zero_point).astype(np.int8)
    else:
        prepared = prepared.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], prepared)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    if output_details["dtype"] == np.int8:
        scale, zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return np.asarray(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Accuracy comparison utility")
    parser.add_argument("--golden-input", required=True, help="Path to golden input .npy")
    parser.add_argument(
        "--golden-output", required=True, help="Path to golden output .npy (PyTorch baseline)"
    )
    parser.add_argument("--onnx", default="", help="Optional ONNX for onnxruntime compare")
    parser.add_argument("--tflite", default="", help="Optional TFLite for compare")
    parser.add_argument(
        "--tflite-input-layout",
        default="nhwc",
        choices=["nchw", "nhwc"],
        help="Layout expected by TFLite model",
    )
    parser.add_argument("--report-json", default="", help="Optional JSON report path")
    args = parser.parse_args()

    x = np.load(Path(args.golden_input).resolve())
    y_golden = np.load(Path(args.golden_output).resolve())

    report = {
        "golden_input_shape": list(x.shape),
        "golden_output_shape": list(y_golden.shape),
        "comparisons": {},
    }

    if args.onnx:
        y_onnx = _run_onnx(Path(args.onnx).resolve(), x)
        report["comparisons"]["onnx_vs_golden"] = _metrics(y_onnx, y_golden)

    if args.tflite:
        y_tflite = _run_tflite(Path(args.tflite).resolve(), x, args.tflite_input_layout)
        report["comparisons"]["tflite_vs_golden"] = _metrics(y_tflite, y_golden)

    print("=" * 72)
    print("Accuracy comparison")
    print("=" * 72)
    for name, metrics in report["comparisons"].items():
        print(f"[{name}]")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    if args.report_json:
        out_path = Path(args.report_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] report written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
