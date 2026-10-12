#!/usr/bin/env python3
"""Convert ONNX to fully-integer INT8 TFLite using onnx2tf."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf


def _load_onnx():
    try:
        import onnx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("onnx is required (`pip install onnx`)") from exc
    return onnx


def _find_onnx2tf_cmd() -> str:
    for cmd in ("onnx2tf",):
        resolved = shutil.which(cmd)
        if resolved:
            return resolved
        local = Path.home() / ".local" / "bin" / cmd
        if local.exists():
            return str(local)
    raise RuntimeError("onnx2tf CLI not found. Install via `pip install onnx2tf`.")


def _load_representative_data(rep_data_path: Path) -> np.ndarray:
    if rep_data_path.is_file():
        data = np.load(rep_data_path)
    else:
        arrays = [np.load(p) for p in sorted(rep_data_path.glob("*.npy"))]
        if not arrays:
            raise RuntimeError(f"No .npy files found under {rep_data_path}")
        data = np.concatenate(arrays, axis=0)

    if data.ndim < 2:
        raise RuntimeError(f"Representative data must be batched. Got shape {data.shape}")
    return data.astype(np.float32)


def _infer_onnx_layout(input_shape: list[int | str]) -> str:
    if len(input_shape) != 4:
        return "unknown"
    c_dim = input_shape[1]
    last_dim = input_shape[3]
    if isinstance(c_dim, int) and c_dim in (1, 3):
        return "nchw"
    if isinstance(last_dim, int) and last_dim in (1, 3):
        return "nhwc"
    return "unknown"


def _normalize_for_quant(data: np.ndarray) -> np.ndarray:
    min_v = float(np.min(data))
    max_v = float(np.max(data))
    if max_v - min_v < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    scaled = (data - min_v) / (max_v - min_v)
    return scaled.astype(np.float32)


def _select_quantized_tflite(
    output_dir: Path,
    model_stem: str,
    allow_non_int8_output: bool,
) -> Path:
    # Strict full-integer preference to avoid accidentally selecting int16_act
    # or float fallback artifacts.
    strict_candidates = [
        output_dir / f"{model_stem}_full_integer_quant.tflite",
        output_dir / "full_integer_quant.tflite",
    ]
    for candidate in strict_candidates:
        if candidate.exists():
            return candidate

    exact_named = sorted(output_dir.glob("*_full_integer_quant.tflite"))
    if exact_named:
        return exact_named[-1]

    if allow_non_int8_output:
        relaxed_patterns = [
            "*integer_quant_with_int16_act*.tflite",
            "*full_integer_quant_with_int16_act*.tflite",
            "*integer_quant*.tflite",
            "*dynamic_range_quant*.tflite",
            "*float16*.tflite",
            "*float32*.tflite",
        ]
        for pattern in relaxed_patterns:
            candidates = sorted(output_dir.glob(pattern))
            if candidates:
                return candidates[-1]

    generated = [p.name for p in sorted(output_dir.glob("*.tflite"))]
    raise RuntimeError(
        "No full integer INT8 TFLite artifact found. "
        "Generated artifacts: " + ", ".join(generated)
    )


def _probe_tflite_io_dtype(model_path: Path) -> tuple[str, str]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_dtype = str(interpreter.get_input_details()[0]["dtype"])
    output_dtype = str(interpreter.get_output_details()[0]["dtype"])
    return input_dtype, output_dtype


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX to full-int8 TFLite")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument(
        "--work-dir",
        default="build/onnx2tf_out",
        help="Working directory for generated artifacts",
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
        help="Layout of representative data before conversion (default: nhwc)",
    )
    parser.add_argument(
        "--max-representative-samples",
        type=int,
        default=100,
        help="Max calibration samples",
    )
    parser.add_argument(
        "--allow-non-int8-output",
        action="store_true",
        help=(
            "Allow selecting non-full-int8 artifacts (int16_act/dynamic/float) "
            "if full_integer_quant is unavailable."
        ),
    )
    parser.add_argument(
        "--transpose-nchw-to-nhwc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transpose representative data from NCHW to NHWC before calibration",
    )
    parser.add_argument(
        "--auto-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply min-max normalization to representative data before calibration",
    )
    parser.add_argument(
        "--quant-norm-mean",
        default="",
        help="Override onnx2tf -qnm argument (e.g. '[[[[0.0,0.0,0.0]]]]')",
    )
    parser.add_argument(
        "--quant-norm-std",
        default="",
        help="Override onnx2tf -qns argument (e.g. '[[[[1.0,1.0,1.0]]]]')",
    )
    args = parser.parse_args()

    onnx = _load_onnx()
    onnx2tf_cmd = _find_onnx2tf_cmd()
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX file not found: {onnx_path}")

    rep_path = Path(args.representative_data).resolve()
    if not rep_path.exists():
        raise RuntimeError(f"representative data path not found: {rep_path}")

    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError("ONNX model has no input tensors")
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    input_shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            input_shape.append(int(dim.dim_value))
        elif dim.dim_param:
            input_shape.append(str(dim.dim_param))
        else:
            input_shape.append("?")

    onnx_layout = _infer_onnx_layout(input_shape)
    rep_data = _load_representative_data(rep_path)[: args.max_representative_samples]

    # onnx2tf calibration path expects data shaped like the converted TF input,
    # which is typically NHWC for image tensors.
    if (
        rep_data.ndim == 4
        and args.input_layout == "nchw"
        and args.transpose_nchw_to_nhwc
    ):
        rep_data = np.transpose(rep_data, (0, 2, 3, 1))

    if args.auto_normalize:
        rep_data = _normalize_for_quant(rep_data)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="onnx2tf_calib_", dir=str(work_dir)) as tmp_dir:
        calib_path = Path(tmp_dir) / "calibration.npy"
        np.save(calib_path, rep_data)

        # Provide neutral normalization vectors expected by onnx2tf -oiqt.
        channels = 1
        if rep_data.ndim == 4:
            channels = int(rep_data.shape[3])
        mean = args.quant_norm_mean or ("[[[[" + ",".join(["0.0"] * channels) + "]]]]")
        std = args.quant_norm_std or ("[[[[" + ",".join(["1.0"] * channels) + "]]]]")

        cmd = [
            onnx2tf_cmd,
            "-i",
            str(onnx_path),
            "-o",
            str(work_dir),
            "-oiqt",
            "-qt",
            "per-channel",
            "-cind",
            input_name,
            str(calib_path),
            mean,
            std,
            "-iqd",
            "int8",
            "-oqd",
            "int8",
            "-tb",
            "tf_converter",
            "-qnm",
            mean,
            "-qns",
            std,
        ]

        env = dict(**os.environ)
        local_bin = str(Path.home() / ".local" / "bin")
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
        log_path = work_dir / "onnx2tf_quant.log"
        log_path.write_text(
            f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "onnx2tf integer quantization failed.\n"
                f"See log: {log_path}\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )

    produced = _select_quantized_tflite(
        output_dir=work_dir,
        model_stem=onnx_path.stem,
        allow_non_int8_output=args.allow_non_int8_output,
    )

    out_path = Path(args.tflite_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_path)
    print(f"[OK] int8 TFLite written: {out_path}")
    input_dtype, output_dtype = _probe_tflite_io_dtype(out_path)
    print(f"[INFO] selected model io dtypes: input={input_dtype}, output={output_dtype}")
    if not args.allow_non_int8_output and (
        "int8" not in input_dtype.lower() or "int8" not in output_dtype.lower()
    ):
        raise RuntimeError(
            "Selected artifact is not full int8 I/O. "
            f"Detected input={input_dtype}, output={output_dtype}."
        )

    meta = {
        "onnx": str(onnx_path),
        "onnx_input_name": input_name,
        "onnx_input_shape": input_shape,
        "onnx_layout": onnx_layout,
        "tflite_out": str(out_path),
        "tflite_source_artifact": str(produced),
        "selected_input_dtype": input_dtype,
        "selected_output_dtype": output_dtype,
        "representative_data": str(rep_path),
        "input_layout": args.input_layout,
        "transpose_nchw_to_nhwc": args.transpose_nchw_to_nhwc,
        "auto_normalize": args.auto_normalize,
        "quant_norm_mean": mean,
        "quant_norm_std": std,
        "allow_non_int8_output": args.allow_non_int8_output,
        "max_representative_samples": args.max_representative_samples,
        "work_dir": str(work_dir),
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
