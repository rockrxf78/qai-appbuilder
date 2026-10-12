#!/usr/bin/env python3
"""
Staged verification loop for PSoC E84 Ethos-U workflow.

Stages (AIPC-inspired):
  1. Preflight
  2. Export baseline (PyTorch -> ONNX + golden tensors)
  3. ONNX checks
  4. ONNX -> TFLite int8
  5. Vela compile + fallback analysis
  6. Accuracy checks (PyTorch/ONNX/TFLite)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("-" * 72)
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"stage failed with exit={result.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PSoC E84 Ethos-U staged pipeline")
    parser.add_argument("--work-dir", default="build/pipeline_demo", help="Output directory")
    parser.add_argument(
        "--target",
        default="psoc_e84",
        help="Target key from _ethosu_targets.py",
    )
    parser.add_argument(
        "--representative-data",
        default="",
        help="Representative dataset path (.npy or folder). If omitted, tinycnn demo data is generated.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight stage",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow compile_with_vela.py to fallback to built-in Vela profile",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = work_dir / "tinycnn.onnx"
    tflite_path = work_dir / "tinycnn_int8.tflite"
    golden_dir = work_dir / "golden"
    rep_data = (
        Path(args.representative_data).resolve()
        if args.representative_data
        else work_dir / "rep_data.npy"
    )
    vela_out_dir = work_dir / "vela"

    if not args.representative_data:
        _run(
            [
                sys.executable,
                str(root_dir / "examples" / "generate_representative_data.py"),
                "--output",
                str(rep_data),
                "--shape",
                "64,3,32,32",
            ],
            cwd=root_dir,
        )

    if not args.skip_preflight:
        _run([sys.executable, str(script_dir / "ethosu_preflight_check.py")], cwd=root_dir)

    _run(
        [
            sys.executable,
            str(script_dir / "export_pytorch_to_onnx.py"),
            "--output-onnx",
            str(onnx_path),
            "--golden-dir",
            str(golden_dir),
        ],
        cwd=root_dir,
    )
    _run(
        [
            sys.executable,
            str(script_dir / "onnx_sanity_check.py"),
            "--onnx",
            str(onnx_path),
            "--json-report",
            str(work_dir / "onnx_check_report.json"),
        ],
        cwd=root_dir,
    )
    _run(
        [
            sys.executable,
            str(script_dir / "onnx_to_tflite_int8.py"),
            "--onnx",
            str(onnx_path),
            "--tflite-out",
            str(tflite_path),
            "--representative-data",
            str(rep_data),
            "--input-layout",
            "nchw",
        ],
        cwd=root_dir,
    )
    vela_cmd = [
        sys.executable,
        str(script_dir / "compile_with_vela.py"),
        "--tflite",
        str(tflite_path),
        "--target",
        args.target,
        "--output-dir",
        str(vela_out_dir),
    ]
    if args.allow_fallback:
        vela_cmd.append("--allow-fallback")
    _run(vela_cmd, cwd=root_dir)
    _run(
        [
            sys.executable,
            str(script_dir / "compare_accuracy.py"),
            "--golden-input",
            str(golden_dir / "golden_input.npy"),
            "--golden-output",
            str(golden_dir / "golden_output.npy"),
            "--onnx",
            str(onnx_path),
            "--tflite",
            str(tflite_path),
            "--tflite-input-layout",
            "nhwc",
            "--report-json",
            str(work_dir / "accuracy_report.json"),
        ],
        cwd=root_dir,
    )

    print("=" * 72)
    print("Pipeline completed")
    print("=" * 72)
    print(f"work_dir: {work_dir}")
    print(f"onnx: {onnx_path}")
    print(f"tflite: {tflite_path}")
    print(f"vela report: {vela_out_dir / 'vela_compile_report.json'}")
    print(f"accuracy report: {work_dir / 'accuracy_report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
