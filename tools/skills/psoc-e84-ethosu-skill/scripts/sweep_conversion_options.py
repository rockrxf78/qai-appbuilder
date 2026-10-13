#!/usr/bin/env python3
"""Small onnx2tf option grid + Vela ranker. Accepts a variant only with no CPU/SRAM/accuracy regression."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_GRID = [
    {"name": "default", "flextranspose": 6, "dsm": False, "keep_nchw": False},
    {"name": "dsm", "flextranspose": 6, "dsm": True, "keep_nchw": False},
    {"name": "nodaftc2", "flextranspose": 2, "dsm": False, "keep_nchw": False},
]


def _run(cmd: list[str], cwd: Path) -> None:
    print("-" * 72)
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cycles(report: dict) -> float | None:
    rows = report.get("summary_rows") or []
    if not rows:
        return None
    try:
        return float(rows[0].get("cycles_total") or 0)
    except (TypeError, ValueError):
        return None


def _cosine(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _tflite_infer(path: Path, nchw: "object") -> "object":
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    x = np.asarray(nchw, dtype=np.float32)
    if x.ndim == 4:
        x = np.transpose(x, (0, 2, 3, 1))
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
    else:
        x = x.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], x)
    interpreter.invoke()
    out = interpreter.get_output_details()[0]
    y = interpreter.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        ys, yz = out["quantization"]
        y = (y.astype(np.float32) - yz) * ys
    return y


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep a small onnx2tf option grid")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--representative-data", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--target", default="psoc_e84")
    parser.add_argument("--input-layout", default="nchw")
    parser.add_argument("--golden-input", default="")
    parser.add_argument("--golden-output", default="")
    parser.add_argument("--max-representative-samples", type=int, default=32)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_hint = script_dir.parents[2]
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for spec in DEFAULT_GRID:
        name = str(spec["name"])
        run_dir = work / name
        tflite = run_dir / "model_int8.tflite"
        vela_dir = run_dir / "vela"
        convert = [
            sys.executable,
            str(script_dir / "onnx_to_tflite_int8.py"),
            "--onnx",
            str(Path(args.onnx).resolve()),
            "--work-dir",
            str(run_dir / "onnx2tf"),
            "--tflite-out",
            str(tflite),
            "--representative-data",
            str(Path(args.representative_data).resolve()),
            "--input-layout",
            args.input_layout,
            "--max-representative-samples",
            str(args.max_representative_samples),
            "--onnx2tf-flextranspose-compression-dim",
            str(spec["flextranspose"]),
        ]
        if spec["dsm"]:
            convert.append("--onnx2tf-disable-strict-mode")
        if spec["keep_nchw"]:
            convert.append("--keep-nchw-input")
        entry: dict = {"name": name, "spec": spec, "ok": False}
        try:
            _run(convert, cwd=repo_hint)
            _run(
                [
                    sys.executable,
                    str(script_dir / "compile_with_vela.py"),
                    "--tflite",
                    str(tflite),
                    "--target",
                    args.target,
                    "--output-dir",
                    str(vela_dir),
                ],
                cwd=repo_hint,
            )
            vela = _read_json(vela_dir / "vela_compile_report.json")
            op = vela.get("operator_breakdown") or {}
            ts = vela.get("transpose_fallback_summary") or {}
            entry.update(
                {
                    "ok": True,
                    "tflite": str(tflite),
                    "vela_report": str(vela_dir / "vela_compile_report.json"),
                    "cpu_ops": int(op.get("cpu_operator_count") or 0),
                    "npu_ops": int(op.get("npu_operator_count") or 0),
                    "npu_percent": op.get("npu_operator_ratio_percent"),
                    "arena_kb": vela.get("arena_estimate_kb"),
                    "cycles": _cycles(vela),
                    "transpose_total": ts.get("total"),
                    "transpose_layout_glue": ts.get("export_qdq_layout_count", ts.get("removable_count")),
                    "transpose_inherent": ts.get("inherent_count"),
                }
            )
            if args.golden_input and args.golden_output:
                import numpy as np

                gin = np.load(args.golden_input)
                gout = np.load(args.golden_output)
                pred = _tflite_infer(tflite, gin)
                entry["cosine_vs_golden"] = _cosine(pred, gout)
                entry["max_abs_vs_golden"] = float(np.max(np.abs(pred.astype(np.float64) - gout.astype(np.float64))))
        except RuntimeError as exc:
            entry["error"] = str(exc)
        results.append(entry)

    ok_rows = [r for r in results if r.get("ok")]
    if not ok_rows:
        raise RuntimeError("All conversion/Vela variants failed. See sweep work-dir logs.")
    baseline = next((r for r in ok_rows if r["name"] == "default"), ok_rows[0])

    def _no_regression(row: dict) -> bool:
        if row["cpu_ops"] > int(baseline["cpu_ops"]):
            return False
        b_arena = baseline.get("arena_kb")
        r_arena = row.get("arena_kb")
        if isinstance(b_arena, (int, float)) and isinstance(r_arena, (int, float)) and r_arena > float(b_arena) * 1.05:
            return False
        b_cos = baseline.get("cosine_vs_golden")
        r_cos = row.get("cosine_vs_golden")
        if isinstance(b_cos, float) and isinstance(r_cos, float) and r_cos + 1e-3 < b_cos:
            return False
        return True

    eligible = [r for r in ok_rows if _no_regression(r)]
    eligible.sort(
        key=lambda r: (
            int(r.get("cpu_ops") or 10**9),
            float(r.get("arena_kb") or 10**9),
            float(r.get("cycles") or 10**9),
            -float(r.get("cosine_vs_golden") or 0.0),
        )
    )
    winner = eligible[0] if eligible else baseline
    summary = {
        "baseline": baseline["name"],
        "winner": winner["name"],
        "winner_accepted_no_regression": winner["name"] in {r["name"] for r in eligible},
        "grid": results,
        "note": "nodaftc2 is diagnostic; rejected automatically if CPU ops regress vs default.",
    }
    out = work / "sweep_report.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] {out}")
    print(f"[INFO] winner={winner['name']} cpu={winner.get('cpu_ops')} arena={winner.get('arena_kb')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
