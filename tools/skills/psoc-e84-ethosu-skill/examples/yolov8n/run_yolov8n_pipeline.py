#!/usr/bin/env python3
"""
Run an end-to-end YOLOv8n pipeline for the PSoC E84 Ethos-U skill:
PyTorch -> ONNX -> full-int8 TFLite -> Vela -> accuracy/fallback reports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def _run(cmd: list[str], cwd: Path) -> None:
    print("-" * 72)
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def _collect_images_from_dataset(dataset_yaml: str) -> list[Path]:
    ds = check_det_dataset(dataset_yaml)
    val = ds.get("val")
    base_path = Path(ds.get("path", ".")).resolve()
    images: list[Path] = []
    if isinstance(val, (list, tuple)):
        images = [Path(x).resolve() for x in val]
    elif isinstance(val, str):
        val_path = Path(val)
        if not val_path.is_absolute():
            val_path = (base_path / val_path).resolve()
        if val_path.is_dir():
            images = sorted(
                [
                    p
                    for p in val_path.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]
            )
        elif val_path.is_file():
            entries = [line.strip() for line in val_path.read_text(encoding="utf-8").splitlines()]
            images = [((base_path / e).resolve() if not Path(e).is_absolute() else Path(e)) for e in entries if e]
    images = [p for p in images if p.exists()]
    if not images:
        raise RuntimeError(f"No validation images resolved from dataset yaml: {dataset_yaml}")
    return images


def _preprocess_rgb(path: Path, imgsz: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _save_representative_data(image_paths: list[Path], output_path: Path, imgsz: int, max_samples: int) -> np.ndarray:
    sample_paths = image_paths[:max_samples]
    data = np.stack([_preprocess_rgb(p, imgsz) for p in sample_paths], axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)
    return data


def _export_yolo_to_onnx(weights: str, onnx_out: Path, imgsz: int) -> Path:
    model = YOLO(weights)
    exported = model.export(format="onnx", imgsz=imgsz, opset=13, dynamic=False, simplify=False)
    exported_path = Path(exported).resolve()
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != onnx_out:
        onnx_out.write_bytes(exported_path.read_bytes())
    return onnx_out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _run_tflite(interpreter, input_nchw: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1))
    if input_detail["dtype"] == np.int8:
        scale, zero = input_detail["quantization"]
        if scale == 0:
            raise RuntimeError("TFLite input scale is zero")
        x = np.clip(np.round(input_nhwc / scale + zero), -128, 127).astype(np.int8)
    else:
        x = input_nhwc.astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail["index"])
    if output_detail["dtype"] == np.int8:
        y_scale, y_zero = output_detail["quantization"]
        y = (y.astype(np.float32) - y_zero) * y_scale
    return np.asarray(y, dtype=np.float32)


def _measure_raw_output_diff(onnx_path: Path, tflite_path: Path, image_paths: list[Path], imgsz: int) -> dict[str, float]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input = sess.get_inputs()[0].name

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    cosines: list[float] = []
    max_abs: list[float] = []
    maes: list[float] = []
    for image_path in image_paths:
        x_nchw = _preprocess_rgb(image_path, imgsz)[None, ...]
        y_onnx = np.asarray(sess.run(None, {onnx_input: x_nchw})[0], dtype=np.float32)
        y_tfl = _run_tflite(interpreter, x_nchw)
        diff = y_tfl - y_onnx
        cosines.append(_cosine(y_tfl, y_onnx))
        max_abs.append(float(np.max(np.abs(diff))))
        maes.append(float(np.mean(np.abs(diff))))
    return {
        "num_images": len(image_paths),
        "cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
        "cosine_min": float(np.min(cosines)) if cosines else 0.0,
        "max_abs_mean": float(np.mean(max_abs)) if max_abs else 0.0,
        "max_abs_max": float(np.max(max_abs)) if max_abs else 0.0,
        "mae_mean": float(np.mean(maes)) if maes else 0.0,
    }


def _run_detection_compare_subprocess(
    repo_root: Path,
    helper_script: Path,
    reference_model: str,
    candidate_model: str,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
    output_json: Path,
) -> None:
    image_list = output_json.with_suffix(".images.txt")
    image_list.write_text("\n".join(str(p) for p in image_paths), encoding="utf-8")
    cmd = [
        sys.executable,
        str(helper_script),
        "--reference-model",
        reference_model,
        "--candidate-model",
        candidate_model,
        "--image-list",
        str(image_list),
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--output-json",
        str(output_json),
    ]
    completed = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if completed.returncode != 0:
        payload = {
            "num_images": len(image_paths),
            "comparison_failed": True,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8n Ethos-U staged pipeline")
    parser.add_argument("--work-dir", default="tools/skills/psoc-e84-ethosu-skill/build/yolov8n")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--dataset-yaml", default="coco128.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--target", default="psoc_e84")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    skill_root = script_path.parents[2]
    scripts_dir = skill_root / "scripts"
    repo_root = skill_root.parents[2]
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = work_dir / "yolov8n.onnx"
    rep_data_path = work_dir / "yolo_rep_data.npy"
    onnx2tf_out = work_dir / "onnx2tf_out"
    tflite_out = work_dir / "yolov8n_int8.tflite"
    vela_dir = work_dir / "vela"
    raw_diff_json = work_dir / "yolov8n_raw_output_diff.json"
    pt_vs_onnx_json = work_dir / "yolov8n_detection_match_pt_vs_onnx.json"
    pt_vs_tfl_json = work_dir / "yolov8n_detection_match_pt_vs_tflite.json"

    image_paths = _collect_images_from_dataset(args.dataset_yaml)
    _save_representative_data(image_paths, rep_data_path, args.imgsz, args.calib_samples)
    eval_paths = image_paths[: args.eval_samples]

    if args.force_export or not onnx_path.exists():
        _export_yolo_to_onnx(args.weights, onnx_path, args.imgsz)

    _run(
        [
            sys.executable,
            str(scripts_dir / "onnx_sanity_check.py"),
            "--onnx",
            str(onnx_path),
            "--json-report",
            str(work_dir / "yolov8n_onnx_check_report.json"),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(scripts_dir / "onnx_to_tflite_int8.py"),
            "--onnx",
            str(onnx_path),
            "--work-dir",
            str(onnx2tf_out),
            "--tflite-out",
            str(tflite_out),
            "--representative-data",
            str(rep_data_path),
            "--input-layout",
            "nchw",
            "--max-representative-samples",
            str(args.calib_samples),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(scripts_dir / "compile_with_vela.py"),
            "--tflite",
            str(tflite_out),
            "--target",
            args.target,
            "--output-dir",
            str(vela_dir),
        ],
        cwd=repo_root,
    )

    raw_diff = _measure_raw_output_diff(onnx_path, tflite_out, eval_paths, args.imgsz)
    raw_diff_json.write_text(json.dumps(raw_diff, indent=2), encoding="utf-8")

    compare_helper = script_path.parent / "detection_match.py"
    _run_detection_compare_subprocess(
        repo_root=repo_root,
        helper_script=compare_helper,
        reference_model=args.weights,
        candidate_model=str(onnx_path),
        image_paths=eval_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        output_json=pt_vs_onnx_json,
    )
    _run_detection_compare_subprocess(
        repo_root=repo_root,
        helper_script=compare_helper,
        reference_model=args.weights,
        candidate_model=str(tflite_out),
        image_paths=eval_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        output_json=pt_vs_tfl_json,
    )

    print("=" * 72)
    print("YOLOv8n pipeline completed")
    print("=" * 72)
    print(f"onnx: {onnx_path}")
    print(f"tflite: {tflite_out}")
    print(f"vela_report: {vela_dir / 'vela_compile_report.json'}")
    print(f"raw_output_diff: {raw_diff_json}")
    print(f"pt_vs_onnx_detection_match: {pt_vs_onnx_json}")
    print(f"pt_vs_tflite_detection_match: {pt_vs_tfl_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
