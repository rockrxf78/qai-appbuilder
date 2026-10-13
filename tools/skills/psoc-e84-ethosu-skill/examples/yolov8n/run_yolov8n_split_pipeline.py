#!/usr/bin/env python3
"""
YOLOv8n pipeline with pre-concat split outputs for independent box/score quantization.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from PIL import Image


def _run(cmd: list[str], cwd: Path) -> None:
    print("-" * 72)
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def _collect_images_from_dataset(dataset_yaml: str) -> list[Path]:
    from ultralytics.data.utils import check_det_dataset

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
            images = [
                ((base_path / e).resolve() if not Path(e).is_absolute() else Path(e))
                for e in entries
                if e
            ]
    images = [p for p in images if p.exists()]
    if not images:
        raise RuntimeError(f"No validation images resolved from dataset yaml: {dataset_yaml}")
    return images


def _preprocess_rgb(path: Path, imgsz: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _save_representative_data(
    image_paths: list[Path], output_path: Path, imgsz: int, max_samples: int
) -> np.ndarray:
    sample_paths = image_paths[:max_samples]
    data = np.stack([_preprocess_rgb(p, imgsz) for p in sample_paths], axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)
    return data


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    shapes: dict[str, list[int | str]] = {}
    for collection in (model.graph.input, model.graph.value_info, model.graph.output):
        for value_info in collection:
            dims: list[int | str] = []
            tensor_type = value_info.type.tensor_type
            if not tensor_type.HasField("shape"):
                continue
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(int(dim.dim_value))
                elif dim.dim_param:
                    dims.append(str(dim.dim_param))
                else:
                    dims.append("?")
            shapes[value_info.name] = dims
    return shapes


def _resolve_head_tensors(model: onnx.ModelProto) -> tuple[str, str]:
    output_name = model.graph.output[0].name
    producer = None
    for node in model.graph.node:
        if output_name in node.output:
            producer = node
            break
    if producer is None or producer.op_type != "Concat" or len(producer.input) < 2:
        raise RuntimeError("Cannot locate final Concat head in YOLO ONNX export.")

    shape_lookup = _shape_map(model)
    a_name, b_name = producer.input[0], producer.input[1]
    a_shape = shape_lookup.get(a_name, [])
    b_shape = shape_lookup.get(b_name, [])
    a_dim = a_shape[1] if len(a_shape) > 1 else None
    b_dim = b_shape[1] if len(b_shape) > 1 else None
    if a_dim == 4:
        return a_name, b_name
    if b_dim == 4:
        return b_name, a_name
    return a_name, b_name


def _export_preconcat_dual_output_onnx(onnx_src: Path, onnx_dst: Path) -> tuple[str, str]:
    model = onnx.load(str(onnx_src))
    box_tensor, score_tensor = _resolve_head_tensors(model)
    shape_lookup = _shape_map(model)
    box_shape = shape_lookup.get(box_tensor, [1, 4, "anchors"])
    score_shape = shape_lookup.get(score_tensor, [1, 80, "anchors"])

    while len(model.graph.output):
        model.graph.output.pop()
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(box_tensor, TensorProto.FLOAT, box_shape),
            helper.make_tensor_value_info(score_tensor, TensorProto.FLOAT, score_shape),
        ]
    )
    onnx_dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(onnx_dst))
    return box_tensor, score_tensor


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _summarize_metrics(values_a: list[np.ndarray], values_b: list[np.ndarray]) -> dict[str, float]:
    cosines: list[float] = []
    max_abs: list[float] = []
    maes: list[float] = []
    for a, b in zip(values_a, values_b):
        diff = a.astype(np.float64) - b.astype(np.float64)
        cosines.append(_cosine(a, b))
        max_abs.append(float(np.max(np.abs(diff))))
        maes.append(float(np.mean(np.abs(diff))))
    return {
        "cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
        "cosine_min": float(np.min(cosines)) if cosines else 0.0,
        "max_abs_mean": float(np.mean(max_abs)) if max_abs else 0.0,
        "max_abs_max": float(np.max(max_abs)) if max_abs else 0.0,
        "mae_mean": float(np.mean(maes)) if maes else 0.0,
    }


def _extract_boxes_scores(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    boxes: np.ndarray | None = None
    scores: np.ndarray | None = None
    for raw in outputs:
        tensor = np.asarray(raw)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            continue
        if 4 in tensor.shape:
            boxes = tensor.T if tensor.shape[0] == 4 else tensor
        if 80 in tensor.shape:
            scores = tensor.T if tensor.shape[0] == 80 else tensor
    if boxes is None or scores is None:
        raise RuntimeError(f"Failed to parse split outputs. Shapes: {[list(o.shape) for o in outputs]}")
    n = min(int(boxes.shape[0]), int(scores.shape[0]))
    return boxes[:n].astype(np.float32), scores[:n].astype(np.float32)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    return np.stack((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0), axis=1)


def _iou_vector(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-9)


def _decode_signature_from_split_outputs(
    outputs: list[np.ndarray], conf_thres: float, iou_thres: float, max_det: int
) -> dict[str, object]:
    boxes_xywh, class_scores = _extract_boxes_scores(outputs)
    if boxes_xywh.size == 0 or class_scores.size == 0:
        return {"count": 0, "classes": [], "top1": None}

    cls = np.argmax(class_scores, axis=1).astype(np.int32)
    conf = class_scores[np.arange(class_scores.shape[0]), cls]
    keep = conf >= conf_thres
    if not np.any(keep):
        return {"count": 0, "classes": [], "top1": None}

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh[keep])
    cls = cls[keep]
    conf = conf[keep]

    kept_indices: list[int] = []
    for cls_id in np.unique(cls):
        cls_idx = np.where(cls == cls_id)[0]
        order = cls_idx[np.argsort(conf[cls_idx])[::-1]]
        while order.size > 0:
            current = int(order[0])
            kept_indices.append(current)
            if len(kept_indices) >= max_det or order.size == 1:
                break
            ious = _iou_vector(boxes_xyxy[current], boxes_xyxy[order[1:]])
            order = order[1:][ious <= iou_thres]
        if len(kept_indices) >= max_det:
            break

    if not kept_indices:
        return {"count": 0, "classes": [], "top1": None}
    kept = np.asarray(kept_indices, dtype=np.int32)
    kept = kept[np.argsort(conf[kept])[::-1]][:max_det]
    classes = cls[kept].astype(int).tolist()
    return {"count": len(classes), "classes": classes, "top1": classes[0] if classes else None}


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / len(sa | sb))


def _compare_signatures(
    reference: list[dict[str, object]], candidate: list[dict[str, object]]
) -> dict[str, float | int | None]:
    n = min(len(reference), len(candidate))
    if n == 0:
        return {
            "num_images": 0,
            "count_match_rate": 0.0,
            "class_jaccard_mean": 0.0,
            "top1_match_rate": None,
            "top1_total": 0,
        }
    count_matches = 0
    jaccards: list[float] = []
    top1_matches = 0
    top1_total = 0
    for ref, cand in zip(reference[:n], candidate[:n]):
        if int(ref["count"]) == int(cand["count"]):
            count_matches += 1
        jaccards.append(_jaccard(list(ref["classes"]), list(cand["classes"])))
        if ref["top1"] is not None and cand["top1"] is not None:
            top1_total += 1
            if int(ref["top1"]) == int(cand["top1"]):
                top1_matches += 1
    return {
        "num_images": n,
        "count_match_rate": float(count_matches / n),
        "class_jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        "top1_match_rate": (float(top1_matches / top1_total) if top1_total else None),
        "top1_total": top1_total,
    }


class TFLiteRunner:
    def __init__(self, model_path: Path) -> None:
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

    def output_quant_summary(self) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for detail in self.output_details:
            summaries.append(
                {
                    "name": str(detail["name"]),
                    "shape": list(np.asarray(detail["shape"]).tolist()),
                    "dtype": str(detail["dtype"]),
                    "quantization": list(detail["quantization"]),
                }
            )
        return summaries

    def infer(self, input_nchw: np.ndarray) -> list[np.ndarray]:
        input_nhwc = np.transpose(input_nchw, (0, 2, 3, 1))
        if self.input_detail["dtype"] == np.int8:
            scale, zero = self.input_detail["quantization"]
            if scale == 0:
                raise RuntimeError("TFLite input scale is zero")
            prepared = np.clip(np.round(input_nhwc / scale + zero), -128, 127).astype(np.int8)
        else:
            prepared = input_nhwc.astype(self.input_detail["dtype"])
        self.interpreter.set_tensor(self.input_detail["index"], prepared)
        self.interpreter.invoke()
        outputs: list[np.ndarray] = []
        for detail in self.output_details:
            y = self.interpreter.get_tensor(detail["index"])
            if detail["dtype"] == np.int8:
                y_scale, y_zero = detail["quantization"]
                y = (y.astype(np.float32) - y_zero) * y_scale
            outputs.append(np.asarray(y, dtype=np.float32))
        return outputs


def _collect_backend_signatures(
    onnx_path: Path,
    tflite_path: Path,
    image_paths: list[Path],
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tfl_runner = TFLiteRunner(tflite_path)

    onnx_sigs: list[dict[str, object]] = []
    tfl_sigs: list[dict[str, object]] = []
    onnx_boxes: list[np.ndarray] = []
    onnx_scores: list[np.ndarray] = []
    tfl_boxes: list[np.ndarray] = []
    tfl_scores: list[np.ndarray] = []

    for path in image_paths:
        x = _preprocess_rgb(path, imgsz)[None, ...]
        onnx_outputs = [np.asarray(o, dtype=np.float32) for o in session.run(None, {input_name: x})]
        tfl_outputs = tfl_runner.infer(x)

        onnx_sigs.append(_decode_signature_from_split_outputs(onnx_outputs, conf_thres, iou_thres, max_det=300))
        tfl_sigs.append(_decode_signature_from_split_outputs(tfl_outputs, conf_thres, iou_thres, max_det=300))

        ob, os = _extract_boxes_scores(onnx_outputs)
        tb, ts = _extract_boxes_scores(tfl_outputs)
        onnx_boxes.append(ob)
        onnx_scores.append(os)
        tfl_boxes.append(tb)
        tfl_scores.append(ts)

    diff_report = {
        "num_images": len(image_paths),
        "boxes": _summarize_metrics(tfl_boxes, onnx_boxes),
        "scores": _summarize_metrics(tfl_scores, onnx_scores),
    }
    return onnx_sigs, tfl_sigs, {
        "output_quantization": tfl_runner.output_quant_summary(),
        "raw_output_diff": diff_report,
    }


def _parse_imgsz_candidates(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise RuntimeError("No valid --imgsz-candidates values provided.")
    return values


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cpu_breakdown_from_vela_report(report: dict[str, object]) -> tuple[int, int]:
    operator = report.get("operator_breakdown", {})
    cpu_total = int(operator.get("cpu_operator_count", 0) or 0)
    transpose = report.get("transpose_fallback_summary", {})
    layout_count = int(
        transpose.get(
            "export_qdq_layout_count",
            transpose.get("removable_count", 0),
        )
        or 0
    )
    non_transpose_cpu = max(0, cpu_total - layout_count)
    return cpu_total, non_transpose_cpu


def _run_for_imgsz(
    args: argparse.Namespace,
    repo_root: Path,
    skill_root: Path,
    image_paths: list[Path],
    imgsz: int,
) -> dict[str, object]:
    scripts_dir = skill_root / "scripts"
    yolo_example_dir = skill_root / "examples" / "yolov8n"
    run_dir = Path(args.work_dir).resolve() / f"imgsz_{imgsz}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rep_data_path = run_dir / "yolo_rep_data.npy"
    onnx_path = run_dir / "yolov8n.onnx"
    split_onnx_path = run_dir / "yolov8n_preconcat.onnx"
    onnx2tf_out = run_dir / "onnx2tf_out"
    tflite_path = run_dir / "yolov8n_int8.tflite"
    vela_dir = run_dir / "vela"
    eval_paths = image_paths[: args.eval_samples]
    eval_list_path = run_dir / "eval_images.txt"
    eval_list_path.write_text("\n".join(str(p) for p in eval_paths), encoding="utf-8")

    _save_representative_data(image_paths, rep_data_path, imgsz, args.calib_samples)

    if args.force_export or not onnx_path.exists():
        _run(
            [
                sys.executable,
                str(yolo_example_dir / "export_yolo_onnx.py"),
                "--weights",
                args.weights,
                "--imgsz",
                str(imgsz),
                "--output-onnx",
                str(onnx_path),
                * (["--simplify"] if args.export_simplify else []),
            ],
            cwd=repo_root,
        )

    box_tensor_name, score_tensor_name = _export_preconcat_dual_output_onnx(onnx_path, split_onnx_path)

    _run(
        [
            sys.executable,
            str(scripts_dir / "onnx_sanity_check.py"),
            "--onnx",
            str(split_onnx_path),
            "--json-report",
            str(run_dir / "yolov8n_onnx_check_report.json"),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(scripts_dir / "onnx_to_tflite_int8.py"),
            "--onnx",
            str(split_onnx_path),
            "--work-dir",
            str(onnx2tf_out),
            "--tflite-out",
            str(tflite_path),
            "--representative-data",
            str(rep_data_path),
            "--input-layout",
            "nchw",
            "--max-representative-samples",
            str(args.calib_samples),
            "--onnx2tf-flextranspose-compression-dim",
            str(args.onnx2tf_flextranspose_compression_dim),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(scripts_dir / "compile_with_vela.py"),
            "--tflite",
            str(tflite_path),
            "--target",
            args.target,
            "--output-dir",
            str(vela_dir),
        ],
        cwd=repo_root,
    )
    baseline_vela_report = _read_json(vela_dir / "vela_compile_report.json")

    selected_tflite = tflite_path
    selected_vela_dir = vela_dir
    transpose_cleanup_attempt: dict[str, object] | None = None

    if args.enable_transpose_cleanup:
        cleanup_tflite = run_dir / "yolov8n_int8_transpose_cleanup.tflite"
        cleanup_onnx2tf_out = run_dir / "onnx2tf_out_transpose_cleanup"
        cleanup_vela_dir = run_dir / "vela_transpose_cleanup"
        _run(
            [
                sys.executable,
                str(scripts_dir / "onnx_to_tflite_int8.py"),
                "--onnx",
                str(split_onnx_path),
                "--work-dir",
                str(cleanup_onnx2tf_out),
                "--tflite-out",
                str(cleanup_tflite),
                "--representative-data",
                str(rep_data_path),
                "--input-layout",
                "nchw",
                "--max-representative-samples",
                str(args.calib_samples),
                "--onnx2tf-flextranspose-compression-dim",
                str(args.transpose_cleanup_flextranspose_dim),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                str(scripts_dir / "compile_with_vela.py"),
                "--tflite",
                str(cleanup_tflite),
                "--target",
                args.target,
                "--output-dir",
                str(cleanup_vela_dir),
            ],
            cwd=repo_root,
        )
        cleanup_vela_report = _read_json(cleanup_vela_dir / "vela_compile_report.json")
        baseline_transpose = baseline_vela_report.get("transpose_fallback_summary", {})
        cleanup_transpose = cleanup_vela_report.get("transpose_fallback_summary", {})
        baseline_layout = int(
            baseline_transpose.get(
                "export_qdq_layout_count",
                baseline_transpose.get("removable_count", 0),
            )
            or 0
        )
        cleanup_layout = int(
            cleanup_transpose.get(
                "export_qdq_layout_count",
                cleanup_transpose.get("removable_count", 0),
            )
            or 0
        )
        baseline_cpu_total, baseline_non_transpose = _cpu_breakdown_from_vela_report(
            baseline_vela_report
        )
        cleanup_cpu_total, cleanup_non_transpose = _cpu_breakdown_from_vela_report(
            cleanup_vela_report
        )
        accepted = (
            cleanup_layout < baseline_layout
            and cleanup_non_transpose <= baseline_non_transpose
            and cleanup_cpu_total <= baseline_cpu_total
        )
        reason = (
            "accepted cleanup variant"
            if accepted
            else (
                "rejected cleanup variant because it did not improve layout-transpose fallback "
                "without increasing other CPU fallback ops"
            )
        )
        if accepted:
            selected_tflite = cleanup_tflite
            selected_vela_dir = cleanup_vela_dir
        transpose_cleanup_attempt = {
            "enabled": True,
            "cleanup_flextranspose_dim": args.transpose_cleanup_flextranspose_dim,
            "accepted": accepted,
            "reason": reason,
            "baseline_report": str(vela_dir / "vela_compile_report.json"),
            "cleanup_report": str(cleanup_vela_dir / "vela_compile_report.json"),
            "baseline_cpu_operator_count": baseline_cpu_total,
            "cleanup_cpu_operator_count": cleanup_cpu_total,
            "baseline_non_transpose_cpu_operator_count": baseline_non_transpose,
            "cleanup_non_transpose_cpu_operator_count": cleanup_non_transpose,
            "baseline_export_qdq_layout_transpose_count": baseline_layout,
            "cleanup_export_qdq_layout_transpose_count": cleanup_layout,
        }
    else:
        transpose_cleanup_attempt = {"enabled": False}

    pt_sig_json = run_dir / "pt_signatures.json"
    _run(
        [
            sys.executable,
            str(yolo_example_dir / "collect_pt_signatures.py"),
            "--weights",
            args.weights,
            "--image-list",
            str(eval_list_path),
            "--imgsz",
            str(imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--output-json",
            str(pt_sig_json),
        ],
        cwd=repo_root,
    )
    pt_signatures = _read_json(pt_sig_json)["signatures"]

    onnx_sigs, tfl_sigs, aux = _collect_backend_signatures(
        onnx_path=split_onnx_path,
        tflite_path=selected_tflite,
        image_paths=eval_paths,
        imgsz=imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )
    onnx_sig_json = run_dir / "onnx_signatures.json"
    tfl_sig_json = run_dir / "tflite_signatures.json"
    onnx_sig_json.write_text(json.dumps({"signatures": onnx_sigs}, indent=2), encoding="utf-8")
    tfl_sig_json.write_text(json.dumps({"signatures": tfl_sigs}, indent=2), encoding="utf-8")

    pt_vs_onnx = _compare_signatures(pt_signatures, onnx_sigs)
    pt_vs_tfl = _compare_signatures(pt_signatures, tfl_sigs)
    onnx_vs_tfl = _compare_signatures(onnx_sigs, tfl_sigs)
    pt_vs_onnx_json = run_dir / "yolov8n_detection_match_pt_vs_onnx.json"
    pt_vs_tfl_json = run_dir / "yolov8n_detection_match_pt_vs_tflite.json"
    onnx_vs_tfl_json = run_dir / "yolov8n_detection_match_onnx_vs_tflite.json"
    pt_vs_onnx_json.write_text(json.dumps(pt_vs_onnx, indent=2), encoding="utf-8")
    pt_vs_tfl_json.write_text(json.dumps(pt_vs_tfl, indent=2), encoding="utf-8")
    onnx_vs_tfl_json.write_text(json.dumps(onnx_vs_tfl, indent=2), encoding="utf-8")

    raw_diff_json = run_dir / "yolov8n_raw_output_diff.json"
    raw_diff_json.write_text(json.dumps(aux["raw_output_diff"], indent=2), encoding="utf-8")
    (run_dir / "yolov8n_tflite_output_quantization.json").write_text(
        json.dumps(aux["output_quantization"], indent=2),
        encoding="utf-8",
    )

    vela_report = _read_json(selected_vela_dir / "vela_compile_report.json")
    arena_kb = vela_report.get("arena_estimate_kb")
    arena_fit = isinstance(arena_kb, (int, float)) and float(arena_kb) <= 4096.0
    transpose_summary = vela_report.get("transpose_fallback_summary", {})
    transpose_classification_json = run_dir / "yolov8n_transpose_fallback_classification.json"
    transpose_classification_json.write_text(
        json.dumps(transpose_summary, indent=2),
        encoding="utf-8",
    )

    quant_summaries = aux["output_quantization"]
    score_scale = None
    box_scale = None
    for entry in quant_summaries:
        shape = entry.get("shape", [])
        quant = entry.get("quantization", [])
        if len(shape) >= 3 and 80 in shape:
            score_scale = quant[0] if quant else None
        if len(shape) >= 3 and 4 in shape:
            box_scale = quant[0] if quant else None
    split_quant_ok = (
        isinstance(score_scale, (int, float))
        and isinstance(box_scale, (int, float))
        and not math.isclose(float(score_scale), float(box_scale), rel_tol=1e-6, abs_tol=1e-9)
    )

    run_report = {
        "imgsz": imgsz,
        "run_dir": str(run_dir),
        "onnx": str(onnx_path),
        "split_onnx": str(split_onnx_path),
        "split_output_tensors": {"boxes": box_tensor_name, "scores": score_tensor_name},
        "tflite": str(selected_tflite),
        "vela_report": str(selected_vela_dir / "vela_compile_report.json"),
        "raw_output_diff_report": str(raw_diff_json),
        "pt_vs_onnx_report": str(pt_vs_onnx_json),
        "pt_vs_tflite_report": str(pt_vs_tfl_json),
        "onnx_vs_tflite_report": str(onnx_vs_tfl_json),
        "transpose_fallback_classification_report": str(transpose_classification_json),
        "pt_vs_onnx": pt_vs_onnx,
        "pt_vs_tflite": pt_vs_tfl,
        "onnx_vs_tflite": onnx_vs_tfl,
        "tflite_output_quantization": quant_summaries,
        "box_output_scale": box_scale,
        "score_output_scale": score_scale,
        "split_output_quantization_ok": split_quant_ok,
        "arena_estimate_kb": arena_kb,
        "arena_fit_under_4096kb": arena_fit,
        "vela_operator_breakdown": vela_report.get("operator_breakdown", {}),
        "vela_cpu_fallback_ops_count": len(vela_report.get("cpu_fallback_ops", [])),
        "vela_transpose_fallback_summary": transpose_summary,
        "vela_warnings": vela_report.get("warnings", []),
        "onnx2tf_flextranspose_compression_dim": args.onnx2tf_flextranspose_compression_dim,
        "transpose_cleanup_attempt": transpose_cleanup_attempt,
    }
    (run_dir / "yolov8n_run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    return run_report


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8n split-output Ethos-U pipeline")
    parser.add_argument("--work-dir", default="tools/skills/psoc-e84-ethosu-skill/build/yolov8n")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--dataset-yaml", default="coco128.yaml")
    parser.add_argument(
        "--imgsz-candidates",
        default="320,256",
        help="Comma-separated candidate sizes tried in order (default: 320,256)",
    )
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--target", default="psoc_e84")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument(
        "--export-simplify",
        action="store_true",
        help="Enable Ultralytics ONNX simplify pass during export.",
    )
    parser.add_argument(
        "--onnx2tf-flextranspose-compression-dim",
        type=int,
        default=6,
        help=(
            "Pass through to onnx2tf -nodaftc. "
            "Lower values can suppress conversion-introduced transpose ops but may increase"
            " fallback in other operator families."
        ),
    )
    parser.add_argument(
        "--enable-transpose-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run an export/layout transpose-cleanup conversion attempt and accept it "
            "only when it improves layout-transpose fallback without increasing other CPU fallback."
        ),
    )
    parser.add_argument(
        "--transpose-cleanup-flextranspose-dim",
        type=int,
        default=2,
        help="onnx2tf -nodaftc value used for cleanup attempt (default: 2).",
    )
    parser.add_argument(
        "--try-all-sizes",
        action="store_true",
        help="Run all candidate sizes even if earlier size already fits SRAM budget",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    skill_root = script_path.parents[2]
    repo_root = skill_root.parents[2]
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images_from_dataset(args.dataset_yaml)
    sizes = _parse_imgsz_candidates(args.imgsz_candidates)

    run_reports: list[dict[str, object]] = []
    for size in sizes:
        report = _run_for_imgsz(args, repo_root, skill_root, image_paths, size)
        run_reports.append(report)
        if report.get("arena_fit_under_4096kb") and not args.try_all_sizes:
            break

    selected = run_reports[-1]
    for report in run_reports:
        if report.get("arena_fit_under_4096kb"):
            selected = report
            break

    summary = {
        "imgsz_candidates": sizes,
        "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples,
        "run_reports": run_reports,
        "selected_run": selected,
    }
    summary_path = work_dir / "yolov8n_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("YOLOv8n split-output pipeline completed")
    print("=" * 72)
    print(f"summary: {summary_path}")
    print(f"selected imgsz: {selected.get('imgsz')}")
    print(f"selected run dir: {selected.get('run_dir')}")
    print(f"pt_vs_tflite report: {selected.get('pt_vs_tflite_report')}")
    print(f"vela report: {selected.get('vela_report')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
