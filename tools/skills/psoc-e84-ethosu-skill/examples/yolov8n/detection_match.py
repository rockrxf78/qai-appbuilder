#!/usr/bin/env python3
"""
Compute detection signature agreement between two YOLO models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def _extract_detection_signature(result) -> dict[str, object]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"count": 0, "classes": [], "top1": None}
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    conf = boxes.conf.cpu().numpy().astype(float).tolist()
    top1 = None
    if conf:
        top1 = cls[int(np.argmax(np.asarray(conf)))]
    return {"count": len(cls), "classes": cls, "top1": top1}


def _jaccard(a: list[int], b: list[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / len(sa | sb))


def compare_detection_models(
    reference_model_path: str,
    candidate_model_path: str,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, float | int | None]:
    if not image_paths:
        return {
            "num_images": 0,
            "count_match_rate": 0.0,
            "class_jaccard_mean": 0.0,
            "top1_match_rate": None,
            "top1_total": 0,
        }
    reference_model = YOLO(reference_model_path)
    candidate_model = YOLO(candidate_model_path)

    count_matches = 0
    jaccards: list[float] = []
    top1_matches = 0
    top1_total = 0
    for image_path in image_paths:
        ref = reference_model.predict(
            source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False, device="cpu"
        )[0]
        cand = candidate_model.predict(
            source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False, device="cpu"
        )[0]
        ref_sig = _extract_detection_signature(ref)
        cand_sig = _extract_detection_signature(cand)
        if ref_sig["count"] == cand_sig["count"]:
            count_matches += 1
        jaccards.append(_jaccard(ref_sig["classes"], cand_sig["classes"]))
        if ref_sig["top1"] is not None and cand_sig["top1"] is not None:
            top1_total += 1
            if ref_sig["top1"] == cand_sig["top1"]:
                top1_matches += 1

    return {
        "num_images": len(image_paths),
        "count_match_rate": float(count_matches / len(image_paths)),
        "class_jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        "top1_match_rate": (float(top1_matches / top1_total) if top1_total else None),
        "top1_total": top1_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare YOLO detection signatures")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--image-list", required=True, help="Text file with one image path per line")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    image_list_path = Path(args.image_list).resolve()
    images = [
        Path(line.strip()).resolve()
        for line in image_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = compare_detection_models(
        reference_model_path=args.reference_model,
        candidate_model_path=args.candidate_model,
        image_paths=images,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] detection comparison saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
