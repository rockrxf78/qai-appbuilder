#!/usr/bin/env python3
"""Collect per-image detection signatures from PyTorch YOLO inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def _signature_from_result(result) -> dict[str, object]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"count": 0, "classes": [], "top1": None}
    classes = boxes.cls.cpu().numpy().astype(int).tolist()
    conf = boxes.conf.cpu().numpy().astype(float).tolist()
    top1 = classes[int(np.argmax(np.asarray(conf)))] if conf else None
    return {"count": len(classes), "classes": classes, "top1": top1}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PT detection signatures")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-list", required=True, help="One image path per line")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    image_paths = [
        Path(line.strip()).resolve()
        for line in Path(args.image_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = YOLO(args.weights)
    signatures: list[dict[str, object]] = []
    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            device="cpu",
        )[0]
        signatures.append(_signature_from_result(result))

    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"num_images": len(image_paths), "signatures": signatures}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] PT signatures written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
