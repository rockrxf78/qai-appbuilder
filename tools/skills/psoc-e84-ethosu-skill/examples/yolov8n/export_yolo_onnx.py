#!/usr/bin/env python3
"""Export YOLO model to ONNX using ultralytics in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--output-onnx", required=True)
    args = parser.parse_args()

    output = Path(args.output_onnx).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=13, dynamic=False, simplify=False)
    exported_path = Path(exported).resolve()
    if exported_path != output:
        output.write_bytes(exported_path.read_bytes())
    print(f"[OK] ONNX exported: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
