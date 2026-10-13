#!/usr/bin/env python3
"""
ONNX structural checks for Ethos-U conversion readiness.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_onnx():
    try:
        import onnx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("onnx is required (`pip install onnx`)") from exc
    return onnx


KNOWN_TFLITE_FRIENDLY_OPS = {
    "Add",
    "AveragePool",
    "Clip",
    "Concat",
    "Conv",
    "DepthToSpace",
    "DepthwiseConv2dNative",
    "Div",
    "Gather",
    "Gemm",
    "HardSigmoid",
    "LeakyRelu",
    "MatMul",
    "MaxPool",
    "Mul",
    "Pad",
    "Relu",
    "Reshape",
    "Resize",
    "Sigmoid",
    "Softmax",
    "Split",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


def _shape_to_list(value_info):
    dims = value_info.type.tensor_type.shape.dim
    out = []
    for d in dims:
        if d.dim_value:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ONNX for Ethos-U pipeline")
    parser.add_argument("--onnx", required=True, help="ONNX file path")
    parser.add_argument(
        "--json-report",
        default="",
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    onnx = _load_onnx()
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    op_counter = Counter(node.op_type for node in model.graph.node)
    input_shapes = {i.name: _shape_to_list(i) for i in model.graph.input}
    output_shapes = {o.name: _shape_to_list(o) for o in model.graph.output}

    dynamic_dims = []
    for name, shape in input_shapes.items():
        for idx, dim in enumerate(shape):
            if isinstance(dim, str):
                dynamic_dims.append(f"{name}[{idx}]={dim}")

    suspected_nchw = []
    for name, shape in input_shapes.items():
        if len(shape) == 4 and shape[1] == 3:
            suspected_nchw.append(name)

    unknown_ops = sorted(op for op in op_counter if op not in KNOWN_TFLITE_FRIENDLY_OPS)

    payload = {
        "onnx": str(onnx_path),
        "node_count": len(model.graph.node),
        "ops": op_counter,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "dynamic_input_dims": dynamic_dims,
        "suspected_nchw_inputs": suspected_nchw,
        "ops_not_in_known_tflite_friendly_set": unknown_ops,
    }

    print("=" * 72)
    print("ONNX sanity check")
    print("=" * 72)
    print(f"file: {onnx_path}")
    print(f"nodes: {payload['node_count']}")
    print(f"inputs: {input_shapes}")
    print(f"outputs: {output_shapes}")
    if dynamic_dims:
        print(f"[WARN] dynamic input dims found: {dynamic_dims}")
    if suspected_nchw:
        print(
            "[WARN] likely NCHW layout detected on inputs "
            f"{suspected_nchw}; Ethos-U deployment usually needs NHWC."
        )
    if unknown_ops:
        print("[WARN] ops that may require rewrite/fallback:")
        for op in unknown_ops:
            print(f"  - {op} (count={op_counter[op]})")
    else:
        print("[OK] all ops are in the known TFLite-friendly set.")

    if args.json_report:
        report_path = Path(args.json_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, indent=2, default=lambda x: dict(x)),
            encoding="utf-8",
        )
        print(f"[OK] report written: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
