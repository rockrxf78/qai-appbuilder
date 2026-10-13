#!/usr/bin/env python3
"""
Export a PyTorch model to ONNX and capture a golden baseline output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is required. Install it with `pip install torch`."
        ) from exc
    return torch


def _build_tinycnn(torch):
    class TinyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(8 * 16 * 16, 4)

        def forward(self, x):
            y = self.net(x)
            y = torch.flatten(y, start_dim=1)
            return self.head(y)

    return TinyCNN()


def _load_external_model(model_py: Path, factory: str, torch):
    spec = importlib.util.spec_from_file_location("user_model", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, factory):
        raise RuntimeError(f"{model_py} does not define `{factory}`")
    fn = getattr(module, factory)
    model = fn()
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(f"{factory}() must return torch.nn.Module")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--output-onnx", required=True, help="Output ONNX path")
    parser.add_argument(
        "--input-shape",
        default="1,3,32,32",
        help="NCHW input shape (default: 1,3,32,32)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--example",
        default="tinycnn",
        choices=["tinycnn", "external"],
        help="Model source",
    )
    parser.add_argument("--model-py", default="", help="External model python file")
    parser.add_argument(
        "--factory",
        default="build_model",
        help="Factory function in --model-py for external mode",
    )
    parser.add_argument(
        "--golden-dir",
        default="",
        help="Directory to save golden input/output tensors (.npy + metadata)",
    )
    args = parser.parse_args()

    torch = _load_torch()
    np.random.seed(7)
    torch.manual_seed(7)

    shape = tuple(int(x) for x in args.input_shape.split(","))
    if len(shape) != 4:
        raise RuntimeError("input shape must be NCHW with 4 dimensions")

    if args.example == "tinycnn":
        model = _build_tinycnn(torch)
    else:
        if not args.model_py:
            raise RuntimeError("--model-py is required when --example external")
        model = _load_external_model(Path(args.model_py), args.factory, torch)

    model.eval()
    dummy_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
    with torch.no_grad():
        golden = model(dummy_input).detach().cpu().numpy()

    output_onnx = Path(args.output_onnx).resolve()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes=None,
    )
    print(f"[OK] ONNX exported: {output_onnx}")

    if args.golden_dir:
        golden_dir = Path(args.golden_dir).resolve()
        golden_dir.mkdir(parents=True, exist_ok=True)
        input_path = golden_dir / "golden_input.npy"
        output_path = golden_dir / "golden_output.npy"
        np.save(input_path, dummy_input.detach().cpu().numpy())
        np.save(output_path, golden)
        metadata = {
            "input_name": "input",
            "output_name": "output",
            "input_shape_nchw": list(shape),
            "layout_warning": (
                "Ethos-U/TFLM typically uses NHWC. "
                "Run NCHW->NHWC conversion before TFLite deployment."
            ),
        }
        (golden_dir / "golden_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"[OK] Golden tensors saved under {golden_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
