#!/usr/bin/env python3
"""
Generate deterministic representative dataset (.npy) for quantization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate representative calibration data")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument(
        "--shape",
        default="64,3,32,32",
        help="Dataset shape (N,C,H,W) or (N,H,W,C)",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    shape = tuple(int(x) for x in args.shape.split(","))
    if len(shape) != 4:
        raise RuntimeError("--shape must be 4D")

    # Keep value range close to normalized image tensors.
    data = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, data)
    print(f"[OK] representative dataset saved: {out} shape={data.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
