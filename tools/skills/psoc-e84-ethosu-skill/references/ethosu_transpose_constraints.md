# Ethos-U55 / Vela TRANSPOSE constraints and YOLOv8n implications

## What Vela reports as NPU-supported TRANSPOSE patterns

From Vela compile warnings (`--show-cpu-operations`) for Ethos-U55:

- rank-2: `WxC -> CxW`
- rank-3:
  - `HxWxC -> WxHxC`
  - `1xWxC -> 1xCxW`
  - `Hx1xC -> Cx1xH`
- rank-4:
  - `1xHxWxC -> 1xWxHxC`
  - `1x1xWxC -> 1x1xCxW`
  - `1xHx1xC -> 1xCx1xW`

Any other permutation falls back to CPU.

## Why YOLOv8n hits CPU TRANSPOSE fallback

For the split-output YOLOv8n 320 pipeline, most fallback transposes are:

- permutation `[0,3,1,2]` (NHWC -> NCHW style)
- permutation `[0,2,3,1]` (NCHW -> NHWC style)

with non-trivial spatial/channel shapes (e.g. `[1,40,40,80]`, `[1,384,10,10]`).
These do not match the Ethos-U rank-4 restrictions above and therefore stay on CPU.

## Removable vs inherent framing used by this skill

`scripts/compile_with_vela.py` emits `transpose_fallback_summary` with this split:

- **removable**: rank-4 layout glue candidates (`[0,3,1,2]` or `[0,2,3,1]`)
  introduced by ONNX->TF/TFLite layout adaptation.
- **inherent**: non-layout swaps (for YOLO run: `[0,1,3,2]`) tied to model-head axis
  semantics.

This classification is heuristic and intentionally conservative: if a transpose
is not an obvious layout glue swap, it is treated as inherent until a validated
graph rewrite proves otherwise.

## Practical mitigation status

Tested on YOLOv8n 320:

- **Baseline conversion (`-nodaftc 6`)**
  - CPU ops: 25
  - TRANSPOSE fallback: 25 (24 removable, 1 inherent)
  - NPU ratio: 92.0%
- **Aggressive FlexTranspose suppression (`-nodaftc 2`)**
  - TRANSPOSE fallback: 0
  - but CPU fallback exploded to 698 ops (NPU ratio down to 69.8%)

Conclusion: for this model/toolchain, forcing away transpose nodes does not
produce a better Ethos-U mapping overall. Keep baseline conversion for now and
focus future optimization on graph-level NHWC-native model export/rewrite.
