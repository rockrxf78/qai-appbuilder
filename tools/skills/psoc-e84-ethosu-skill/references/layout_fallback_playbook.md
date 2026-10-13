# Playbook: export/quantization layout ops falling back to CPU on Ethos-U

Read this **before** brute-forcing converter flags on a new model.

## Problem

ONNX (NCHW) → onnx2tf/TFLite (NHWC) + QDQ often injects `TRANSPOSE` (and related reshape/concat glue). Ethos-U55 only maps a **tiny** set of transpose permutations to NPU; the rest stay on CPU. See `ethosu_transpose_constraints.md`.

## Root causes seen (YOLOv8n 320)

| Source | What it looks like | Typical perm / shape |
| --- | --- | --- |
| onnx2tf NCHW↔NHWC glue | `tf.compat.v1.transpose_*` around conv/concat | `[0,3,1,2]` or `[0,2,3,1]` on rank-4 with H,W,C all > 1 |
| QDQ / FlexTranspose compression | `-nodaftc 2` can **delete** those transposes | TRANSPOSE 24→0, but other CPU ops explode |
| Head reshape/concat (DFL / split head) | axis swap before softmax/reshape | `[0,1,3,2]` on `[1,4,16,2100]` — **inherent** |

Measured tradeoff on YOLOv8n 320 (`psoc_e84`):

- Baseline (`-nodaftc 6`): CPU 25 (8%), TRANSPOSE 25 = 24 layout_glue + 1 inherent, SRAM 2800 KB, NPU 92%
- Aggressive (`-nodaftc 2`): TRANSPOSE 0, CPU **698** (30.2%), NPU 69.8% — **rejected**

ONNX itself had **1** Transpose; the other 24 were conversion glue.

tinycnn (same toolchain): ONNX has 0 Transpose, Vela still shows **1 layout_glue** TRANSPOSE. `-nodaftc 2` removed it but CPU 1→9 — sweep rejected.

## Classification

- **layout_glue** — NHWC↔NCHW ping-pong from export/onnx2tf. Candidate to *remove from the graph*, not “map to NPU”.
- **qdq** — quantize/dequantize wrapping a layout op. Safe to drop only if the pair is numerically identity.
- **inherent** — topology needs that axis order (detection DFL, RNN time/channel, etc.). Document as CPU fallback; do not claim elimination.
- **unsupported_op** — GATHER/WHILE/LSTM/… not a layout problem.

## Decision tree

1. Parse Vela log (`analyze_cpu_fallback.py`). Split counts: layout_glue vs inherent vs unsupported.
2. If layout_glue == 0: stop. Remaining CPU ops are inherent/unsupported — patch the model, not flags.
3. ONNX pre-pass (`onnx_layout_optimizer.py`): cancel inverse Transpose pairs, push through elementwise, drop identity QDQ. Keep only if ORT max-abs is within tolerance.
4. Reconvert with **default** onnx2tf. Re-analyze. Do **not** jump to `-nodaftc 2`.
5. Optional **small** sweep (`sweep_conversion_options.py`): default vs `-dsm` vs one other flag. Accept a variant only if CPU ops, SRAM, and accuracy do not regress.
6. `-nodaftc 2` (or similar “delete transposes”) is last-resort diagnostic, not a default. Reject if non-transpose CPU ops rise.

## Recommended order of attempts

1. Playbook + `analyze_cpu_fallback.py`
2. `onnx_layout_optimizer.py` on the **ONNX** (before onnx2tf)
3. NHWC-native export if the framework allows (avoid NCHW→NHWC glue)
4. Tiny conversion sweep with regression guard
5. Model rewrite for **inherent** transposes (move decode/NMS off-graph)

## Guardrails

- Never accept a graph that trades TRANSPOSE for a larger CPU family (GATHER/SLICE/…).
- Rank candidates by `(cpu_ops, sram_kb, cycles)` then accuracy; require no CPU/SRAM/accuracy regression vs baseline.
- Do not brute-force onnx2tf grids on large detectors; one baseline + one diagnostic is enough to learn the tradeoff.
- Inherent fallback is an acceptable documented outcome.
