# Operator patching for Ethos-U / TFLite Micro

## Goal

Reduce CPU fallback ops reported by Vela (`--show-cpu-operations`) while
preserving model semantics and acceptable accuracy.

## Priority order

1. Patch in PyTorch source model (preferred)
2. Patch during ONNX export (custom symbolic/export settings)
3. ONNX graph surgery (fallback)

## Common rewrite patterns

### 1) Unsupported activation variants
- Replace exotic activations with supported combinations:
  - `SiLU` -> `x * sigmoid(x)` (verify numerical parity)
  - `HardSwish` may map to supported ops depending on exporter

### 2) Complex reshape/transposes
- Fold redundant reshape/transpose sequences before export.
- Keep tensors NHWC-friendly for downstream TFLite conversion.

### 3) Einsum / advanced matmul forms
- Rewrite to explicit `MatMul` + `Add` where possible.

### 4) Dynamic-shape control flow
- Freeze dynamic dimensions for MCU deployment paths.
- Remove data-dependent control flow from hot path.

### 5) Unsupported pooling / interpolation modes
- Replace with supported pooling/interp variants accepted by TFLite.

## Acceptance rule

- CPU fallback can be tolerated for non-critical ops, but must be explicit in
  report and justified by latency/power impact.
- For production acceptance, track fallback percentage trend per model commit.

## Patch validation gates

After each patch:
1. ONNX checker passes
2. ONNX runtime matches golden baseline
3. TFLite int8 conversion succeeds
4. Vela fallback list shrinks or remains acceptable
5. Accuracy metrics remain within tolerance
