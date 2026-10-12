# Troubleshooting (PSoC E84 Ethos-U skill)

## 1) `ModuleNotFoundError` for torch/onnx/tensorflow/onnx2tf

Run:

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/ethosu_preflight_check.py
```

Install missing modules shown by preflight.

## 2) ONNX export succeeds but conversion fails in onnx2tf

- Check ONNX ops and layout:
  - `scripts/onnx_sanity_check.py`
- Patch unsupported/dynamic ops before retry.
- If conversion still fails, capture full stdout/stderr and keep failing ONNX
  as an artifact for manual surgery.

## 3) Vela compile fails

- Validate target config names (`system-config` and `memory-mode`) match `.ini`.
- Ensure input model is quantized int8 TFLite.
- Inspect generated log: `vela_stdout_stderr.log`.

## 4) High CPU fallback

- Read fallback list in `vela_compile_report.json`.
- Patch operators using `operator_patching_ethosu.md`.
- Re-run conversion and compare fallback delta.

## 5) Arena estimate exceeds SoCMEM budget

- Reduce activation footprint:
  - Lower input resolution
  - Narrow channel widths
  - Re-balance stage boundaries
- Revisit memory mode placement (SRAM vs PSRAM cache/arena split).

## 6) FVP stage not available

- This VM may not include:
  - Arm Corstone-300 FVP
  - Arm embedded compiler
  - TFLM Ethos-U app source tree
- Use `run_fvp_harness.py --dry-run` to obtain explicit missing prerequisites.
