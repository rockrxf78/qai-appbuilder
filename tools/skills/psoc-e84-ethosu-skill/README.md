# PSoC E84 Ethos-U Skill (AIPC extension)

This directory adds an Ethos-U deployment skill path, modeled after
`tools/skills/qai-runner-skill` (`aipc-toolkit`) but targeting:

- **Infineon PSoC Edge E84** (assumed profile: Cortex-M55 + Ethos-U55-128)
- **Bare-metal / RTOS runtime**
- **TFLite Micro + Ethos-U driver + CMSIS-NN fallback**

It does **not** modify existing Qualcomm QNN workflow files.

---

## Pipeline covered

1. PyTorch -> ONNX export and golden baseline capture
2. ONNX structural checks (dynamic dims/layout/operators)
3. ONNX -> TFLite int8 full-integer quantization
4. Vela compile (`ethos-u55-128`) with target config
5. CPU fallback + arena estimate report
6. Accuracy comparison (PyTorch/ONNX/TFLite)
7. Corstone-300 FVP harness hooks (build/run glue)

---

## File layout

- `SKILL.md` - skill entry and stage workflow
- `configs/vela_psoc_e84.ini` - Vela target/system/memory profiles
- `scripts/_ethosu_targets.py` - target table (extensible for more SoCs)
- `scripts/ethosu_preflight_check.py` - dependency/tool preflight
- `scripts/export_pytorch_to_onnx.py` - export + golden tensors
- `scripts/onnx_sanity_check.py` - ONNX checks
- `scripts/onnx_to_tflite_int8.py` - ONNX->TFLite int8 conversion
- `scripts/compile_with_vela.py` - Vela compile and report
- `scripts/compare_accuracy.py` - golden parity checks
- `scripts/run_fvp_harness.py` - Corstone-300 build/run hooks
- `scripts/run_pipeline.py` - staged loop orchestration
- `references/` - operator patching, quantization, troubleshooting docs
- `examples/` - tiny CNN demo helpers

---

## Quick start

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/ethosu_preflight_check.py

python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84
```

If you already have a model and calibration data, call stage scripts directly.

---

## Assumptions

- Host-side Python environment can install/use:
  - `torch`, `onnx`, `onnxruntime`, `tensorflow`, `onnx2tf`, `ethos-u-vela`
- Device-side runtime integration (TFLM + Ethos-U) is handled by project BSP.
- Vela memory profiles are an initial approximation and must be tuned with
  board-specific timing + linker constraints.

---

## Verified vs unverified in this repository

### Verified in-cloud (this branch)

- Skill and script structure added under a new directory.
- End-to-end script wiring for staged loop is implemented.
- Preflight, conversion, compile, reporting, and harness scripts are executable
  Python files with explicit error guidance.

### Not yet verified in-cloud (requires external toolchains/hardware assets)

- Corstone-300 FVP execution with a real TFLM Ethos-U app image.
- Hardware execution on an actual PSoC Edge E84 board.
- ModusToolbox-specific build integration and flashing pipeline.
- Board-accurate SRAM/PSRAM/Flash latency parameters for Vela config.

---

## Facts that should be re-validated by users

The following are treated as target assumptions and should be validated against
official board documentation/BSP before production use:

- PSoC E84 memory split and exact accessible arena budget.
- Ethos-U55-128 deployment clock/latency profile and bus characteristics.
- Final FVP model and flags used for your runtime stack.
