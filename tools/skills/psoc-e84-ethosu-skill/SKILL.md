---
name: aipc-ethosu-toolkit
description: AIPC-style workflow for PSoC Edge E84 (Cortex-M55 + Ethos-U55-128). Covers PyTorch->ONNX export, ONNX checks/patching, ONNX->TFLite int8 conversion, Vela compile, CPU-fallback diagnosis, memory budgeting for SoCMEM/PSRAM/Flash, and staged verification on Corstone-300 FVP or hardware.
---

# AIPC Ethos-U Toolkit (PSoC Edge E84)

## Trigger Phrases

Always activate this skill when the user mentions:
- "PSoC E84"
- "Ethos-U55" / "ethos-u"
- "Vela compile"
- "TFLite Micro + Ethos-U"
- "Corstone-300 FVP"
- "ONNX to TFLite int8 for MCU"
- "CPU fallback ops on Ethos-U"
- "SoCMEM / PSRAM / Flash placement"

## When to Use

Use this skill for Arm MCU deployment pipelines where the target NPU is Ethos-U
(especially `ethos-u55-128`) and the runtime is TFLite Micro + Ethos-U driver,
with CMSIS-NN/CPU fallback for unsupported ops.

## Required Guardrails

- Do not modify existing QNN/HTP scripts or tables unless explicitly asked.
- Treat PSoC E84 memory split as a deployment contract:
  - SoCMEM: 5MB (SRAM budget for tensor arena + scratch)
  - PSRAM: 32MB (large buffers/cache, slower)
  - Flash: 128MB (weights/constants/model blob)
- Always build a **golden baseline** first (PyTorch output + fixed test input).
- Enforce **full integer quantization** for NPU path:
  - `target_spec.supported_ops = TFLITE_BUILTINS_INT8`
  - representative dataset is mandatory
- If ONNX input is NCHW, explicitly handle NCHW->NHWC before TFLite validation.
- Always run Vela with `--show-cpu-operations`; CPU fallback list is required output.
- FVP/hardware stage must fail fast with setup guidance if dependencies are missing
  (FVP binary, Arm toolchain, Ethos-U app workspace).

## Staged Workflow (AIPC Method)

1. **Preflight & target resolution**
   - `scripts/ethosu_preflight_check.py`
   - Resolve target profile from `scripts/_ethosu_targets.py`

2. **Golden baseline + ONNX export**
   - `scripts/export_pytorch_to_onnx.py`
   - Save `golden_input.npy`, `golden_output.npy`

3. **ONNX structural checks + patch loop**
   - Read `references/layout_fallback_playbook.md` **first** if Vela shows TRANSPOSE/CPU glue
   - `scripts/onnx_sanity_check.py`
   - Optional: `scripts/onnx_layout_optimizer.py` (`run_pipeline.py --optimize-onnx-layout`)
   - Detect dynamic dims, NCHW usage, and risky ops
   - Apply model patching rules from `references/operator_patching_ethosu.md`
   - Repeat until conversion-risk ops are removed or accepted as CPU fallback

4. **ONNX -> TFLite int8 conversion**
   - `scripts/onnx_to_tflite_int8.py`
   - Use representative dataset and force full-int path by default
   - Optional small grid: `scripts/sweep_conversion_options.py`
     (`run_pipeline.py --sweep-conversion`). Accept a variant only with no
     CPU/SRAM/accuracy regression. Do **not** brute-force `-nodaftc 2` on large detectors.

5. **Vela compile + memory/fallback inspection**
   - `scripts/compile_with_vela.py`
   - Always include:
     - `--accelerator-config ethos-u55-128`
     - `--config configs/vela_psoc_e84.ini`
     - `--system-config psoc_e84_syscfg`
     - `--memory-mode psoc_e84_balanced`
     - `--show-cpu-operations`
   - Analyze generated report:
     - CPU fallback operators
     - arena estimate vs SoCMEM budget
     - `transpose_fallback_summary` (split into export/QDQ/layout vs inherent)
   - `scripts/analyze_cpu_fallback.py` → per-op table (`layout_glue` / `qdq` / `inherent` / `unsupported_op`)
   - Default compile mode uses `--debug-force-legacy-core` for the custom
     PSoC profile on current Vela release.
   - Built-in profile fallback is opt-in only (`--allow-fallback`).

6. **Accuracy checks**
   - `scripts/compare_accuracy.py`
   - Compare ONNX/TFLite outputs to PyTorch golden
   - Track cosine similarity / MAE / RMSE

7. **FVP/hardware validation**
   - `scripts/run_fvp_harness.py`
   - Build + run TFLM Ethos-U app on Corstone-300 FVP
   - If unavailable on current machine, keep a clear "unverified" mark

## Quick Start

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84
```

## Real-model examples

- YOLOv8n pipeline:
  - `examples/yolov8n/run_yolov8n_split_pipeline.py`
  - exposes pre-concat box/score tensors separately before int8 quantization
- CRNN pipeline:
  - `examples/crnn/run_crnn_pipeline.py`
  - runs strict full-int8 attempt first, then optional relaxed conversion for diagnosis

## Failure Diagnosis Shortcuts

- **onnx2tf / tensorflow missing**
  - Install required packages and rerun preflight.
- **Vela compile fails**
  - Confirm `ethos-u-vela` install and target config names.
  - Inspect `vela_primary_stdout_stderr.log`.
  - If explicitly enabled, inspect `vela_fallback_stdout_stderr.log`.
  - Built-in fallback is not automatic; pass `--allow-fallback` only when you
    intentionally want fallback behavior.
- **High CPU fallback ratio**
  - Follow `references/layout_fallback_playbook.md` before changing converter flags.
  - Run `scripts/analyze_cpu_fallback.py` on the Vela log.
  - layout_glue → ONNX layout optimizer / NHWC export; inherent → document CPU fallback.
- **Arena overflow**
  - Reduce activation footprint (resolution/channels, block fusion choices),
    adjust memory mode placement, or split model stages.
- **Accuracy collapse after int8**
  - Increase representative set diversity, inspect quant scales/zero-points,
    and review layout conversions.

## Reference Map

- Workflow overview: `README.md`
- Operator patching: `references/operator_patching_ethosu.md`
- Layout CPU-fallback playbook: `references/layout_fallback_playbook.md`
- Ethos-U transpose constraints: `references/ethosu_transpose_constraints.md`
- Quantization + accuracy: `references/accuracy_and_quantization.md`
- Troubleshooting: `references/troubleshooting.md`
- FVP notes: `references/fvp_corstone300.md`

## Script Index

- `scripts/_ethosu_targets.py` - target catalog (SoC config abstraction)
- `scripts/ethosu_preflight_check.py` - environment/toolchain checks
- `scripts/export_pytorch_to_onnx.py` - baseline and ONNX export
- `scripts/onnx_sanity_check.py` - ONNX checks and risk report
- `scripts/onnx_to_tflite_int8.py` - ONNX->TFLite int8 conversion
- `scripts/onnx_layout_optimizer.py` - cancel/fold layout Transpose + identity QDQ
- `scripts/analyze_cpu_fallback.py` - classify Vela CPU fallback ops
- `scripts/sweep_conversion_options.py` - small converter grid with regression guard
- `scripts/compile_with_vela.py` - Vela compile + report generation
- `scripts/summarize_vela.py` - compact compile report summary
- `scripts/compare_accuracy.py` - cross-framework accuracy checks
- `scripts/run_fvp_harness.py` - Corstone-300 build/run hooks
- `scripts/run_pipeline.py` - staged loop orchestrator
- `examples/yolov8n/run_yolov8n_split_pipeline.py` - real-model split-output YOLO validation
