---
name: aipc-ethosu-toolkit
description: ARM Ethos-U NPU workflow for Ethos-U55 and Ethos-U85. Installs this skill plugin, registers both NPUs, and compiles TFLite int8 with Vela for PSoC Edge E84 (U55-128) or Corstone-320 (U85-256).
use_for: Deploying models onto an Arm Ethos-U55 or Ethos-U85 NPU, including PSoC E84, Vela compile, TFLite Micro, and Corstone FVP.
tags: ethos-u, ethos-u55, ethos-u85, npu, vela, tflite-micro, psoc-e84
---

# AIPC Ethos-U Toolkit (Ethos-U55 and Ethos-U85)

This file is the AppBuilder skill plugin. `scripts/install_ethosu_plugin.py` generates it from `tools/skills/psoc-e84-ethosu-skill` so install, registration, and NPU target selection stay on one path.

Registered NPU families: `ethos-u55`, `ethos-u85`.

## Registered Ethos-U NPU targets

| Target | NPU | Accelerator | Vela config | Legacy core | FVP env |
| --- | --- | --- | --- | --- | --- |
| `corstone300_u55_128_reference` | `ethos-u55` | `ethos-u55-128` | `vela_psoc_e84.ini` | yes | `FVP_CORSTONE300_PATH` |
| `corstone320_u85_256_reference` | `ethos-u85` | `ethos-u85-256` | `vela_ethos_u85.ini` | no (Regor) | `FVP_CORSTONE320_PATH` |
| `psoc_e84` | `ethos-u55` | `ethos-u55-128` | `vela_psoc_e84.ini` | yes | `FVP_CORSTONE300_PATH` |

Select a target with `--target <key>` on `run_pipeline.py`, `compile_with_vela.py`, `ethosu_preflight_check.py`, and `run_fvp_harness.py`. U55 targets default to Vela's legacy core. U85 targets use Regor and do not pass `--debug-force-legacy-core`.

## Canonical workflow


# AIPC Ethos-U Toolkit (PSoC Edge E84)

## Trigger Phrases

Always activate this skill when the user mentions:
- "PSoC E84"
- "Ethos-U55" / "Ethos-U85" / "ethos-u"
- "Vela compile"
- "TFLite Micro + Ethos-U"
- "Corstone-300 FVP" / "Corstone-320 FVP"
- "ONNX to TFLite int8 for MCU"
- "CPU fallback ops on Ethos-U"
- "SoCMEM / PSRAM / Flash placement"

## When to Use

Use this skill for Arm MCU deployment pipelines where the target NPU is Ethos-U55
or Ethos-U85 and the runtime is TFLite Micro + Ethos-U driver, with CMSIS-NN/CPU
fallback for unsupported ops. Pick the NPU from `scripts/_ethosu_targets.py`
(`psoc_e84` and `corstone300_u55_128_reference` are U55; `corstone320_u85_256_reference`
is U85). Do not hard-code `ethos-u55-128` once a U85 target is selected.

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

1. **Install, register, then preflight**
   - `scripts/install_ethosu_plugin.py` writes the AppBuilder skill plugin and the U55/U85 target registry
   - `scripts/ethosu_preflight_check.py --target <key>`
   - Resolve the same target from `scripts/_ethosu_targets.py` for every later stage

2. **Golden baseline + ONNX export**
   - `scripts/export_pytorch_to_onnx.py`
   - Save `golden_input.npy`, `golden_output.npy`

3. **ONNX structural checks + patch loop**
   - `scripts/onnx_sanity_check.py`
   - Detect dynamic dims, NCHW usage, and risky ops
   - Apply model patching rules from `references/operator_patching_ethosu.md`
   - Repeat until conversion-risk ops are removed or accepted as CPU fallback

4. **ONNX -> TFLite int8 conversion**
   - `scripts/onnx_to_tflite_int8.py`
   - Use representative dataset and force full-int path by default
   - For transpose experiments, use `--onnx2tf-flextranspose-compression-dim`
     cautiously. `2` can suppress transpose nodes but may increase fallback for
     other op families.
   - `examples/yolov8n/run_yolov8n_split_pipeline.py` can run an automatic
     export-layout transpose cleanup attempt (`--enable-transpose-cleanup`).

5. **Vela compile + memory/fallback inspection**
   - `scripts/compile_with_vela.py --target <key>`
   - The script fills accelerator, ini, system config, and memory mode from
     `scripts/_ethosu_targets.py`. Do not override those with U55 values when
     the selected target is Ethos-U85.
   - U55 custom profiles default to `--debug-force-legacy-core`. U85 uses Regor
     and does not pass that flag.
   - Built-in profile fallback is opt-in (`--allow-fallback`) and uses the
     target's fallback system config (`Ethos_U55_High_End_Embedded` or
     `Ethos_U85_SYS_DRAM_Mid`).
   - Analyze generated report:
     - CPU fallback operators
     - arena estimate vs the target budget
     - `transpose_fallback_summary` (split into export/QDQ/layout vs inherent)
     - `npu` and `accelerator_config` fields

6. **Accuracy checks**
   - `scripts/compare_accuracy.py`
   - Compare ONNX/TFLite outputs to PyTorch golden
   - Track cosine similarity / MAE / RMSE

7. **FVP/hardware validation**
   - `scripts/run_fvp_harness.py --target <key>`
   - U55 uses Corstone-300 (`FVP_CORSTONE300_PATH`); U85 uses Corstone-320
     (`FVP_CORSTONE320_PATH`)
   - If unavailable on current machine, keep a clear "unverified" mark

## Quick Start

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/install_ethosu_plugin.py

python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84

python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_u85 \
  --target corstone320_u85_256_reference
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
  - Check unsupported ops and rewrite pattern in references.
  - For transpose-heavy models (e.g. YOLOv8n), inspect
    `transpose_fallback_summary`:
    - export/QDQ/layout = layout glue from export/conversion
    - inherent = model-topology axis swaps
- **Arena overflow**
  - Reduce activation footprint (resolution/channels, block fusion choices),
    adjust memory mode placement, or split model stages.
- **Accuracy collapse after int8**
  - Increase representative set diversity, inspect quant scales/zero-points,
    and review layout conversions.

## Reference Map

- Workflow overview: `README.md`
- Operator patching: `references/operator_patching_ethosu.md`
- Ethos-U transpose constraints: `references/ethosu_transpose_constraints.md`
- Quantization + accuracy: `references/accuracy_and_quantization.md`
- Troubleshooting: `references/troubleshooting.md`
- FVP notes: `references/fvp_corstone300.md`

## Script Index

- `scripts/_ethosu_targets.py` - U55 and U85 target catalog
- `scripts/install_ethosu_plugin.py` - install plugin and register both NPUs
- `scripts/ethosu_preflight_check.py` - environment/toolchain checks
- `scripts/export_pytorch_to_onnx.py` - baseline and ONNX export
- `scripts/onnx_sanity_check.py` - ONNX checks and risk report
- `scripts/onnx_to_tflite_int8.py` - ONNX->TFLite int8 conversion
- `scripts/compile_with_vela.py` - Vela compile + report generation
- `scripts/summarize_vela.py` - compact compile report summary
- `scripts/compare_accuracy.py` - cross-framework accuracy checks
- `scripts/run_fvp_harness.py` - Corstone-300 (U55) / Corstone-320 (U85) hooks
- `scripts/run_pipeline.py` - staged loop orchestrator
- `examples/yolov8n/run_yolov8n_split_pipeline.py` - real-model split-output YOLO validation
