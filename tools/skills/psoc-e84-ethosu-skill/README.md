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
- `examples/` - tiny CNN and real-model pipeline helpers

---

## Quick start

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/ethosu_preflight_check.py

python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84
```

If you already have a model and calibration data, call stage scripts directly.

Real-model scripts:

```bash
python3 tools/skills/psoc-e84-ethosu-skill/examples/yolov8n/run_yolov8n_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/yolov8n \
  --dataset-yaml coco128.yaml \
  --calib-samples 128 --eval-samples 32

python3 tools/skills/psoc-e84-ethosu-skill/examples/crnn/run_crnn_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/crnn \
  --calib-samples 160 --eval-samples 120
```

### Vela compile modes

- Default behavior (custom profile, no fallback):
  - `compile_with_vela.py` uses `--debug-force-legacy-core`
  - built-in fallback is disabled
- Optional fallback (explicit only):
  - add `--allow-fallback`
- Regor-only experiment (expected to fail for current custom profile):
  - add `--no-legacy-core`

---

## Assumptions

- Host-side Python environment can install/use:
  - `torch`, `onnx`, `onnxruntime`, `tensorflow`, `onnx2tf`, `ethos-u-vela`
- Device-side runtime integration (TFLM + Ethos-U) is handled by project BSP.
- Vela memory profiles are an initial approximation and must be tuned with
  board-specific timing + linker constraints.

---

## Real-model validation snapshot (cloud VM)

The following runs were executed in this branch with the scripts under `examples/`.

### YOLOv8n (ultralytics pretrained, COCO128 images)

- Calibration data: 128 real images (`build/yolov8n/yolo_rep_data.npy`)
- Validation subset: first 32 images from COCO128 val set
- Full pipeline status:
  - PyTorch -> ONNX: success
  - ONNX -> TFLite full-int8: success (`int8` input/output)
  - Vela custom `psoc_e84` compile: success (no built-in fallback)

Key outputs:
- `build/yolov8n/yolov8n.onnx`
- `build/yolov8n/yolov8n_int8.tflite`
- `build/yolov8n/vela/vela_compile_report.json`
- `build/yolov8n/yolov8n_raw_output_diff.json`
- `build/yolov8n/yolov8n_detection_match_pt_vs_onnx.json`
- `build/yolov8n/yolov8n_detection_match_pt_vs_tflite.json`

Observed metrics:
- Operator placement: NPU `291 (92.1%)`, CPU `25 (7.9%)`
  - CPU fallback is mostly `TRANSPOSE`
- Memory/cycles from Vela summary:
  - SRAM arena estimate: `11200 KB` (**exceeds** 5 MB SoCMEM / 4096 KB arena budget)
  - Flash weights estimate: `2856.69 KB` (within 128 MB Flash budget)
  - Estimated cycles/inference: `98,716,312`
- Accuracy drift:
  - ONNX vs TFLite raw outputs: cosine mean `0.9988`, max-abs max `549.19`
  - PT vs ONNX detection signature match (count/class-jaccard/top1): `0.75 / 0.948 / 1.0`
  - PT vs TFLite detection signature match: count `0.0`, class-jaccard `0.0`

Interpretation:
- Quantization numerics stay close at tensor level, but post-processing/detection
  behavior diverges severely for this export path and settings.

### CRNN (rockrxf78/crnn-pytorch)

- Repo path: `build/crnn-pytorch` (cloned)
- Pretrained checkpoint used: `checkpoints/crnn_synth90k.pt`
- Calibration/eval data: 160 synthetic text-line images (grayscale, 32x100)

Key outputs:
- `build/crnn/crnn_synth90k.onnx`
- `build/crnn/crnn_strict_int8_attempt.json`
- `build/crnn/crnn_relaxed_int8_attempt.json`
- `build/crnn/crnn_synth90k_relaxed.tflite`
- `build/crnn/vela-relaxed/vela_compile_report.json`
- `build/crnn/crnn_accuracy_report.json`
- `build/crnn/crnn_vs_pytorch_report.json`

Observed metrics:
- Strict full-int8 quantization: **failed** (onnx2tf conversion stopped in RNN/LSTM section)
- Relaxed conversion (`--allow-non-int8-output`): rerun also failed; script reused existing relaxed artifact from prior successful conversion and recorded this in `crnn_relaxed_int8_attempt.json`
- Vela (relaxed model) operator placement:
  - NPU `0 (0.0%)`, CPU `152 (100.0%)`
  - includes `WHILE`, LSTM-related ops, reshape/gather/split etc.
- Memory/cycles from Vela summary:
  - SRAM estimate: `1000 KB` (within 5 MB SoCMEM)
  - Flash estimate: `0 KB`
  - cycles reported as `0` (not meaningful for this all-CPU relaxed graph)
- Accuracy drift vs PyTorch (120 synthetic samples):
  - ONNX vs PyTorch: sequence agreement `1.0`, CER `0.0`
  - TFLite(relaxed) vs PyTorch: sequence agreement `0.525`, CER `0.0969`

Interpretation:
- Current toolchain path cannot provide production-usable full-int8/NPU execution
  for this CRNN architecture without model redesign.

## Compile mode and fallback policy

- Primary custom profile:
  - `System_Config=psoc_e84_syscfg`
  - `Memory_Mode=psoc_e84_balanced`
- Scripts use `--debug-force-legacy-core` by default because current Vela Regor
  path still rejects this custom profile (`Feature Map memory not configured`).
- Built-in profile fallback remains **opt-in only** (`--allow-fallback`).

## Not verified in this repository

- Corstone-300 FVP execution of a real TFLite Micro + Ethos-U firmware image
- Hardware execution on actual PSoC Edge E84 board
- ModusToolbox integration, flashing, and board-level perf/latency/power
- Board-accurate memory timing parameters for Vela config

## Facts to re-check against official sources

- Exact E84 deployable tensor-arena budget and linker placement constraints
- Final Ethos-U55-128 frequency/timing profile used in product firmware
- FVP model/version/flags matching your runtime integration
