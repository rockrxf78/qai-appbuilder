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
- `scripts/onnx_layout_optimizer.py` - ONNX layout-glue cleanup (optional)
- `scripts/analyze_cpu_fallback.py` - classify Vela CPU fallback ops
- `scripts/sweep_conversion_options.py` - small converter grid, no-regression pick
- `scripts/compile_with_vela.py` - Vela compile and report
- `scripts/compare_accuracy.py` - golden parity checks
- `scripts/run_fvp_harness.py` - Corstone-300 build/run hooks
- `scripts/run_pipeline.py` - staged loop orchestration
- `references/` - operator patching, layout-fallback playbook, quantization, troubleshooting docs
- `examples/` - tiny CNN and real-model pipeline helpers

---

## Quick start

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/ethosu_preflight_check.py

python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84 \
  --optimize-onnx-layout
```

Layout CPU-fallback tooling (generic, not YOLO-specific):

1. Read `references/layout_fallback_playbook.md`
2. `scripts/analyze_cpu_fallback.py --vela-report .../vela_compile_report.json`
3. Optional `scripts/onnx_layout_optimizer.py` before conversion
4. Optional `scripts/sweep_conversion_options.py` (tiny grid; rejects CPU regressions)

If you already have a model and calibration data, call stage scripts directly.

Real-model scripts:

```bash
python3 tools/skills/psoc-e84-ethosu-skill/examples/yolov8n/run_yolov8n_split_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/yolov8n \
  --dataset-yaml coco128.yaml \
  --calib-samples 128 --eval-samples 32 \
  --imgsz-candidates 320,256 \
  --onnx2tf-flextranspose-compression-dim 6

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

## Layout-fallback tooling validation

- **YOLOv8n 320 (existing artifacts, no reconvert):**
  - `analyze_cpu_fallback.py`: CPU 25 = **24 layout_glue + 1 inherent**
  - `onnx_layout_optimizer.py` on preconcat ONNX: Transpose **1→1** (no inverse pairs to cancel; glue is in onnx2tf, not ONNX). ORT max-abs `0.0`
- **tinycnn (`run_pipeline.py --optimize-onnx-layout --sweep-conversion`):**
  - ONNX has 0 Transpose; Vela still reports **1 layout_glue** TRANSPOSE (conversion-introduced)
  - Sweep winner: `default` (CPU 1). `nodaftc2` dropped TRANSPOSE to 0 but CPU rose **1→9** and was rejected

## Real-model validation snapshot (cloud VM)

The following runs were executed in this branch with the scripts under `examples/`.

### YOLOv8n (ultralytics pretrained, COCO128 images)

- Calibration data: 128 real images (`build/yolov8n/imgsz_320/yolo_rep_data.npy`)
- Validation subset: first 32 images from COCO128 val set
- Full pipeline status:
  - PyTorch -> ONNX: success
  - ONNX output split (pre-concat): success
    - boxes: `/model.22/Mul_2_output_0`
    - scores: `/model.22/Sigmoid_output_0`
  - ONNX -> TFLite full-int8: success (`int8` input/output, split outputs keep independent scales)
  - Vela custom `psoc_e84` compile: success (no built-in fallback)

Key outputs:
- `build/yolov8n/yolov8n_pipeline_summary.json`
- `build/yolov8n/imgsz_320/yolov8n.onnx`
- `build/yolov8n/imgsz_320/yolov8n_preconcat.onnx`
- `build/yolov8n/imgsz_320/yolov8n_int8.tflite`
- `build/yolov8n/imgsz_320/vela/vela_compile_report.json`
- `build/yolov8n/imgsz_320/yolov8n_tflite_output_quantization.json`
- `build/yolov8n/imgsz_320/yolov8n_raw_output_diff.json`
- `build/yolov8n/imgsz_320/yolov8n_detection_match_pt_vs_onnx.json`
- `build/yolov8n/imgsz_320/yolov8n_detection_match_pt_vs_tflite.json`
- `build/yolov8n/imgsz_320/yolov8n_transpose_fallback_classification.json`

Observed metrics:
- Selected run: `imgsz=320` (`256` fallback not needed because 320 already fits arena budget)
- Operator placement: NPU `289 (92.0%)`, CPU `25 (8.0%)`
  - TRANSPOSE fallback split: total `25` = export/QDQ/layout `24` + inherent `1`
  - export/QDQ/layout class is mostly ONNX->TF layout glue (`[0,3,1,2]` / `[0,2,3,1]`)
  - inherent class is one head-topology swap (`[0,1,3,2]`)
- Memory/cycles from Vela summary:
  - SRAM arena estimate: `2800 KB` (**fits** 5 MB SoCMEM / 4096 KB arena budget)
  - Flash weights estimate: `2839.64 KB` (within 128 MB Flash budget)
  - Estimated cycles/inference: `25,077,420`
- Output quantization (split-output check):
  - class scores tensor scale: `0.00390625`
  - box tensor scale: `1.6232509613`
  - scales are separated (`split_output_quantization_ok: true`)
- Accuracy drift:
  - ONNX vs TFLite raw outputs (split tensors):
    - boxes cosine mean `0.9989`, max-abs max `239.35`
    - scores cosine mean `0.9499`, max-abs max `0.7323`
  - PT vs ONNX detection signature (count/class-jaccard/top1): `0.3125 / 0.7542 / 0.8571`
  - PT vs TFLite detection signature (count/class-jaccard/top1): `0.3438 / 0.7198 / 0.8276`

TRANSPOSE mitigation experiments (same YOLOv8n 320 graph):
- Baseline (`-nodaftc 6`): 25 TRANSPOSE fallback, CPU ops 25 (8.0%)
- Aggressive suppression (`-nodaftc 2`): TRANSPOSE fallback became 0, but CPU fallback
  exploded to 698 ops (30.2%), so this is **not accepted** as default.
- Export simplify (`--export-simplify`) did not reduce TRANSPOSE fallback in this model.
- The YOLO pipeline now performs a built-in cleanup attempt (`--enable-transpose-cleanup`)
  and accepts it only if export/QDQ/layout transpose count improves **without**
  increasing non-transpose CPU fallback.
- Study artifacts:
  - `build/yolov8n-transpose-study/baseline/`
  - `build/yolov8n-transpose-study/optimized/`
  - `build/yolov8n-transpose-study/simplify/`

Interpretation:
- Previous fused-output int8 path had severe detection collapse (class-jaccard near 0).
- Split-output quantization recovered meaningful detections and non-zero agreement,
  but parity with PT is still not production-grade and needs further tuning.
- Current Vela/Ethos-U constraints do not allow direct NPU execution of most YOLO
  NHWC<->NCHW transpose glue; focus should be on graph/layout rewrites, not forcing
  unsupported transposes onto NPU.

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
