# Examples

This directory provides runnable examples for three model scales.

## 1) tinycnn sanity pipeline

```bash
python3 tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84
```

## 2) YOLOv8n real-model pipeline

Runs:
- PyTorch (ultralytics pretrained) -> ONNX
- ONNX -> full-int8 TFLite (representative data from COCO images)
- Vela compile with custom `psoc_e84` profile
- Output diff + detection match reports

```bash
python3 tools/skills/psoc-e84-ethosu-skill/examples/yolov8n/run_yolov8n_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/yolov8n \
  --dataset-yaml coco128.yaml \
  --calib-samples 128 \
  --eval-samples 32
```

Key outputs:
- `build/yolov8n/yolov8n.onnx`
- `build/yolov8n/yolov8n_int8.tflite`
- `build/yolov8n/vela/vela_compile_report.json`
- `build/yolov8n/yolov8n_raw_output_diff.json`
- `build/yolov8n/yolov8n_detection_match_pt_vs_onnx.json`
- `build/yolov8n/yolov8n_detection_match_pt_vs_tflite.json`
- `yolov8n/detection_match.py` is used as a subprocess helper for robust
  detection-signature comparison (isolated backend crashes/errors).

## 3) CRNN real-model pipeline

Runs:
- Clone/load `rockrxf78/crnn-pytorch` and pretrained checkpoint (if available)
- PyTorch -> ONNX
- Strict full-int8 quantization attempt (expected to fail on current toolchain)
- Relaxed quantization (explicit) for diagnosis
- Vela compile + CTC-based accuracy reports

```bash
python3 tools/skills/psoc-e84-ethosu-skill/examples/crnn/run_crnn_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/crnn \
  --calib-samples 160 \
  --eval-samples 120
```

Key outputs:
- `build/crnn/crnn_synth90k.onnx`
- `build/crnn/crnn_strict_int8_attempt.json`
- `build/crnn/crnn_relaxed_int8_attempt.json`
- `build/crnn/crnn_synth90k_relaxed.tflite`
- `build/crnn/vela-relaxed/vela_compile_report.json`
- `build/crnn/crnn_accuracy_report.json`
- `build/crnn/crnn_vs_pytorch_report.json`

Notes:
- The script records strict/relaxed onnx2tf failures explicitly.
- If relaxed conversion fails and an existing relaxed artifact is present, the
  script reuses it and marks this in `crnn_relaxed_int8_attempt.json`.

If your host lacks TensorFlow/onnx2tf/Vela, run preflight first:

```bash
python3 tools/skills/psoc-e84-ethosu-skill/scripts/ethosu_preflight_check.py
```
