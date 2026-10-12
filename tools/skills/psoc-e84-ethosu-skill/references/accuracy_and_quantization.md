# Accuracy and full-int8 quantization checklist

## Objective

Keep model behavior stable across:
- PyTorch golden baseline
- ONNX Runtime
- TFLite int8
- (optional) TFLM + Ethos-U on FVP/hardware

## Calibration data requirements

- Use representative data from the true input distribution.
- Do not use only random noise for final acceptance.
- Prefer at least 100 samples for initial runs; increase for unstable models.
- Preserve preprocessing parity (normalization, channel order, resize rules).

## Full integer conversion settings

Use `onnx_to_tflite_int8.py` with:
- `supported_ops = TFLITE_BUILTINS_INT8`
- `inference_input_type = int8` and `inference_output_type = int8` (default)
- representative dataset iterator

If debugging host scripts, `--force-float-io` can isolate preprocessing issues.

## Accuracy gates (recommended)

- cosine similarity >= 0.99 for classification heads
- MAE and RMSE trend monitored per model version
- no catastrophic class flip on sanity inputs

Thresholds may be relaxed for detector/segmenter heads with post-processing.

## Typical degradation causes

1. Layout mismatch (NCHW not converted to NHWC)
2. Wrong representative data normalization/range
3. Unsupported ops rewritten with non-equivalent approximations
4. Post-processing drift (softmax/NMS outside model graph)
5. Activation saturation under int8

## Remediation loop

1. Validate ONNX parity with PyTorch golden.
2. Validate float TFLite parity (if available).
3. Enable int8 conversion and compare again.
4. Inspect per-layer scales/zero-points for outliers.
5. Update representative data and retry.
