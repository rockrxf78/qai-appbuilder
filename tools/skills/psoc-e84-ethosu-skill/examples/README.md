# Example: tinycnn staged run

This skill ships a tiny CNN demo to exercise the staged loop:

1. PyTorch -> ONNX export
2. ONNX checks
3. ONNX -> TFLite int8 (full integer quantization)
4. Vela compile for `ethos-u55-128`
5. Accuracy comparison against the PyTorch golden output

Run:

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/run_pipeline.py \
  --work-dir tools/skills/psoc-e84-ethosu-skill/build/pipeline_demo \
  --target psoc_e84
```

If your host lacks TensorFlow/onnx2tf/Vela, run preflight first:

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/ethosu_preflight_check.py
```
