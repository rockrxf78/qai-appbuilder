# Corstone-300 FVP validation notes

## Scope

Corstone-300 (Cortex-M55 + Ethos-U55) is the closest widely available
simulation target for PSoC E84 class deployments.

## What this skill automates

- Host-side preflight checks (FVP/toolchain presence)
- Build/run hook orchestration (`run_fvp_harness.py`)
- Structured failure messages when prerequisites are missing

## What remains project-specific

- TFLM + Ethos-U application source tree
- BSP/startup/linker setup
- Exact FVP runtime flags (MPS3 image paths, memory maps, semihosting)
- PMU and latency extraction scripts

## Suggested command shape

```bash
python tools/skills/psoc-e84-ethosu-skill/scripts/run_fvp_harness.py \
  --app-dir <ethosu_app_root> \
  --build-cmd "cmake -S . -B build && cmake --build build -j" \
  --run-cmd "$FVP_CORSTONE300_PATH -a build/app.axf --timelimit 60"
```

## Verification outputs to capture

- Build log with compiler flags
- FVP console log
- Model output tensors for golden comparison
- Cycle/latency numbers (if instrumented)
