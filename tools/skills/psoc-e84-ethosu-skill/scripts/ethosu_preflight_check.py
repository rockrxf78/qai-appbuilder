#!/usr/bin/env python3
"""
Preflight checker for PSoC E84 Ethos-U workflow.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str | None = None


def _find_cmd(candidates: list[str]) -> str | None:
    for cmd in candidates:
        resolved = shutil.which(cmd)
        if resolved:
            return resolved
        local_bin = Path.home() / ".local" / "bin" / cmd
        if local_bin.exists():
            return str(local_bin)
    return None


def check_python_modules() -> list[CheckResult]:
    required = [
        ("onnx", "pip install onnx"),
        ("numpy", "pip install numpy"),
    ]
    optional = [
        ("torch", "pip install torch"),
        ("tensorflow", "pip install tensorflow"),
        ("onnx2tf", "pip install onnx2tf"),
        ("onnxruntime", "pip install onnxruntime"),
        ("ultralytics", "pip install ultralytics"),
        ("PIL", "pip install Pillow"),
        ("scipy", "pip install scipy"),
    ]
    results: list[CheckResult] = []
    for mod_name, fix in required:
        try:
            importlib.import_module(mod_name)
            results.append(CheckResult(f"python:{mod_name}", True, "import ok"))
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    f"python:{mod_name}",
                    False,
                    f"import failed: {exc}",
                    fix=fix,
                )
            )
    for mod_name, fix in optional:
        try:
            importlib.import_module(mod_name)
            results.append(CheckResult(f"python:{mod_name}", True, "import ok"))
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    f"python:{mod_name}",
                    False,
                    f"optional module missing: {exc}",
                    fix=fix,
                )
            )
    return results


def check_vela_tools() -> list[CheckResult]:
    results: list[CheckResult] = []
    vela_cmd = _find_cmd(["vela", "ethos-u-vela", "ethosu-vela"])
    if vela_cmd:
        results.append(CheckResult("tool:vela", True, f"found at {vela_cmd}"))
    else:
        results.append(
            CheckResult(
                "tool:vela",
                False,
                "Vela CLI not found in PATH",
                fix="pip install ethos-u-vela",
            )
        )
    return results


def check_fvp() -> list[CheckResult]:
    results: list[CheckResult] = []
    env_hint = os.environ.get("FVP_CORSTONE300_PATH")
    candidates = [
        "FVP_Corstone_SSE-300_Ethos-U55",
        "FVP_Corstone_SSE-300_Ethos-U55.exe",
    ]
    if env_hint:
        hinted = Path(env_hint).expanduser()
        if hinted.exists():
            results.append(CheckResult("tool:fvp", True, f"found via env: {hinted}"))
        else:
            results.append(
                CheckResult(
                    "tool:fvp",
                    False,
                    f"FVP_CORSTONE300_PATH is set but missing: {hinted}",
                    fix="Install Arm Corstone-300 FVP and update FVP_CORSTONE300_PATH",
                )
            )
        return results

    resolved = _find_cmd(candidates)
    if resolved:
        results.append(CheckResult("tool:fvp", True, f"found at {resolved}"))
    else:
        results.append(
            CheckResult(
                "tool:fvp",
                False,
                "Corstone-300 FVP binary not found",
                fix=(
                    "Install Arm Corstone-300 FVP and set FVP_CORSTONE300_PATH "
                    "or add the binary to PATH"
                ),
            )
        )
    return results


def check_arm_toolchain() -> list[CheckResult]:
    results: list[CheckResult] = []
    cmd = _find_cmd(["arm-none-eabi-gcc", "armclang"])
    if cmd:
        results.append(CheckResult("tool:arm-toolchain", True, f"found at {cmd}"))
    else:
        results.append(
            CheckResult(
                "tool:arm-toolchain",
                False,
                "No Arm embedded compiler found",
                fix="Install Arm GNU toolchain (arm-none-eabi-gcc) or Arm Compiler",
            )
        )
    return results


def run_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.extend(check_python_modules())
    checks.extend(check_vela_tools())
    checks.extend(check_fvp())
    checks.extend(check_arm_toolchain())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ethos-U preflight checker")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat optional checks as fatal for CI-style gating",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report",
    )
    args = parser.parse_args()

    checks = run_checks()
    hard_failures = [
        c
        for c in checks
        if not c.ok and (args.strict or c.name.startswith(("python:onnx", "python:numpy", "tool:vela")))
    ]

    if args.json:
        payload = {
            "python": sys.version,
            "strict": args.strict,
            "checks": [asdict(c) for c in checks],
            "hard_failure_count": len(hard_failures),
        }
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 72)
        print("PSoC E84 Ethos-U Preflight")
        print("=" * 72)
        for c in checks:
            status = "PASS" if c.ok else "FAIL"
            print(f"[{status}] {c.name}: {c.detail}")
            if c.fix and not c.ok:
                print(f"       fix: {c.fix}")
        print("-" * 72)
        print(
            f"hard failures: {len(hard_failures)} "
            f"(strict={'on' if args.strict else 'off'})"
        )

    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
