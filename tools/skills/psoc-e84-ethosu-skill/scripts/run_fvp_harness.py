#!/usr/bin/env python3
"""
Build/run harness template for Corstone-300 FVP + TFLM Ethos-U app.

This script is intentionally a hook layer: it validates host prerequisites and
then executes user-provided build/run commands.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_fvp(user_value: str) -> str | None:
    if user_value:
        p = Path(user_value).expanduser()
        return str(p) if p.exists() else None
    env_path = os.environ.get("FVP_CORSTONE300_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in (
        "FVP_Corstone_SSE-300_Ethos-U55",
        "FVP_Corstone_SSE-300_Ethos-U55.exe",
    ):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _check_tool(name: str) -> tuple[bool, str]:
    resolved = shutil.which(name)
    return (resolved is not None, resolved or "")


def _run_shell(cmd: str, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), shell=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit={completed.returncode}: {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Corstone-300 FVP harness")
    parser.add_argument(
        "--app-dir",
        required=True,
        help="Directory of TFLM + Ethos-U application",
    )
    parser.add_argument(
        "--build-cmd",
        default="",
        help="Build command executed in --app-dir (optional)",
    )
    parser.add_argument(
        "--run-cmd",
        default="",
        help="Run command executed in --app-dir (optional)",
    )
    parser.add_argument(
        "--fvp-bin",
        default="",
        help="Path to FVP_Corstone_SSE-300_Ethos-U55 binary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prerequisites and print commands only",
    )
    args = parser.parse_args()

    app_dir = Path(args.app_dir).resolve()
    if not app_dir.exists():
        raise RuntimeError(f"--app-dir not found: {app_dir}")

    fvp = _resolve_fvp(args.fvp_bin)
    gcc_ok, gcc_path = _check_tool("arm-none-eabi-gcc")
    cmake_ok, cmake_path = _check_tool("cmake")

    print("=" * 72)
    print("Corstone-300 FVP harness preflight")
    print("=" * 72)
    print(f"app_dir: {app_dir}")
    print(f"fvp: {fvp or 'NOT FOUND'}")
    print(f"arm-none-eabi-gcc: {gcc_path or 'NOT FOUND'}")
    print(f"cmake: {cmake_path or 'NOT FOUND'}")

    failures = []
    if not fvp:
        failures.append(
            "Install Arm Corstone-300 FVP and set FVP_CORSTONE300_PATH "
            "or pass --fvp-bin."
        )
    if not gcc_ok:
        failures.append(
            "Install Arm GNU Embedded Toolchain so arm-none-eabi-gcc is in PATH."
        )
    if not cmake_ok:
        failures.append("Install CMake.")

    if failures:
        print("[ERROR] prerequisites missing:")
        for item in failures:
            print(f"  - {item}")
        return 1

    if args.build_cmd:
        print(f"[BUILD] {args.build_cmd}")
        if not args.dry_run:
            _run_shell(args.build_cmd, app_dir)
    else:
        print("[BUILD] no --build-cmd provided; skipped")

    if args.run_cmd:
        print(f"[RUN] {args.run_cmd}")
        if not args.dry_run:
            _run_shell(args.run_cmd, app_dir)
    else:
        print(
            "[RUN] no --run-cmd provided. Example:\n"
            "      --run-cmd '<fvp-bin> -a build/app.axf --timelimit 30'"
        )

    print("[OK] harness completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
