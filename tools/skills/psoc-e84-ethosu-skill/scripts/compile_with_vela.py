#!/usr/bin/env python3
"""
Compile TFLite for Ethos-U with Vela and summarize fallback/memory status.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from _ethosu_targets import format_human_memory, get_target


def _find_vela() -> str:
    for cmd in ("vela", "ethos-u-vela", "ethosu-vela"):
        resolved = shutil.which(cmd)
        if resolved:
            return resolved
        local_bin = Path.home() / ".local" / "bin" / cmd
        if local_bin.exists():
            return str(local_bin)
    raise RuntimeError("Vela CLI not found. Install with `pip install ethos-u-vela`.")


def _extract_cpu_fallback_ops(log_text: str) -> list[str]:
    patterns = [
        r"CPU operation[:\s]+(.+)",
        r"will run on CPU[:\s]+(.+)",
        r"placed on cpu[:\s]+(.+)",
    ]
    ops: list[str] = []
    for line in log_text.splitlines():
        lowered = line.lower()
        if "cpu" not in lowered:
            continue
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                ops.append(match.group(1).strip())
                break
    return sorted(set(ops))


def _read_summary_csv(output_dir: Path) -> list[dict[str, str]]:
    candidates = sorted(output_dir.glob("*summary*.csv"))
    if not candidates:
        return []
    rows: list[dict[str, str]] = []
    with candidates[-1].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def _first_float(row: dict[str, str], candidate_keys: list[str]) -> float | None:
    for key in candidate_keys:
        if key not in row:
            continue
        try:
            return float(str(row[key]).strip())
        except ValueError:
            continue
    return None


def _estimate_arena_usage_kb(summary_rows: list[dict[str, str]]) -> float | None:
    for row in summary_rows:
        value = _first_float(
            row,
            [
                "arena_size",
                "arena_memory",
                "arena_memory_used",
                "sram_used",
                "total_arena_memory",
            ],
        )
        if value is not None:
            return value / 1024.0 if value > 4096 else value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile model with Vela for Ethos-U")
    parser.add_argument("--tflite", required=True, help="Input int8 TFLite")
    parser.add_argument(
        "--target",
        default="psoc_e84",
        help="Target key from _ethosu_targets.py (default: psoc_e84)",
    )
    parser.add_argument(
        "--vela-config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs" / "vela_psoc_e84.ini"
        ),
        help="Path to Vela config .ini",
    )
    parser.add_argument(
        "--output-dir", default="build/vela", help="Directory for Vela output"
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Additional Vela arguments appended at the end",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable fallback to built-in Vela system/memory profile on failure",
    )
    parser.add_argument(
        "--fallback-system-config",
        default="Ethos_U55_High_End_Embedded",
        help="Fallback built-in Vela system config",
    )
    parser.add_argument(
        "--fallback-memory-mode",
        default="Shared_Sram",
        help="Fallback built-in Vela memory mode",
    )
    args = parser.parse_args()

    target = get_target(args.target)
    vela = _find_vela()
    input_model = Path(args.tflite).resolve()
    if not input_model.exists():
        raise RuntimeError(f"TFLite model not found: {input_model}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_cmd = [
        vela,
        str(input_model),
        "--accelerator-config",
        target.accelerator_config,
        "--config",
        str(Path(args.vela_config).resolve()),
        "--system-config",
        target.system_config,
        "--memory-mode",
        target.memory_mode,
        "--show-cpu-operations",
        "--output-dir",
        str(output_dir),
    ] + args.extra_args

    print("=" * 72)
    print("Vela compile command")
    print("=" * 72)
    print(" ".join(primary_cmd))
    print("-" * 72)
    print(f"target memory profile: {format_human_memory(target)}")

    completed = subprocess.run(primary_cmd, capture_output=True, text=True)
    primary_log_path = output_dir / "vela_primary_stdout_stderr.log"
    primary_log_path.write_text(
        f"CMD: {' '.join(primary_cmd)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}",
        encoding="utf-8",
    )

    used_fallback = False
    active_completed = completed
    active_cmd = primary_cmd
    log_path = primary_log_path
    initial_error = None

    if completed.returncode != 0 and not args.disable_fallback:
        initial_error = completed.stderr.strip() or completed.stdout.strip()
        fallback_cmd = [
            vela,
            str(input_model),
            "--accelerator-config",
            target.accelerator_config,
            "--system-config",
            args.fallback_system_config,
            "--memory-mode",
            args.fallback_memory_mode,
            "--show-cpu-operations",
            "--output-dir",
            str(output_dir),
        ] + args.extra_args
        fallback_completed = subprocess.run(fallback_cmd, capture_output=True, text=True)
        fallback_log_path = output_dir / "vela_fallback_stdout_stderr.log"
        fallback_log_path.write_text(
            f"CMD: {' '.join(fallback_cmd)}\n\nSTDOUT:\n{fallback_completed.stdout}\n\nSTDERR:\n{fallback_completed.stderr}",
            encoding="utf-8",
        )
        if fallback_completed.returncode == 0:
            used_fallback = True
            active_completed = fallback_completed
            active_cmd = fallback_cmd
            log_path = fallback_log_path
        else:
            raise RuntimeError(
                "Vela compile failed for both primary and fallback profiles.\n"
                f"Primary log: {primary_log_path}\n"
                f"Fallback log: {fallback_log_path}\n"
                f"Primary stderr:\n{completed.stderr}\n\n"
                f"Fallback stderr:\n{fallback_completed.stderr}"
            )
    elif completed.returncode != 0:
        raise RuntimeError(
            "Vela compile failed.\n"
            f"See {primary_log_path}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    compiled_models = sorted(output_dir.glob("*_vela.tflite"))
    cpu_fallback_ops = _extract_cpu_fallback_ops(
        active_completed.stdout + "\n" + active_completed.stderr
    )
    summary_rows = _read_summary_csv(output_dir)
    arena_kb = _estimate_arena_usage_kb(summary_rows)

    warnings = []
    if used_fallback:
        warnings.append(
            "Primary PSoC custom config failed; compiled using fallback built-in "
            f"profile ({args.fallback_system_config}/{args.fallback_memory_mode})."
        )
        if initial_error:
            warnings.append(f"Primary config error: {initial_error.splitlines()[-1]}")
    if cpu_fallback_ops:
        warnings.append(
            "CPU fallback ops detected. Inspect operator patching guidance before acceptance."
        )
    if arena_kb is not None and arena_kb > target.arena_budget_kb:
        warnings.append(
            f"Arena estimate {arena_kb:.1f}KB exceeds target budget {target.arena_budget_kb}KB."
        )

    report = {
        "input_tflite": str(input_model),
        "target": target.key,
        "accelerator_config": target.accelerator_config,
        "system_config": target.system_config,
        "memory_mode": target.memory_mode,
        "used_fallback_profile": used_fallback,
        "active_command": active_cmd,
        "compiled_models": [str(p) for p in compiled_models],
        "cpu_fallback_ops": cpu_fallback_ops,
        "arena_estimate_kb": arena_kb,
        "target_arena_budget_kb": target.arena_budget_kb,
        "summary_rows": summary_rows,
        "warnings": warnings,
        "vela_log_path": str(log_path),
        "primary_log_path": str(primary_log_path),
    }
    report_path = output_dir / "vela_compile_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Vela compile report: {report_path}")
    if warnings:
        print("[WARN] compile completed with warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("[OK] no fallback/memory warning detected by script heuristics.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
