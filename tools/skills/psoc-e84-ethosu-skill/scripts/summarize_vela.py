#!/usr/bin/env python3
"""
Read Vela output artifacts and print a compact summary for review comments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Vela compile output")
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path to vela_compile_report.json produced by compile_with_vela.py",
    )
    args = parser.parse_args()

    report_path = Path(args.report_json).resolve()
    if not report_path.exists():
        raise RuntimeError(f"report file not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("=" * 72)
    print("Vela Summary")
    print("=" * 72)
    print(f"target: {report['target']} ({report['accelerator_config']})")
    print(f"system_config: {report['system_config']}")
    print(f"memory_mode: {report['memory_mode']}")
    print(f"compiled_models: {len(report.get('compiled_models', []))}")
    for model in report.get("compiled_models", []):
        print(f"  - {model}")

    cpu_ops = report.get("cpu_fallback_ops", [])
    print(f"cpu_fallback_ops: {len(cpu_ops)}")
    for op in cpu_ops:
        print(f"  - {op}")

    arena_kb = report.get("arena_estimate_kb")
    budget_kb = report.get("target_arena_budget_kb")
    if arena_kb is not None:
        print(f"arena_estimate_kb: {arena_kb:.2f}")
    if budget_kb is not None:
        print(f"target_arena_budget_kb: {budget_kb}")

    warnings = report.get("warnings", [])
    if warnings:
        print("[WARNINGS]")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
