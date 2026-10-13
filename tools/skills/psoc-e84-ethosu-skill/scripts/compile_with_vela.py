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
        r"for (.+?)\.?\s*Placing on CPU instead",
        r"Unsupported .* for ([^']+)'?\.?\s*Placing on CPU instead",
        r"Warning:\s*([A-Z0-9_]+)\s*'([^']+)'.*Placing on CPU instead",
    ]
    ops: list[str] = []
    for line in log_text.splitlines():
        lowered = line.lower()
        if "cpu" not in lowered:
            continue
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                if match.lastindex and match.lastindex >= 2:
                    ops.append(f"{match.group(1).strip()}::{match.group(2).strip()}")
                else:
                    ops.append(match.group(1).strip())
                break
    return sorted(set(ops))


def _parse_int_list(raw: str) -> list[int]:
    return [int(token) for token in re.findall(r"-?\d+", raw)]


def _parse_cpu_operator_details(log_text: str) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    pattern = re.compile(
        r"CPU:\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s+\(inputs\s+(.+?),\s*outputs\s+(.+?)\)",
        flags=re.IGNORECASE,
    )
    for line in log_text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        details.append(
            {
                "op_type": match.group(1),
                "op_name": match.group(2).strip(),
                "inputs": match.group(3).strip(),
                "outputs": match.group(4).strip(),
                "raw_line": line.strip(),
            }
        )
    return details


def _classify_transpose(shape: list[int], perm: list[int]) -> tuple[str, str]:
    # Heuristic split for reporting:
    # - layout-glue candidates are commonly introduced by ONNX->TF layout adaptation.
    # - non-layout swaps are usually tied to model topology (decoder/axis semantics).
    is_rank4 = len(shape) == 4
    is_layout_swap = tuple(perm) in {(0, 3, 1, 2), (0, 2, 3, 1)}
    if is_rank4 and is_layout_swap:
        if shape[1] > 1 and shape[2] > 1 and shape[3] > 1:
            return (
                "removable",
                "NHWC<->NCHW layout-conversion candidate introduced by export/conversion tooling.",
            )
    if tuple(perm) == (0, 1, 3, 2):
        return (
            "inherent",
            "Axis swap is tied to detection-head topology (not a pure NHWC<->NCHW glue pattern).",
        )
    return (
        "inherent",
        "Permutation is not a common removable layout-glue swap; treat as model-topology fallback unless proven otherwise.",
    )


def _parse_transpose_fallbacks(log_text: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r"Warning:\s*TRANSPOSE\s*'([^']+)'.*?Op has ifm_shape:\s*\[([^\]]+)\]\s*and permutation is:\s*\[([^\]]+)\]",
        flags=re.IGNORECASE | re.DOTALL,
    )
    entries: list[dict[str, object]] = []
    for match in pattern.finditer(log_text):
        op_name = match.group(1).strip()
        ifm_shape = _parse_int_list(match.group(2))
        permutation = _parse_int_list(match.group(3))
        classification, reason = _classify_transpose(ifm_shape, permutation)
        entries.append(
            {
                "op_name": op_name,
                "ifm_shape": ifm_shape,
                "permutation": permutation,
                "classification": classification,
                "reason": reason,
            }
        )
    deduped: dict[str, dict[str, object]] = {}
    for entry in entries:
        deduped[str(entry["op_name"])] = entry
    return sorted(deduped.values(), key=lambda item: str(item["op_name"]))


def _summarize_transpose_fallbacks(
    transpose_fallbacks: list[dict[str, object]],
) -> dict[str, object]:
    removable = [entry for entry in transpose_fallbacks if entry["classification"] == "removable"]
    inherent = [entry for entry in transpose_fallbacks if entry["classification"] != "removable"]
    return {
        "total": len(transpose_fallbacks),
        "removable_count": len(removable),
        "inherent_count": len(inherent),
        "classification_method": (
            "Heuristic split based on permutation/shape from Vela warnings: "
            "rank-4 [0,3,1,2] or [0,2,3,1] with non-trivial H/W/C -> removable layout glue; "
            "other permutations -> inherent."
        ),
        "removable_examples": removable[:8],
        "inherent_examples": inherent[:8],
        "all": transpose_fallbacks,
    }


def _parse_operator_breakdown(log_text: str) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    cpu_match = re.search(r"CPU operators\s*=\s*(\d+)\s*\(([\d\.]+)%\)", log_text)
    npu_match = re.search(r"NPU operators\s*=\s*(\d+)\s*\(([\d\.]+)%\)", log_text)
    if cpu_match:
        payload["cpu_operator_count"] = int(cpu_match.group(1))
        payload["cpu_operator_ratio_percent"] = float(cpu_match.group(2))
    if npu_match:
        payload["npu_operator_count"] = int(npu_match.group(1))
        payload["npu_operator_ratio_percent"] = float(npu_match.group(2))
    return payload


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
        for key in [
            "sram_memory_used",      # KiB in Vela summary CSV
            "arena_memory_used",     # usually KiB
            "arena_size",            # tool-dependent
            "arena_memory",
            "sram_used",
            "total_arena_memory",    # may be bytes in some tool versions
        ]:
            if key not in row:
                continue
            try:
                value = float(str(row[key]).strip())
            except ValueError:
                continue
            if key == "total_arena_memory" and value > 1024 * 1024:
                return value / 1024.0
            return value
    return None


def _estimate_flash_usage_kb(summary_rows: list[dict[str, str]]) -> float | None:
    for row in summary_rows:
        offchip = _first_float(row, ["off_chip_flash_memory_used"])
        onchip = _first_float(row, ["on_chip_flash_memory_used"])
        if offchip is not None or onchip is not None:
            return float((offchip or 0.0) + (onchip or 0.0))
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
        "--allow-fallback",
        action="store_true",
        help="Explicitly allow fallback to built-in Vela profile when primary compile fails",
    )
    parser.add_argument(
        "--legacy-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Vela legacy Python core for primary compile "
            "(default: enabled, because current Regor path rejects custom psoc_e84 profile)."
        ),
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
    if args.legacy_core:
        primary_cmd.append("--debug-force-legacy-core")

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

    if completed.returncode != 0 and args.allow_fallback:
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
    cpu_fallback_details = _parse_cpu_operator_details(
        active_completed.stdout + "\n" + active_completed.stderr
    )
    transpose_fallbacks = _parse_transpose_fallbacks(
        active_completed.stdout + "\n" + active_completed.stderr
    )
    transpose_summary = _summarize_transpose_fallbacks(transpose_fallbacks)
    summary_rows = _read_summary_csv(output_dir)
    arena_kb = _estimate_arena_usage_kb(summary_rows)
    flash_kb = _estimate_flash_usage_kb(summary_rows)
    operator_breakdown = _parse_operator_breakdown(
        active_completed.stdout + "\n" + active_completed.stderr
    )

    warnings = []
    if args.legacy_core:
        warnings.append(
            "Primary compile used --debug-force-legacy-core to support custom psoc_e84 profile "
            "on current Vela release."
        )
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
    if transpose_summary["total"]:
        warnings.append(
            "TRANSPOSE fallback split: "
            f"total={transpose_summary['total']}, "
            f"removable={transpose_summary['removable_count']}, "
            f"inherent={transpose_summary['inherent_count']}."
        )
    if arena_kb is not None and arena_kb > target.arena_budget_kb:
        warnings.append(
            f"Arena estimate {arena_kb:.1f}KB exceeds target budget {target.arena_budget_kb}KB."
        )
    sram_exceed_match = re.search(
        r"SRAM target for arena memory area exceeded\.\s*Target\s*=\s*(\d+)\s*Bytes,\s*Actual\s*=\s*(\d+)\s*Bytes",
        active_completed.stdout + "\n" + active_completed.stderr,
        flags=re.IGNORECASE,
    )
    if sram_exceed_match:
        target_bytes = int(sram_exceed_match.group(1))
        actual_bytes = int(sram_exceed_match.group(2))
        warnings.append(
            f"SRAM target exceeded in Vela log: target={target_bytes} bytes, actual={actual_bytes} bytes."
        )

    report = {
        "input_tflite": str(input_model),
        "target": target.key,
        "accelerator_config": target.accelerator_config,
        "system_config": target.system_config,
        "memory_mode": target.memory_mode,
        "used_legacy_core": args.legacy_core,
        "used_fallback_profile": used_fallback,
        "active_command": active_cmd,
        "compiled_models": [str(p) for p in compiled_models],
        "cpu_fallback_ops": cpu_fallback_ops,
        "cpu_fallback_details": cpu_fallback_details,
        "operator_breakdown": operator_breakdown,
        "transpose_fallback_summary": transpose_summary,
        "arena_estimate_kb": arena_kb,
        "flash_estimate_kb": flash_kb,
        "target_arena_budget_kb": target.arena_budget_kb,
        "target_flash_budget_kb": target.flash_mb * 1024,
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
