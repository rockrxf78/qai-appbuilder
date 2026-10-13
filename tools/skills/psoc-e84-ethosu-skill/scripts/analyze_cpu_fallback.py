#!/usr/bin/env python3
"""Parse Vela CPU-fallback logs into a classified per-op table (JSON + markdown)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LAYOUT_PERMS = {(0, 3, 1, 2), (0, 2, 3, 1)}
QDQ_OP_TYPES = {"QUANTIZE", "DEQUANTIZE", "QUANTIZELINEAR", "DEQUANTIZELINEAR"}
UNSUPPORTED_HINTS = (
    "GATHER",
    "WHILE",
    "LSTM",
    "UNIDIRECTIONALLSTM",
    "BIDIRECTIONALLSTM",
    "RNN",
    "GRU",
    "NONMAXSUPPRESSION",
    "WHERE",
    "SCATTER",
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok) for tok in re.findall(r"-?\d+", raw)]


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_warnings(log_text: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r"Warning:\s*([A-Z0-9_]+)\s+'([^']+)'(.*?)(?:\n(?=Warning:|\nNetwork summary)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    shape_re = re.compile(
        r"ifm_shape:\s*\[([^\]]+)\]\s*and permutation is:\s*\[([^\]]+)\]",
        flags=re.IGNORECASE,
    )
    entries: list[dict[str, object]] = []
    for match in pattern.finditer(log_text):
        op_type = match.group(1).upper()
        op_name = match.group(2).strip()
        blob = match.group(3)
        item: dict[str, object] = {
            "op_type": op_type,
            "op_name": op_name,
            "warning": (" ".join(blob.split())[:400] if blob else ""),
        }
        shape_match = shape_re.search(blob)
        if shape_match:
            item["ifm_shape"] = _parse_int_list(shape_match.group(1))
            item["permutation"] = _parse_int_list(shape_match.group(2))
        entries.append(item)
    return entries


def _parse_cpu_listing(log_text: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r"CPU:\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s+\(inputs\s+(.+?),\s*outputs\s+(.+?)\)",
        flags=re.IGNORECASE,
    )
    rows: list[dict[str, object]] = []
    for line in log_text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        rows.append(
            {
                "op_type": match.group(1).upper(),
                "op_name": match.group(2).strip(),
                "inputs": match.group(3).strip(),
                "outputs": match.group(4).strip(),
            }
        )
    return rows


def _classify(op_type: str, perm: list[int] | None, shape: list[int] | None, name: str) -> tuple[str, str]:
    upper = op_type.upper()
    if upper in QDQ_OP_TYPES or "QUANT" in upper:
        return "qdq", "Quantize/dequantize op placed on CPU."
    if upper == "TRANSPOSE":
        perm_t = tuple(perm or [])
        if perm_t in LAYOUT_PERMS and shape and len(shape) == 4 and all(d > 1 for d in shape[1:4]):
            return (
                "layout_glue",
                "NCHW<->NHWC conversion glue (unsupported Ethos-U permutation).",
            )
        if perm_t == (0, 1, 3, 2):
            return "inherent", "Head/axis swap (not a pure layout-glue permutation)."
        return "inherent", "Transpose permutation is not a known removable layout-glue swap."
    if any(hint in upper for hint in UNSUPPORTED_HINTS):
        return "unsupported_op", f"{upper} is not an Ethos-U layout-glue issue."
    lname = name.lower()
    if "quant" in lname:
        return "qdq", "Op name suggests QDQ wrapping."
    return "unsupported_op", f"{upper} fell back to CPU (not classified as layout glue)."


def _onnx_inventory(onnx_path: Path | None) -> dict[str, object]:
    if onnx_path is None or not onnx_path.exists():
        return {}
    import onnx

    model = onnx.load(str(onnx_path))
    producers = {out: node for node in model.graph.node for out in node.output if out}
    consumers: dict[str, list] = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    transposes = []
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        perm = []
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        pred = producers.get(node.input[0]).op_type if node.input and node.input[0] in producers else "graph_input"
        succs = [n.op_type for n in consumers.get(node.output[0], [])]
        transposes.append(
            {
                "name": node.name or node.output[0],
                "perm": perm,
                "producer": pred,
                "consumers": succs,
            }
        )
    return {
        "onnx_node_count": len(model.graph.node),
        "onnx_transpose_count": len(transposes),
        "onnx_transposes": transposes,
    }


def _to_markdown(report: dict[str, object]) -> str:
    counts = report.get("classification_counts", {})
    lines = [
        "# CPU fallback analysis",
        "",
        f"- source_log: `{report.get('vela_log_path', '')}`",
        f"- cpu_ops: `{report.get('cpu_operator_count')}`  npu_ops: `{report.get('npu_operator_count')}`",
        f"- counts: {json.dumps(counts)}",
        "",
        "| classification | op_type | permutation | ifm_shape | neighbors | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("ops", []):
        lines.append(
            "| {classif} | {op_type} | {perm} | {shape} | {neigh} | {reason} |".format(
                classif=row.get("classification"),
                op_type=row.get("op_type"),
                perm=row.get("permutation") or "",
                shape=row.get("ifm_shape") or row.get("inputs") or "",
                neigh=row.get("neighbors") or "",
                reason=str(row.get("reason", "")).replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify Vela CPU fallback ops")
    parser.add_argument("--vela-log", default="", help="vela_primary_stdout_stderr.log")
    parser.add_argument("--vela-report", default="", help="vela_compile_report.json")
    parser.add_argument("--onnx", default="", help="Optional ONNX for transpose neighbor inventory")
    parser.add_argument("--output-json", default="", help="JSON report path")
    parser.add_argument("--output-md", default="", help="Markdown report path")
    parser.add_argument("--output-dir", default="", help="Directory for both reports")
    args = parser.parse_args()

    log_path = Path(args.vela_log).resolve() if args.vela_log else None
    report_path = Path(args.vela_report).resolve() if args.vela_report else None
    if report_path and report_path.exists() and log_path is None:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        log_path = Path(str(payload.get("vela_log_path") or payload.get("primary_log_path") or ""))

    if log_path is None or not log_path.exists():
        raise RuntimeError("Need --vela-log or a --vela-report that points at a log file.")

    log_text = _read_text(log_path)
    warnings = _parse_warnings(log_text)
    listing = _parse_cpu_listing(log_text)
    warn_by_name = {str(w["op_name"]): w for w in warnings}

    ops: list[dict[str, object]] = []
    seen: set[str] = set()
    sequence = listing or [
        {"op_type": w.get("op_type", "UNKNOWN"), "op_name": w["op_name"], "inputs": "", "outputs": ""}
        for w in warnings
    ]
    for idx, row in enumerate(sequence):
        name = str(row["op_name"])
        if name in seen:
            continue
        seen.add(name)
        extra = warn_by_name.get(name, {})
        perm = extra.get("permutation")
        shape = extra.get("ifm_shape")
        classification, reason = _classify(str(row["op_type"]), perm if isinstance(perm, list) else None, shape if isinstance(shape, list) else None, name)
        prev_name = sequence[idx - 1]["op_name"] if idx else None
        next_name = sequence[idx + 1]["op_name"] if idx + 1 < len(sequence) else None
        ops.append(
            {
                "op_type": row["op_type"],
                "op_name": name,
                "ifm_shape": shape,
                "permutation": perm,
                "inputs": row.get("inputs"),
                "outputs": row.get("outputs"),
                "neighbors": {"prev": prev_name, "next": next_name},
                "classification": classification,
                "reason": reason,
                "warning": extra.get("warning", ""),
            }
        )

    cpu_match = re.search(r"CPU operators\s*=\s*(\d+)\s*\(([\d\.]+)%\)", log_text)
    npu_match = re.search(r"NPU operators\s*=\s*(\d+)\s*\(([\d\.]+)%\)", log_text)
    counts = dict(Counter(str(op["classification"]) for op in ops))
    report: dict[str, object] = {
        "vela_log_path": str(log_path),
        "vela_report_path": str(report_path) if report_path else None,
        "cpu_operator_count": int(cpu_match.group(1)) if cpu_match else len(ops),
        "cpu_operator_ratio_percent": float(cpu_match.group(2)) if cpu_match else None,
        "npu_operator_count": int(npu_match.group(1)) if npu_match else None,
        "npu_operator_ratio_percent": float(npu_match.group(2)) if npu_match else None,
        "classification_counts": counts,
        "ops": ops,
        "onnx_inventory": _onnx_inventory(Path(args.onnx).resolve() if args.onnx else None),
        "playbook": "references/layout_fallback_playbook.md",
    }

    out_dir = Path(args.output_dir).resolve() if args.output_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output_json).resolve() if args.output_json else out_dir / "cpu_fallback_analysis.json"
    md_path = Path(args.output_md).resolve() if args.output_md else out_dir / "cpu_fallback_analysis.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"[OK] {json_path}")
    print(f"[OK] {md_path}")
    print(f"[INFO] classification_counts={counts}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
