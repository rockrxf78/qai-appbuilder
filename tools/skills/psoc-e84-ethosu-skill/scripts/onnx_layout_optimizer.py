#!/usr/bin/env python3
"""ONNX layout cleanup before TF conversion: cancel/fold/push Transpose, drop identity QDQ."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

UNARY_OPS = {
    "Relu",
    "Sigmoid",
    "Tanh",
    "Clip",
    "Exp",
    "Log",
    "Sqrt",
    "Neg",
    "Abs",
    "HardSigmoid",
    "LeakyRelu",
    "HardSwish",
    "Celu",
    "Elu",
}
ELEMENTWISE_BINARY = {"Add", "Mul", "Sub", "Div"}


def _perm(node) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == "perm":
            return list(attr.ints)
    return None


def _identity(perm: list[int]) -> bool:
    return perm == list(range(len(perm)))


def _compose(first: list[int], second: list[int]) -> list[int] | None:
    if len(first) != len(second):
        return None
    try:
        return [first[i] for i in second]
    except IndexError:
        return None


def _maps(graph):
    producers = {}
    consumers: dict[str, list] = defaultdict(list)
    for node in graph.node:
        for out in node.output:
            if out:
                producers[out] = node
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    return producers, consumers


def _initializer_names(graph) -> set[str]:
    return {t.name for t in graph.initializer}


def _is_scalar_or_const(name: str, graph, producers: dict) -> bool:
    if name in _initializer_names(graph):
        for tensor in graph.initializer:
            if tensor.name == name:
                dims = list(tensor.dims) if tensor.dims else []
                return (not dims) or all(d == 1 for d in dims)
        return True
    node = producers.get(name)
    return node is not None and node.op_type == "Constant"


def _rewire_consumers(graph, old: str, new: str) -> None:
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new
    for output in graph.output:
        if output.name == old:
            output.name = new


def _remove_nodes(graph, victims) -> int:
    victim_ids = {id(n) for n in victims}
    keep = [n for n in graph.node if id(n) not in victim_ids]
    removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    return removed


def _pass_drop_identity_qdq(graph) -> list[str]:
    producers, consumers = _maps(graph)
    changes: list[str] = []
    victims = []
    for node in list(graph.node):
        if node.op_type != "QuantizeLinear":
            continue
        outs = consumers.get(node.output[0], [])
        if len(outs) != 1 or outs[0].op_type != "DequantizeLinear":
            continue
        dq = outs[0]
        if node.input[1:] == dq.input[1:] or (
            len(node.input) > 1 and len(dq.input) > 1 and node.input[1] == dq.input[1]
        ):
            _rewire_consumers(graph, dq.output[0], node.input[0])
            victims.extend([node, dq])
            changes.append(f"drop identity QDQ around {node.output[0]}")
    _remove_nodes(graph, set(victims))
    return changes


def _pass_drop_identity_transpose(graph) -> list[str]:
    changes: list[str] = []
    victims = []
    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue
        perm = _perm(node)
        if perm and _identity(perm):
            _rewire_consumers(graph, node.output[0], node.input[0])
            victims.append(node)
            changes.append(f"drop identity Transpose {node.name or node.output[0]}")
    _remove_nodes(graph, set(victims))
    return changes


def _pass_cancel_inverse_transposes(graph) -> list[str]:
    producers, consumers = _maps(graph)
    changes: list[str] = []
    victims = []
    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue
        outs = consumers.get(node.output[0], [])
        if len(outs) != 1 or outs[0].op_type != "Transpose":
            continue
        nxt = outs[0]
        p1, p2 = _perm(node), _perm(nxt)
        if not p1 or not p2:
            continue
        composed = _compose(p1, p2)
        if composed and _identity(composed):
            _rewire_consumers(graph, nxt.output[0], node.input[0])
            victims.extend([node, nxt])
            changes.append(
                f"cancel inverse Transpose {node.name or node.output[0]} + {nxt.name or nxt.output[0]}"
            )
    _remove_nodes(graph, set(victims))
    return changes


def _pass_push_through_unary(graph) -> list[str]:
    _producers, consumers = _maps(graph)
    changes: list[str] = []
    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue
        outs = consumers.get(node.output[0], [])
        if len(outs) != 1 or outs[0].op_type not in UNARY_OPS:
            continue
        unary = outs[0]
        after = list(consumers.get(unary.output[0], []))
        data_in = node.input[0]
        t_out = node.output[0]
        u_out = unary.output[0]
        unary.input[0] = data_in
        node.input[0] = u_out
        for consumer in after:
            for i, inp in enumerate(consumer.input):
                if inp == u_out:
                    consumer.input[i] = t_out
        for output in graph.output:
            if output.name == u_out:
                output.name = t_out
        changes.append(f"push Transpose through {unary.op_type} ({unary.name or u_out})")
    return changes


def _pass_push_through_binary_scalar(graph) -> list[str]:
    producers, consumers = _maps(graph)
    changes: list[str] = []
    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue
        outs = consumers.get(node.output[0], [])
        if len(outs) != 1 or outs[0].op_type not in ELEMENTWISE_BINARY:
            continue
        binary = outs[0]
        if len(binary.input) != 2:
            continue
        other = binary.input[1] if binary.input[0] == node.output[0] else binary.input[0]
        if not _is_scalar_or_const(other, graph, producers):
            continue
        # Transpose -> Add(scalar) => Add(scalar) -> Transpose
        data_in = node.input[0]
        t_out = node.output[0]
        b_out = binary.output[0]
        after = consumers.get(b_out, [])
        if binary.input[0] == t_out:
            binary.input[0] = data_in
        else:
            binary.input[1] = data_in
        node.input[0] = b_out
        for consumer in after:
            for i, inp in enumerate(consumer.input):
                if inp == b_out:
                    consumer.input[i] = t_out
        for output in graph.output:
            if output.name == b_out:
                output.name = t_out
        changes.append(f"push Transpose through scalar {binary.op_type}")
    return changes


def _pass_fold_transpose_reshape(graph) -> list[str]:
    """If Transpose only reorders size-1 axes, replace with Reshape of the same output shape."""
    from onnx import helper, numpy_helper

    changes: list[str] = []
    shape_lookup = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        dims = []
        if vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                dims.append(int(dim.dim_value) if dim.dim_value else -1)
            shape_lookup[vi.name] = dims

    rebuilt = []
    for node in list(graph.node):
        if node.op_type != "Transpose":
            rebuilt.append(node)
            continue
        perm = _perm(node)
        if not perm:
            rebuilt.append(node)
            continue
        in_shape = shape_lookup.get(node.input[0])
        if not in_shape or any(d < 0 for d in in_shape) or len(in_shape) != len(perm):
            rebuilt.append(node)
            continue
        nontrivial = [
            axis
            for axis, src in enumerate(perm)
            if src != axis and in_shape[perm[axis]] != 1 and in_shape[axis] != 1
        ]
        if nontrivial:
            rebuilt.append(node)
            continue
        out_shape = [in_shape[i] for i in perm]
        shape_name = node.output[0] + "_fold_shape"
        graph.initializer.extend([numpy_helper.from_array(np.asarray(out_shape, dtype=np.int64), shape_name)])
        rebuilt.append(
            helper.make_node(
                "Reshape",
                inputs=[node.input[0], shape_name],
                outputs=[node.output[0]],
                name=(node.name or node.output[0]) + "_as_reshape",
            )
        )
        changes.append(f"fold size-1 Transpose {node.name or node.output[0]} into Reshape {out_shape}")
    if changes:
        del graph.node[:]
        graph.node.extend(rebuilt)
    return changes


def _count_op(graph, op_type: str) -> int:
    return sum(1 for n in graph.node if n.op_type == op_type)


def _infer_feed(session, sample: np.ndarray | None) -> dict[str, np.ndarray]:
    inp = session.get_inputs()[0]
    shape = [int(dim) if isinstance(dim, int) and dim > 0 else 1 for dim in (inp.shape or [])]
    if sample is not None:
        arr = np.asarray(sample, dtype=np.float32)
        if tuple(arr.shape) == tuple(shape):
            return {inp.name: arr}
        if arr.ndim == len(shape) and shape and shape[0] == 1 and arr.shape[0] >= 1:
            return {inp.name: arr[:1]}
    rng = np.random.default_rng(0)
    return {inp.name: rng.random(shape or [1], dtype=np.float32)}


def _max_abs_diff(onnx_a: Path, onnx_b: Path, sample: np.ndarray | None) -> float:
    import onnxruntime as ort

    sess_a = ort.InferenceSession(str(onnx_a), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(onnx_b), providers=["CPUExecutionProvider"])
    feed = _infer_feed(sess_a, sample)
    # Align name if needed
    name_b = sess_b.get_inputs()[0].name
    feed_b = {name_b: next(iter(feed.values()))}
    out_a = sess_a.run(None, feed)
    out_b = sess_b.run(None, feed_b)
    diffs = []
    for a, b in zip(out_a, out_b):
        aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape:
            diffs.append(float("inf"))
        else:
            diffs.append(float(np.max(np.abs(aa - bb))))
    return max(diffs) if diffs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize ONNX layout ops before conversion")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--sample-input", default="", help="Optional .npy for ORT numeric check")
    parser.add_argument("--rtol-max-abs", type=float, default=1e-4)
    args = parser.parse_args()

    import onnx
    from onnx import checker

    src = Path(args.onnx).resolve()
    dst = Path(args.output_onnx).resolve()
    model = onnx.load(str(src))
    graph = model.graph
    before = {"nodes": len(graph.node), "transpose": _count_op(graph, "Transpose"), "qdq": _count_op(graph, "QuantizeLinear") + _count_op(graph, "DequantizeLinear")}

    changes: list[str] = []
    for _ in range(8):
        step = []
        step += _pass_drop_identity_qdq(graph)
        step += _pass_drop_identity_transpose(graph)
        step += _pass_cancel_inverse_transposes(graph)
        step += _pass_push_through_unary(graph)
        step += _pass_push_through_binary_scalar(graph)
        step += _pass_fold_transpose_reshape(graph)
        step += _pass_cancel_inverse_transposes(graph)
        if not step:
            break
        changes.extend(step)

    after = {"nodes": len(graph.node), "transpose": _count_op(graph, "Transpose"), "qdq": _count_op(graph, "QuantizeLinear") + _count_op(graph, "DequantizeLinear")}
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        checker.check_model(model, full_check=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"optimized ONNX failed checker: {exc}") from exc
    onnx.save(model, str(dst))

    sample = None
    if args.sample_input:
        sample = np.load(args.sample_input)
    max_abs = _max_abs_diff(src, dst, sample)
    ok = max_abs <= args.rtol_max_abs or max_abs == 0.0
    report = {
        "input_onnx": str(src),
        "output_onnx": str(dst),
        "before": before,
        "after": after,
        "changes": changes,
        "max_abs_diff": max_abs,
        "numeric_ok": ok,
        "tolerance": args.rtol_max_abs,
    }
    report_path = Path(args.report_json).resolve() if args.report_json else dst.with_suffix(".layout_opt.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {dst}")
    print(f"[OK] report {report_path}")
    print(f"[INFO] transpose {before['transpose']} -> {after['transpose']}; max_abs={max_abs} numeric_ok={ok}")
    if not ok:
        print("[WARN] numeric check exceeded tolerance; inspect report before using optimized ONNX.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
