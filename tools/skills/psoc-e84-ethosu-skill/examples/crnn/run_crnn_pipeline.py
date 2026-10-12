#!/usr/bin/env python3
"""
Run an end-to-end CRNN pipeline for the PSoC E84 Ethos-U skill:
PyTorch -> ONNX -> (strict int8 attempt + relaxed fallback) -> Vela -> accuracy reports.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont


CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
LABEL2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("-" * 72)
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return completed


def _ensure_crnn_repo(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url, str(repo_dir)], cwd=repo_dir.parent)


def _random_text(rng: random.Random, min_len: int, max_len: int) -> str:
    text_len = rng.randint(min_len, max_len)
    return "".join(rng.choice(CHARSET) for _ in range(text_len))


def _make_synthetic_line(text: str, width: int, height: int) -> Image.Image:
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((4, max(0, (height - 10) // 2)), text, fill=0, font=font)
    return image


def _generate_synthetic_samples(output_dir: Path, count: int, width: int, height: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[Path, str]] = []
    for idx in range(count):
        text = _random_text(rng, 4, 10)
        img = _make_synthetic_line(text, width, height)
        file_path = output_dir / f"{idx:04d}_{text}.png"
        img.save(file_path)
        samples.append((file_path, text))
    return samples


def _image_to_crnn_input(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).reshape(1, height, width)
    arr = (arr / 127.5) - 1.0
    return arr


def _save_representative_data(samples: list[tuple[Path, str]], out_npy: Path, width: int, height: int) -> np.ndarray:
    stacked = np.stack([_image_to_crnn_input(path, width, height) for path, _ in samples], axis=0).astype(np.float32)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, stacked)
    return stacked


def _load_crnn_model(repo_dir: Path, checkpoint: Path):
    src_dir = repo_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import common_config  # type: ignore
    from model import CRNN  # type: ignore

    num_class = len(CHARSET) + 1
    model = CRNN(
        1,
        common_config["img_height"],
        common_config["img_width"],
        num_class,
        map_to_seq_hidden=common_config["map_to_seq_hidden"],
        rnn_hidden=common_config["rnn_hidden"],
        leaky_relu=common_config["leaky_relu"],
    )
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    else:
        raise RuntimeError(f"CRNN checkpoint not found: {checkpoint}")
    model.eval()
    return model, common_config


def _export_to_onnx(model, width: int, height: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, height, width, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=13,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )


def _ctc_greedy_decode(logits: np.ndarray) -> str:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D logits, got {arr.shape}")
    if arr.shape[1] == 1:
        seq = arr[:, 0, :]
    elif arr.shape[0] == 1:
        seq = arr[0, :, :]
    else:
        seq = arr[:, 0, :]
    labels = np.argmax(seq, axis=-1).tolist()
    decoded: list[int] = []
    prev = None
    for label in labels:
        if label != prev:
            decoded.append(int(label))
            prev = label
    decoded = [d for d in decoded if d != 0]
    return "".join(LABEL2CHAR.get(i, "") for i in decoded)


def _edit_distance(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _cer(preds: list[str], refs: list[str]) -> float:
    total_dist = 0
    total_chars = 0
    for pred, ref in zip(preds, refs):
        total_dist += _edit_distance(pred, ref)
        total_chars += max(1, len(ref))
    return float(total_dist / total_chars) if total_chars else 0.0


def _sequence_accuracy(preds: list[str], refs: list[str]) -> float:
    if not refs:
        return 0.0
    correct = sum(int(p == r) for p, r in zip(preds, refs))
    return float(correct / len(refs))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _run_tflite(interpreter, x_nchw: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    x = np.transpose(x_nchw, (0, 2, 3, 1))
    if input_detail["dtype"] == np.int8:
        scale, zero = input_detail["quantization"]
        if scale == 0:
            raise RuntimeError("TFLite input scale is zero")
        x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
    else:
        x = x.astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail["index"])
    if output_detail["dtype"] == np.int8:
        s, z = output_detail["quantization"]
        y = (y.astype(np.float32) - z) * s
    return np.asarray(y, dtype=np.float32)


def _evaluate_predictions(
    model,
    onnx_path: Path,
    tflite_path: Path,
    eval_samples: list[tuple[Path, str]],
    width: int,
    height: int,
) -> tuple[dict[str, object], dict[str, object]]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_name = sess.get_inputs()[0].name

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    gt_texts: list[str] = []
    pt_preds: list[str] = []
    onnx_preds: list[str] = []
    tfl_preds: list[str] = []
    cosines: list[float] = []
    max_abs: list[float] = []
    maes: list[float] = []

    for image_path, gt in eval_samples:
        x = _image_to_crnn_input(image_path, width, height)[None, ...]
        gt_texts.append(gt)

        with torch.no_grad():
            pt_logits = model(torch.from_numpy(x).float()).cpu().numpy()
        onnx_logits = np.asarray(sess.run(None, {onnx_input_name: x.astype(np.float32)})[0], dtype=np.float32)
        tfl_logits = _run_tflite(interpreter, x)

        pt_preds.append(_ctc_greedy_decode(pt_logits))
        onnx_preds.append(_ctc_greedy_decode(onnx_logits))
        tfl_preds.append(_ctc_greedy_decode(tfl_logits))

        diff = tfl_logits - onnx_logits
        cosines.append(_cosine(tfl_logits, onnx_logits))
        max_abs.append(float(np.max(np.abs(diff))))
        maes.append(float(np.mean(np.abs(diff))))

    accuracy_report = {
        "num_samples": len(eval_samples),
        "ground_truth_source": "synthetic text lines generated in-skill",
        "sequence_accuracy": {
            "pytorch": _sequence_accuracy(pt_preds, gt_texts),
            "onnx": _sequence_accuracy(onnx_preds, gt_texts),
            "tflite_relaxed": _sequence_accuracy(tfl_preds, gt_texts),
        },
        "cer": {
            "pytorch": _cer(pt_preds, gt_texts),
            "onnx": _cer(onnx_preds, gt_texts),
            "tflite_relaxed": _cer(tfl_preds, gt_texts),
        },
        "tflite_vs_onnx_output_diff": {
            "cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
            "cosine_min": float(np.min(cosines)) if cosines else 0.0,
            "max_abs_mean": float(np.mean(max_abs)) if max_abs else 0.0,
            "max_abs_max": float(np.max(max_abs)) if max_abs else 0.0,
            "mae_mean": float(np.mean(maes)) if maes else 0.0,
        },
        "samples": [
            {"gt": g, "pt": p, "onnx": o, "tfl": t}
            for g, p, o, t in zip(gt_texts[:10], pt_preds[:10], onnx_preds[:10], tfl_preds[:10])
        ],
    }

    vs_pt_report = {
        "num_samples": len(eval_samples),
        "onnx_vs_pytorch": {
            "sequence_agreement": _sequence_accuracy(onnx_preds, pt_preds),
            "cer": _cer(onnx_preds, pt_preds),
        },
        "tflite_vs_pytorch": {
            "sequence_agreement": _sequence_accuracy(tfl_preds, pt_preds),
            "cer": _cer(tfl_preds, pt_preds),
        },
    }
    return accuracy_report, vs_pt_report


def main() -> int:
    parser = argparse.ArgumentParser(description="CRNN Ethos-U staged pipeline")
    parser.add_argument("--work-dir", default="tools/skills/psoc-e84-ethosu-skill/build/crnn")
    parser.add_argument(
        "--crnn-repo",
        default="tools/skills/psoc-e84-ethosu-skill/build/crnn-pytorch",
        help="Path to cloned CRNN repo",
    )
    parser.add_argument("--repo-url", default="https://github.com/rockrxf78/crnn-pytorch")
    parser.add_argument("--target", default="psoc_e84")
    parser.add_argument("--calib-samples", type=int, default=160)
    parser.add_argument("--eval-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    skill_root = script_path.parents[2]
    scripts_dir = skill_root / "scripts"
    repo_root = skill_root.parents[2]

    work_dir = Path(args.work_dir).resolve()
    crnn_repo = Path(args.crnn_repo).resolve()
    _ensure_crnn_repo(crnn_repo, args.repo_url)

    checkpoint = crnn_repo / "checkpoints" / "crnn_synth90k.pt"
    model, cfg = _load_crnn_model(crnn_repo, checkpoint)
    width = int(cfg["img_width"])
    height = int(cfg["img_height"])

    calib_samples = _generate_synthetic_samples(work_dir / "calib_images", args.calib_samples, width, height, args.seed)
    eval_samples = calib_samples[: args.eval_samples]
    rep_data_path = work_dir / "crnn_rep_data.npy"
    _save_representative_data(calib_samples, rep_data_path, width, height)

    onnx_path = work_dir / "crnn_synth90k.onnx"
    _export_to_onnx(model, width, height, onnx_path)

    _run(
        [
            sys.executable,
            str(scripts_dir / "onnx_sanity_check.py"),
            "--onnx",
            str(onnx_path),
            "--json-report",
            str(work_dir / "crnn_onnx_check_report.json"),
        ],
        cwd=repo_root,
    )

    strict_tflite_out = work_dir / "crnn_synth90k_int8.tflite"
    strict_workdir = work_dir / "onnx2tf_out_strict"
    strict_cmd = [
        sys.executable,
        str(scripts_dir / "onnx_to_tflite_int8.py"),
        "--onnx",
        str(onnx_path),
        "--work-dir",
        str(strict_workdir),
        "--tflite-out",
        str(strict_tflite_out),
        "--representative-data",
        str(rep_data_path),
        "--input-layout",
        "nchw",
        "--max-representative-samples",
        str(args.calib_samples),
    ]
    strict_res = _run(strict_cmd, cwd=repo_root, check=False)
    strict_failure = {
        "command": strict_cmd,
        "returncode": strict_res.returncode,
        "stdout_tail": strict_res.stdout[-4000:],
        "stderr_tail": strict_res.stderr[-4000:],
        "strict_full_int8_succeeded": strict_res.returncode == 0,
    }
    (work_dir / "crnn_strict_int8_attempt.json").write_text(json.dumps(strict_failure, indent=2), encoding="utf-8")

    relaxed_tflite_out = work_dir / "crnn_synth90k_relaxed.tflite"
    relaxed_workdir = work_dir / "onnx2tf_out_relaxed"
    relaxed_cmd = [
        sys.executable,
        str(scripts_dir / "onnx_to_tflite_int8.py"),
        "--onnx",
        str(onnx_path),
        "--work-dir",
        str(relaxed_workdir),
        "--tflite-out",
        str(relaxed_tflite_out),
        "--representative-data",
        str(rep_data_path),
        "--input-layout",
        "nchw",
        "--max-representative-samples",
        str(args.calib_samples),
        "--allow-non-int8-output",
    ]
    relaxed_res = _run(relaxed_cmd, cwd=repo_root, check=False)
    relaxed_attempt = {
        "command": relaxed_cmd,
        "returncode": relaxed_res.returncode,
        "stdout_tail": relaxed_res.stdout[-4000:],
        "stderr_tail": relaxed_res.stderr[-4000:],
        "relaxed_conversion_succeeded": relaxed_res.returncode == 0,
        "used_existing_relaxed_tflite": False,
    }
    if relaxed_res.returncode != 0:
        if relaxed_tflite_out.exists():
            relaxed_attempt["used_existing_relaxed_tflite"] = True
        else:
            (work_dir / "crnn_relaxed_int8_attempt.json").write_text(
                json.dumps(relaxed_attempt, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                "Relaxed ONNX->TFLite conversion failed and no prior relaxed artifact is available."
            )
    (work_dir / "crnn_relaxed_int8_attempt.json").write_text(
        json.dumps(relaxed_attempt, indent=2), encoding="utf-8"
    )

    _run(
        [
            sys.executable,
            str(scripts_dir / "compile_with_vela.py"),
            "--tflite",
            str(relaxed_tflite_out),
            "--target",
            args.target,
            "--output-dir",
            str(work_dir / "vela-relaxed"),
        ],
        cwd=repo_root,
    )

    accuracy_report, vs_pt_report = _evaluate_predictions(
        model=model,
        onnx_path=onnx_path,
        tflite_path=relaxed_tflite_out,
        eval_samples=eval_samples,
        width=width,
        height=height,
    )
    (work_dir / "crnn_accuracy_report.json").write_text(json.dumps(accuracy_report, indent=2), encoding="utf-8")
    (work_dir / "crnn_vs_pytorch_report.json").write_text(json.dumps(vs_pt_report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("CRNN pipeline completed")
    print("=" * 72)
    print(f"onnx: {onnx_path}")
    print(f"strict_int8_attempt: {work_dir / 'crnn_strict_int8_attempt.json'}")
    print(f"relaxed_tflite: {relaxed_tflite_out}")
    print(f"vela_report: {work_dir / 'vela-relaxed' / 'vela_compile_report.json'}")
    print(f"accuracy_report: {work_dir / 'crnn_accuracy_report.json'}")
    print(f"vs_pytorch_report: {work_dir / 'crnn_vs_pytorch_report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
