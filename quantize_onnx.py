"""
quantize_onnx.py — INT8 dynamic quantization of exported CLIP ONNX (W9)
=======================================================================
Applies onnxruntime.quantization.quantize_dynamic (weight-only INT8) to the
FP32 encoders from export_onnx.py.

Primary deliverable (W9 "3.1x speedup"):
    onnx/clip_int8.onnx          <- quantized text encoder (query-time path)

Also produced for Image->Caption / reverse-search:
    onnx/clip_visual_int8.onnx   <- quantized visual encoder

Usage:
    python export_onnx.py      # once, if FP32 models are missing
    python quantize_onnx.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

OUT_DIR = "onnx"
TEXT_FP32 = os.path.join(OUT_DIR, "clip_text.onnx")
VISUAL_FP32 = os.path.join(OUT_DIR, "clip_visual.onnx")
TEXT_PRE = os.path.join(OUT_DIR, "clip_text_pre.onnx")
VISUAL_PRE = os.path.join(OUT_DIR, "clip_visual_pre.onnx")
TEXT_INT8 = os.path.join(OUT_DIR, "clip_int8.onnx")          # W9 named deliverable
VISUAL_INT8 = os.path.join(OUT_DIR, "clip_visual_int8.onnx")

# Weight-only INT8 on transformers is lossy; require usable embedding fidelity
# (real CLIP-tokenized prompts), not FP32 abs-error.
MIN_COSINE_TEXT = 0.85
MIN_COSINE_VISUAL = 0.98

SAMPLE_TEXTS = [
    "a dog on the beach",
    "a red car parked outside",
    "two people walking in the snow",
    "sunset over mountains",
]


def _mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def _require_fp32(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing '{path}'. Run `python export_onnx.py` first."
        )


def _pre_process(model_input: str, model_output: str) -> str:
    """ORT-recommended shape inference / optimization before quantize_dynamic."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    print(f"[INT8] Pre-processing {model_input} ...")
    quant_pre_process(
        input_model=model_input,
        output_model_path=model_output,
        skip_symbolic_shape=True,  # dynamic batch axes — skip symbolic infer
    )
    return model_output


def quantize_model(model_input: str, model_output: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[INT8] Quantizing {model_input} -> {model_output} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_dynamic(
            model_input=model_input,
            model_output=model_output,
            weight_type=QuantType.QInt8,  # weight-only INT8
            per_channel=False,
            reduce_range=False,
        )
    src_mb = _mb(model_input) if "pre.onnx" not in model_input else None
    # Report compression vs original FP32 sibling when input is a *_pre.onnx
    fp32_sibling = model_input.replace("_pre.onnx", ".onnx")
    if os.path.isfile(fp32_sibling):
        src_mb = _mb(fp32_sibling)
    print(
        f"[INT8] Wrote {model_output} "
        f"({src_mb:.1f} MB FP32 -> {_mb(model_output):.1f} MB INT8, "
        f"{_mb(model_output) / src_mb:.0%} size)"
    )


def _ort_session(path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den == 0, 1e-12, den)
    return num / den


def verify_text(fp32_path: str, int8_path: str) -> float:
    """Compare FP32 vs INT8 on real CLIP-tokenized prompts; return min cosine."""
    import open_clip

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(SAMPLE_TEXTS).numpy().astype(np.int64)

    fp32 = _ort_session(fp32_path)
    int8 = _ort_session(int8_path)
    out_fp32 = fp32.run(None, {"tokens": tokens})[0]
    out_int8 = int8.run(None, {"tokens": tokens})[0]
    cos = _cosine_rows(out_fp32, out_int8)
    print(
        f"[VERIFY] text   min cosine={float(cos.min()):.4f}  "
        f"avg={float(cos.mean()):.4f}  (n={len(SAMPLE_TEXTS)} prompts)"
    )
    return float(cos.min())


def verify_visual(fp32_path: str, int8_path: str) -> float:
    fp32 = _ort_session(fp32_path)
    int8 = _ort_session(int8_path)
    rng = np.random.default_rng(0)
    images = rng.standard_normal((2, 3, 224, 224), dtype=np.float32)
    out_fp32 = fp32.run(None, {"images": images})[0]
    out_int8 = int8.run(None, {"images": images})[0]
    cos = _cosine_rows(out_fp32, out_int8)
    print(
        f"[VERIFY] visual min cosine={float(cos.min()):.4f}  "
        f"avg={float(cos.mean()):.4f}"
    )
    return float(cos.min())


def _cleanup_preprocessed() -> None:
    for path in (TEXT_PRE, VISUAL_PRE):
        if os.path.isfile(path):
            os.remove(path)


def main() -> int:
    try:
        from onnxruntime.quantization import quantize_dynamic  # noqa: F401
    except ImportError:
        print("[ERROR] onnxruntime is not installed. Run:")
        print("        pip install 'onnxruntime>=1.18.1'")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        _require_fp32(TEXT_FP32)
        _require_fp32(VISUAL_FP32)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    text_pre = _pre_process(TEXT_FP32, TEXT_PRE)
    visual_pre = _pre_process(VISUAL_FP32, VISUAL_PRE)

    # W9 deliverable: clip_int8.onnx = quantized text encoder (query latency path)
    quantize_model(text_pre, TEXT_INT8)
    quantize_model(visual_pre, VISUAL_INT8)
    _cleanup_preprocessed()

    print("\n[INT8] Verifying with onnxruntime.InferenceSession ...")
    text_cos = verify_text(TEXT_FP32, TEXT_INT8)
    vis_cos = verify_visual(VISUAL_FP32, VISUAL_INT8)

    print()
    ok_text = text_cos >= MIN_COSINE_TEXT
    ok_vis = vis_cos >= MIN_COSINE_VISUAL
    if ok_text and ok_vis:
        print(
            f"[OK] INT8 fidelity OK "
            f"(text cosine>={MIN_COSINE_TEXT:g}, visual>={MIN_COSINE_VISUAL:g})"
        )
        print(f"[OK] W9 deliverable ready: {TEXT_INT8}")
        return 0

    print(
        f"[FAIL] Fidelity below threshold "
        f"(text={text_cos:.4f} need>={MIN_COSINE_TEXT:g}, "
        f"visual={vis_cos:.4f} need>={MIN_COSINE_VISUAL:g})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
