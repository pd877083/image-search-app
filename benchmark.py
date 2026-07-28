"""
benchmark.py — Measured query latency (Master WPR)
==================================================
Times 50 text queries under:
  (a) PyTorch FP32
  (b) PyTorch FP16   (skipped on CPU if half() is unavailable / slower-only)
  (c) ONNX Runtime   (clip_text.onnx, CPUExecutionProvider)
  (d) ONNX INT8      (clip_int8.onnx)

Writes:
  embeddings/latency.json   — table of mean / median / p95 ms + speedups
  embeddings/latency_chart.png — bar chart of mean encode latency

Usage:
    python export_onnx.py      # if onnx/*.onnx missing
    python quantize_onnx.py    # if onnx/clip_int8.onnx missing
    python benchmark.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

N_QUERIES = 50
N_WARMUP = 5
TOP_K = 5

ONNX_DIR = "onnx"
TEXT_ONNX = os.path.join(ONNX_DIR, "clip_text.onnx")
TEXT_INT8 = os.path.join(ONNX_DIR, "clip_int8.onnx")
LIGHT_WEIGHTS = "light_clip_heads.pt"

EMBEDDINGS_DIR = "embeddings"
IMAGE_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npz")
OUT_JSON = os.path.join(EMBEDDINGS_DIR, "latency.json")
OUT_CHART = os.path.join(EMBEDDINGS_DIR, "latency_chart.png")

# Diverse Flickr8k-style prompts (cycled to reach N_QUERIES)
SEED_QUERIES = [
    "a dog on the beach",
    "children playing football",
    "a woman cooking in the kitchen",
    "snow covered mountains",
    "a man riding a bike",
    "people sitting at a table",
    "a cat sitting on a chair",
    "a baby playing outdoors",
    "a red car parked on the street",
    "two birds flying in the sky",
    "a group of people hiking",
    "sunset over the ocean",
    "a black and white dog running",
    "kids playing in a park",
    "a person skiing downhill",
    "boats on a lake",
    "a girl holding an umbrella",
    "horses in a field",
    "a man climbing a rock wall",
    "city street at night",
]


def _build_queries(n: int) -> List[str]:
    return [SEED_QUERIES[i % len(SEED_QUERIES)] for i in range(n)]


def _stats(ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(ms, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "std_ms": round(float(arr.std()), 2),
        "n": int(arr.size),
    }


def _time_calls(fn: Callable[[], None], n: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1e-10, n)
    return x / n


def _ort_session(path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Pin CPU provider — matches W9 cloud / Streamlit deploy constraint
    return ort.InferenceSession(
        path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def _maybe_inject_heads(model) -> None:
    import torch

    if not os.path.isfile(LIGHT_WEIGHTS):
        return
    state = torch.load(LIGHT_WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"[BENCH] Injected {LIGHT_WEIGHTS}")


def _load_gallery() -> Optional[np.ndarray]:
    if not os.path.isfile(IMAGE_EMBED_PATH):
        print(f"[BENCH] No {IMAGE_EMBED_PATH} — encode-only timings (no e2e search).")
        return None
    emb = np.load(IMAGE_EMBED_PATH, allow_pickle=True)["embeddings"].astype(np.float32)
    print(f"[BENCH] Gallery embeddings: {emb.shape}")
    return emb


def _search(query_emb: np.ndarray, gallery: np.ndarray) -> None:
    sims = (query_emb.reshape(1, -1) @ gallery.T).ravel()
    np.argpartition(-sims, min(TOP_K, len(sims) - 1))[:TOP_K]


# ── Backends ─────────────────────────────────────────────────────────────────

def bench_pytorch_fp32(queries: List[str], gallery: Optional[np.ndarray]) -> Dict:
    import torch
    from model import load_clip_model
    import model as model_mod

    print("\n[a] PyTorch FP32 ...")
    model, _ = load_clip_model("ViT-B/32")
    model = model.cpu().float().eval()
    _maybe_inject_heads(model)
    tok = model_mod._tokenizer

    encode_ms, e2e_ms = [], []
    # Warmup
    for q in queries[:N_WARMUP]:
        with torch.no_grad():
            _ = model.encode_text(tok([q]))

    for q in queries:
        tokens = tok([q])
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = model.encode_text(tokens)
        emb_np = _l2(emb.cpu().numpy().astype(np.float32))
        encode_ms.append((time.perf_counter() - t0) * 1000.0)

        if gallery is not None:
            t1 = time.perf_counter()
            with torch.no_grad():
                emb = model.encode_text(tokens)
            emb_np = _l2(emb.cpu().numpy().astype(np.float32))
            _search(emb_np, gallery)
            e2e_ms.append((time.perf_counter() - t1) * 1000.0)

    return {
        "backend": "pytorch_fp32",
        "label": "PyTorch FP32",
        "available": True,
        "encode": _stats(encode_ms),
        "e2e": _stats(e2e_ms) if e2e_ms else None,
    }


def bench_pytorch_fp16(queries: List[str], gallery: Optional[np.ndarray]) -> Dict:
    """FP16 is only meaningful on CUDA (tensor-core path). On CPU, half() runs
    but is *slower* than FP32 (no accelerated fp16 kernels) — reporting that
    number would be misleading and contradict the Master WPR's CUDA-only FP16
    claim. Skip cleanly on CPU with an explicit reason instead.
    """
    import torch
    from model import load_clip_model
    import model as model_mod

    print("\n[b] PyTorch FP16 ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[BENCH] No CUDA — skipping FP16 (CPU fp16 is slower than fp32, "
              "would be misleading). Run this backend on Colab T4 for the real "
              "Master-WPR FP16 number.")
        return {
            "backend": "pytorch_fp16",
            "label": "PyTorch FP16",
            "available": False,
            "skip_reason": "CUDA required for meaningful FP16 (CPU fp16 has no tensor-core path; would report a misleading slowdown). Run on Colab T4.",
            "encode": None,
            "e2e": None,
            "device": device,
        }

    model, _ = load_clip_model("ViT-B/32")
    model = model.to(device).eval()
    _maybe_inject_heads(model)
    try:
        model = model.half()
    except Exception as exc:
        return {
            "backend": "pytorch_fp16",
            "label": "PyTorch FP16",
            "available": False,
            "skip_reason": f"model.half() failed: {exc}",
            "encode": None,
            "e2e": None,
        }

    tok = model_mod._tokenizer
    encode_ms, e2e_ms = [], []

    try:
        for q in queries[:N_WARMUP]:
            with torch.no_grad():
                t = tok([q]).to(device)
                _ = model.encode_text(t)

        for q in queries:
            tokens = tok([q]).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                emb = model.encode_text(tokens)
            if device == "cuda":
                torch.cuda.synchronize()
            emb_np = _l2(emb.float().cpu().numpy().astype(np.float32))
            encode_ms.append((time.perf_counter() - t0) * 1000.0)

            if gallery is not None:
                t1 = time.perf_counter()
                with torch.no_grad():
                    emb = model.encode_text(tokens)
                if device == "cuda":
                    torch.cuda.synchronize()
                emb_np = _l2(emb.float().cpu().numpy().astype(np.float32))
                _search(emb_np, gallery)
                e2e_ms.append((time.perf_counter() - t1) * 1000.0)
    except Exception as exc:
        return {
            "backend": "pytorch_fp16",
            "label": "PyTorch FP16",
            "available": False,
            "skip_reason": str(exc),
            "encode": None,
            "e2e": None,
            "device": device,
        }

    return {
        "backend": "pytorch_fp16",
        "label": "PyTorch FP16",
        "available": True,
        "device": device,
        "encode": _stats(encode_ms),
        "e2e": _stats(e2e_ms) if e2e_ms else None,
    }


def bench_onnx(path: str, backend: str, label: str,
               queries: List[str], gallery: Optional[np.ndarray]) -> Dict:
    import open_clip

    print(f"\n[{backend}] {label} ({path}) ...")
    if not os.path.isfile(path):
        return {
            "backend": backend,
            "label": label,
            "available": False,
            "skip_reason": f"Missing {path} — run export_onnx.py / quantize_onnx.py",
            "encode": None,
            "e2e": None,
        }

    session = _ort_session(path)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    in_name = session.get_inputs()[0].name

    encode_ms, e2e_ms = [], []
    for q in queries[:N_WARMUP]:
        tokens = tokenizer([q]).numpy().astype(np.int64)
        session.run(None, {in_name: tokens})

    for q in queries:
        tokens = tokenizer([q]).numpy().astype(np.int64)
        t0 = time.perf_counter()
        out = session.run(None, {in_name: tokens})[0]
        emb = _l2(out.astype(np.float32))
        encode_ms.append((time.perf_counter() - t0) * 1000.0)

        if gallery is not None:
            t1 = time.perf_counter()
            out = session.run(None, {in_name: tokens})[0]
            emb = _l2(out.astype(np.float32))
            _search(emb, gallery)
            e2e_ms.append((time.perf_counter() - t1) * 1000.0)

    return {
        "backend": backend,
        "label": label,
        "available": True,
        "model_path": path,
        "model_mb": round(os.path.getsize(path) / 1e6, 1),
        "encode": _stats(encode_ms),
        "e2e": _stats(e2e_ms) if e2e_ms else None,
    }


# ── Chart + report ────────────────────────────────────────────────────────────

def save_chart(results: List[Dict], baseline_ms: float, path: str) -> None:
    """Bar chart via Pillow (no matplotlib / kaleido dependency)."""
    from PIL import Image, ImageDraw, ImageFont

    labels, means, colors = [], [], []
    palette = {
        "pytorch_fp32": (74, 85, 104),
        "pytorch_fp16": (113, 128, 150),
        "onnx": (43, 108, 176),
        "onnx_int8": (47, 133, 90),
    }
    for r in results:
        if not r.get("available") or not r.get("encode"):
            continue
        labels.append(r["label"])
        means.append(float(r["encode"]["mean_ms"]))
        colors.append(palette.get(r["backend"], (160, 174, 192)))

    if not labels:
        print("[BENCH] No data for chart.")
        return

    W, H = 900, 520
    pad_l, pad_r, pad_t, pad_b = 70, 30, 70, 90
    img = Image.new("RGB", (W, H), (250, 250, 252))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 12)
        font_title = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = font_sm = font_title = ImageFont.load_default()

    draw.text(
        (pad_l, 18),
        f"CLIP ViT-B/32 query latency — {N_QUERIES} prompts (measured)",
        fill=(26, 32, 44),
        font=font_title,
    )
    draw.text(
        (pad_l, 44),
        "Mean text-encode latency (ms)",
        fill=(74, 85, 104),
        font=font_sm,
    )

    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    max_ms = max(means) * 1.18
    n = len(labels)
    gap = 28
    bar_w = max(40, (plot_w - gap * (n + 1)) // n)

    # Axis line
    draw.line(
        [(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)],
        fill=(160, 174, 192),
        width=2,
    )

    for i, (label, ms, color) in enumerate(zip(labels, means, colors)):
        x0 = pad_l + gap + i * (bar_w + gap)
        bar_h = int((ms / max_ms) * plot_h)
        y0 = pad_t + plot_h - bar_h
        draw.rectangle([x0, y0, x0 + bar_w, pad_t + plot_h], fill=color)
        speedup = baseline_ms / ms if ms > 0 else 0
        caption = f"{ms:.0f} ms\n({speedup:.2f}x)"
        # Center label under bar
        draw.text(
            (x0 + 4, pad_t + plot_h + 10),
            label,
            fill=(45, 55, 72),
            font=font_sm,
        )
        draw.text(
            (x0 + 8, max(pad_t + 4, y0 - 36)),
            caption,
            fill=(26, 32, 44),
            font=font_sm,
        )

    img.save(path)
    print(f"[BENCH] Chart -> {path}")


def _merge_fp16_cuda(fp16_path: str, results: List[Dict], payload: Dict) -> None:
    """Merge a Colab-measured CUDA FP16 result into the local latency table.

    The local FP16 row was skipped (CPU fp16 is slower/misleading). This pulls
    the real CUDA number from `fp16_cuda_benchmark.json` (produced in the Colab
    notebook) and slots it into the FP16 row so the report has an honest,
    measured FP16 latency alongside the CPU numbers.
    """
    if not os.path.isfile(fp16_path):
        print(f"[BENCH] --merge-fp16: {fp16_path} not found, leaving FP16 skipped.")
        return
    with open(fp16_path, encoding="utf-8") as f:
        fp16 = json.load(f)

    fp16_stats = fp16["fp16_cuda"]
    fp32_cuda = fp16["fp32_cuda"]
    device = fp16.get("device", "CUDA")
    speedup_cuda = fp16.get("fp16_speedup_vs_fp32")  # FP16 vs FP32 *on the same CUDA device*

    # Patch the FP16 result row in place.
    for r in results:
        if r["backend"] == "pytorch_fp16":
            r["available"] = True
            r["skip_reason"] = None
            r["device"] = device
            r["encode"] = fp16_stats
            r["e2e"] = None
            r["note"] = (f"Measured on {device} (Colab T4). CPU fp16 was skipped locally "
                         f"as misleading; this is the real CUDA fp16 number.")

    # Patch the table rows + payload notes.
    for row in payload["table"]:
        if row["backend"] == "pytorch_fp16":
            row["available"] = True
            row.pop("skip_reason", None)
            row["device"] = device
            row["encode_mean_ms"] = fp16_stats["mean_ms"]
            row["encode_median_ms"] = fp16_stats["median_ms"]
            row["encode_p95_ms"] = fp16_stats["p95_ms"]
            row["speedup_vs_fp32"] = speedup
            row["measured_on"] = device
    payload["device_note"] = (
        f"FP32/ONNX/INT8 measured on local CPU; FP16 measured on {device} "
        "(Colab T4) — CUDA fp16 is the only meaningful fp16 path."
    )
    payload.setdefault("cuda_reference", {
        "fp32_cuda_mean_ms": fp32_cuda["mean_ms"],
        "fp16_cuda_mean_ms": fp16_stats["mean_ms"],
        "device": device,
    })
    print(f"[BENCH] Merged CUDA FP16: {fp16_stats['mean_ms']} ms on {device} ({speedup}x vs CUDA FP32)")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Latency benchmark for CLIP text encode.")
    ap.add_argument("--merge-fp16", metavar="JSON",
                    help="Path to fp16_cuda_benchmark.json (from Colab) to merge into the FP16 row.")
    args = ap.parse_args()

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    queries = _build_queries(N_QUERIES)
    gallery = _load_gallery()

    print("=" * 60)
    print(f"Latency benchmark — {N_QUERIES} queries, warmup={N_WARMUP}")
    print("=" * 60)

    results = [
        bench_pytorch_fp32(queries, gallery),
        bench_pytorch_fp16(queries, gallery),
        bench_onnx(TEXT_ONNX, "onnx", "ONNX Runtime", queries, gallery),
        bench_onnx(TEXT_INT8, "onnx_int8", "ONNX + INT8", queries, gallery),
    ]

    baseline = None
    for r in results:
        if r["backend"] == "pytorch_fp32" and r.get("encode"):
            baseline = r["encode"]["mean_ms"]
            break
    if baseline is None or baseline <= 0:
        baseline = 1.0

    table = []
    for r in results:
        row = {
            "backend": r["backend"],
            "label": r["label"],
            "available": r.get("available", False),
        }
        if r.get("skip_reason"):
            row["skip_reason"] = r["skip_reason"]
        if r.get("encode"):
            row["encode_mean_ms"] = r["encode"]["mean_ms"]
            row["encode_median_ms"] = r["encode"]["median_ms"]
            row["encode_p95_ms"] = r["encode"]["p95_ms"]
            row["speedup_vs_fp32"] = round(baseline / r["encode"]["mean_ms"], 2)
        if r.get("e2e"):
            row["e2e_mean_ms"] = r["e2e"]["mean_ms"]
            row["e2e_median_ms"] = r["e2e"]["median_ms"]
        if r.get("model_mb") is not None:
            row["model_mb"] = r["model_mb"]
        if r.get("device"):
            row["device"] = r["device"]
        table.append(row)

    payload = {
        "n_queries": N_QUERIES,
        "warmup": N_WARMUP,
        "metric": "text encoder latency (ms) for CLIP ViT-B/32 single-query encode",
        "device_note": "All ONNX runs use CPUExecutionProvider; PyTorch FP16 uses CUDA when present",
        "notes": [
            "Absolute ms depend on host CPU load; use speedup_vs_fp32 for cross-machine comparison.",
            "PyTorch FP16 on CPU is typically slower than FP32 (no Tensor-Core path) — reported when it runs.",
            "ONNX / INT8 are the W8/W9 inference paths; gallery search cost is negligible vs encode.",
        ],
        "baseline": "pytorch_fp32",
        "baseline_mean_ms": round(baseline, 2),
        "table": table,
        "details": results,
    }

    # Optional: merge the Colab-measured CUDA FP16 number into the FP16 row.
    if args.merge_fp16:
        _merge_fp16_cuda(args.merge_fp16, results, payload)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[BENCH] JSON -> {OUT_JSON}")

    save_chart(results, baseline, OUT_CHART)

    # Console table
    print("\n" + "=" * 60)
    print(f"{'Backend':<18} {'mean ms':>10} {'median':>10} {'p95':>10} {'vs FP32':>10}")
    print("-" * 60)
    for row in table:
        if not row.get("available") or "encode_mean_ms" not in row:
            reason = row.get("skip_reason", "n/a")[:28]
            print(f"{row['label']:<18} {'SKIP':>10}  ({reason})")
            continue
        device_tag = ""
        if row.get("device") and row["device"] != "cpu":
            device_tag = f" ({row['device']})"
        print(
            f"{row['label']:<18} {row['encode_mean_ms']:>10.1f} "
            f"{row['encode_median_ms']:>10.1f} {row['encode_p95_ms']:>10.1f} "
            f"{row['speedup_vs_fp32']:>9.2f}x{device_tag}"
        )
    print("=" * 60)
    print("[OK] Measured latency table ready (replaces fabricated WPR numbers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
