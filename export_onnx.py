"""
export_onnx.py — Export CLIP visual + text encoders to ONNX
===========================================================
Exports OpenCLIP ViT-B/32 image and text encoders via torch.onnx.export
(opset 17) with dynamic batch axes, then verifies outputs against PyTorch
within 1e-3 using onnxruntime.InferenceSession.

Usage:
    python export_onnx.py

Outputs:
    onnx/clip_visual.onnx
    onnx/clip_text.onnx
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from model import load_clip_model

OPSET = 17
OUT_DIR = "onnx"
VISUAL_ONNX = os.path.join(OUT_DIR, "clip_visual.onnx")
TEXT_ONNX = os.path.join(OUT_DIR, "clip_text.onnx")
LIGHT_WEIGHTS = "light_clip_heads.pt"
ATOL = 1e-3


class VisualEncoder(nn.Module):
    """Thin wrapper so encode_image is a valid ONNX graph root."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(images)


class TextEncoder(nn.Module):
    """Thin wrapper so encode_text is a valid ONNX graph root."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(tokens)


def _maybe_inject_heads(model: nn.Module) -> None:
    """Load lightweight fine-tuned projection heads when present (matches app.py)."""
    if not os.path.isfile(LIGHT_WEIGHTS):
        print(f"[ONNX] No '{LIGHT_WEIGHTS}' found - exporting base OpenAI weights.")
        return
    print(f"[ONNX] Injecting fine-tuned heads from '{LIGHT_WEIGHTS}' ...")
    state = torch.load(LIGHT_WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print("[ONNX] Heads injected (strict=False).")


def _onnx_export(encoder: nn.Module, dummy: torch.Tensor, path: str, **kwargs) -> None:
    """Export with the TorchScript ONNX path (opset 17 + dynamic_axes)."""
    export_kwargs = dict(
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        **kwargs,
    )
    # PyTorch ≥2.5 defaults to the dynamo exporter; force classic path for opset/dynamic_axes.
    try:
        torch.onnx.export(encoder, dummy, path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(encoder, dummy, path, **export_kwargs)


def export_visual(model: nn.Module, path: str) -> None:
    encoder = VisualEncoder(model).eval()
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[ONNX] Exporting visual encoder -> {path}")
    _onnx_export(
        encoder,
        dummy,
        path,
        input_names=["images"],
        output_names=["image_features"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
    )
    print(f"[ONNX] Wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def export_text(model: nn.Module, path: str) -> None:
    encoder = TextEncoder(model).eval()
    # CLIP context length is 77; vocab size ~49k - zeros are a valid dummy batch
    dummy = torch.zeros(1, 77, dtype=torch.long)

    print(f"[ONNX] Exporting text encoder -> {path}")
    _onnx_export(
        encoder,
        dummy,
        path,
        input_names=["tokens"],
        output_names=["text_features"],
        dynamic_axes={
            "tokens": {0: "batch_size"},
            "text_features": {0: "batch_size"},
        },
    )
    print(f"[ONNX] Wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def _ort_session(path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def verify_visual(model: nn.Module, path: str) -> float:
    """Compare PyTorch vs ONNX for batch sizes 1 and 2; return max abs error."""
    encoder = VisualEncoder(model).eval()
    session = _ort_session(path)
    max_err = 0.0

    for batch in (1, 2):
        images = torch.randn(batch, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            pt_out = encoder(images).cpu().numpy()
        ort_out = session.run(None, {"images": images.numpy()})[0]
        err = float(np.max(np.abs(pt_out - ort_out)))
        max_err = max(max_err, err)
        print(f"[VERIFY] visual  batch={batch}  max|d|={err:.2e}")

    return max_err


def verify_text(model: nn.Module, path: str) -> float:
    """Compare PyTorch vs ONNX for batch sizes 1 and 2; return max abs error."""
    encoder = TextEncoder(model).eval()
    session = _ort_session(path)
    max_err = 0.0

    for batch in (1, 2):
        # Non-trivial token ids exercise the embedding + transformer path
        tokens = torch.randint(0, 1000, (batch, 77), dtype=torch.long)
        with torch.no_grad():
            pt_out = encoder(tokens).cpu().numpy()
        ort_out = session.run(None, {"tokens": tokens.numpy()})[0]
        err = float(np.max(np.abs(pt_out - ort_out)))
        max_err = max(max_err, err)
        print(f"[VERIFY] text    batch={batch}  max|d|={err:.2e}")

    return max_err


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Export on CPU for portable ORT CPUExecutionProvider graphs
    model, _ = load_clip_model("ViT-B/32")
    model = model.cpu().eval()
    _maybe_inject_heads(model)

    export_visual(model, VISUAL_ONNX)
    export_text(model, TEXT_ONNX)

    print("\n[ONNX] Verifying with onnxruntime.InferenceSession ...")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[ERROR] onnxruntime is not installed. Run:")
        print("        pip install 'onnxruntime>=1.18.1'")
        return 1

    vis_err = verify_visual(model, VISUAL_ONNX)
    txt_err = verify_text(model, TEXT_ONNX)

    print()
    if vis_err < ATOL and txt_err < ATOL:
        print(f"[OK] PyTorch <-> ONNX match within {ATOL:g} "
              f"(visual={vis_err:.2e}, text={txt_err:.2e})")
        print(f"[OK] Files ready: {VISUAL_ONNX}, {TEXT_ONNX}")
        return 0

    print(f"[FAIL] Max error exceeds {ATOL:g} "
          f"(visual={vis_err:.2e}, text={txt_err:.2e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
