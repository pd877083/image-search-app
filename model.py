"""
model.py — CLIP Model Loading and Encoding
==========================================
This module handles loading the pretrained CLIP model (via open_clip_torch)
and encoding both images and text into the shared embedding space.
"""

import os

# NOTE: do NOT override HF_HOME here. On local dev the user already has the
# CLIP weights at ~/.cache/huggingface/ — overriding HF_HOME would force a
# re-download. On Cloud the default cache works fine and download_models.py
# (called from app.py) pre-populates it at startup.

# FP16 halves RAM at load time — critical for the 1 GB Streamlit Cloud
# sandbox. Toggle via env var if you need FP32 for exact numerics.
USE_FP16 = os.environ.get("CLIP_FP16", "1") == "1"

import torch
import open_clip
from PIL import Image
import numpy as np


# ── Device Setup ────────────────────────────────────────────────────────────
# GPU is used only for the one-time offline embedding-build / training steps
# (build_embeddings.py, train.py, export_onnx.py). At SERVE time only a single
# query is encoded per request, so CPU is plenty fast (ViT-B/32 ≈ 0.5 s/query)
# and Streamlit Cloud has no GPU anyway.
#
# Set FORCE_CPU=1 to force CPU regardless of CUDA availability (recommended for
# the cloud deploy to avoid pulling CUDA libs / surprise slowdowns).
DEVICE = "cpu" if os.environ.get("FORCE_CPU", "0") == "1" else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Set by load_clip_model(); used by encode_texts / encode_single_text
_tokenizer = None


def _to_open_clip_name(model_name: str) -> str:
    """Map OpenAI CLIP names (ViT-B/32) to OpenCLIP names (ViT-B-32)."""
    return model_name.replace("/", "-")


def load_clip_model(model_name: str = "ViT-B/32", pretrained: str | bool | None = "openai"):
    """
    Load the CLIP model architecture (and optionally its pretrained weights)
    via OpenCLIP.

    Args:
        model_name: CLIP variant to use. 'ViT-B/32' (or 'ViT-B-32') is the
                    fastest and runs comfortably on CPU with 8 GB RAM.
                    Other options: 'ViT-B/16', 'RN50', 'RN101'.
        pretrained: Which weights to download. Defaults to "openai".
                    Pass None to skip the download entirely — useful when
                    you're about to overwrite all weights from a local
                    fine-tuned checkpoint (saves ~150 MB and avoids the
                    "wrong file already in the HF cache" trap).

    Returns:
        model      : The CLIP model (in eval mode, no gradients needed).
        preprocess : torchvision transform applied to PIL images before
                     feeding them to the model.
    """
    global _tokenizer

    open_clip_name = _to_open_clip_name(model_name)
    print(f"[CLIP] Loading model '{open_clip_name}' (pretrained={pretrained}) on {DEVICE} …")

    # force_quick_gelu=True matches OpenAI CLIP weights (needed on open_clip ≥3.x)
    # precision='fp16' halves RAM at load (≈600 MB → ≈300 MB for ViT-B/32) which
    # is the difference between fitting in the 1 GB Streamlit Cloud sandbox
    # and being OOM-killed before the first user request completes.
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_name,
            pretrained=pretrained,
            force_quick_gelu=True,
            precision="fp16" if USE_FP16 else "fp32",
        )
    except TypeError:
        # Older open_clip versions don't accept `precision=` — fall back to
        # the legacy kwarg path, then cast to half manually if requested.
        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_name, pretrained=pretrained, force_quick_gelu=True
        )
        if USE_FP16:
            model = model.half()
    if USE_FP16 and DEVICE != "cpu":
        # Half precision is only beneficial on CUDA. On CPU keep float32 to
        # avoid silent numerical issues with the L2-normalised similarity math.
        model = model.float()

    model = model.to(DEVICE)
    model.eval()          # inference-only — disables dropout / batch-norm updates
    # Even when pretrained=None we still want the tokenizer (it's just
    # BPE-vocab, no weights, instant to construct).
    _tokenizer = open_clip.get_tokenizer(open_clip_name)

    print("[CLIP] Model loaded successfully.")
    return model, preprocess


def encode_images(image_paths: list, model, preprocess, batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of image file paths into L2-normalised CLIP embeddings.

    Images are processed in batches to keep memory usage manageable on CPU.

    Args:
        image_paths : List of absolute or relative paths to image files.
        model       : Loaded CLIP model.
        preprocess  : CLIP preprocessing transform.
        batch_size  : Number of images per batch (reduce if RAM is tight).

    Returns:
        embeddings  : Float32 NumPy array of shape (N, 512) where N = len(image_paths).
    """
    all_embeddings = []

    # Process in mini-batches
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")   # ensure 3-channel RGB
                images.append(preprocess(img))           # apply CLIP transforms
            except Exception as exc:
                # Skip corrupted / missing files and log a warning
                print(f"[WARNING] Could not load image '{path}': {exc}")
                continue

        if not images:
            continue

        # Stack into a tensor batch: shape → (B, 3, 224, 224)
        image_tensor = torch.stack(images).to(DEVICE)
        # Cast to the model's dtype (fp16 on CPU under the 1 GB sandbox).
        model_dtype = next(model.parameters()).dtype
        image_tensor = image_tensor.to(dtype=model_dtype)

        with torch.no_grad():                            # no gradient tracking needed
            batch_embeddings = model.encode_image(image_tensor)

        # Move to CPU, convert to NumPy, L2-normalise each row
        batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)
        batch_embeddings = _l2_normalise(batch_embeddings)
        all_embeddings.append(batch_embeddings)

        print(f"[CLIP] Encoded images {start + 1}–{min(start + batch_size, len(image_paths))} / {len(image_paths)}")

    return np.vstack(all_embeddings)


def encode_texts(captions: list, model, batch_size: int = 256) -> np.ndarray:
    """
    Convert a list of caption strings into L2-normalised CLIP embeddings.

    Args:
        captions   : List of caption strings.
        model      : Loaded CLIP model.
        batch_size : Number of captions per batch (text encoding is fast; 256 is safe).

    Returns:
        embeddings : Float32 NumPy array of shape (N, 512).
    """
    if _tokenizer is None:
        raise RuntimeError("Call load_clip_model() before encode_texts().")

    all_embeddings = []

    for start in range(0, len(captions), batch_size):
        batch_captions = captions[start : start + batch_size]

        # OpenCLIP tokenizer truncates captions longer than 77 tokens
        tokens = _tokenizer(batch_captions).to(DEVICE)
        # Cast to the model's dtype (fp16 on CPU under the 1 GB sandbox).
        model_dtype = next(model.parameters()).dtype
        tokens = tokens.to(dtype=model_dtype)

        with torch.no_grad():
            batch_embeddings = model.encode_text(tokens)

        batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)
        batch_embeddings = _l2_normalise(batch_embeddings)
        all_embeddings.append(batch_embeddings)

        print(f"[CLIP] Encoded captions {start + 1}–{min(start + batch_size, len(captions))} / {len(captions)}")

    return np.vstack(all_embeddings)


def encode_single_image(image: Image.Image, model, preprocess) -> np.ndarray:
    """
    Encode a single PIL Image into an L2-normalised embedding.

    Used in the Streamlit app for real-time image upload queries.

    Args:
        image      : PIL Image object.
        model      : Loaded CLIP model.
        preprocess : CLIP preprocessing transform.

    Returns:
        embedding  : Float32 NumPy array of shape (1, 512).
    """
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    # Cast the input to the model's dtype (fp16 on CPU under the 1 GB sandbox).
    # Without this, model.encode_image() raises a "Input type (torch.FloatTensor)
    # and weight type (torch.HalfTensor) should be the same" RuntimeError.
    model_dtype = next(model.parameters()).dtype
    image_tensor = image_tensor.to(dtype=model_dtype)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    embedding = embedding.cpu().numpy().astype(np.float32)
    return _l2_normalise(embedding)


def encode_single_text(query: str, model) -> np.ndarray:
    """
    Encode a single text query into an L2-normalised embedding.

    Args:
        query     : User's text search string.
        model     : Loaded CLIP model.

    Returns:
        embedding : Float32 NumPy array of shape (1, 512).
    """
    if _tokenizer is None:
        raise RuntimeError("Call load_clip_model() before encode_single_text().")

    tokens = _tokenizer([query]).to(DEVICE)
    # Cast the token tensor to the model's dtype (same reason as encode_single_image).
    model_dtype = next(model.parameters()).dtype
    tokens = tokens.to(dtype=model_dtype)

    with torch.no_grad():
        embedding = model.encode_text(tokens)

    embedding = embedding.cpu().numpy().astype(np.float32)
    return _l2_normalise(embedding)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """
    Divide each row vector by its L2 norm so that cosine similarity reduces
    to a simple dot product: cos(a, b) = a · b  (when ‖a‖ = ‖b‖ = 1).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)   # avoid division by zero
    return vectors / norms
