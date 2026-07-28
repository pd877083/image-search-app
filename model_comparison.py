"""
model_comparison.py — CLIP vs BLIP vs ALIGN using OpenCLIP
All three models loaded via open_clip — no transformers needed.

Model mapping:
- CLIP  → ViT-B/32   (OpenAI)
- BLIP  → ViT-L-14   (OpenAI, larger — simulates BLIP scale)
- ALIGN → ViT-H-14   (LAION — simulates ALIGN scale)
"""

import os
import torch
import numpy as np
import open_clip
from PIL import Image

# CPU-first at serve time (see model.py for rationale). FORCE_CPU=1 forces CPU.
DEVICE = "cpu" if os.environ.get("FORCE_CPU", "0") == "1" else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


def _l2_normalise(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms


# Generic loader 

def load_openclip_model(model_name, pretrained):
    print(f"[MODEL] Loading {model_name} ({pretrained}) ...")
    # OpenAI checkpoints use QuickGELU; LAION / others use standard GELU
    kwargs = {"force_quick_gelu": True} if pretrained == "openai" else {}
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, **kwargs
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(DEVICE)
    model.eval()
    print(f"[MODEL] {model_name} loaded successfully.")
    return model, preprocess, tokenizer


def load_blip_model():
    """BLIP simulated via ViT-L-14 OpenAI — larger and more powerful than ViT-B/32"""
    return load_openclip_model("ViT-L-14", "openai")


def load_align_model():
    """ALIGN simulated via ViT-H-14 LAION — trained on 2B image-text pairs like ALIGN"""
    return load_openclip_model("ViT-H-14", "laion2b_s32b_b79k")


# Batch encoders 

def encode_images_model(image_paths, model, preprocess, batch_size=16, label="MODEL"):
    """Encode a list of image file paths.

    Raises on any image that fails to load — the previous silent ``except:
    continue`` dropped rows invisibly and corrupted the saved npz (gallery row
    order no longer matched image order). Failing loud keeps the row count
    honest.
    """
    all_embeddings = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = []
        for path in batch_paths:
            try:
                images.append(preprocess(Image.open(path).convert("RGB")))
            except Exception as exc:
                raise RuntimeError(f"[{label}] Failed to load '{path}': {exc}") from exc
        if not images:
            continue
        tensor = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        emb = emb.cpu().numpy().astype(np.float32)
        all_embeddings.append(_l2_normalise(emb))
        print(f"[{label}] Encoded images {start+1}-{min(start+batch_size, len(image_paths))} / {len(image_paths)}")
    return np.vstack(all_embeddings)


def encode_texts_model(captions, model, tokenizer, batch_size=128, label="MODEL"):
    all_embeddings = []
    for start in range(0, len(captions), batch_size):
        batch = captions[start:start + batch_size]
        tokens = tokenizer(batch).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
        emb = emb.cpu().numpy().astype(np.float32)
        all_embeddings.append(_l2_normalise(emb))
        print(f"[{label}] Encoded captions {start+1}-{min(start+batch_size, len(captions))} / {len(captions)}")
    return np.vstack(all_embeddings)


#  Single query encoders 

def encode_single_image_model(image, model, preprocess):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(tensor)
    return _l2_normalise(emb.cpu().numpy().astype(np.float32))


def encode_single_text_model(query, model, tokenizer):
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    return _l2_normalise(emb.cpu().numpy().astype(np.float32))
