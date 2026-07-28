"""
build_embeddings.py — Precompute and Save CLIP Embeddings (FINETUNED VERSION)
==========================================================
TRUNCATED VERSION: Limited to 8000 captions for blazingly fast CPU performance.
"""

import os
import sys
import numpy as np
import torch

from model import load_clip_model, encode_images, encode_texts
from utils import save_embeddings, load_flickr8k_captions

DATA_DIR        = os.path.join("data", "Flickr8k_Dataset", "Flicker8k_Dataset")
CAPTIONS_FILE   = os.path.join("data", "Flickr8k_Dataset", "Flickr8k_text", "Flickr8k.token.txt")
EMBEDDINGS_DIR  = "embeddings"

IMAGE_EMBED_OUT = os.path.join(EMBEDDINGS_DIR, "image_embeddings")
TEXT_EMBED_OUT  = os.path.join(EMBEDDINGS_DIR, "text_embeddings")
META_OUT        = os.path.join(EMBEDDINGS_DIR, "metadata")

WEIGHTS_PATH    = "my_finetuned_clip.pt"
CLIP_MODEL      = "ViT-B/32"
BATCH_SIZE      = 32

def main():
    if not os.path.isdir(DATA_DIR) or not os.path.isfile(CAPTIONS_FILE):
        print("\n[ERROR] Dataset files missing under data/ directory!")
        sys.exit(1)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    model, preprocess = load_clip_model(CLIP_MODEL)

    if os.path.isfile(WEIGHTS_PATH):
        print(f"\n[WEIGHTS] Loading finetuned weights from '{WEIGHTS_PATH}' ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("[WEIGHTS] Custom fine-tuned weights successfully injected!")

    print("\n[DATA] Parsing captions …")
    image_names, captions = load_flickr8k_captions(CAPTIONS_FILE)

    # Fast 8000 slicing to match dimensions across all files smoothly
    image_names = image_names[:8000]
    captions = captions[:8000]

    image_paths_for_caption = []
    valid_image_names = []
    valid_captions = []

    for name, cap in zip(image_names, captions):
        full_path = os.path.join(DATA_DIR, name)
        if os.path.isfile(full_path):
            image_paths_for_caption.append(full_path)
            valid_image_names.append(name)
            valid_captions.append(cap)

    unique_names, unique_paths = [], []
    seen = set()
    for name, path in zip(valid_image_names, image_paths_for_caption):
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
            unique_paths.append(path)

    print(f"[DATA] {len(unique_names)} unique images will be encoded.")
    image_embeddings = encode_images(unique_paths, model, preprocess, batch_size=BATCH_SIZE)

    print("\n[EMBED] Encoding captions …")
    text_embeddings = encode_texts(valid_captions, model, batch_size=256)

    print("\n[SAVE] Writing embeddings to disk …")
    save_embeddings(IMAGE_EMBED_OUT, embeddings=image_embeddings)
    save_embeddings(TEXT_EMBED_OUT,  embeddings=text_embeddings)

    np.savez_compressed(
        META_OUT,
        image_names=np.array(unique_names,   dtype=object),
        image_paths=np.array(unique_paths,   dtype=object),
        captions   =np.array(valid_captions, dtype=object),
        caption_image_names=np.array(valid_image_names, dtype=object),
    )
    print(f"[SAVE] Metadata saved smoothly to '{META_OUT}.npz'")

if __name__ == "__main__":
    main()