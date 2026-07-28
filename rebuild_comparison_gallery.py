"""
Rebuild BLIP/ALIGN *image* embeddings for the 1,600 unique captioned
Flickr8k images so they match the current OpenCLIP weights used at query time.

(The previous npz files were out of sync with the installed open_clip models —
fresh vs stored cosine was ~0.2–0.5 — which made BLIP/ALIGN Hit@K ≈ 0.)
"""

import os
import gc
import numpy as np

from model_comparison import (
    load_blip_model,
    load_align_model,
    encode_images_model,
)

EMBEDDINGS_DIR = "embeddings"
META_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.npz")
BLIP_OUT = os.path.join(EMBEDDINGS_DIR, "blip_image_embeddings")
ALIGN_OUT = os.path.join(EMBEDDINGS_DIR, "align_image_embeddings")


def captioned_paths(meta):
    image_names = meta["image_names"].tolist()
    image_paths = meta["image_paths"].tolist()
    caption_image_names = meta["caption_image_names"].tolist()
    name_to_path = dict(zip(image_names, image_paths))
    seen, paths, names = set(), [], []
    for name in caption_image_names:
        if name not in seen:
            seen.add(name)
            names.append(name)
            paths.append(name_to_path[name])
    return names, paths


def main():
    meta = np.load(META_PATH, allow_pickle=True)
    names, paths = captioned_paths(meta)
    print(f"[DATA] Rebuilding comparison gallery for {len(paths)} captioned images")

    print("\n=== BLIP ViT-L-14 ===")
    blip_model, blip_pre, _ = load_blip_model()
    blip_img = encode_images_model(
        paths, blip_model, blip_pre, batch_size=8, label="BLIP"
    )
    np.savez_compressed(BLIP_OUT, embeddings=blip_img.astype(np.float16))
    print(f"[SAVE] {BLIP_OUT}.npz  shape={blip_img.shape}")
    del blip_model, blip_pre
    gc.collect()

    print("\n=== ALIGN ViT-H-14 ===")
    align_model, align_pre, _ = load_align_model()
    align_img = encode_images_model(
        paths, align_model, align_pre, batch_size=4, label="ALIGN"
    )
    np.savez_compressed(ALIGN_OUT, embeddings=align_img.astype(np.float16))
    print(f"[SAVE] {ALIGN_OUT}.npz  shape={align_img.shape}")

    print("\n✅ Comparison gallery embeddings rebuilt.")


if __name__ == "__main__":
    main()
