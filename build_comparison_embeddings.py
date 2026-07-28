"""
build_comparison_embeddings.py
Builds BLIP and ALIGN embeddings using optimized parameters for CPU execution.
"""

import os
import numpy as np
from model_comparison import (
    load_blip_model, load_align_model,
    encode_images_model, encode_texts_model,
)

EMBEDDINGS_DIR = "embeddings"
META_PATH      = os.path.join(EMBEDDINGS_DIR, "metadata.npz")

def main():
    if not os.path.exists(META_PATH):
        print(f"[ERROR] Run build_embeddings.py first! Metadata file missing.")
        return

    meta        = np.load(META_PATH, allow_pickle=True)
    image_paths = meta["image_paths"].tolist()
    captions    = meta["captions"].tolist()
    print(f"\nImages to process: {len(image_paths)} | Captions to process: {len(captions)}")

    # ── BLIP (ViT-L-14) ───────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("BUILDING BLIP EMBEDDINGS (Optimized Batch Size = 8)")
    print("="*50)
    blip_model, blip_preprocess, blip_tokenizer = load_blip_model()

    blip_img = encode_images_model(image_paths, blip_model, blip_preprocess, batch_size=8, label="BLIP")
    np.savez_compressed(os.path.join(EMBEDDINGS_DIR, "blip_image_embeddings"), embeddings=blip_img)

    blip_txt = encode_texts_model(captions, blip_model, blip_tokenizer, batch_size=128, label="BLIP")
    np.savez_compressed(os.path.join(EMBEDDINGS_DIR, "blip_text_embeddings"), embeddings=blip_txt)
    print("✅ BLIP embeddings saved!")
    del blip_model, blip_preprocess, blip_tokenizer

    # ── ALIGN (ViT-H-14) ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("BUILDING ALIGN EMBEDDINGS (Optimized Batch Size = 4)")
    print("="*50)
    align_model, align_preprocess, align_tokenizer = load_align_model()

    align_img = encode_images_model(image_paths, align_model, align_preprocess, batch_size=4, label="ALIGN")
    np.savez_compressed(os.path.join(EMBEDDINGS_DIR, "align_image_embeddings"), embeddings=align_img)

    align_txt = encode_texts_model(captions, align_model, align_tokenizer, batch_size=128, label="ALIGN")
    np.savez_compressed(os.path.join(EMBEDDINGS_DIR, "align_text_embeddings"), embeddings=align_txt)
    print("✅ ALIGN embeddings saved!")

    print("\n🎉 All comparison embeddings configured and built successfully!")

if __name__ == "__main__":
    main()