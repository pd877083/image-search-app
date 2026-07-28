"""
rebuild_comparison_embeddings.py — Robust rebuild of BLIP/ALIGN embeddings
==========================================================================

WHY THIS EXISTS
---------------
The previous build_comparison_embeddings.py iterated over *all* 8,091
image_paths and dropped corrupted files silently, which left the saved
gallery with an unknown row order. Fresh re-encoding of the same image gave
cosine 0.22-0.59 against the stored row (should be 1.0000) — i.e. the rows
were misaligned/corrupt, so BLIP P@5 came out as 0%.

This script fixes that by:
  1. Encoding exactly the 1,600 captioned-gallery images — the first-seen
     unique names from metadata.npz:caption_image_names, which coincide with
     image_names[0:1600] (= image_paths[0:1600]). This matches the rows that
     evaluate.py's captioned_gallery() selects from the CLIP array.
  2. Saving image_names ALONGSIDE the embeddings inside each npz, so any
     future consumer can verify alignment without metadata.
  3. Storing float32 (not float16) — fp16 caused numerical drift in the
     previous BLIP file.

Output rows are GUARANTEED to match the gallery slice the app/eval use.

GPU-FIRST (intended for Colab T4)
---------------------------------
  - BLIP ViT-L-14 : 1600 imgs ~ 1 min on T4, ~35 min on CPU
  - ALIGN ViT-H-14: 1600 imgs ~ 3 min on T4, ~90 min on CPU

Usage:
    python rebuild_comparison_embeddings.py            # both models, GPU if avail
    python rebuild_comparison_embeddings.py --model blip
    python rebuild_comparison_embeddings.py --model align
    python rebuild_comparison_embeddings.py --limit 50 # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from model_comparison import (
    load_blip_model, load_align_model,
    encode_images_model, encode_texts_model,
)

EMBEDDINGS_DIR = "embeddings"
META_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.npz")


def _save(path_noext: str, embeddings: np.ndarray, image_names: list) -> None:
    """Persist as float32 + keep image_names inside the npz for self-describing alignment."""
    out = os.path.join(EMBEDDINGS_DIR, os.path.basename(path_noext))
    np.savez_compressed(
        out,
        embeddings=embeddings.astype(np.float32),
        image_names=np.array(image_names, dtype=object),
    )
    mb = os.path.getsize(out + ".npz") / 1e6
    print(f"[SAVE] {out}.npz  ({embeddings.shape}, {mb:.1f} MB)")


def _build_image(model_name: str, load_fn, paths: list, batch_size: int):
    print(f"\n[{model_name}] Encoding {len(paths)} images (batch={batch_size}) ...")
    model, preprocess, tokenizer = load_fn()
    t0 = time.time()
    emb = encode_images_model(paths, model, preprocess, batch_size=batch_size, label=model_name)
    dt = time.time() - t0
    print(f"[{model_name}] {len(paths)} images in {dt:.1f}s ({dt/max(len(paths),1):.2f}s/img)")
    del model, preprocess, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


def _build_text(model_name: str, load_fn, captions: list, batch_size: int):
    model, preprocess, tokenizer = load_fn()
    print(f"[{model_name}] Encoding {len(captions)} captions ...")
    t0 = time.time()
    emb = encode_texts_model(captions, model, tokenizer, batch_size=batch_size, label=model_name)
    print(f"[{model_name}] {len(captions)} captions in {time.time()-t0:.1f}s")
    del model, preprocess, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


def build_blip(paths, names, captions, do_text=True):
    emb_img = _build_image("BLIP", load_blip_model, paths, batch_size=16)
    _save("blip_image_embeddings", emb_img, names)
    if do_text:
        emb_txt = _build_text("BLIP", load_blip_model, captions, batch_size=128)
        _save("blip_text_embeddings", emb_txt, [])
    return emb_img


def build_align(paths, names, captions, do_text=True):
    emb_img = _build_image("ALIGN", load_align_model, paths, batch_size=8)
    _save("align_image_embeddings", emb_img, names)
    if do_text:
        emb_txt = _build_text("ALIGN", load_align_model, captions, batch_size=128)
        _save("align_text_embeddings", emb_txt, [])
    return emb_img


def _sanity_check(path_noext: str, load_fn, image_paths: list, names: list) -> None:
    """Re-encode image[0] and confirm cosine with stored row[0] ~ 1.0."""
    from model_comparison import encode_single_image_model
    emb = np.load(os.path.join(EMBEDDINGS_DIR, os.path.basename(path_noext) + ".npz"),
                  allow_pickle=True)["embeddings"].astype(np.float32)
    if emb.shape[0] != len(names):
        print(f"[CHECK] {path_noext}: row count {emb.shape[0]} != names {len(names)}  FAIL")
        return
    model, preprocess, _ = load_fn()
    fresh = encode_single_image_model(Image.open(image_paths[0]).convert("RGB"),
                                      model, preprocess)[0]
    cos = float(np.dot(fresh, emb[0]))
    status = "OK" if cos > 0.99 else "FAIL"
    print(f"[CHECK] {path_noext}: row[0] cosine = {cos:.4f}  {status}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blip", "align", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only first N images (0 = all). Smoke-test mode.")
    ap.add_argument("--no-text", action="store_true", help="skip caption embeddings")
    ap.add_argument("--check", action="store_true", help="run sanity check after build")
    args = ap.parse_args()

    if not os.path.isfile(META_PATH):
        print(f"[ERROR] {META_PATH} missing. Run build_embeddings.py first.")
        return 1

    meta = np.load(META_PATH, allow_pickle=True)
    all_image_names = meta["image_names"].tolist()      # 8091 (every file)
    all_image_paths = meta["image_paths"].tolist()      # aligned 1:1 with above
    caption_image_names = meta["caption_image_names"].tolist()  # 8000 with dupes
    captions = meta["captions"].tolist()

    # Gallery = first-seen unique captioned images = image_names[0:1600].
    # This is the exact slice evaluate.py's captioned_gallery() picks, so the
    # BLIP/ALIGN npz rows line up with the CLIP gallery rows it slices.
    gallery_names = list(dict.fromkeys(caption_image_names))
    image_names = gallery_names

    # Resolve gallery paths in an OS-agnostic way. metadata.npz:image_paths stores
    # Windows backslash paths from the build machine; on Linux/Colab those won't
    # resolve. Try (a) the stored path as-is, (b) a data/ root with the bare
    # filename, (c) a Flickr8k_Dataset/Flicker8k_Dataset/ root. Bail loudly if a
    # path is missing — the OLD silent except:continue is what corrupted the
    # embeddings in the first place.
    def _resolve(name: str, stored: str) -> str:
        candidates = [
            stored,
            os.path.join("data", "Flickr8k_Dataset", "Flicker8k_Dataset", name),
            os.path.join("Flickr8k_Dataset", name),
            os.path.join("data", name),
            name,
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"Cannot locate image '{name}'. Tried: {candidates}"
        )

    name_to_stored = dict(zip(all_image_names, all_image_paths))
    image_paths = [_resolve(n, name_to_stored.get(n, "")) for n in image_names]
    print(f"[META] gallery (unique captioned) images={len(image_names)}  captions={len(captions)}")
    print(f"[META] device={'cuda (GPU)' if torch.cuda.is_available() else 'cpu'}")
    print(f"[META] resolved path sample: {image_paths[0]}")

    if args.limit > 0:
        image_paths = image_paths[:args.limit]
        image_names = image_names[:args.limit]
        print(f"[META] --limit {args.limit} applied")

    do_text = not args.no_text
    if args.model in ("blip", "both"):
        build_blip(image_paths, image_names, captions, do_text)
        if args.check:
            _sanity_check("blip_image_embeddings", load_blip_model, image_paths, image_names)
    if args.model in ("align", "both"):
        build_align(image_paths, image_names, captions, do_text)
        if args.check:
            _sanity_check("align_image_embeddings", load_align_model, image_paths, image_names)

    print("\n[OK] Rebuild complete. Row order matches metadata.npz image_names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
