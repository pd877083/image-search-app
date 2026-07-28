"""
prepare_demo_subset.py — Build a compact, self-contained demo dataset
====================================================================
The full Flickr8k gallery (8091 images, ~1.1 GB) cannot be pushed to
GitHub / Streamlit Cloud. This script selects a diverse ~300-image subset,
slices the corresponding rows out of EVERY embedding file, copies just those
images, and rewrites all stored paths to POSIX (forward-slash) form so they
resolve identically on Windows, Linux, and Streamlit Cloud.

Output (all under data_demo/ + embeddings_demo/, ~45 MB total):
  data_demo/Flickr8k_Dataset/Flicker8k_Dataset/*.jpg
  embeddings_demo/image_embeddings.npz
  embeddings_demo/text_embeddings.npz
  embeddings_demo/metadata.npz
  embeddings_demo/blip_image_embeddings.npz
  embeddings_demo/blip_text_embeddings.npz
  embeddings_demo/align_image_embeddings.npz
  embeddings_demo/align_text_embeddings.npz
  embeddings_demo/precision_scores.json   (copied verbatim — precomputed)

Run once locally, then commit the *_demo folders. The cloud app points at
them via the DEMO_MODE flag in app.py.
"""

import json
import os
import random
import shutil

import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_IMAGES = 300
SRC_EMB = "embeddings"
SRC_IMG = os.path.join("data", "Flickr8k_Dataset", "Flicker8k_Dataset")
DST_EMB = "embeddings_demo"
DST_IMG = os.path.join("data_demo", "Flickr8k_Dataset", "Flicker8k_Dataset")


def to_posix(p: str) -> str:
    """Normalise Windows backslashes to forward slashes (Linux-safe)."""
    return p.replace("\\", "/")


# ── 1. Load metadata and pick a diverse image subset ──────────────────────────
meta = np.load(os.path.join(SRC_EMB, "metadata.npz"), allow_pickle=True)
image_names_all = meta["image_names"].tolist()        # 8091 unique filenames
image_paths_all = [to_posix(p) for p in meta["image_paths"].tolist()]
caption_image_names = meta["caption_image_names"].tolist()  # 8000 (5 per image)

# Prefer images that actually have captions (the 1600 with captions), so the
# image->caption and model-comparison tabs stay meaningful. Then top up.
captioned = sorted(set(caption_image_names))
random.shuffle(captioned)
chosen_names = captioned[:NUM_IMAGES]
if len(chosen_names) < NUM_IMAGES:
    # Top up from non-captioned images (image->image search still works).
    extras = [n for n in image_names_all if n not in set(chosen_names)]
    random.shuffle(extras)
    chosen_names += extras[: NUM_IMAGES - len(chosen_names)]

chosen_set = set(chosen_names)
print(f"[subset] Selected {len(chosen_names)} images "
      f"({sum(1 for n in chosen_names if n in set(caption_image_names))} captioned).")

# Map chosen filename -> POSIX path (relative, forward-slash)
name_to_path = {n: p for n, p in zip(image_names_all, image_paths_all)}

# Index of kept image rows in image_embeddings (order = image_names_all order)
kept_img_idx = np.array(
    [i for i, n in enumerate(image_names_all) if n in chosen_set], dtype=np.int64
)

# Final subset image ordering (used by both the metadata writer and the
# BLIP/ALIGN name-aligned slicers below).
subset_image_names = [image_names_all[i] for i in kept_img_idx]
subset_image_paths = [name_to_path[n] for n in subset_image_names]
# name -> new local index in the subset (for caption_image_names remap)
name_to_newidx = {image_names_all[i]: new_i for new_i, i in enumerate(kept_img_idx)}

# ── 2. Prepare output dirs ────────────────────────────────────────────────────
for d in (DST_EMB, DST_IMG):
    os.makedirs(d, exist_ok=True)

# ── 3. Copy the chosen images ─────────────────────────────────────────────────
copied = 0
missing = 0
for n in chosen_names:
    src = os.path.join(SRC_IMG, n)
    dst = os.path.join(DST_IMG, n)
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
        copied += 1
    else:
        missing += 1
print(f"[subset] Copied {copied} images, {missing} missing.")

# ── 4. Slice image embeddings ─────────────────────────────────────────────────
# CLIP image_embeddings is aligned positionally with image_names_all (8091 rows).
# BLIP/ALIGN only encoded the 1600 captioned-gallery images and store their own
# `image_names` array alongside the embeddings — so subset those by name match.
def slice_clip_img(npz_path, save_name):
    arr = np.load(npz_path, allow_pickle=True)
    emb = arr["embeddings"]
    assert emb.shape[0] == len(image_names_all), \
        f"{npz_path}: {emb.shape[0]} rows vs {len(image_names_all)} expected"
    sub = emb[kept_img_idx]
    out = os.path.join(DST_EMB, save_name)
    np.savez(out, embeddings=sub)
    print(f"[subset] {save_name}: {sub.shape}  ({os.path.getsize(out) // 1024} KB)")
    return sub


def slice_named_img(npz_path, save_name):
    arr = np.load(npz_path, allow_pickle=True)
    emb = arr["embeddings"]
    names = arr["image_names"].tolist()
    name_to_row = {n: i for i, n in enumerate(names)}
    rows = []
    out_names = []
    for n in subset_image_names:        # final CLIP-subset order
        if n in name_to_row:
            rows.append(name_to_row[n])
            out_names.append(n)
    sub = emb[np.array(rows, dtype=np.int64)]
    out = os.path.join(DST_EMB, save_name)
    np.savez(out, embeddings=sub, image_names=np.array(out_names, dtype=object))
    print(f"[subset] {save_name}: {sub.shape}  "
          f"({os.path.getsize(out) // 1024} KB, {len(rows)}/{len(subset_image_names)} matched)")
    return sub


clip_img = slice_clip_img(os.path.join(SRC_EMB, "image_embeddings.npz"), "image_embeddings.npz")
blip_img = slice_named_img(os.path.join(SRC_EMB, "blip_image_embeddings.npz"), "blip_image_embeddings.npz")
align_img = slice_named_img(os.path.join(SRC_EMB, "align_image_embeddings.npz"), "align_image_embeddings.npz")

# ── 5. Slice text embeddings + remap caption_image_names to subset ────────────
# captions list (8000) maps 1:1 to caption_image_names. Keep only captions whose
# image survived the subset, and remap their image index into the new frame.
cap_mask = [c in chosen_set for c in caption_image_names]
kept_cap_idx = np.array([i for i, k in enumerate(cap_mask) if k], dtype=np.int64)

def slice_text(npz_path, save_name):
    arr = np.load(npz_path, allow_pickle=True)
    emb = arr["embeddings"]
    sub = emb[kept_cap_idx]
    out = os.path.join(DST_EMB, save_name)
    np.savez(out, embeddings=sub)
    print(f"[subset] {save_name}: {sub.shape}  ({os.path.getsize(out) // 1024} KB)")
    return sub


clip_txt = slice_text(os.path.join(SRC_EMB, "text_embeddings.npz"), "text_embeddings.npz")
blip_txt = slice_text(os.path.join(SRC_EMB, "blip_text_embeddings.npz"), "blip_text_embeddings.npz")
align_txt = slice_text(os.path.join(SRC_EMB, "align_text_embeddings.npz"), "align_text_embeddings.npz")

kept_captions = [meta["captions"].tolist()[i] for i in kept_cap_idx]
kept_cap_img_names = [caption_image_names[i] for i in kept_cap_idx]

# ── 6. Write subset metadata (POSIX paths) ────────────────────────────────────
np.savez(
    os.path.join(DST_EMB, "metadata.npz"),
    image_names=subset_image_names,
    image_paths=subset_image_paths,
    captions=kept_captions,
    caption_image_names=kept_cap_img_names,
    allow_pickle=True,
)
print(f"[subset] metadata.npz: {len(subset_image_names)} images, "
      f"{len(kept_captions)} captions  "
      f"({os.path.getsize(os.path.join(DST_EMB, 'metadata.npz')) // 1024} KB)")

# ── 7. Copy precomputed precision scores (chart still renders) ────────────────
src_prec = os.path.join(SRC_EMB, "precision_scores.json")
if os.path.isfile(src_prec):
    shutil.copyfile(src_prec, os.path.join(DST_EMB, "precision_scores.json"))
    print(f"[subset] precision_scores.json copied.")

# ── 8. Summary ────────────────────────────────────────────────────────────────
total_kb = sum(
    os.path.getsize(os.path.join(DST_EMB, f))
    for f in os.listdir(DST_EMB)
)
img_kb = sum(
    os.path.getsize(os.path.join(DST_IMG, f))
    for f in os.listdir(DST_IMG)
)
print("\n" + "=" * 60)
print(f"DONE. embeddings_demo = {total_kb // 1024} MB, "
      f"data_demo images = {img_kb // 1024} MB")
print(f"Total demo payload ≈ {(total_kb + img_kb) / (1024 * 1024):.1f} MB")
print("Commit these two folders to deploy on Streamlit Cloud.")
