r"""Generate docs/architecture.png - block diagram of the encode -> project -> compare pipeline.

Reproducibility: run from the repo root with the project venv:
    venv\Scripts\python.exe docs\generate_architecture_diagram.py

Uses only matplotlib (no graphviz binaries required).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ----------------------------------------------------------------------------
# Canvas: 100 x 60 data units, 10.8 x 6.48 in @ 150 dpi -> 1620 x 972 px
# ----------------------------------------------------------------------------
W, H = 100.0, 60.0
fig = plt.figure(figsize=(10.8, 6.48), dpi=150)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# Colour scheme per stage
ONLINE_FC, ONLINE_EC = "#DBEAFE", "#2563EB"   # online query path (blue)
OFFLINE_FC, OFFLINE_EC = "#FFEDD5", "#EA580C"  # offline indexing (orange)
COMPARE_FC, COMPARE_EC = "#DCFCE7", "#16A34A"  # compare & rank (green)
ONNX_FC, ONNX_EC = "#F3E8FF", "#9333EA"        # optional ONNX path (purple, dashed)
TEXT_DARK, TEXT_SUB = "#111827", "#374151"


def add_block(cx, cy, w, h, title, sub, fc, ec, dashed=False):
    """Rounded rectangle centred at (cx, cy) with a bold title and a sub-label."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
    )
    ax.add_patch(box)
    n_sub = sub.count("\n") + 1 if sub else 0
    if sub:
        ax.text(cx, cy + h * 0.24, title, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=TEXT_DARK)
        ax.text(cx, cy - h * (0.14 if n_sub == 1 else 0.20), sub,
                ha="center", va="center", fontsize=7.2, color=TEXT_SUB,
                linespacing=1.25)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=TEXT_DARK)


def add_arrow(p0, p1, color, dashed=False, rad=0.0):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.5,
        color=color, linestyle=(0, (4, 2.5)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1,
    )
    ax.add_patch(arr)


# ----------------------------------------------------------------------------
# Titles
# ----------------------------------------------------------------------------
ax.text(W / 2, 57.6, "CLIP Image Search \u2014 Encode \u2192 Project \u2192 Compare Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT_DARK)
ax.text(W / 2, 54.9,
        "CLIP ViT-B-32 (open_clip_torch), fine-tuned weights: my_finetuned_clip.pt \u2022 shared 512-d embedding space",
        ha="center", va="center", fontsize=8, color=TEXT_SUB)

# ----------------------------------------------------------------------------
# Blocks - online query path (blue)
# ----------------------------------------------------------------------------
BH = 7.0  # standard block height
# lane centres
Y_TXT, Y_MID, Y_IMG, Y_OFF, Y_CMP = 46.0, 36.0, 26.0, 12.0, 24.0

add_block(8, Y_TXT, 12, BH, "Text Query", "search tab (Tab 1)", ONLINE_FC, ONLINE_EC)
add_block(23.5, Y_TXT, 12, BH, "Tokenize", "open_clip ViT-B-32\ntokenizer, 77 tokens", ONLINE_FC, ONLINE_EC)
add_block(40, Y_TXT, 14, BH, "Text Encoder", "CLIP ViT-B-32 fine-tuned\n(PyTorch)", ONLINE_FC, ONLINE_EC)

add_block(8, Y_IMG, 12, BH, "Uploaded Image", "reverse search (Tab 5)", ONLINE_FC, ONLINE_EC)
add_block(23.5, Y_IMG, 12, BH, "Preprocess", "resize + centre crop\n224 \u00d7 224", ONLINE_FC, ONLINE_EC)
add_block(40, Y_IMG, 14, BH, "Image Encoder", "CLIP ViT-B-32 fine-tuned\n(PyTorch)", ONLINE_FC, ONLINE_EC)

# Optional ONNX text-query path (purple, dashed) - sits between the two lanes
add_block(40, Y_MID + 0.0, 14, BH, "ONNX Runtime (optional)",
          "clip_text.onnx FP32 /\nclip_int8.onnx INT8, opset 17", ONNX_FC, ONNX_EC, dashed=True)

# Shared projection: native 512-d embedding + L2 norm (no extra MLP head)
add_block(57.5, Y_MID, 13, BH, "512-d Embedding",
          "L2 normalization\n(native CLIP projection)", ONLINE_FC, ONLINE_EC)

# ----------------------------------------------------------------------------
# Blocks - offline indexing lane (orange)
# ----------------------------------------------------------------------------
add_block(8, Y_OFF, 12, BH, "Flickr8k Gallery", "8,091 images\n(demo subset: 300)", OFFLINE_FC, OFFLINE_EC)
add_block(23.5, Y_OFF, 12, BH, "Batch Encode", "build_embeddings.py", OFFLINE_FC, OFFLINE_EC)
add_block(40, Y_OFF, 14, BH, "Gallery Embeddings", "512-d, L2-normalized\nembeddings\\*.npy", OFFLINE_FC, OFFLINE_EC)

# ----------------------------------------------------------------------------
# Blocks - compare & rank (green)
# ----------------------------------------------------------------------------
add_block(75.5, Y_CMP, 15, BH, "Cosine Similarity",
          "FAISS IndexFlatIP\n(NumPy dot-product fallback)", COMPARE_FC, COMPARE_EC)
add_block(92.75, Y_CMP, 12.5, BH, "Top-K Ranking",
          "Streamlit gallery\nwith similarity scores", COMPARE_FC, COMPARE_EC)

# ----------------------------------------------------------------------------
# Arrows
# ----------------------------------------------------------------------------
# Text lane
add_arrow((14, Y_TXT), (17.5, Y_TXT), ONLINE_EC)
add_arrow((29.5, Y_TXT), (33, Y_TXT), ONLINE_EC)
add_arrow((47, Y_TXT), (51, Y_MID + 2.2), ONLINE_EC)          # text encoder -> 512-d
# Image lane
add_arrow((14, Y_IMG), (17.5, Y_IMG), ONLINE_EC)
add_arrow((29.5, Y_IMG), (33, Y_IMG), ONLINE_EC)
add_arrow((47, Y_IMG), (51, Y_MID - 2.2), ONLINE_EC)          # image encoder -> 512-d
# Optional ONNX text path (dashed purple)
add_arrow((24.5, Y_TXT - BH / 2), (33, Y_MID - 1.2), ONNX_EC, dashed=True)  # tokenize -> onnx
add_arrow((47, Y_MID), (51, Y_MID), ONNX_EC, dashed=True)                    # onnx -> 512-d
# Offline lane
add_arrow((14, Y_OFF), (17.5, Y_OFF), OFFLINE_EC)
add_arrow((29.5, Y_OFF), (33, Y_OFF), OFFLINE_EC)
add_arrow((47, Y_OFF), (68, Y_CMP - 2.2), OFFLINE_EC)         # embeddings -> compare
# Query embedding -> compare, compare -> rank
add_arrow((64, Y_MID), (68, Y_CMP + 2.2), ONLINE_EC)
add_arrow((83, Y_CMP), (86.5, Y_CMP), COMPARE_EC)

# Small arrow labels (offset so they never touch the arrows)
ax.text(68.6, 31.6, "query\n(512-d)", ha="left", va="center", fontsize=6.8,
        color=ONLINE_EC, style="italic", linespacing=1.1)
ax.text(57.5, 13.6, "precomputed index (512-d)", ha="center", va="center",
        fontsize=6.8, color=OFFLINE_EC, style="italic")

# ----------------------------------------------------------------------------
# Legend (bottom right)
# ----------------------------------------------------------------------------
lg = FancyBboxPatch((67.5, 2.2), 31, 13.6, boxstyle="round,pad=0,rounding_size=0.8",
                    facecolor="#F9FAFB", edgecolor="#9CA3AF", linewidth=0.9)
ax.add_patch(lg)
ax.text(69, 14.1, "Legend", ha="left", va="center", fontsize=7.8,
        fontweight="bold", color=TEXT_DARK)
legend_items = [
    ("Online query path (app.py)", ONLINE_FC, ONLINE_EC, False),
    ("Offline indexing (build_embeddings.py)", OFFLINE_FC, OFFLINE_EC, False),
    ("Compare & rank (FAISS / NumPy)", COMPARE_FC, COMPARE_EC, False),
    ("Optional ONNX query path", ONNX_FC, ONNX_EC, True),
]
for i, (label, fc, ec, dashed) in enumerate(legend_items):
    y = 11.9 - i * 2.5
    sw = FancyBboxPatch((69, y - 0.8), 3.2, 1.6,
                        boxstyle="round,pad=0,rounding_size=0.4",
                        facecolor=fc, edgecolor=ec, linewidth=1.1,
                        linestyle=(0, (3, 2)) if dashed else "solid")
    ax.add_patch(sw)
    ax.text(73.2, y, label, ha="left", va="center", fontsize=7.2, color=TEXT_SUB)

# ----------------------------------------------------------------------------
# Footnotes (optional annotations: benchmark + evaluation)
# ----------------------------------------------------------------------------
ax.text(2, 4.6, "Benchmark: PyTorch vs ONNX FP32 vs ONNX INT8 query latency (benchmark.py)",
        ha="left", va="center", fontsize=7, color="#6B7280", style="italic")
ax.text(2, 2.6, "Evaluation: Hit@K of Top-K ranking vs ground-truth caption\u2013image pairs (evaluate.py)",
        ha="left", va="center", fontsize=7, color="#6B7280", style="italic")

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
fig.savefig(out_path, dpi=150, facecolor="white")
print(f"Saved {out_path}")
