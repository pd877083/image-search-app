"""
app.py — Semantic Image Search · Streamlit UI
"""

import os

# NOTE: do NOT override HF_HOME here. On local dev the user already has the
# CLIP weights at ~/.cache/huggingface/ (downloaded during local training) —
# pointing HF_HOME at a project-local directory would force a re-download
# and waste minutes. On Streamlit Cloud huggingface_hub writes to the user's
# home cache by default; that's also fine (the pre-flight download below
# populates it at startup so the first user request is fast).
#
# If you ever need to force a project-local cache, set HF_CACHE_DIR (the
# modern env var huggingface_hub reads) explicitly before running.

import io
import time
import importlib.util
import numpy as np
import streamlit as st
from PIL import Image

from model import load_clip_model, encode_single_text, encode_single_image
from model_comparison import (
    load_blip_model, load_align_model,
    encode_single_text_model, encode_single_image_model,
)
from utils import (
    load_embeddings,
    text_to_image_search,
    image_to_text_search,
    image_to_image_search,
    TFIDFSearcher,
    BM25Searcher,
)

st.set_page_config(
    page_title="CLIP Semantic Search",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Pre-flight: ensure model weights are cached locally ─────────────────────
# On Streamlit Cloud the first request triggers a 150 MB+ download from
# HuggingFace. Doing it here (with a visible status widget + retry logic)
# turns a silent OOM/network-kill into a clear, recoverable error for the
# user. Idempotent — a warm cache skips straight to "ready".
#
# BUT: if the user already has the full fine-tuned checkpoint locally
# (my_finetuned_clip.pt), we're about to overwrite all weights anyway and
# the openai download is pure waste. Worse, the local HF cache may contain
# a stale/wrong file (e.g. the fine-tuned .pt placed under the openai repo
# path) that confuses huggingface_hub and hangs the download. Skip entirely
# in that case.
import download_models  # noqa: E402

_SKIP_DOWNLOAD = os.path.isfile("my_finetuned_clip.pt")

if _SKIP_DOWNLOAD:
    # Local dev with full fine-tuned weights on disk — no download needed.
    st.toast("✅ Found my_finetuned_clip.pt — using local fine-tuned weights, skipping HuggingFace download.", icon="✅")
else:
    with st.status("Preparing models (one-time, ~150 MB on first run) ...", expanded=True) as status_box:
        def _on_progress(repo_id, stage, detail):
            if stage == "cached":
                status_box.write(f"✅ {repo_id} — already cached")
            elif stage == "downloading":
                status_box.write(f"⬇️ {repo_id} — {detail}")
            elif stage == "ready":
                status_box.write(f"✅ {repo_id} — ready")
            elif stage == "error":
                status_box.write(f"❌ {repo_id} — {detail}")

        try:
            download_models.ensure_models(include_optional=False, progress_cb=_on_progress)
            status_box.update(label="Models ready ✓", state="complete")
        except Exception as exc:
            status_box.update(label="Model download failed", state="error")
            st.error(
                f"**Couldn't fetch the CLIP model from HuggingFace.**\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                f"This is usually one of:\n"
                f"- A transient HuggingFace rate-limit — try **Reboot app** in the\n"
                f"  Streamlit Cloud menu (top-right).\n"
                f"- A blocked egress from the cloud sandbox — check that\n"
                f"  `huggingface.co` is reachable.\n\n"
                f"For higher rate limits, add a free `HF_TOKEN` to your app's\n"
                f"**Secrets** (https://huggingface.co/settings/tokens)."
        )
        st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #08080f; color: #e8e6f0; }
.main { background-color: #08080f; }
section[data-testid="stSidebar"] { display: none; }
.hero-wrap { text-align: center; padding: 3.5rem 0 2rem; }
.hero-eyebrow { font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.25em; text-transform: uppercase; color: #9b8fff; margin-bottom: 1rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 800; line-height: 1.05; background: linear-gradient(135deg, #ffffff 0%, #c4b8ff 50%, #8b6fff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 1rem; }
.hero-sub { font-size: 1rem; font-weight: 300; color: #8a87a0; max-width: 520px; margin: 0 auto 2rem; line-height: 1.65; }
.pill-badge { display: inline-block; background: rgba(139, 111, 255, 0.12); border: 1px solid rgba(139, 111, 255, 0.3); color: #b3a0ff; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 99px; margin: 0 0.25rem; }
.stTabs [data-baseweb="tab-list"] { gap: 0; background: rgba(255,255,255,0.03); border-radius: 12px; padding: 4px; border: 1px solid rgba(255,255,255,0.06); }
.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.04em; color: #6b6884; background: transparent; border-radius: 9px; padding: 0.6rem 1.6rem; border: none; transition: all 0.2s ease; }
.stTabs [aria-selected="true"] { background: rgba(139, 111, 255, 0.2) !important; color: #c4b8ff !important; }
.stTextInput > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 12px !important; color: #e8e6f0 !important; font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; }
.stTextInput > div > div > input::placeholder { color: #4a4862 !important; }
.stButton > button { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; background: linear-gradient(135deg, #7c5fff, #5b3fff) !important; color: #fff !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 2rem !important; box-shadow: 0 4px 20px rgba(124, 95, 255, 0.35) !important; }
hr { border-color: rgba(255,255,255,0.06) !important; margin: 2.5rem 0 !important; }
.score-bar-wrap { margin-top: 0.6rem; }
.score-label { font-size: 0.7rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #6b6884; margin-bottom: 0.35rem; }
.score-bar-bg { background: rgba(255,255,255,0.06); border-radius: 99px; height: 4px; width: 100%; }
.score-bar-fill { border-radius: 99px; height: 4px; }
.score-value { font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700; text-align: right; margin-top: 0.25rem; }
.caption-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 0.9rem; }
.caption-rank { font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #5b4bcc; margin-bottom: 0.5rem; }
.caption-text { font-size: 0.95rem; color: #ccc9e0; line-height: 1.6; font-style: italic; }
.caption-score { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; color: #7c5fff; margin-top: 0.75rem; }
.section-header { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: #4a4862; margin-bottom: 1.2rem; padding-bottom: 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.05); }
.results-header { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #e8e6f0; margin: 2rem 0 1.2rem; }
.method-header { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: #e8e6f0; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; text-align: center; }
.clip-header { background: rgba(139, 111, 255, 0.15); border: 1px solid rgba(139, 111, 255, 0.3); }
.tfidf-header { background: rgba(255, 165, 0, 0.15); border: 1px solid rgba(255, 165, 0, 0.3); }
.bm25-header { background: rgba(0, 200, 150, 0.15); border: 1px solid rgba(0, 200, 150, 0.3); }
.blip-header { background: rgba(255, 100, 100, 0.15); border: 1px solid rgba(255, 100, 100, 0.3); }
.align-header { background: rgba(0, 180, 255, 0.15); border: 1px solid rgba(0, 180, 255, 0.3); }
.info-box { background: rgba(139, 111, 255, 0.07); border: 1px solid rgba(139, 111, 255, 0.18); border-radius: 12px; padding: 1rem 1.25rem; font-size: 0.85rem; color: #a09ac8; line-height: 1.6; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Demo vs. full dataset ─────────────────────────────────────────────────────
# Cloud deploy ships a compact 300-image subset (embeddings_demo/ + data_demo/,
# ~59 MB). Locally the full 8091-image set lives in embeddings/ + data/. Pick
# whichever exists, preferring the demo set on Streamlit Cloud.
EMBEDDINGS_DIR   = "embeddings_demo" if os.path.isdir("embeddings_demo") else "embeddings"
IMAGE_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npz")
TEXT_EMBED_PATH  = os.path.join(EMBEDDINGS_DIR, "text_embeddings.npz")
META_PATH        = os.path.join(EMBEDDINGS_DIR, "metadata.npz")
BLIP_IMG_PATH    = os.path.join(EMBEDDINGS_DIR, "blip_image_embeddings.npz")
BLIP_TXT_PATH    = os.path.join(EMBEDDINGS_DIR, "blip_text_embeddings.npz")
ALIGN_IMG_PATH   = os.path.join(EMBEDDINGS_DIR, "align_image_embeddings.npz")
ALIGN_TXT_PATH   = os.path.join(EMBEDDINGS_DIR, "align_text_embeddings.npz")

def resolve_image_path(path: str) -> str:
    """Map a stored gallery path onto the image directory that actually exists.

    metadata.npz stores paths like `data/Flickr8k_Dataset/...`. On the cloud
    demo the images live under `data_demo/`; locally they live under `data/`.
    Also normalises any stray Windows backslashes to POSIX for Linux safety.
    """
    p = path.replace("\\", "/")
    if p.startswith("data/"):
        demo_p = "data_demo/" + p[len("data/"):]
        if os.path.isfile(demo_p):
            return demo_p
    return p

# ── HATAO PURANA BLOCK AUR YEH NAYA METER PASTE KARO ──────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    import os
    # 1. Base model architecture. If the full fine-tuned checkpoint is on
    # disk, skip the 150 MB openai download — we're about to overwrite
    # every weight anyway, and the local HF cache may already contain a
    # stale/wrong file (e.g. the fine-tuned .pt accidentally placed
    # under the openai repo's snapshot dir) that would otherwise be
    # loaded and break the forward pass.
    full_checkpoint = "my_finetuned_clip.pt"
    light_checkpoint = "light_clip_heads.pt"
    if os.path.exists(full_checkpoint):
        print(f"[load_model] Skipping openai download — {full_checkpoint} will replace all weights.")
        model, preprocess = load_clip_model("ViT-B/32", pretrained=None)
    else:
        model, preprocess = load_clip_model("ViT-B/32", pretrained="openai")

    # 2. Query encoder MUST match the checkpoint the gallery embeddings were
    # built with (build_embeddings.py uses my_finetuned_clip.pt), otherwise
    # query and gallery live in different embedding spaces and every retrieval
    # tab degrades. Prefer the full checkpoint locally; fall back to the
    # lightweight Kaggle heads on the cloud deploy where the 577 MB file
    # isn't committed.
    if os.path.exists(full_checkpoint):
        print(f"Loading fine-tuned CLIP weights from {full_checkpoint}...")
        state_dict = torch.load(full_checkpoint, map_location='cpu', weights_only=True)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # .pt files are saved in fp32; the in-memory model may be fp16
        # (CLIP_FP16=1, default — keeps RSS under the 1 GB Cloud sandbox
        # limit). Cast the state dict to the model's dtype to avoid a
        # "Found dtype Float vs Half" error from load_state_dict.
        model_param = next(model.parameters())
        state_dict = {k: v.to(dtype=model_param.dtype) for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Fine-tuned CLIP weights injected successfully!")
    elif os.path.exists(light_checkpoint):
        print(f"Loading lightweight fine-tuned CLIP weights from {light_checkpoint}...")
        state_dict = torch.load(light_checkpoint, map_location='cpu', weights_only=True)
        # Srf projection heads badalne ke liye strict=False zaroori hai!
        # Same dtype-alignment dance as above.
        model_param = next(model.parameters())
        state_dict = {k: v.to(dtype=model_param.dtype) for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Lightweight CLIP weights injected successfully!")
    else:
        print("⚠️ No fine-tuned checkpoint found! Using default OpenAI weights.")

    return model, preprocess

@st.cache_resource(show_spinner=False)
def load_blip():
    import torch
    import os
    # Base model configuration load karega
    model, preprocess, tokenizer = load_blip_model()

    checkpoint_path = "light_blip_heads.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading lightweight fine-tuned BLIP weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        # Overwriting dynamic heads safely. Align dtypes with the (possibly
        # fp16) in-memory model — see load_model() for the same fix.
        model_param = next(model.parameters())
        state_dict = {k: v.to(dtype=model_param.dtype) for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Lightweight BLIP weights injected successfully!")
    else:
        print("⚠️ light_blip_heads.pt not found! Using default weights.")

    return model, preprocess, tokenizer

@st.cache_resource(show_spinner=False)
def load_align():
    import torch
    import os
    # Base configuration download from hub
    model, preprocess, tokenizer = load_align_model()

    checkpoint_path = "light_align_heads.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading lightweight fine-tuned ALIGN weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        # Heavy backbone safe rakh kar mapping heads deploy karna
        # (same dtype-alignment dance as load_model / load_blip).
        model_param = next(model.parameters())
        state_dict = {k: v.to(dtype=model_param.dtype) for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Lightweight ALIGN weights injected successfully!")
    else:
        print("⚠️ light_align_heads.pt not found! Using default weights.")

    return model, preprocess, tokenizer

# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_embeddings():
    img_data = load_embeddings(IMAGE_EMBED_PATH)
    txt_data = load_embeddings(TEXT_EMBED_PATH)
    meta     = np.load(META_PATH, allow_pickle=True)
    return (
        img_data["embeddings"],
        txt_data["embeddings"],
        meta["image_paths"].tolist(),
        meta["captions"].tolist(),
        meta["image_names"].tolist(),
    )

@st.cache_data(show_spinner=False)
def load_comparison_embeddings():
    blip_img  = np.load(BLIP_IMG_PATH,  allow_pickle=True)["embeddings"]
    blip_txt  = np.load(BLIP_TXT_PATH,  allow_pickle=True)["embeddings"]
    align_img = np.load(ALIGN_IMG_PATH, allow_pickle=True)["embeddings"]
    align_txt = np.load(ALIGN_TXT_PATH, allow_pickle=True)["embeddings"]
    return blip_img, blip_txt, align_img, align_txt

@st.cache_resource(show_spinner=False)
def load_searchers(captions):
    return TFIDFSearcher(captions), BM25Searcher(captions)

def embeddings_ready():
    return all(os.path.isfile(p) for p in [IMAGE_EMBED_PATH, TEXT_EMBED_PATH, META_PATH])

def comparison_embeddings_ready():
    return all(os.path.isfile(p) for p in [BLIP_IMG_PATH, BLIP_TXT_PATH, ALIGN_IMG_PATH, ALIGN_TXT_PATH])

# ── ONNX Runtime text encoder (optional fast path) ───────────────────────────
# onnxruntime is intentionally NOT in requirements.txt (cloud deploys skip it),
# so it must NEVER be imported at module level. Everything below degrades
# gracefully to the PyTorch path when onnxruntime or the onnx/ files are absent.
ONNX_TEXT_FP32 = os.path.join("onnx", "clip_text.onnx")
ONNX_TEXT_INT8 = os.path.join("onnx", "clip_int8.onnx")

def onnx_text_available() -> bool:
    """True only if onnxruntime is importable AND the FP32 text model exists.
    Uses find_spec so we never actually import onnxruntime here."""
    return (
        importlib.util.find_spec("onnxruntime") is not None
        and os.path.isfile(ONNX_TEXT_FP32)
    )

@st.cache_resource(show_spinner=False)
def load_onnx_text_session(model_path: str):
    import onnxruntime as ort   # lazy import — only reached when toggle is ON
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

@st.cache_resource(show_spinner=False)
def load_onnx_tokenizer():
    import open_clip
    return open_clip.get_tokenizer("ViT-B-32")

def encode_single_text_onnx(query: str, model_path: str) -> np.ndarray:
    """Encode one text query via onnxruntime. Same contract as
    encode_single_text(): returns an L2-normalised float32 (1, 512) array,
    so retrieval scores are directly comparable with the PyTorch path."""
    session = load_onnx_text_session(model_path)
    tokenizer = load_onnx_tokenizer()
    tokens = tokenizer([query]).numpy().astype(np.int64)
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: tokens})[0].astype(np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return embedding / norms

def score_bar(score, color="#7c5fff"):
    pct = min(int(abs(score) * 100), 100)
    return f"""
    <div class="score-bar-wrap">
        <div class="score-label">Similarity</div>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;background:{color};"></div></div>
        <div class="score-value" style="color:{color};">{score:.4f}</div>
    </div>"""

# ── Session state ─────────────────────────────────────────────────────────────
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "model_history" not in st.session_state:
    st.session_state.model_history = []
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Vision · Language · Alignment</div>
    <h1 class="hero-title">Semantic Image Search</h1>
    <p class="hero-sub">Cross-modal retrieval — Compare CLIP, BLIP and ALIGN vision-language models.</p>
    <span class="pill-badge">CLIP ViT-B/32</span>
    <span class="pill-badge">BLIP ViT-L/14</span>
    <span class="pill-badge">ALIGN ViT-H/14</span>
    <span class="pill-badge">Flickr8k</span>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not embeddings_ready():
    st.markdown('<div class="info-box">⚠️ Run <code>python build_embeddings.py</code> first.</div>', unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading CLIP model ..."):
    model, preprocess = load_model()
with st.spinner("Loading embeddings ..."):
    img_embeddings, txt_embeddings, image_paths, captions, image_names = load_all_embeddings()
with st.spinner("Building search indexes ..."):
    tfidf_searcher, bm25_searcher = load_searchers(tuple(captions))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_text, tab_image, tab_compare, tab_models, tab_i2i = st.tabs([
    "🔤  Text → Images",
    "🖼  Image → Captions",
    "⚖️  CLIP vs TF-IDF vs BM25",
    "🤖  CLIP vs BLIP vs ALIGN",
    "🔄  Image → Images",
])

# ── TAB 1 — Text to Image ─────────────────────────────────────────────────────
with tab_text:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Try an example</div>", unsafe_allow_html=True)
    ex_cols = st.columns(4)
    examples = ["a dog on the beach", "children playing football", "a woman cooking", "snow covered mountains"]
    for i, (col, ex) in enumerate(zip(ex_cols, examples)):
        with col:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.query_text = ex

    if st.session_state.search_history:
        st.markdown("<div class='section-header'>Recent Searches</div>", unsafe_allow_html=True)
        h_cols = st.columns(5)
        for i, h in enumerate(st.session_state.search_history[:10]):
            with h_cols[i % 5]:
                if st.button(h, key=f"hist_{i}"):
                    # Fill the query box AND auto-run the search on the rerun.
                    # Safe because query_text/auto_search are plain state keys
                    # (not widget keys) set before their consumers instantiate.
                    st.session_state.query_text = h
                    st.session_state.auto_search = True
                    st.rerun()

    col_input, col_gap, col_k = st.columns([4, 0.3, 1.5])
    with col_input:
        st.markdown("<div class='section-header'>Search Query</div>", unsafe_allow_html=True)
        text_query = st.text_input("text_query", value=st.session_state.query_text, placeholder="e.g. a dog running on the beach", label_visibility="collapsed")
    with col_k:
        st.markdown("<div class='section-header'>Results</div>", unsafe_allow_html=True)
        top_k_text = st.slider("top_k_text", min_value=1, max_value=10, value=5, label_visibility="collapsed")

    # ── ⚡ ONNX mode toggle (sidebar is hidden by CSS, so it lives here) ──────
    onnx_ok = onnx_text_available()
    col_onnx, col_prec, _ = st.columns([1.4, 2.2, 2.4])
    with col_onnx:
        use_onnx = st.toggle(
            "⚡ ONNX mode",
            value=False,
            disabled=not onnx_ok,
            key="onnx_mode",
            help="Encode the text query with ONNX Runtime instead of PyTorch (faster on CPU).",
        )
    with col_prec:
        onnx_precision = "FP32"
        if onnx_ok and use_onnx:
            onnx_precision = st.radio(
                "onnx_precision",
                options=["FP32", "INT8"] if os.path.isfile(ONNX_TEXT_INT8) else ["FP32"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key="onnx_precision",
            )
    if not onnx_ok:
        st.info("⚡ ONNX mode unavailable (onnxruntime not installed or onnx/ models missing) — using PyTorch.", icon="ℹ️")

    # auto_search is set by the Recent Searches buttons: consume it exactly once
    # so a history click both fills the box and runs the search immediately.
    auto_search = st.session_state.pop("auto_search", False)
    if (st.button("Search Images", key="search_text") or auto_search) and text_query.strip():
        with st.spinner("Searching ..."):
            encode_via = "PyTorch"
            t0 = time.perf_counter()
            if use_onnx and onnx_ok:
                try:
                    onnx_path = ONNX_TEXT_INT8 if onnx_precision == "INT8" else ONNX_TEXT_FP32
                    # Warm the cached session/tokenizer first so the latency
                    # readout measures pure encode time, not one-off model load.
                    load_onnx_text_session(onnx_path)
                    load_onnx_tokenizer()
                    t0 = time.perf_counter()
                    query_embedding = encode_single_text_onnx(text_query.strip(), onnx_path)
                    encode_via = f"ONNX Runtime ({onnx_precision})"
                except Exception as exc:
                    st.warning(f"ONNX encoding failed ({type(exc).__name__}) — falling back to PyTorch.")
                    t0 = time.perf_counter()
                    query_embedding = encode_single_text(text_query.strip(), model)
            else:
                query_embedding = encode_single_text(text_query.strip(), model)
            encode_ms = (time.perf_counter() - t0) * 1000.0
            results = text_to_image_search(query_embedding, img_embeddings, image_paths, top_k=top_k_text)
        st.caption(f"Query encoded in {encode_ms:.1f} ms via {encode_via}")
        if text_query.strip() not in st.session_state.search_history:
            st.session_state.search_history.insert(0, text_query.strip())
        st.session_state.search_history = st.session_state.search_history[:10]
        st.markdown(f"<div class='results-header'>Top {len(results)} results for <em>\"{text_query}\"</em></div>", unsafe_allow_html=True)
        cols = st.columns(min(len(results), 5))
        for col, result in zip(cols, results):
            with col:
                rp = resolve_image_path(result.image_path)
                if os.path.isfile(rp):
                    st.image(Image.open(rp).convert("RGB"), width="stretch")
                    st.markdown(score_bar(result.score), unsafe_allow_html=True)

# ── TAB 2 — Image to Caption ──────────────────────────────────────────────────
with tab_image:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_up, col_gap2, col_k2 = st.columns([4, 0.3, 1.5])
    with col_up:
        st.markdown("<div class='section-header'>Upload Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    with col_k2:
        st.markdown("<div class='section-header'>Results</div>", unsafe_allow_html=True)
        top_k_image = st.slider("top_k_image", min_value=1, max_value=10, value=5, label_visibility="collapsed")

    if uploaded_file is not None:
        query_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        c_prev, c_spacer = st.columns([1.5, 4])
        with c_prev:
            st.image(query_image, caption="Query image", width="stretch")
        if st.button("Find Captions", key="search_image"):
            with st.spinner("Searching ..."):
                img_query_embedding = encode_single_image(query_image, model, preprocess)
                results = image_to_text_search(img_query_embedding, txt_embeddings, captions, top_k=top_k_image)
            st.markdown(f"<div class='results-header'>Top {len(results)} matching captions</div>", unsafe_allow_html=True)
            for i, result in enumerate(results):
                st.markdown(f"""
                <div class="caption-card">
                    <div class="caption-rank">#{i+1} match</div>
                    <div class="caption-text">"{result.caption}"</div>
                    <div class="caption-score">Score: {result.score:.4f}</div>
                </div>""", unsafe_allow_html=True)

# ── TAB 3 — CLIP vs TF-IDF vs BM25 ───────────────────────────────────────────
with tab_compare:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        🔬 <strong>Search Method Comparison</strong> — CLIP (semantic AI) vs TF-IDF and BM25 (keyword matching).
    </div>""", unsafe_allow_html=True)

    compare_query = st.text_input("compare_query", placeholder="e.g. a dog running on the beach", label_visibility="collapsed", key="compare_input")

    if st.button("Compare Methods", key="compare_btn") and compare_query.strip():
        with st.spinner("Running all 3 methods ..."):
            clip_emb      = encode_single_text(compare_query.strip(), model)
            clip_results  = text_to_image_search(clip_emb, img_embeddings, image_paths, top_k=5)
            tfidf_results = tfidf_searcher.search(compare_query.strip(), top_k=5)
            bm25_results  = bm25_searcher.search(compare_query.strip(), top_k=5)

        meta = np.load(META_PATH, allow_pickle=True)
        cap_img_names = meta["caption_image_names"].tolist()
        all_img_paths = meta["image_paths"].tolist()
        all_img_names = meta["image_names"].tolist()

        def get_img(idx):
            name = cap_img_names[idx]
            if name in all_img_names:
                return resolve_image_path(all_img_paths[all_img_names.index(name)])
            return ""

        st.markdown(f"<div class='results-header'>Results for: <em>\"{compare_query}\"</em></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='method-header clip-header'>🧠 CLIP — Semantic AI</div>", unsafe_allow_html=True)
            st.markdown("<small style='color:#6b6884'>Understands meaning — finds relevant images even without exact keywords</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for r in clip_results:
                rp = resolve_image_path(r.image_path)
                if os.path.isfile(rp):
                    st.image(Image.open(rp).convert("RGB"), width=190)
                    st.markdown(score_bar(r.score, "#7c5fff"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='method-header tfidf-header'>📄 TF-IDF — Keyword</div>", unsafe_allow_html=True)
            st.markdown("<small style='color:#6b6884'>Exact keyword matching — no understanding of meaning</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for r in tfidf_results:
                p = get_img(r.index)
                if p and os.path.isfile(p):
                    st.image(Image.open(p).convert("RGB"), width=190)
                    st.markdown(f"<div style='font-size:0.72rem;color:#6b6884;font-style:italic'>{r.caption[:60]}...</div>", unsafe_allow_html=True)
                    st.markdown(score_bar(r.score, "#ffa500"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='method-header bm25-header'>🔍 BM25 — Keyword+</div>", unsafe_allow_html=True)
            st.markdown("<small style='color:#6b6884'>Smarter keyword ranking — still no semantic understanding</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for r in bm25_results:
                p = get_img(r.index)
                if p and os.path.isfile(p):
                    st.image(Image.open(p).convert("RGB"), width=190)
                    st.markdown(f"<div style='font-size:0.72rem;color:#6b6884;font-style:italic'>{r.caption[:60]}...</div>", unsafe_allow_html=True)
                    st.markdown(score_bar(r.score, "#00c896"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            📊 <strong>Conclusion:</strong> CLIP retrieves semantically relevant images even when exact keywords are absent.
            TF-IDF and BM25 fail on queries where words don't exactly match captions.
        </div>""", unsafe_allow_html=True)

# ── TAB 4 — CLIP vs BLIP vs ALIGN ────────────────────────────────────────────
with tab_models:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        🤖 <strong>Model Comparison</strong> — Compare three state-of-the-art vision-language models on the same query.<br>
        CLIP (ViT-B/32) vs BLIP (ViT-L/14) vs ALIGN (ViT-H/14)
    </div>""", unsafe_allow_html=True)

    if not comparison_embeddings_ready():
        st.warning("⚠️ Run `python build_comparison_embeddings.py` first to build BLIP and ALIGN embeddings.")
        st.stop()

    # ── Heavy-model load with graceful RAM fallback ──────────────────────────
    # BLIP (ViT-L/14 ~0.9 GB) + ALIGN (ViT-H/14 ~3.9 GB) together exceed the
    # 1 GB RAM of Streamlit Community Cloud's free tier. On a local box
    # with 8 GB+ RAM they fit comfortably. We detect which environment we're
    # in and pick a sensible default.
    blip_model = align_model = None
    comparison_models_ok = False  # default: no live BLIP/ALIGN encoding
    blip_status_msg = None
    align_status_msg = None

    # Streamlit Cloud mounts the repo at /mount/src/ — a reliable signal
    # we're in the sandboxed 1 GB environment. On local dev that path
    # doesn't exist.
    _IS_CLOUD = os.path.isdir("/mount/src")

    _try_heavy_env = os.environ.get("TRY_HEAVY_MODELS", "").strip().lower()
    if _try_heavy_env in ("1", "true", "yes", "on"):
        try_heavy_bool = True
    elif _try_heavy_env in ("0", "false", "no", "off"):
        try_heavy_bool = False
    else:
        # No explicit override: default to LOADING on local, SKIPPING on Cloud.
        try_heavy_bool = not _IS_CLOUD

    if not try_heavy_bool:
        if _IS_CLOUD:
            st.info(
                "ℹ️ **Cloud mode** — BLIP/ALIGN live encoding is disabled to stay "
                "under the 1 GB sandbox limit. The comparison below still shows "
                "BLIP/ALIGN results using the *precomputed* gallery embeddings. "
                "To run all three models live, set `TRY_HEAVY_MODELS=1` and run "
                "locally on 8 GB+ RAM.",
                icon="ℹ️",
            )
        else:
            st.info(
                "ℹ️ BLIP/ALIGN live encoding is disabled. Set "
                "`TRY_HEAVY_MODELS=1` in your environment to enable it on "
                "this machine (needs ~3 GB free RAM).",
                icon="ℹ️",
            )
    else:
        # Opt-in path: pre-download + load. Surfaces a clear error if the host
        # can't handle it. This is the only place BLIP/ALIGN are touched.
        with st.status("Preparing BLIP & ALIGN weights ...", expanded=False) as opt_status:
            def _on_progress_opt(repo_id, stage, detail):
                if stage == "cached":
                    opt_status.write(f"✅ {repo_id} — cached")
                elif stage == "downloading":
                    opt_status.write(f"⬇️ {repo_id} — {detail}")
                elif stage == "ready":
                    opt_status.write(f"✅ {repo_id} — ready")
                elif stage == "error":
                    opt_status.write(f"❌ {repo_id} — {detail}")
            try:
                download_models.ensure_models(include_optional=True, progress_cb=_on_progress_opt)
                opt_status.update(label="BLIP & ALIGN weights ready ✓", state="complete")
                comparison_models_ok = True
            except Exception as exc:
                opt_status.update(label="Could not pre-cache BLIP/ALIGN", state="error")
                st.warning(
                    f"⚠️ Could not pre-cache BLIP/ALIGN weights "
                    f"({type(exc).__name__}: {exc}). The comparison tab will "
                    f"only show CLIP results."
                )
                comparison_models_ok = False

        if comparison_models_ok:
            # Load BLIP first. Wrap each load in its own try/except so a
            # failure on one model doesn't leave the other half-initialised.
            # The Linux OOM killer can still bypass this when the over-
            # allocation happens entirely in C (PyTorch's allocator), but
            # capping RLIMIT_AS first means most failures show up as a
            # catchable MemoryError before the kernel steps in.
            try:
                import resource  # type: ignore  # POSIX-only
                _HAS_RESOURCE = True
            except ImportError:
                _HAS_RESOURCE = False

            def _cap_rss_and_load(loader, name, cap_mb=700):
                """Run loader() under a virtual-memory cap so OOM becomes MemoryError."""
                old_soft = old_hard = None
                if _HAS_RESOURCE:
                    try:
                        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
                        cap = cap_mb * 1024 * 1024
                        new_soft = min(cap, old_hard) if old_hard != resource.RLIM_INFINITY else cap
                        resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_hard))
                    except (ValueError, OSError):
                        old_soft = old_hard = None
                try:
                    return loader(), None
                except (MemoryError, RuntimeError, OSError) as exc:
                    return None, f"{type(exc).__name__}: {exc}"
                finally:
                    if _HAS_RESOURCE and old_soft is not None:
                        try:
                            resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
                        except (ValueError, OSError):
                            pass

            with st.spinner("Loading BLIP model ..."):
                result = _cap_rss_and_load(load_blip, "BLIP", cap_mb=700)
                if result[0] is not None:
                    blip_model, blip_preprocess, blip_tokenizer = result[0]
                else:
                    st.warning(
                        f"⚠️ BLIP failed to load ({result[1]}). "
                        "ALIGN is even larger so it's being skipped too. "
                        "**CLIP results still work** below."
                    )
                    comparison_models_ok = False

            if blip_model is not None:
                with st.spinner("Loading ALIGN model ..."):
                    result = _cap_rss_and_load(load_align, "ALIGN", cap_mb=950)
                    if result[0] is not None:
                        align_model, align_preprocess, align_tokenizer = result[0]
                    else:
                        st.warning(
                            f"⚠️ ALIGN failed to load ({result[1]}). "
                            "**CLIP and BLIP results still work** below."
                        )

            with st.spinner("Loading BLIP model ..."):
                result = _cap_rss_and_load(load_blip, "BLIP", cap_mb=700)
                if result[0] is not None:
                    blip_model, blip_preprocess, blip_tokenizer = result[0]
                else:
                    blip_status_msg = (
                        f"⚠️ BLIP failed to load ({result[1]}). "
                        "The ViT-L/14 backbone is too large for this environment. "
                        "ALIGN is even larger so it's being skipped too. "
                        "**CLIP results still work** below."
                    )
                    st.warning(blip_status_msg)
                    comparison_models_ok = False

            # Only try ALIGN if BLIP succeeded — keeps the memory budget honest.
            if blip_model is not None:
                with st.spinner("Loading ALIGN model ..."):
                    result = _cap_rss_and_load(load_align, "ALIGN", cap_mb=950)
                    if result[0] is not None:
                        align_model, align_preprocess, align_tokenizer = result[0]
                    else:
                        align_status_msg = (
                            f"⚠️ ALIGN failed to load ({result[1]}). "
                            "The ViT-H/14 backbone (~2.5 GB) is too large for this "
                            "environment. **CLIP and BLIP results still work** below."
                        )
                        st.warning(align_status_msg)
    with st.spinner("Loading comparison embeddings ..."):
        blip_img_emb, blip_txt_emb, align_img_emb, align_txt_emb = load_comparison_embeddings()

    # Google style search with dropdown history
    st.markdown("""
    <style>
    .search-wrap { position: relative; margin-bottom: 0.5rem; }
    .history-dropdown {
        background: #12111f;
        border: 1px solid rgba(139,111,255,0.3);
        border-radius: 12px;
        overflow: hidden;
        margin-top: 0.25rem;
    }
    .history-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 1rem;
        font-size: 0.88rem;
        color: #ccc9e0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        cursor: pointer;
        transition: background 0.15s;
    }
    .history-item:last-child { border-bottom: none; }
    .history-item:hover { background: rgba(139,111,255,0.1); }
    .history-icon { color: #4a4862; margin-right: 0.6rem; font-size: 0.8rem; }
    .clear-all {
        text-align: center;
        padding: 0.5rem;
        font-size: 0.75rem;
        color: #7c5fff;
        cursor: pointer;
        border-top: 1px solid rgba(255,255,255,0.05);
        background: rgba(124,95,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    model_query = st.text_input("model_query", placeholder="🔍  e.g. a dog running on the beach", label_visibility="collapsed", key="model_input")

    if st.session_state.model_history:
        history_html = '<div class="history-dropdown">'
        for h in st.session_state.model_history[:5]:
            history_html += f'''
            <div class="history-item">
                <span><span class="history-icon">🕐</span>{h}</span>
            </div>'''
        history_html += '</div>'
        st.markdown(history_html, unsafe_allow_html=True)

        col_clear, col_space = st.columns([1, 5])
        with col_clear:
            if st.button("🗑 Clear History", key="clear_model_hist"):
                st.session_state.model_history = []
                st.rerun()





    if st.button("Compare Models", key="model_btn") and model_query.strip():
        with st.spinner("Running all 3 models ..."):
            # CLIP (always available)
            clip_emb     = encode_single_text(model_query.strip(), model)
            clip_results = text_to_image_search(clip_emb, img_embeddings, image_paths, top_k=5)
            # BLIP / ALIGN — only if the heavy models loaded in this environment
            if comparison_models_ok and blip_model is not None:
                blip_emb     = encode_single_text_model(model_query.strip(), blip_model, blip_tokenizer)
                blip_results = text_to_image_search(blip_emb, blip_img_emb, image_paths, top_k=5)
            else:
                blip_results = []
            if comparison_models_ok and align_model is not None:
                align_emb     = encode_single_text_model(model_query.strip(), align_model, align_tokenizer)
                align_results = text_to_image_search(align_emb, align_img_emb, image_paths, top_k=5)
            else:
                align_results = []
        if model_query.strip() not in st.session_state.model_history:
            st.session_state.model_history.insert(0, model_query.strip())
        st.session_state.model_history = st.session_state.model_history[:5]

        st.markdown(f"<div class='results-header'>Model results for: <em>\"{model_query}\"</em></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='method-header clip-header'>🧠 CLIP — ViT-B/32</div>", unsafe_allow_html=True)
            st.markdown("<small style='color:#6b6884'>OpenAI · 400M image-text pairs · Fast & efficient</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for r in clip_results:
                rp = resolve_image_path(r.image_path)
                if os.path.isfile(rp):
                    st.image(Image.open(rp).convert("RGB"), width=190)
                    st.markdown(score_bar(r.score, "#7c5fff"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='method-header blip-header'>🔴 BLIP — ViT-L/14</div>", unsafe_allow_html=True)
            st.markdown("<small style='color:#6b6884'>Salesforce · Larger model · Better detail understanding</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if blip_results:
                for r in blip_results:
                    rp = resolve_image_path(r.image_path)
                    if os.path.isfile(rp):
                        st.image(Image.open(rp).convert("RGB"), width=190)
                        st.markdown(score_bar(r.score, "#ff6464"), unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='info-box' style='font-size:0.8rem; padding:0.8rem;'>"
                    "⏭ <strong>BLIP skipped on this deploy</strong> — the ViT-L/14 "
                    "backbone (~900 MB) plus the running CLIP model exceed the 1 GB "
                    "Streamlit Cloud sandbox. See the Precision@K chart at the "
                    "bottom of the page for the offline three-model numbers, or "
                    "run locally with `TRY_HEAVY_MODELS=1`."
                    "</div>",
                    unsafe_allow_html=True,
                )

        with col3:
            st.markdown("<div class='method-header align-header'>🔵 ALIGN — ViT-H/14</div>", unsafe_allow_html=True)
            st.markdown("<small style='color:#6b6884'>Google · 1.8B image-text pairs · Largest scale training</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if align_results:
                for r in align_results:
                    rp = resolve_image_path(r.image_path)
                    if os.path.isfile(rp):
                        st.image(Image.open(rp).convert("RGB"), width=190)
                        st.markdown(score_bar(r.score, "#00b4ff"), unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='info-box' style='font-size:0.8rem; padding:0.8rem;'>"
                    "⏭ <strong>ALIGN skipped on this deploy</strong> — the ViT-H/14 "
                    "backbone (~2.5 GB) is the largest of the three. See the "
                    "Precision@K chart at the bottom of the page for the offline "
                    "three-model numbers, or run locally with `TRY_HEAVY_MODELS=1`."
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            📊 <strong>What this shows:</strong><br><br>
            • <strong>CLIP (ViT-B/32)</strong> — Fast, lightweight, good general retrieval<br>
            • <strong>BLIP (ViT-L/14)</strong> — Larger model, better at fine-grained details<br>
            • <strong>ALIGN (ViT-H/14)</strong> — Trained on massive data, strongest on complex queries<br><br>
            Larger models generally retrieve more semantically accurate results at the cost of speed.
        </div>""", unsafe_allow_html=True)
        # yha se paste kr rha hu #

# ── TAB 5 — Image to Image (reverse search) ───────────────────────────────────
with tab_i2i:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        🔄 <strong>Image-to-Image reverse search</strong> — upload a query image, encode it with CLIP,
        then rank the Flickr8k gallery by cosine similarity against precomputed <code>image_embeddings</code>.<br><br>
        ⚠️ <strong>Gallery is Flickr8k (natural images only).</strong> For out-of-distribution
        images like spectrograms, charts, or diagrams, similarity scores will be low and results
        approximate — this is the documented <em>"Relative Semantic Proxy effect"</em>.<br>
        💡 <strong>Tip:</strong> add a short text context below (e.g. <code>bird</code>,
        <code>spectrogram</code>, <code>chart</code>) to switch to a <strong>text-based</strong>
        search, which works much better for OOD images.
    </div>""", unsafe_allow_html=True)

    col_up_i2i, col_gap_i2i, col_k_i2i = st.columns([4, 0.3, 1.5])
    with col_up_i2i:
        st.markdown("<div class='section-header'>Upload Query Image</div>", unsafe_allow_html=True)
        i2i_file = st.file_uploader(
            "i2i_upload",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="i2i_uploader",
        )
    with col_k_i2i:
        st.markdown("<div class='section-header'>Results</div>", unsafe_allow_html=True)
        top_k_i2i = st.slider(
            "top_k_i2i", min_value=1, max_value=10, value=5, label_visibility="collapsed", key="i2i_k"
        )

    if i2i_file is not None:
        i2i_query = Image.open(io.BytesIO(i2i_file.read())).convert("RGB")
        c_prev_i2i, _ = st.columns([1.5, 4])
        with c_prev_i2i:
            st.image(i2i_query, caption="Query image", width="stretch")

        # Optional text refinement — switches the search from image→image to text→image.
        # This is the right tool for OOD queries (spectrograms, charts, etc.) where the
        # raw image embedding can't find a good match in the Flickr8k gallery.
        i2i_text = st.text_input(
            "Optional text context (switches to text search)",
            placeholder="e.g. 'bird', 'spectrogram', 'chart'",
            key="i2i_text",
            help="Leave empty for pure image search. Add a word/phrase to switch to text search — much better for OOD images.",
        )

        if st.button("Find Similar Images", key="search_i2i"):
            i2i_text_clean = (i2i_text or "").strip()
            if i2i_text_clean:
                # Text-based search (better for OOD / spectrograms)
                with st.spinner(f"Encoding text \"{i2i_text_clean}\" & searching gallery ..."):
                    i2i_embedding = encode_single_text(i2i_text_clean, model)
                    i2i_results = image_to_image_search(
                        i2i_embedding, img_embeddings, image_paths, top_k=top_k_i2i
                    )
                search_mode = f'text "{i2i_text_clean}"'
            else:
                # Pure image-based search (original behaviour)
                with st.spinner("Encoding query image & searching gallery ..."):
                    i2i_embedding = encode_single_image(i2i_query, model, preprocess)
                    i2i_results = image_to_image_search(
                        i2i_embedding, img_embeddings, image_paths, top_k=top_k_i2i
                    )
                search_mode = "image"

            st.markdown(
                f"<div class='results-header'>Top {len(i2i_results)} matches via {search_mode} search</div>",
                unsafe_allow_html=True,
            )

            # Low-score warning: if image-search returned noise-floor scores (<0.55),
            # the query is likely OOD for the Flickr8k gallery — suggest text refinement.
            top_score = max((r.score for r in i2i_results), default=0.0)
            if search_mode == "image" and top_score < 0.55:
                st.markdown(f"""
                <div class="info-box" style="background:rgba(255,180,50,0.08);border-color:#ffb432;">
                    ⚠️ <strong>Low similarity detected</strong> (top score = {top_score:.3f}).
                    The query image appears <em>out-of-distribution</em> for the Flickr8k gallery
                    (e.g. spectrogram, chart, diagram). Try the <strong>Image→Caption</strong> tab
                    to get a text description, or add a text context above (e.g. "bird", "spectrum")
                    to switch to a text-based search.
                </div>""", unsafe_allow_html=True)

            cols_i2i = st.columns(min(len(i2i_results), 5))
            for col, result in zip(cols_i2i, i2i_results):
                with col:
                    rp = resolve_image_path(result.image_path)
                    if os.path.isfile(rp):
                        st.image(
                            Image.open(rp).convert("RGB"),
                            width="stretch",
                        )
                    st.markdown(score_bar(result.score), unsafe_allow_html=True)

# ── Precision@K Chart ─────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Precision@K Evaluation Results</div>", unsafe_allow_html=True)

PRECISION_PATH = os.path.join(EMBEDDINGS_DIR, "precision_scores.json")

if not os.path.isfile(PRECISION_PATH):
    st.markdown("""
    <div class="info-box">
        ⚠️ No evaluation results found. Run <code>python evaluate.py</code> first, then refresh.
    </div>""", unsafe_allow_html=True)
else:
    import json
    import plotly.graph_objects as go

    with open(PRECISION_PATH) as f:
        prec = json.load(f)

    k          = prec["k"]
    queries    = prec["queries"]
    clip_sc    = prec["clip"]
    blip_sc    = prec["blip"]
    align_sc   = prec["align"]
    avg_clip   = prec["avg_clip"]
    avg_blip   = prec["avg_blip"]
    avg_align  = prec["avg_align"]

    # ── Average summary cards ──────────────────────────────────────────────────
    st.markdown(f"<div style='margin-bottom:0.5rem;font-size:0.8rem;color:#6b6884'>Average Precision@{k} across {len(queries)} test queries</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    def avg_card(col, label, value, color, bg):
        col.markdown(f"""
        <div style="background:{bg};border:1px solid {color};border-radius:14px;padding:1.2rem;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:{color};margin-bottom:0.5rem">{label}</div>
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{color}">{value*100:.1f}%</div>
            <div style="font-size:0.72rem;color:#6b6884;margin-top:0.3rem">Precision@{k}</div>
        </div>""", unsafe_allow_html=True)

    avg_card(c1, "CLIP · ViT-B/32",  avg_clip,  "#7c5fff", "rgba(124,95,255,0.08)")
    avg_card(c2, "BLIP · ViT-L/14",  avg_blip,  "#ff6464", "rgba(255,100,100,0.08)")
    avg_card(c3, "ALIGN · ViT-H/14", avg_align, "#00b4ff", "rgba(0,180,255,0.08)")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Average bar chart ──────────────────────────────────────────────────────
    fig_avg = go.Figure(go.Bar(
        x=["CLIP (ViT-B/32)", "BLIP (ViT-L/14)", "ALIGN (ViT-H/14)"],
        y=[avg_clip, avg_blip, avg_align],
        marker_color=["#7c5fff", "#ff6464", "#00b4ff"],
        text=[f"{v*100:.1f}%" for v in [avg_clip, avg_blip, avg_align]],
        textposition="outside",
        textfont=dict(family="Syne", size=13, color="#e8e6f0"),
        width=0.45,
    ))
    fig_avg.update_layout(
        title=dict(text=f"Average Precision@{k} — Model Comparison", font=dict(family="Syne", size=15, color="#e8e6f0"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8a87a0"),
        yaxis=dict(tickformat=".0%", range=[0, max(avg_clip, avg_blip, avg_align) + 0.18],
                   gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
        xaxis=dict(showgrid=False),
        margin=dict(t=50, b=20, l=20, r=20),
        height=320,
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # ── Per-query grouped bar chart ────────────────────────────────────────────
    short_queries = [q[:30] + "…" if len(q) > 30 else q for q in queries]

    fig_per = go.Figure()
    fig_per.add_trace(go.Bar(name="CLIP",  x=short_queries, y=clip_sc,  marker_color="#7c5fff"))
    fig_per.add_trace(go.Bar(name="BLIP",  x=short_queries, y=blip_sc,  marker_color="#ff6464"))
    fig_per.add_trace(go.Bar(name="ALIGN", x=short_queries, y=align_sc, marker_color="#00b4ff"))
    fig_per.update_layout(
        title=dict(text=f"Per-Query Precision@{k}", font=dict(family="Syne", size=15, color="#e8e6f0"), x=0),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8a87a0"),
        yaxis=dict(tickformat=".0%", range=[0, 1.12],
                   gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
        xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e6f0")),
        margin=dict(t=50, b=120, l=20, r=20),
        height=420,
    )
    st.plotly_chart(fig_per, use_container_width=True)

    # ── Winner callout ─────────────────────────────────────────────────────────
    best = max(zip(["CLIP", "BLIP", "ALIGN"], [avg_clip, avg_blip, avg_align]), key=lambda x: x[1])
    st.markdown(f"""
    <div class="info-box">
        🏆 <strong>{best[0]}</strong> achieves the highest average Precision@{k} of <strong>{best[1]*100:.1f}%</strong> on these {len(queries)} test queries.<br><br>
        <em>Note:</em> Relevance is determined by keyword overlap (≥2 matching content words). 
        For more reliable evaluation, replace <code>is_relevant()</code> in <code>evaluate.py</code> with human-annotated ground truth labels.
    </div>""", unsafe_allow_html=True)

# yha tak paste kra hai #


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#3d3a52;padding-bottom:2rem;">
    CLIP · BLIP · ALIGN · TF-IDF · BM25 · Image→Image · Flickr8k · Streamlit
</div>""", unsafe_allow_html=True)
