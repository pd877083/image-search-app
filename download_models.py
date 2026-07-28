"""
download_models.py — Pre-download CLIP / BLIP / ALIGN weights to a project-local
HuggingFace cache so the deployed app never hits the network at serve time.

Why this exists
---------------
On Streamlit Cloud's free tier (1 GB RAM, no GPU, unauthenticated HF Hub access)
the first request that triggers `open_clip.create_model_and_transforms(...)`
downloads ~150 MB of CLIP weights from HuggingFace. Two failure modes are
common and both look like a "silent crash" because they happen after the
Uvicorn server has already reported "started":

  1. **Rate-limited / network failure** — HF returns 429 or the download
     stalls. open_clip surfaces this as a Python exception, but if it happens
     inside a `st.spinner(...)` the Streamlit worker can be killed before
     the traceback reaches the log.
  2. **OOM at load** — PyTorch + a freshly-downloaded ~600 MB float32
     ViT-B/32 model pushes peak RSS past the 1 GB sandbox limit and the Linux
     OOM killer terminates the process with no Python traceback at all.

This script solves both:

  * Downloads to `./hf_cache/` so the runtime HF_HOME env var points at a
    directory the deploys already have on disk — no network call on first
    user request.
  * Retries on transient failures with exponential backoff (HF rate limits
    often clear within a few seconds).
  * Uses HF_TOKEN from the environment (or Streamlit secrets) to raise rate
    limits dramatically when available.
  * Runs at the very top of `app.py` inside a Streamlit status widget so any
    failure is shown to the user, not swallowed by the worker.

Run standalone:
    python download_models.py
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Do NOT override HF_HOME — we want to use the default cache so:
#   * Local dev picks up weights already at ~/.cache/huggingface/ (instant).
#   * Streamlit Cloud populates ~/.cache/huggingface/ once and the same
#     default location is read by open_clip via huggingface_hub.
# If the user needs a project-local cache for some reason, they can set
# HF_HOME themselves before running.

# (repo_id, expected_file_substring) — only ViT-B/32 is required at startup;
# BLIP/ALIGN are loaded lazily by the comparison tab so we still pre-download
# them but the failure of one shouldn't block the app.
REQUIRED_MODELS = [
    ("timm/vit_base_patch32_clip_224.openai", "open_clip_pytorch_model.bin"),
]

# Lazy / on-demand: only downloaded when the comparison tab is opened.
OPTIONAL_MODELS = [
    ("timm/vit_large_patch14_clip_224.openai", "open_clip_pytorch_model.bin"),
    ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_pytorch_model.bin"),
]


def _hf_token() -> str | None:
    """Read HF token from env or Streamlit secrets if available."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    # Streamlit secrets: only import streamlit here so this script is also
    # usable from the CLI (e.g. for local pre-caching).
    try:
        import streamlit as st  # type: ignore
        token = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
    except Exception:
        token = None
    return token


def _already_cached(repo_id: str, filename_substring: str) -> bool:
    """Return True if the model weights are already present in the default cache.

    Checks both the modern and legacy HF cache layouts:
      * ~/.cache/huggingface/hub/  (current default)
      * ~/.cache/huggingface/       (older)
    """
    needle = filename_substring.lower()
    home = Path.home()
    search_roots = [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and needle in path.name.lower():
                return True
    return False


def _download_with_retry(repo_id: str, filename_substring: str, max_attempts: int = 4) -> None:
    """Download a single repo with exponential backoff on transient failures."""
    from huggingface_hub import snapshot_download, login
    from huggingface_hub.utils import (
        GatedRepoError,
        RepositoryNotFoundError,
    )

    token = _hf_token()
    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            # Non-fatal — anonymous downloads still work, just rate-limited.
            pass

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"[download_models] ({attempt}/{max_attempts}) {repo_id}",
                flush=True,
            )
            # No `cache_dir=` — use the default location so open_clip's
            # later lookup hits the same path. huggingface_hub respects
            # HF_HOME / HF_HUB_CACHE if set, else ~/.cache/huggingface/hub.
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{filename_substring}*", "*.json", "*.txt"],
                token=token,
            )
            print(f"[download_models] ✓ {repo_id} ready", flush=True)
            return
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            # Permanent — no point retrying.
            raise RuntimeError(
                f"Cannot download {repo_id}: {exc}. "
                "If this is a gated repo, set HF_TOKEN in Streamlit secrets."
            ) from exc
        except Exception as exc:  # network blips, 429s, etc.
            last_exc = exc
            wait = min(2 ** attempt, 30)  # 2, 4, 8, 16, 30, 30...
            print(
                f"[download_models] ⚠️ attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}. Retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {repo_id} after {max_attempts} attempts. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    )


def ensure_models(include_optional: bool = False, progress_cb=None) -> dict:
    """
    Make sure all required (and optionally the comparison-tab) model weights
    are present in the default cache.

    Args:
        include_optional: also pre-download BLIP/ALIGN (use this when the
            caller is the model-compare tab).
        progress_cb: optional callable(model_name, status, detail) invoked
            after each step — used by app.py to render a Streamlit status
            widget so the user sees what's happening.

    Returns:
        dict mapping repo_id -> "cached" | "downloaded".
    """
    def _report(name, status, detail=""):
        if progress_cb is not None:
            try:
                progress_cb(name, status, detail)
            except Exception:
                pass

    results: dict[str, str] = {}
    plans = list(REQUIRED_MODELS)
    if include_optional:
        plans += OPTIONAL_MODELS

    for repo_id, filename in plans:
        if _already_cached(repo_id, filename):
            _report(repo_id, "cached", "already on disk")
            results[repo_id] = "cached"
            continue
        _report(repo_id, "downloading", f"~{_approx_size_mb(repo_id)} MB")
        try:
            _download_with_retry(repo_id, filename)
        except Exception as exc:
            _report(repo_id, "error", str(exc))
            raise
        _report(repo_id, "ready", "")
        results[repo_id] = "downloaded"
    return results


def _approx_size_mb(repo_id: str) -> int:
    """Very rough size hint shown in the UI."""
    table = {
        "timm/vit_base_patch32_clip_224.openai": 150,
        "timm/vit_large_patch14_clip_224.openai": 890,
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 2500,
    }
    return table.get(repo_id, 200)


# ── CLI mode ────────────────────────────────────────────────────────────────
# `python download_models.py` works locally too — useful for pre-caching on
# a dev box before pushing to Streamlit Cloud.

def _cli():
    include_opt = "--all" in sys.argv
    print("[download_models] using default cache (~/.cache/huggingface/)")
    print(f"[download_models] include_optional = {include_opt}")
    ensure_models(include_optional=include_opt)
    home = Path.home() / ".cache" / "huggingface"
    if home.exists():
        size_mb = sum(p.stat().st_size for p in home.rglob("*") if p.is_file()) / (1024 * 1024)
        print(f"[download_models] ✓ done. cache size = {size_mb:.0f} MB")


if __name__ == "__main__":
    _cli()
