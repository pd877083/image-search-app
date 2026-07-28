# 🔭 Semantic Image Search — CLIP Cross-Modal Retrieval

A beginner-friendly but professionally structured deep learning project demonstrating **cross-modal retrieval** between images and text using a pretrained **OpenAI CLIP** model.

---

## What This Project Does

| Mode | Input | Output |
|------|-------|--------|
| **Text → Images** | A natural-language query (e.g. *"a dog running on the beach"*) | Top-5 most semantically similar images from Flickr8k |
| **Image → Captions** | An uploaded image | Top-5 most semantically relevant captions from the dataset |

Everything is **retrieval-only** — no generative model is involved. CLIP maps images and text into a shared 512-dimensional embedding space, and cosine similarity does the rest.

---

## Project Structure

```
clip_search/
├── app.py                  ← Streamlit frontend (run this to launch the app)
├── model.py                ← CLIP loading, image & text encoding
├── utils.py                ← Cosine similarity, retrieval logic, data helpers
├── build_embeddings.py     ← One-time script: precomputes & saves embeddings
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── data/                   ← Download Flickr8k here (see Step 3)
│   ├── Flickr8k_Dataset/
│   │   └── Flicker8k_Dataset/   ← ~8 000 .jpg images
│   └── Flickr8k_text/
│       └── Flickr8k.token.txt   ← 40 000 image-caption pairs
│
└── embeddings/             ← Auto-created by build_embeddings.py
    ├── image_embeddings.npz
    ├── text_embeddings.npz
    └── metadata.npz
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.9 – 3.11 | 3.10 recommended |
| RAM | 8 GB+ | Embeddings fit comfortably in memory |
| Disk | ~2 GB free | Dataset (~1.1 GB) + embeddings (~300 MB) |
| GPU | Optional | CPU is fully supported; ViT-B/32 runs well on a laptop |

---

## Step-by-Step Setup

### Step 1 — Open in VS Code

```bash
# Clone or copy the project folder, then open it
code clip_search
```

### Step 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

### Step 3 — Install dependencies

```bash
# 1. Install PyTorch (CPU-only, smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest (includes open_clip_torch — one CLIP library for the whole project)
pip install -r requirements.txt
```

> All CLIP usage (`model.py`, `model_comparison.py`, `train.py`) goes through **`open_clip_torch`** (PyPI). No GitHub `git+https` install is required.

### Step 4 — Download the Flickr8k dataset

1. Go to **https://www.kaggle.com/datasets/adityajn105/flickr8k** and sign in.
2. Click **Download** (ZIP, ~1.1 GB).
3. Extract and arrange the files like this:

```
data/
├── Flickr8k_Dataset/
│   └── Flicker8k_Dataset/    ← all 8 091 .jpg files go here
└── Flickr8k_text/
    └── Flickr8k.token.txt    ← 40 455-line caption file
```

> Alternatively you can use the [official Illinois dataset page](https://illinois.edu/fb/sec/1713398).
> The folder names must match exactly (note the typo *Flicker* in the official archive — keep it).

### Step 5 — Build embeddings (one-time, ~15–30 min on CPU)

```bash
python build_embeddings.py
```

This will:
- Load CLIP `ViT-B/32`
- Encode ~8 000 images in batches of 32
- Encode ~40 000 captions in batches of 256
- Save three `.npz` files to `embeddings/`

You only need to run this **once**. The app loads from disk on every subsequent launch.

### Step 6 — Launch the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## How It Works

```
        ┌─────────────┐      CLIP encoder      ┌──────────────────────┐
        │  Text query  │ ──────────────────────▶│  512-dim embedding   │
        └─────────────┘                         └──────────┬───────────┘
                                                           │  cosine
                                                           │  similarity
        ┌─────────────┐      CLIP encoder      ┌──────────▼───────────┐
        │  8k images  │ ──────────────────────▶│  image embedding DB  │
        └─────────────┘   (precomputed once)   └──────────────────────┘
```

1. **CLIP** (Contrastive Language–Image Pretraining) was trained to pull matching image-text pairs close together in embedding space.
2. We encode all dataset images and captions **once** and save them.
3. At query time we encode only the single query, then rank corpus embeddings by **cosine similarity** — a simple dot product since all vectors are L2-normalised.
4. Top-K indices map back to image paths or caption strings and are displayed in the UI.

---

## Performance Notes

| Setting | Value |
|---------|-------|
| Model | `ViT-B/32` (fastest CLIP variant) |
| Batch size (images) | 32 (reduce to 16 if RAM is tight) |
| Batch size (captions) | 256 |
| Query latency | < 1 second (embedding already on disk) |
| RAM at runtime | ~2–3 GB |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: open_clip` | Run `pip install open_clip_torch==2.24.1` (or `pip install -r requirements.txt`) |
| `FileNotFoundError: embeddings/…` | Run `python build_embeddings.py` first |
| Images not displayed in app | Check that `data/Flickr8k_Dataset/Flicker8k_Dataset/` exists |
| Out of memory during build | Lower `BATCH_SIZE` in `build_embeddings.py` to `8` or `16` |
| Slow on Windows | Make sure you're in the venv and using the CPU wheel of PyTorch |

---

## Key Concepts Demonstrated

- **Vision-Language Alignment** — CLIP's shared embedding space
- **Cross-Modal Retrieval** — querying across modalities (text ↔ image)
- **Zero-Shot Generalisation** — no fine-tuning on Flickr8k, purely pretrained
- **Efficient Nearest-Neighbour Search** — cosine similarity with L2-normalised vectors
- **Batch Processing** — memory-efficient encoding of large datasets on CPU

---

## License

Dataset: Flickr8k is provided for non-commercial research / educational use.  
CLIP / OpenCLIP model weights: [MIT License](https://github.com/mlfoundations/open_clip).  
Project code: MIT.
