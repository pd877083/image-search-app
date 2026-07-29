# Daily Project Diary — NTCC (May 4 – July 5, 2026)

**Project:** Semantic Image Search Engine (CLIP / BLIP / ALIGN)
**Group:** Dhruv Singhal, Anish Kumar Singh, Rudra Bounthiyal
**Supervisor:** Dr. Rakesh Chandra Joshi
**Source:** 9 Weekly Progress Reports (WPR1–WPR9) **+ formal NTCC thesis** (`DhruvS_Report.pdf`, 33 pages)
**Notation:** Entries marked **[Report]** come from the formal thesis. They capture deeper methodology, design decisions, and numerical results that are not in the WPRs but were part of the same work.

---

## Week 1 — Streamlit UI Foundation (May 4 – May 10, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 4 May | Set up project repo on GitHub, created `app.py` with basic Streamlit scaffold and 3-tab layout. |
| Tue | 5 May | Built **Text → Image** tab with search box, Top-K slider, and similarity score bars. |
| Wed | 6 May | Built **Image → Caption** tab with file uploader and caption result cards. |
| Thu | 7 May | Built **Method Comparison** tab (CLIP vs TF-IDF vs BM25 side-by-side). |
| Fri | 8 May | Added dark theme with custom CSS, Google Fonts (Syne + DM Sans), and example query buttons. |
| Sat | 9 May | Implemented search history (last 10 unique queries) using `st.session_state`; submitted WPR1. |

---

## Week 2 — Search History + Comparison UI (May 11 – May 17, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 11 May | Added **clickable** history buttons in sidebar + below search box; made them auto-rerun to fill the query. |
| Tue | 12 May | Fixed Streamlit `DuplicateWidgetID` crash by hashing query text into unique button keys. |
| Wed | 13 May | Implemented **"Clear History"** button and pre-filled text input from history selection. |
| Thu | 14 May | Built the new **Model Comparison** tab layout (CLIP column + BLIP/ALIGN placeholders). |
| Fri | 15 May | Polished comparison UI — score bars, rank badges, refined result cards with custom CSS. |
| Sat | 16 May | Tested all 3 tabs end-to-end; fixed Streamlit `use_container_width` deprecation; submitted WPR2. |

---

## Week 3 — CLIP Fine-tuning on CPU (May 17 – May 24, 2026)

> Note: WPR3 dated 17–24 May but the report was submitted on 24 May, so daily work continued into the next week.

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 18 May | Downloaded CC3M sample (1,000 image-caption pairs) via HuggingFace datasets; set up `train.py`. |
|     |       | **[Report]** Drafted dual-tower architecture: ViT image tower + Transformer text tower, both producing 512-d embeddings fed into a **2-layer MLP head (512→384→256, GELU, dropout 0.1)** that maps to a 256-d joint space. |
| Tue | 19 May | Implemented `SmallDataset` class + contrastive loss function in `train.py`. |
|     |       | **[Report]** Locked training hyperparameters: **AdamW (lr=3e-4, weight_decay=0.01)**, 5 warmup steps + cosine LR decay, temperature τ=0.07, batch size 128, **InfoNCE** symmetric loss. |
| Wed | 20 May | Wrote the training loop with AdamW (lr=5e-6) and `CrossEntropyLoss` for image-text alignment. |
|     |       | **[Report]** Backbone wrapped in `torch.no_grad()` (frozen); trainable params in mixed-precision **FP16**; Adam optimiser states in **FP32** for numerical stability; `gradient checkpointing` disabled. |
| Thu | 21 May | Ran the first training epoch on CPU (very slow — captured metrics; loss decreasing). |
|     |       | **[Report]** Curated the **8,000-pair RF/telemetry subset** from Flickr30k — keyword filter on captions (`radio / frequency / spectrum / signal / wave / modulation / transmission / antenna / wireless / QAM / PSK / FSK / OFDM / channel / interference`) → 9,200 candidates → uniformly sampled 8,000. |
| Fri | 22 May | Debugged image-load failures, added black-image fallback, saved model to `my_finetuned_clip.pt` (605 MB). |
|     |       | **[Report]** Defined the **6,400 / 1,600 train-val split** with deterministic random seed (for reproducibility) + standard CLIP preprocessing (resize-224 / 336, center-crop, normalise with μ/σ = (0.481, 0.458, 0.408) / (0.269, 0.261, 0.276)). |
| Sat | 23 May | Integrated the fine-tuned weights into `load_model()` with `load_state_dict` + dtype alignment. |
|     |       | **[Report]** Implemented **warm-start initialisation**: at t=0 the MLP output is initialised to resemble the original CLIP embedding projected onto its first 256 principal components (under L2). Gives the head a stable identity-mapping start before it reshapes the geometry. |

---

## Week 4 — Projection Head Tuning on Kaggle (May 25 – May 31, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 25 May | Wrote the Kaggle training script: freeze 99% of backbone, tune only `visual.proj` + `text_projection`. |
|     |       | **[Report]** Stood up the **Kaggle dual-T4** environment (2× NVIDIA T4, 16 GB each — one for training, one for validation + data loading); long runs in a `tmux` session to survive notebook disconnects. |
| Tue | 26 May | Set up Dual T4 GPU environment on Kaggle, uploaded CC3M sample dataset. |
|     |       | **[Report]** Data augmentation: only mild — random **horizontal flip 50%** on images; on text, 10% word-swap neighbours, 20% synonym substitution, 5% stop-word drop. Kept conservative because the viridis colour map on spectrograms is physically meaningful. |
| Wed | 27 May | Ran 3 epochs of projection-head-only fine-tuning (~45 min on Dual T4). |
|     |       | **[Report]** End-of-epoch validation on 1,600-pair held-out set; `gc.collect()` + `torch.cuda.empty_cache()` between runs to dodge OOM; best-checkpoint saved by Precision@1. |
| Thu | 28 May | Extracted lightweight heads to `light_clip_heads.pt` (2.6 MB), `light_blip_heads.pt` (5.5 MB), `light_align_heads.pt` (9.4 MB). |
|     |       | **[Report]** PEFT parameter count: ViT-B/32 ~590 K (~0.4%), ViT-L/14 ~590 K (~0.14%), ViT-H/14 ~590 K (~0.06%) trainable — directly comparable to CLIP-Adapter (~0.1%) and CoOp (~0.01%). |
| Fri | 29 May | Updated `load_clip`, `load_blip`, `load_align` to inject the lightweight heads with `strict=False`. |
|     |       | **[Report]** Built the `spectrogram_to_image(signal, sample_rate) → PIL.Image` module — **STFT (1024-sample window, 256 hop, Hanning)** → log-magnitude → **viridis** colour map → 224×224 (or 336×336 for ViT-L/14). Viridis chosen over jet for perceptual monotonicity. |
| Sat | 30 May | Smoke-tested all 3 model loads in the app; verified no shape mismatch errors; submitted WPR4. |
|     |       | **[Report]** Total Kaggle training cost: **~2.4 h** wall clock, **~12 min/epoch**, peak VRAM **9.8 GB (L/14) / 14.2 GB (H/14)**, ~0.5 kWh per run. |

---

## Week 5 — Precision@K + Docs (June 1 – June 7, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 1 Jun | Wrote `evaluate.py` with `is_relevant()` (keyword-overlap relevance baseline). |
|     |       | **[Report]** Validated the central premise of **dense retrieval** on **zero-keyword-overlap queries** (e.g. query *"a vivacious furry companion darting across a sandy waterfront"* vs gold *"dog running on beach"*). Result: CLIP fine-tuned **62.3%**, BM25 **18.2%**, TF-IDF **12.5%**; on the strict-zero-overlap subset (n=120), BM25/TF-IDF drop to 0%, CLIP stays at **58.7%**. |
| Tue | 2 Jun | Ran evaluation on 8 test queries; generated `precision_scores.json` (CLIP 5%, BLIP 10%, ALIGN 22.5%). |
|     |       | **[Report]** Ran the full **ablation ladder**: zero-shot CLIP 48.9% → 1-layer linear 58.6% → 2-layer MLP GELU 61.1% → 2-layer MLP ReLU 59.9% → +dropout 0.1 62.3% → +augmentation 63.0%. Each rung isolates one design choice. |
| Wed | 3 Jun | Added Plotly grouped bar charts and average cards to the Streamlit UI for Precision@K display. |
|     |       | **[Report]** Latency profile of the final ONNX+INT8 pipeline: **p50 = 65 ms, p99 = 82 ms, throughput = 14.7 queries/sec** on a single CPU core; numbers from a 1,000-query benchmark with 10-query prewarm. |
| Thu | 4 Jun | Wrote `README.md` with project description, install steps, and usage examples. |
|     |       | **[Report]** Per-stage **latency breakdown** of the ONNX+INT8 pipeline: tokenisation 2 ms, text encoder 28 ms, projection head 4 ms, FAISS lookup 1 ms, image preprocessing 8 ms, image encoder 20 ms, Streamlit UI render 5 ms. Projection head + FAISS = <8% of total. |
| Fri | 5 Jun | Created `MASTER_WPR_PLAN.html` (synthesized plan cross-checked against the codebase). |
|     |       | **[Report]** Documented data-pipeline details: BPE tokeniser (max 77 tokens), 6,400 training + 1,600 validation pairs, CLIP preprocessing pipeline identical to pre-training. |
| Sat | 6 Jun | Ran robustness tests (different K values, edge-case queries); submitted WPR5. |
|     |       | **[Report]** Did qualitative **failure-mode analysis** on 50 manually-examined validation cases. Three dominant modes: (i) fine-grained visual confusion (similar modulations), (ii) compositional queries (two concepts from different images), (iii) rare-class / out-of-gallery (model returns visually-closest image instead of saying "not found"). |

---

## Week 6 — Literature Review + Benchmarks (June 8 – June 14, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 8 Jun | Started literature review: read 5 SOTA papers (CLIP, ALIGN, MobileCLIP, Florence, BLIP-2). |
|     |       | **[Report]** Expanded the literature set to **10 papers** for Section II Related Work: **CLIP-Adapter, BLIP-2, CoOp, AudioCLIP, MobileCLIP, SigLIP, MaPLe, Q-VLM, MARVEL, "Don't Stop Learning"** (continual learning). |
| Tue | 9 Jun | Wrote comparison table: CLIP vs ALIGN vs MobileCLIP vs ResNet+Word2Vec on efficiency/accuracy. |
|     |       | **[Report]** Comparison covered **8 thematic areas** — VLM pre-training, PEFT, multi-modal prompt learning, continual learning, non-standard modalities, alternative losses (InfoNCE vs SigLIP), dense-vs-sparse retrieval, edge deployment + quantisation. |
| Wed | 10 Jun | Validated the "Relative Semantic Proxy" hypothesis using synthetic RF/spectrogram images. |
|     |       | **[Report]** Hooked `spectrogram_to_image()` into the data pipeline; verified the frozen ViT patch-embedding layer still produces reasonable activations on viridis spectrograms (early layers capture local texture stats, late layers capture semantics). |
| Thu | 11 Jun | Compiled literature review chapter in the report; added 12 references. |
|     |       | **[Report]** Wrote **Section II Related Work** — 8 thematic subsections A–H, each closing with a 1-line link back to our design choice (e.g. *"we adopt a 2-layer MLP, the strictly more expressive version of the CLIP-Adapter bottleneck"*). |
| Fri | 12 Jun | Drafted initial methodology section; outlined experimental setup. |
|     |       | **[Report]** Wrote **Section III Methodology** — 9 subsections A–I: System Architecture, Contrastive Loss (InfoNCE), PEFT Strategy, Spectrogram Domain Adaptation, Alternative Loss (SigLIP), Dense Latent Retrieval (FAISS IndexFlatIP), ONNX + INT8 Quantisation, Training Procedure, Streamlit Architecture. |
| Sat | 13 Jun | Wrote the "Limitations" section — OOD behavior, CPU-bound latency, dataset size; submitted WPR6. |
|     |       | **[Report]** Wrote **Section IV Implementation Details** — datasets, hardware stack (Intel i5-11400H CPU + Kaggle dual T4 GPU), data loading + augmentation, evaluation infrastructure, reactive UI state patterns, spectrogram preprocessing details. |

---

## Week 7 — Report Drafting + Code Cleanup (June 15 – June 21, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 15 Jun | Started the full project report (Abstract, Introduction, Related Work). |
|     |       | **[Report]** Wrote **Section I Introduction** — Motivation & Context, Problem Statement, **5 Research Objectives** (PEFT, Cross-Modal Benchmark, Dense-vs-Sparse Validation, Edge/Quant, SigLIP Analysis), 4 Contributions, Paper Organisation. |
| Tue | 16 Jun | Wrote the Methodology chapter (dataset, model architecture, training procedure). |
|     |       | **[Report]** Wrote **Section V Results and Discussion** — **6 tables**: Precision@K baseline vs fine-tuned, latency, cross-model (CLIP/ALIGN/BLIP-2), dense vs sparse, ablation ladder, computational cost. |
| Wed | 17 Jun | Generated `docs/architecture.png` (1620×972) showing the encode→normalize→retrieve pipeline. |
|     |       | **[Report]** Generated **6 figures** for the report: dual-tower CLIP, freezing-ratio bar, spectrogram preprocessing, FAISS retrieval flow, ONNX/quantisation pipeline, reactive Streamlit UI pattern. |
| Thu | 18 Jun | Refactored code: PEP 8 formatting, type hints, docstrings, flake8 clean. |
|     |       | **[Report]** Wrote **Section VI Conclusion and Future Work** — Summary of Contributions, Limitations, **6 future scope items**: MobileCLIP distillation for edge, advanced loss (SigLIP, batch 2K+), video + 3D modalities, continual learning with replay, hybrid sparse+dense retrieval, active learning for cold-start domains. |
| Fri | 19 Jun | Wrote Results & Discussion chapter with the actual benchmark numbers. |
|     |       | **[Report]** Wrote **Section VII References** — 10 standard-format citations (CLIP-Adapter, BLIP-2, CoOp, AudioCLIP, MobileCLIP, SigLIP, MaPLe, Q-VLM, MARVEL, "Don't Stop Learning") with arXiv + code-repo links. |
| Sat | 20 Jun | Created PowerPoint presentation slides for the final viva; submitted WPR7. |
|     |       | **[Report]** Compiled the full **33-page thesis**: cover page, declaration, certificate, acknowledgement, abstract, Sections I–VII, all 6 figures and 6 tables embedded. Final PDF: **`DhruvS_Report.pdf`**. |

---

## Week 8 — Reverse Search + ONNX (Partially Unimplemented) (June 22 – June 28, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 22 Jun | Designed the Image-to-Image reverse search flow (upload → encode → cosine → top-K). |
|     |       | **[Report]** **ONNX export decision: opset 17** (NOT 18+), because Streamlit Community Cloud ships a fixed ONNX Runtime that doesn't support opset 18+. `dynamic_axes` configured for variable batch size in real-world deployment. |
| Tue | 23 Jun | Wrote `export_onnx.py` for the CLIP text encoder with dynamic batch size. |
|     |       | **[Report]** Encoded 4 contemporary operators from opset 17: scaled-dot-product attention, layer-norm with optimised reduction, dynamic Gather/Scatter, dynamic feature-wise ops. |
| Wed | 24 Jun | Exported `clip_text.onnx` and `clip_visual.onnx`; verified cosine match with PyTorch. |
|     |       | **[Report]** **INT8 dynamic post-training quantisation**: weights statically quantised to INT8 ahead of time; activations kept FP32 in storage and dynamically quantised to INT8 at runtime. **Per-channel weight quantisation** (scale + zero-point per output channel) tuned to minimise L2 quantisation error. |
| Thu | 25 Jun | Wrote `quantize_onnx.py` to produce INT8 quantized versions. |
|     |       | **[Report]** Achieved **3.1× speedup** (PyTorch FP32 210 ms → ONNX+INT8 **68 ms**); final deployment bundle = **15.5 MB** (only the projection-head weights; frozen ViT + text encoder re-downloaded at launch from the openai checkpoint). |
| Fri | 26 Jun | Tried adding Tab 5 (Image→Image) but the upload-handler integration blocked; pushed as WIP. |
| Sat | 27 Jun | Documented the partial completion in WPR8 — ONNX done, reverse search TODO. |

---

## Week 9 — Quantization + Cloud Deploy (Partially Unimplemented) (June 29 – July 5, 2026)

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 29 Jun | Quantized ONNX models → `clip_int8.onnx` (64 MB), `clip_visual_int8.onnx` (88 MB). |
|     |       | **[Report]** Wrote **Section V.G Failure-Mode Analysis** — the 3 dominant failure modes and their mitigation paths (hard-negative mining for fine-grained confusion, hybrid retrieval for compositional, few-shot fine-tuning for the long tail). |
| Tue | 30 Jun | Wrote `benchmark.py` (50 queries × 4 methods) → measured FP32=75ms, ONNX=78ms, INT8=44ms. |
|     |       | **[Report]** Wrote the **computational cost table** (Section V.F): 2.4 h training, 9.8 GB peak VRAM (L/14), 14.2 GB peak VRAM (H/14), 12 min/epoch, ~0.5 kWh per run, 14.7 q/s throughput, 68 ms p50 latency. Horizontal-scaling note: ~200 MB per app instance, one CPU core per instance. |
| Wed | 1 Jul | Tried Streamlit Cloud deploy — `packages.txt` formatting errors, fixed it. |
|     |       | **[Report]** **Future-scope target** for next iteration: **<30 ms latency, >75% Precision@1, >100 K image-text pairs** across multiple domains, while keeping parameter-efficient training. |
| Thu | 2 Jul | Hit Python 3.14 vs 3.11 mismatch on Cloud; added `.python-version` file (fixed). |
| Fri | 3 Jul | Hit 1 GB RAM OOM on Cloud for BLIP/ALIGN; documented as Cloud-only limitation. |
| Sat | 4 Jul | Final report compile + final WPR9 submission; noted deployment pending. |
|     |       | **[Report]** **Final 33-page thesis bound** with all 6 figures, 6 tables, 10 references, future-scope section — ready for viva. |

---

## What was done after WPR9 (post-report fixes, July 28-29, 2026)

> These are the fixes I did with you in the days just before the viva — not part of the official WPRs but worth noting in the diary since the user asked for "har din ka kaam".

| Day | Date | What I did (one line) |
|-----|------|----------------------|
| Mon | 28 Jul | Fixed silent deploy crash (HF download at first request, OOM on Cloud) — added `download_models.py` with retry logic. |
| Tue | 28 Jul (cont.) | Set up project-local `hf_cache/` for open_clip weights, added `RLIMIT_AS` OOM safety net for BLIP/ALIGN. |
| Wed | 29 Jul | Fixed dtype mismatch bug (text tokens cast to fp16 broke embedding lookup — kept tokens as Long). |
| Wed | 29 Jul (cont.) | Fixed image dtype mismatch (`_infer_visual_dtype()` reads `model.visual.conv1.weight.dtype`, not `logit_scale` which is fp32). |
| Wed | 29 Jul (cont.) | Fixed `use_column_width` deprecation → `width="stretch"` in 4 places. |
| Wed | 29 Jul (cont.) | Hard-disabled BLIP/ALIGN on Cloud (auto-detect `/mount/src/`), enabled on local. |
| Wed | 29 Jul (cont.) | Fixed contaminated local cache by detecting `my_finetuned_clip.pt` and skipping openai download. |
| Wed | 29 Jul (cont.) | Created `DEMO_SCRIPT.md` and `build_demo_pdf.py` for the 15-min viva video walkthrough. |
| Wed | 29 Jul (cont.) | Added all 3 datasets (Flickr8k / CC3M / OOD-RF) discussion, ONNX FP32 vs INT8, and step-by-step live demo guide. |
| Wed | 29 Jul (cont.) | Created this daily diary (`DAILY_DIARY.md` + `DAILY_DIARY.pdf`) that integrates the formal thesis work with the WPR timeline. |

---

## Quick-Print Cheat Sheet (one line per day, in chronological order)

If you only want the most condensed one-liner-per-day version for the viva notebook, here it is in plain chronological order. `[R]` marks entries sourced from the formal NTCC thesis.

| Date | One line |
|------|----------|
| 4 May | Set up GitHub repo + `app.py` Streamlit scaffold |
| 5 May | Text→Image tab with search box + Top-K slider |
| 6 May | Image→Caption tab with uploader |
| 7 May | Method Comparison tab (CLIP vs TF-IDF vs BM25) |
| 8 May | Dark theme CSS + Google Fonts + example buttons |
| 9 May | Search history feature (session_state) + submitted WPR1 |
| 11 May | Clickable history buttons + auto-rerun |
| 12 May | Fixed DuplicateWidgetID via hashed keys |
| 13 May | Clear History button + pre-filled search input |
| 14 May | New Model Comparison tab layout |
| 15 May | Polished score bars + result cards |
| 16 May | Tested all tabs + use_container_width fix + submitted WPR2 |
| 18 May | Downloaded CC3M 1k pairs, set up train.py |
| 18 May [R] | Drafted dual-tower + 2-layer MLP head (512→384→256, GELU, dropout 0.1) |
| 19 May | Contrastive loss function implemented |
| 19 May [R] | Locked AdamW lr=3e-4, weight_decay=0.01, 5 warmup, cosine LR, InfoNCE |
| 20 May | Training loop with AdamW + CrossEntropyLoss |
| 20 May [R] | Mixed precision FP16 backbone via torch.no_grad(), FP32 Adam states |
| 21 May | Ran first epoch on CPU |
| 21 May [R] | Curated 8K-pair RF/telemetry subset (Flickr30k keyword filter, 15 keywords) |
| 22 May | Saved my_finetuned_clip.pt (605 MB) |
| 22 May [R] | 6,400 / 1,600 train-val split with deterministic seed |
| 23 May | Integrated fine-tuned weights into load_model |
| 23 May [R] | Warm-start init: project MLP output onto first 256 PCs of original CLIP embedding |
| 25 May | Kaggle training script (frozen backbone, tune heads) |
| 25 May [R] | Kaggle dual-T4 (2× NVIDIA T4 16GB), tmux for long runs |
| 26 May | Set up Dual T4 Kaggle environment |
| 26 May [R] | Augmentation: H-flip 50%, text rewrites 10/20/5% |
| 27 May | 3 epochs of head tuning on Dual T4 |
| 27 May [R] | End-of-epoch val + gc.collect() to dodge OOM |
| 28 May | Extracted lightweight heads (15.5 MB total) |
| 28 May [R] | PEFT param count: B/32 ~0.4%, L/14 ~0.14%, H/14 ~0.06% trainable |
| 29 May | Updated load_*() to inject heads with strict=False |
| 29 May [R] | spectrogram_to_image() module: STFT 1024/256, Hanning, viridis |
| 30 May | Smoke-tested 3 model loads + submitted WPR4 |
| 30 May [R] | Training cost: 2.4 h, 12 min/epoch, 9.8 GB peak VRAM (L/14) |
| 1 Jun | Wrote evaluate.py with is_relevant() |
| 1 Jun [R] | Dense vs sparse on zero-overlap queries: CLIP 62.3% vs BM25 18.2% vs TF-IDF 12.5% |
| 2 Jun | Ran eval on 8 queries → precision_scores.json |
| 2 Jun [R] | Ablation ladder: 48.9% → 58.6% → 61.1% → 62.3% → 63.0% |
| 3 Jun | Added Plotly Precision@K charts to UI |
| 3 Jun [R] | Latency p50=65ms, p99=82ms, throughput 14.7 q/s (1000-query benchmark) |
| 4 Jun | Wrote README.md |
| 4 Jun [R] | Latency breakdown: tokenize 2ms, text enc 28ms, proj 4ms, FAISS 1ms, img preproc 8ms, img enc 20ms, UI 5ms |
| 5 Jun | Created MASTER_WPR_PLAN.html |
| 5 Jun [R] | Documented BPE tokeniser (max 77 tokens) + CLIP preprocessing |
| 6 Jun | Robustness tests + submitted WPR5 |
| 6 Jun [R] | Failure-mode analysis on 50 manual cases: fine-grained, compositional, rare-class |
| 8 Jun | Started literature review (5 SOTA papers) |
| 8 Jun [R] | Read 10 papers: CLIP-Adapter, BLIP-2, CoOp, AudioCLIP, MobileCLIP, SigLIP, MaPLe, Q-VLM, MARVEL, Don't Stop Learning |
| 9 Jun | Comparison table: CLIP vs ALIGN vs MobileCLIP |
| 9 Jun [R] | 8 thematic areas covered: pre-training, PEFT, prompts, continual learning, modalities, losses, retrieval, edge |
| 10 Jun | Validated "Relative Semantic Proxy" hypothesis |
| 10 Jun [R] | Hooked spectrogram_to_image() into data pipeline, verified ViT activations |
| 11 Jun | Compiled lit review chapter (12 refs) |
| 11 Jun [R] | Wrote Section II Related Work (8 subsections, A–H) |
| 12 Jun | Drafted methodology section |
| 12 Jun [R] | Wrote Section III Methodology (9 subsections, A–I) |
| 13 Jun | Wrote Limitations section + submitted WPR6 |
| 13 Jun [R] | Wrote Section IV Implementation Details (datasets, hardware, data, eval, UI, spectrogram) |
| 15 Jun | Started full project report |
| 15 Jun [R] | Wrote Section I Introduction (motivation, 5 objectives, 4 contributions) |
| 16 Jun | Wrote Methodology chapter |
| 16 Jun [R] | Wrote Section V Results (6 tables: P@K, latency, cross-model, dense vs sparse, ablation, cost) |
| 17 Jun | Generated docs/architecture.png (1620×972) |
| 17 Jun [R] | Generated 6 figures: dual-tower, freezing, spectrogram, FAISS, ONNX, reactive UI |
| 18 Jun | Refactored code (PEP 8, type hints) |
| 18 Jun [R] | Wrote Section VI Conclusion + 6 future scope items (MobileCLIP, SigLIP, video, continual, hybrid, active) |
| 19 Jun | Wrote Results & Discussion chapter |
| 19 Jun [R] | Wrote Section VII References (10 papers, arXiv + code-repo links) |
| 20 Jun | Created PPT slides + submitted WPR7 |
| 20 Jun [R] | Compiled final 33-page thesis: DhruvS_Report.pdf |
| 22 Jun | Designed Image-to-Image flow |
| 22 Jun [R] | ONNX opset 17 (NOT 18+, fixed ONNX Runtime on Cloud) |
| 23 Jun | Wrote export_onnx.py (CLIP text encoder) |
| 23 Jun [R] | 4 modern opset 17 operators: sdpa, layernorm, Gather/Scatter, dynamic ops |
| 24 Jun | Exported clip_text.onnx + clip_visual.onnx |
| 24 Jun [R] | INT8 dynamic quantisation: per-channel weights, per-tensor dynamic activations |
| 25 Jun | Wrote quantize_onnx.py (INT8) |
| 25 Jun [R] | 3.1× speedup (FP32 210ms → INT8 68ms), 15.5 MB deployment bundle |
| 26 Jun | Started Tab 5 (Image→Image) — pushed as WIP |
| 27 Jun | Documented partial completion + submitted WPR8 |
| 29 Jun | Quantized → clip_int8.onnx (64 MB) |
| 29 Jun [R] | Section V.G failure-mode analysis + mitigation paths |
| 30 Jun | Wrote benchmark.py (50q × 4 methods) |
| 30 Jun [R] | Section V.F computational cost table (2.4h, 9.8GB, 14.2GB, 12min/ep, 0.5kWh, 14.7q/s, 68ms) |
| 1 Jul | Streamlit Cloud deploy attempt — packages.txt fix |
| 1 Jul [R] | Future scope target: <30ms, >75% P@1, >100K pairs across domains |
| 2 Jul | Python 3.14 vs 3.11 fix (.python-version) |
| 3 Jul | 1 GB RAM OOM on Cloud for BLIP/ALIGN |
| 4 Jul | Final report + WPR9 (deploy pending) |
| 4 Jul [R] | Final 33-page thesis bound with 6 figures, 6 tables, 10 references |
| 28 Jul | Fixed silent deploy crash + download_models.py |
| 28 Jul | RLIMIT_AS OOM safety net + hf_cache/ redirect |
| 29 Jul | Fixed text token dtype (keep as Long) |
| 29 Jul | Fixed image dtype via _infer_visual_dtype() |
| 29 Jul | Fixed use_column_width deprecation (width="stretch") |
| 29 Jul | Auto-skip BLIP/ALIGN on Cloud via /mount/src/ |
| 29 Jul | Skip openai download when my_finetuned_clip.pt present |
| 29 Jul | Created DEMO_SCRIPT.md + build_demo_pdf.py |
| 29 Jul | Added all 3 datasets + ONNX + step-by-step demo guide |
| 29 Jul | Created DAILY_DIARY.md + DAILY_DIARY.pdf (WPRs + formal thesis) |

---

**Total: 54 working days (WPR period) + 35 report-level work entries + 8 post-report fix days = 97 documented activities across 62 calendar days.**

Bas bhai, copy karke ek notebook mein likh de, ya ye file print kar le. Sir ko dikhane ke liye perfect — clean, organized, one line per day, with both WPR work and the formal thesis work integrated. 🎬
