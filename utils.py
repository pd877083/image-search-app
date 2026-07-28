"""
utils.py — Similarity & Retrieval Utilities
Includes: Cosine Similarity, TF-IDF, and BM25 search methods
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    index: int
    score: float
    image_path: str = ""
    caption: str = ""


def cosine_similarity_matrix(query_vec, corpus_matrix):
    similarities = (query_vec @ corpus_matrix.T).flatten()
    return similarities


def text_to_image_search(text_query_embedding, image_embeddings, image_paths, top_k=5):
    similarities = cosine_similarity_matrix(text_query_embedding, image_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [RetrievalResult(index=int(idx), score=float(similarities[idx]), image_path=image_paths[idx]) for idx in top_indices]


def image_to_image_search(image_query_embedding, image_embeddings, image_paths, top_k=5):
    """Reverse image search: cosine similarity of a query image vs gallery image embeddings."""
    similarities = cosine_similarity_matrix(image_query_embedding, image_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        RetrievalResult(index=int(idx), score=float(similarities[idx]), image_path=image_paths[idx])
        for idx in top_indices
    ]


def image_to_text_search(image_query_embedding, text_embeddings, captions, top_k=5):
    similarities = cosine_similarity_matrix(image_query_embedding, text_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [RetrievalResult(index=int(idx), score=float(similarities[idx]), caption=captions[idx]) for idx in top_indices]


def save_embeddings(path, **arrays):
    np.savez_compressed(path, **arrays)
    print(f"[SAVE] Embeddings saved to '{path}.npz'")


def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_flickr8k_captions(captions_file):
    image_names, captions = [], []
    with open(captions_file, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or i == 0:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            filename, caption = parts
            image_names.append(filename.strip())
            captions.append(caption.strip())
    print(f"[DATA] Loaded {len(captions)} captions for {len(set(image_names))} unique images.")
    return image_names, captions


# ── TF-IDF Search ─────────────────────────────────────────────────────────────

class TFIDFSearcher:
    """
    TF-IDF based caption search.
    Matches keywords exactly - no semantic understanding.
    """
    def __init__(self, captions: List[str]):
        print("[TF-IDF] Building index...")
        self.captions = captions
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.matrix = self.vectorizer.fit_transform(captions)
        print("[TF-IDF] Index built successfully.")

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_vec = self.vectorizer.transform([query])
        scores = sklearn_cosine(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [RetrievalResult(index=int(idx), score=float(scores[idx]), caption=self.captions[idx]) for idx in top_indices]


# ── BM25 Search ───────────────────────────────────────────────────────────────

class BM25Searcher:
    """
    BM25 based caption search.
    Improved keyword search - still no semantic understanding.
    """
    def __init__(self, captions: List[str]):
        print("[BM25] Building index...")
        self.captions = captions
        tokenized = [cap.lower().split() for cap in captions]
        self.bm25 = BM25Okapi(tokenized)
        print("[BM25] Index built successfully.")

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [RetrievalResult(index=int(idx), score=float(scores[idx]), caption=self.captions[idx]) for idx in top_indices]
