"""
embeddings.py
-------------
Builds and persists a FAISS vector index over AmazonHelp conversation
pairs. At query time, retrieves the top-k most similar (customer_text,
brand_reply) pairs to ground reply generation.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"   # 22 MB; fast & good
EMBED_DIM   = 384
INDEX_PATH  = Path(__file__).parent.parent / "data" / "faiss_index.bin"
META_PATH   = Path(__file__).parent.parent / "data" / "faiss_meta.pkl"
BATCH_SIZE  = 512


# ── Embedder singleton ────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embeddings] Loading SentenceTransformer ({MODEL_NAME}) …")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Return L2-normalised embeddings of shape (N, EMBED_DIM)."""
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


# ── Index builders ────────────────────────────────────────────────────────────

def build_index(convs: pd.DataFrame) -> None:
    """
    Build a flat inner-product (cosine) FAISS index from customer_text.
    Saves both the index and metadata (customer_text, brand_reply, intent).
    """
    texts = convs["customer_text"].tolist()
    print(f"[embeddings] Embedding {len(texts):,} texts …")
    vecs = embed(texts)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "customer_text": convs["customer_text"].tolist(),
        "brand_reply":   convs["brand_reply"].tolist(),
        "intent":        convs.get("intent", pd.Series(["UNKNOWN"] * len(convs))).tolist(),
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"[embeddings] Index saved → {INDEX_PATH}  ({index.ntotal:,} vectors)")


def load_index() -> Tuple[faiss.IndexFlatIP, dict]:
    """Load pre-built FAISS index and metadata."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Run `python src/embeddings.py` to build it first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    meta: dict,
    k: int = 5,
) -> List[dict]:
    """
    Return the top-k similar historical (customer_text, brand_reply) pairs.
    """
    vec = embed([query])
    scores, ids = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append({
            "customer_text": meta["customer_text"][idx],
            "brand_reply":   meta["brand_reply"][idx],
            "intent":        meta["intent"][idx],
            "similarity":    float(score),
        })
    return results


if __name__ == "__main__":
    from data_pipeline import run as build_conversations
    convs = build_conversations()
    build_index(convs)
