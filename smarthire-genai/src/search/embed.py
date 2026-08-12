"""
MODULE 2a — Embeddings.
Wraps a local sentence-transformers model so both jobs and career notes
use one consistent embedding function (same model = comparable vectors).
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return a (n, dim) float32 array of embeddings, L2-normalized for cosine similarity."""
    model = get_embedder()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return (vectors / norms).astype("float32")


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
