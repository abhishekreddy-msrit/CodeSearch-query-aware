"""Sentence-BERT embeddings with on-disk cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ARTIFACTS, SBERT_BATCH, SBERT_MODEL, SEED


def load_sbert(model_name: str = SBERT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = SBERT_BATCH,
    show_progress: bool = True,
) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def cached_encode(
    model: SentenceTransformer,
    texts: list[str],
    cache_path: Path,
    batch_size: int = SBERT_BATCH,
) -> np.ndarray:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            return arr
    emb = encode_texts(model, texts, batch_size=batch_size)
    np.save(cache_path, emb)
    return emb


def code_repr(row) -> str:
    """
    Candidate text for semantic / lexical encoding.

    Uses docstring-stripped `code` plus function name so naming signals remain,
    without feeding the gold query docstring (label leakage).
    """
    name = str(row.get("func_name", "") or "")
    code = str(row.get("code", "") or "")
    return f"{name}\n{code}"


def default_cache_dir() -> Path:
    return ARTIFACTS / "embeddings"
