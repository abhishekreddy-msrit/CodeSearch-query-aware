"""Richer query characterization for query-aware fusion."""
from __future__ import annotations

import numpy as np

from .text import WORD, query_id_density, query_len_norm, word_tokens


def build_code_vocab(code_texts: list[str]) -> set[str]:
    vocab: set[str] = set()
    for t in code_texts:
        vocab.update(word_tokens(t))
    return vocab


def mean_idf(query: str, df: dict[str, int], n_docs: int) -> float:
    toks = word_tokens(query)
    if not toks:
        return 0.0
    vals = []
    for t in set(toks):
        d = df.get(t, 0)
        vals.append(np.log((n_docs + 1) / (d + 1)) + 1.0)
    return float(np.mean(vals))


def query_code_overlap(query: str, code_vocab: set[str]) -> float:
    toks = set(word_tokens(query))
    if not toks:
        return 0.0
    return len(toks & code_vocab) / len(toks)


def query_feature_vector(
    query: str,
    *,
    code_vocab: set[str],
    df: dict[str, int],
    n_docs: int,
) -> np.ndarray:
    """
    Compact query vector:
      [bias=1, Qid, Qlen, Qspec(IDF), Qcode(vocab overlap), Quniq(#unique/len)]
    """
    toks = word_tokens(query)
    qid = query_id_density(query)
    qlen = query_len_norm(query)
    qspec = mean_idf(query, df, n_docs)
    # normalize IDF roughly into [0,1]
    qspec_n = min(qspec / 8.0, 1.0)
    qcode = query_code_overlap(query, code_vocab)
    quniq = (len(set(toks)) / len(toks)) if toks else 0.0
    return np.array([1.0, qid, qlen, qspec_n, qcode, quniq], dtype=np.float64)


def batch_query_features(queries: list[str], code_vocab, df, n_docs) -> np.ndarray:
    return np.vstack([query_feature_vector(q, code_vocab=code_vocab, df=df, n_docs=n_docs) for q in queries])


def pool_peakiness(scores: np.ndarray) -> np.ndarray:
    """
    Per-query peakiness of a score matrix (n_q, n_c): (max - mean) / (std + eps).
    High => signal has a sharp top candidate (more trustworthy for that query).
    """
    mx = scores.max(axis=1)
    mu = scores.mean(axis=1)
    sd = scores.std(axis=1)
    return ((mx - mu) / (sd + 1e-6)).astype(np.float64)


def zscore_rows(M: np.ndarray) -> np.ndarray:
    """Per-query z-score so L/I/S are commensurate within each query."""
    M = np.asarray(M, dtype=np.float32)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    return ((M - mu) / (sd + 1e-6)).astype(np.float32)


def zscore_row(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    sd = float(v.std()) + 1e-6
    return (v - float(v.mean())) / sd
