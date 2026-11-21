"""Run ranking systems on a benchmark pool and collect metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import check_scores
from .metrics import bootstrap_mrr_ci, metrics_from_matrices
from .models import predict_proba_positive


def label_matrix_from_split(n: int) -> np.ndarray:
    """In our full-split design the positive is always on the diagonal (query i → func i)."""
    return np.eye(n, dtype=np.int8)


def evaluate_matrix_system(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    func_ids: np.ndarray,
    pool: pd.DataFrame | None = None,
    bootstrap: bool = True,
) -> dict:
    metrics = metrics_from_matrices(scores, labels, k=5)
    out = {"system": name, **metrics}
    if bootstrap:
        mrr, lo, hi = bootstrap_mrr_ci(scores, labels, n_boot=400, seed=42)
        out["MRR_ci95_lo"] = lo
        out["MRR_ci95_hi"] = hi
        out["MRR"] = mrr
    if pool is not None:
        # Materialize score table for check_scores (may be large — optional)
        pass
    out["n_queries"] = int(scores.shape[0])
    out["n_candidates"] = int(scores.shape[1])
    return out


def materialize_scores(
    scores: np.ndarray,
    func_ids: np.ndarray,
) -> pd.DataFrame:
    """Build long-form score table matching eval_full schema keys."""
    n = len(func_ids)
    q_ids = np.array([f"q_{fid}" for fid in func_ids])
    # rows: query index, cols: candidate index
    qi = np.repeat(np.arange(n), n)
    ci = np.tile(np.arange(n), n)
    return pd.DataFrame(
        {
            "query_id": q_ids[qi],
            "candidate_func_id": func_ids[ci],
            "score": scores.ravel().astype(np.float32),
        }
    )


def verify_against_pool(score_df: pd.DataFrame, pool: pd.DataFrame) -> None:
    check_scores(score_df, pool)


def score_supervised_on_matrices(
    model,
    mats: dict[str, np.ndarray],
    feature_names: list[str],
    scaler=None,
) -> np.ndarray:
    """Apply a sklearn classifier to full-split feature matrices."""
    shape = mats["L"].shape
    cols = [mats[n].ravel() for n in feature_names]
    X = np.column_stack(cols).astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    proba = predict_proba_positive(model, X)
    return proba.reshape(shape)


def filter_ambiguous(scores: np.ndarray, labels: np.ndarray, amb_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For sensitivity analysis: set ambiguous negatives' scores to -inf so they
    cannot enter the ranking (positives untouched).
    amb_mask True where pair is ambiguous negative.
    """
    s = scores.copy()
    s = np.where(amb_mask & (labels == 0), -1e9, s)
    return s, labels
