"""Ranking metrics: MRR, Precision@k, Recall@k (+ optional bootstrap CIs)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ranks_of_positives(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    For each query row, rank of the (first) positive candidate.
    Higher score = better. Ties broken by stable argsort.
    """
    n = scores.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(-scores[i], kind="mergesort")
        ranked_labels = labels[i, order]
        pos = np.flatnonzero(ranked_labels == 1)
        ranks[i] = float(pos[0] + 1) if len(pos) else np.nan
    return ranks


def mrr_from_matrices(scores: np.ndarray, labels: np.ndarray) -> float:
    ranks = _ranks_of_positives(scores, labels)
    return float(np.nanmean(1.0 / ranks))


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    n = scores.shape[0]
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(-scores[i], kind="mergesort")[:k]
        vals[i] = labels[i, order].sum() / k
    return float(vals.mean())


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """With a single relevant doc this equals Success@k."""
    n = scores.shape[0]
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        rel = labels[i].sum()
        if rel <= 0:
            vals[i] = np.nan
            continue
        order = np.argsort(-scores[i], kind="mergesort")[:k]
        vals[i] = labels[i, order].sum() / rel
    return float(np.nanmean(vals))


def metrics_from_matrices(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> dict[str, float]:
    return {
        "MRR": mrr_from_matrices(scores, labels),
        f"P@{k}": precision_at_k(scores, labels, k),
        f"R@{k}": recall_at_k(scores, labels, k),
        "Success@k": recall_at_k(scores, labels, k),
    }


def scores_df_to_matrices(
    scores: pd.DataFrame,
    pool: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Pivot score/label tables into dense (n_query, n_cand) matrices with aligned order.
    Assumes every query shares the same candidate set (true for full-split).
    """
    merged = pool[["query_id", "candidate_func_id", "label"]].merge(
        scores[["query_id", "candidate_func_id", "score"]],
        on=["query_id", "candidate_func_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["score"].isna().any():
        raise ValueError("missing scores after merge")

    qids = sorted(merged["query_id"].unique())
    cids = sorted(merged["candidate_func_id"].unique())
    q_index = {q: i for i, q in enumerate(qids)}
    c_index = {c: i for i, c in enumerate(cids)}

    S = np.zeros((len(qids), len(cids)), dtype=np.float32)
    Y = np.zeros((len(qids), len(cids)), dtype=np.int8)
    for row in merged.itertuples(index=False):
        i = q_index[row.query_id]
        j = c_index[row.candidate_func_id]
        S[i, j] = row.score
        Y[i, j] = row.label
    return S, Y, qids


def evaluate_score_table(scores: pd.DataFrame, pool: pd.DataFrame, k: int = 5) -> dict[str, float]:
    S, Y, _ = scores_df_to_matrices(scores, pool)
    return metrics_from_matrices(S, Y, k=k)


def bootstrap_mrr_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Query-level bootstrap CI for MRR."""
    rng = np.random.default_rng(seed)
    ranks = _ranks_of_positives(scores, labels)
    rr = 1.0 / ranks
    n = len(rr)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = rr[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(rr.mean()), float(lo), float(hi)
