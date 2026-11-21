"""Baselines and supervised rankers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import ALL_FEATURES, BASE_FEATURES, SEED


@dataclass
class ScorerResult:
    name: str
    scores: np.ndarray  # same shape as ranking matrix or 1d pair scores


def fixed_hybrid_score(L: np.ndarray, I: np.ndarray, S: np.ndarray, w: tuple[float, float, float]) -> np.ndarray:
    wl, wi, ws = w
    return wl * L + wi * I + ws * S


def tune_hybrid_weights(
    L: np.ndarray,
    I: np.ndarray,
    S: np.ndarray,
    labels: np.ndarray,
    grid: list[tuple[float, float, float]] | None = None,
) -> tuple[tuple[float, float, float], float]:
    """
    Choose fixed hybrid weights on VALIDATION only by maximizing MRR.
    labels: (n_query, n_cand) with a single 1 per row (identity for our setup).
    """
    from .metrics import mrr_from_matrices

    if grid is None:
        grid = []
        for wl in (0.2, 0.33, 0.4, 0.5, 0.6):
            for wi in (0.1, 0.2, 0.33):
                ws = 1.0 - wl - wi
                if ws < 0.05:
                    continue
                grid.append((round(wl, 3), round(wi, 3), round(ws, 3)))
        grid.append((1 / 3, 1 / 3, 1 / 3))

    best_w, best = grid[0], -1.0
    for w in grid:
        scores = fixed_hybrid_score(L, I, S, w)
        mrr = mrr_from_matrices(scores, labels)
        if mrr > best:
            best, best_w = mrr, w
    return best_w, best


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    class_weight: str | dict | None = "balanced",
) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=SEED,
        C=1.0,
    )
    clf.fit(X, y)
    return clf


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return model.decision_function(X).astype(np.float32)


def save_model(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: Path):
    return joblib.load(path)


def select_feature_columns(X: np.ndarray, all_names: list[str], keep: list[str]) -> np.ndarray:
    idx = [all_names.index(n) for n in keep]
    return X[:, idx]


# Convenience: coefficient inspection for query-aware LR
def explain_lr(model: LogisticRegression, feature_names: list[str]) -> list[tuple[str, float]]:
    coefs = model.coef_.ravel()
    return sorted(zip(feature_names, coefs.tolist()), key=lambda x: -abs(x[1]))
