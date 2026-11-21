"""
Query-aware fusion: learn query-conditioned weights over (L, I, S).

score(q,c) = wL(q)*L + wI(q)*I + wS(q)*S
[wL,wI,wS] = softmax(W @ [1, Qid, Qlen]^T)

This is the direct operationalization of "query-aware adaptation": the fusion
weights change with query characteristics. Unlike pair-level LR trained on a
10-candidate pool, this is trained with a pairwise ranking loss so the
objective matches retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


@dataclass
class QueryAwareFusion:
    """3×3 matrix W maps [1, Qid, Qlen] → logits for (L, I, S) weights."""

    W: np.ndarray  # shape (3, 3)
    lr: float = 0.5
    l2: float = 1e-3

    @classmethod
    def from_uniform(cls) -> "QueryAwareFusion":
        # Start near equal weights
        W = np.zeros((3, 3), dtype=np.float64)
        return cls(W=W)

    def weights(self, Qid: np.ndarray, Qlen: np.ndarray) -> np.ndarray:
        """Return (n, 3) weights."""
        Qid = np.asarray(Qid, dtype=np.float64).ravel()
        Qlen = np.asarray(Qlen, dtype=np.float64).ravel()
        X = np.column_stack([np.ones(len(Qid)), Qid, Qlen])  # (n, 3)
        logits = X @ self.W.T  # (n, 3)
        return _softmax(logits, axis=1)

    def score(
        self,
        L: np.ndarray,
        I: np.ndarray,
        S: np.ndarray,
        Qid: np.ndarray,
        Qlen: np.ndarray,
    ) -> np.ndarray:
        """
        L,I,S: (n_query, n_cand) or (n_pairs,)
        Qid,Qlen: broadcastable to L's query axis (n_query,) or (n_pairs,)
        """
        L = np.asarray(L, dtype=np.float64)
        I = np.asarray(I, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        if L.ndim == 2:
            w = self.weights(Qid[:, 0] if Qid.ndim == 2 else Qid, Qlen[:, 0] if Qlen.ndim == 2 else Qlen)
            return (w[:, 0:1] * L + w[:, 1:2] * I + w[:, 2:3] * S).astype(np.float32)
        w = self.weights(Qid, Qlen)
        return (w[:, 0] * L + w[:, 1] * I + w[:, 2] * S).astype(np.float32)

    def fit_pairwise(
        self,
        L: np.ndarray,
        I: np.ndarray,
        S: np.ndarray,
        Qid: np.ndarray,
        Qlen: np.ndarray,
        y: np.ndarray,
        query_ids: np.ndarray,
        epochs: int = 40,
        seed: int = 42,
    ) -> dict:
        """
        Pairwise logistic ranking loss over each query's positive vs negatives.
        """
        rng = np.random.default_rng(seed)
        L = L.astype(np.float64)
        I = I.astype(np.float64)
        S = S.astype(np.float64)
        Qid = Qid.astype(np.float64)
        Qlen = Qlen.astype(np.float64)
        y = y.astype(np.int8)
        query_ids = np.asarray(query_ids)

        # group indices
        groups: dict = {}
        for idx, qid in enumerate(query_ids):
            groups.setdefault(qid, []).append(idx)

        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            n_pairs = 0
            grad = np.zeros_like(self.W)
            q_items = list(groups.items())
            rng.shuffle(q_items)
            for qid, idxs in q_items:
                idxs = np.asarray(idxs)
                labels = y[idxs]
                pos = idxs[labels == 1]
                neg = idxs[labels == 0]
                if len(pos) == 0 or len(neg) == 0:
                    continue
                p = pos[0]
                # sample up to 4 negatives
                if len(neg) > 4:
                    neg = rng.choice(neg, size=4, replace=False)
                qid_v = Qid[p]
                qlen_v = Qlen[p]
                x = np.array([1.0, qid_v, qlen_v])
                logits = self.W @ x
                w = _softmax(logits)

                for n in neg:
                    # s_pos - s_neg
                    dL = L[p] - L[n]
                    dI = I[p] - I[n]
                    dS = S[p] - S[n]
                    margin = w[0] * dL + w[1] * dI + w[2] * dS
                    # softplus(-margin) = log(1+exp(-margin))
                    loss = np.logaddexp(0.0, -margin)
                    total_loss += loss
                    n_pairs += 1
                    # d(loss)/d(margin) = -sigmoid(-margin) = sigmoid(margin)-1
                    sig = 1.0 / (1.0 + np.exp(-margin))
                    d_loss_d_margin = sig - 1.0
                    # d(margin)/d(w) = [dL, dI, dS]
                    d_margin_dw = np.array([dL, dI, dS])
                    d_loss_dw = d_loss_d_margin * d_margin_dw
                    # softmax jacobian: dw_i/d(logit_j)
                    # dsoftmax: diag(w) - w w^T
                    J = np.diag(w) - np.outer(w, w)
                    d_loss_d_logits = J.T @ d_loss_dw
                    # logits = W @ x  => d(logits_k)/d(W_kj) = x_j
                    grad += np.outer(d_loss_d_logits, x)

            if n_pairs == 0:
                break
            grad /= n_pairs
            grad += self.l2 * self.W
            self.W -= self.lr * grad
            avg = total_loss / n_pairs
            history.append(avg)
            # mild lr decay
            if (epoch + 1) % 10 == 0:
                self.lr *= 0.8
        return {"loss_history": history, "final_loss": history[-1] if history else None}
