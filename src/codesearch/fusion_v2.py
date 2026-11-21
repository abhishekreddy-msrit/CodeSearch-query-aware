"""
QueryAwareFusionV2 — ranking-oriented, query-conditioned fusion.

Improvements over v1:
1. Per-query z-normalization of L/I/S (commensurate scales).
2. Richer query features (IDF specificity, code-vocab overlap, uniqueness).
3. Optional pool peakiness features (signal reliability for this query).
4. Optional soft gate to down-weight identifier when it is noisy.
5. Training on many in-split negatives with rank-weighted pairwise loss
   (emphasizes fixing top-of-list mistakes → closer to MRR).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .query_feats import pool_peakiness, zscore_row, zscore_rows


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


@dataclass
class FusionV2Config:
    name: str = "fusion_v2"
    use_zscore: bool = True
    use_peakiness: bool = True
    use_i_gate: bool = True  # learn scalar gate on I based on Qid/peak_I
    n_neg: int = 16
    hard_frac: float = 0.5  # fraction of negatives from top of max(L,S)
    epochs: int = 60
    lr: float = 0.35
    l2: float = 5e-4
    rank_weight: bool = True  # weight pairs by soft approx of rank impact
    seed: int = 42


@dataclass
class QueryAwareFusionV2:
    """
    q_feat dims:
      base: [1, Qid, Qlen, Qspec, Qcode, Quniq] = 6
      + peak_L, peak_I, peak_S if use_peakiness => 9
    W: (3, d) softmax weights over zscored (L,I,S)
    optional i_gate: sigmoid(g · q_feat_subset) multiplies I contribution
    """

    W: np.ndarray
    config: FusionV2Config = field(default_factory=FusionV2Config)
    gate_v: np.ndarray | None = None  # shape (d_gate,)
    feature_dim: int = 6

    @classmethod
    def create(cls, cfg: FusionV2Config, feature_dim: int) -> "QueryAwareFusionV2":
        rng = np.random.default_rng(cfg.seed)
        W = rng.normal(0, 0.01, size=(3, feature_dim))
        # bias toward semantic a bit (matches strong SBERT baseline)
        W[2, 0] = 0.15
        W[0, 0] = 0.05
        W[1, 0] = -0.05
        gate = rng.normal(0, 0.01, size=(feature_dim,)) if cfg.use_i_gate else None
        return cls(W=W, config=cfg, gate_v=gate, feature_dim=feature_dim)

    def _weights(self, Q: np.ndarray) -> np.ndarray:
        return _softmax(Q @ self.W.T, axis=1)

    def _i_gate(self, Q: np.ndarray) -> np.ndarray:
        if self.gate_v is None:
            return np.ones(len(Q), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-Q @ self.gate_v))

    def prepare_signals(
        self, L: np.ndarray, I: np.ndarray, S: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        L = np.asarray(L, dtype=np.float64)
        I = np.asarray(I, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        if self.config.use_zscore:
            Lz, Iz, Sz = zscore_rows(L), zscore_rows(I), zscore_rows(S)
        else:
            Lz, Iz, Sz = L, I, S
        if self.config.use_peakiness:
            peaks = np.column_stack(
                [pool_peakiness(L), pool_peakiness(I), pool_peakiness(S)]
            )
        else:
            peaks = None
        return Lz, Iz, Sz, peaks

    def build_Q(self, base_Q: np.ndarray, peaks: np.ndarray | None) -> np.ndarray:
        if self.config.use_peakiness:
            assert peaks is not None
            # mild scale for peakiness
            return np.hstack([base_Q, peaks / 5.0])
        return base_Q

    def score_matrices(
        self,
        L: np.ndarray,
        I: np.ndarray,
        S: np.ndarray,
        base_Q: np.ndarray,
    ) -> np.ndarray:
        Lz, Iz, Sz, peaks = self.prepare_signals(L, I, S)
        Q = self.build_Q(base_Q, peaks)
        w = self._weights(Q)
        g = self._i_gate(Q)
        return (
            w[:, 0:1] * Lz + (w[:, 1:2] * g[:, None]) * Iz + w[:, 2:3] * Sz
        ).astype(np.float32)

    def fit(
        self,
        L: np.ndarray,
        I: np.ndarray,
        S: np.ndarray,
        base_Q: np.ndarray,
        labels: np.ndarray,
        epochs: int | None = None,
        max_queries: int | None = 2500,
    ) -> dict:
        """
        Train on full-split style matrices (n_q x n_c) with diagonal positives.
        Uses rank-weighted pairwise loss against hard+random negatives.
        Z-scores each query row on the fly (avoids huge float64 copies).
        """
        cfg = self.config
        epochs = epochs or cfg.epochs
        rng = np.random.default_rng(cfg.seed)
        L = np.asarray(L, dtype=np.float32)
        I = np.asarray(I, dtype=np.float32)
        S = np.asarray(S, dtype=np.float32)
        n, m = L.shape
        assert labels.shape == (n, m)
        pos_idx = labels.argmax(axis=1)

        # subsample queries for memory/time; still enough for stable gradients
        q_idx = np.arange(n)
        if max_queries is not None and n > max_queries:
            q_idx = rng.choice(n, size=max_queries, replace=False)
            q_idx.sort()

        # peaks / Q only for used queries
        if cfg.use_peakiness:
            peaks_all = np.column_stack(
                [pool_peakiness(L), pool_peakiness(I), pool_peakiness(S)]
            ).astype(np.float64)
        else:
            peaks_all = None
        Q_all = self.build_Q(base_Q, peaks_all)

        hard_pools = {}
        for i in q_idx:
            strength = np.maximum(L[i], S[i]).copy()
            strength[pos_idx[i]] = -np.inf
            hard_pools[int(i)] = np.argsort(-strength)[:50]

        history = []
        lr = cfg.lr
        for epoch in range(epochs):
            total = 0.0
            n_pairs = 0
            grad_W = np.zeros_like(self.W)
            grad_g = np.zeros_like(self.gate_v) if self.gate_v is not None else None
            order = rng.permutation(q_idx)
            for i in order:
                i = int(i)
                p = int(pos_idx[i])
                Lz = zscore_row(L[i]) if cfg.use_zscore else L[i].astype(np.float64)
                Iz = zscore_row(I[i]) if cfg.use_zscore else I[i].astype(np.float64)
                Sz = zscore_row(S[i]) if cfg.use_zscore else S[i].astype(np.float64)

                n_hard = int(cfg.n_neg * cfg.hard_frac)
                n_rand = cfg.n_neg - n_hard
                hard = hard_pools[i]
                if len(hard) > n_hard:
                    hard = rng.choice(hard, size=n_hard, replace=False)
                else:
                    hard = hard[:n_hard]
                mask = np.ones(m, dtype=bool)
                mask[p] = False
                mask[hard] = False
                pool = np.flatnonzero(mask)
                rnd = pool if len(pool) <= n_rand else rng.choice(pool, size=n_rand, replace=False)
                negs = np.concatenate([hard, rnd]) if len(hard) else rnd

                qi = Q_all[i]
                logits = self.W @ qi
                w = _softmax(logits)
                gate = 1.0 / (1.0 + np.exp(-self.gate_v @ qi)) if self.gate_v is not None else 1.0

                if cfg.rank_weight:
                    sc = w[0] * Lz + (w[1] * gate) * Iz + w[2] * Sz
                    rpos = 1 + int(np.sum(sc > sc[p]))
                else:
                    rpos = 1

                for n_idx in negs:
                    dL = float(Lz[p] - Lz[n_idx])
                    dI = float(Iz[p] - Iz[n_idx])
                    dS = float(Sz[p] - Sz[n_idx])
                    margin = w[0] * dL + (w[1] * gate) * dI + w[2] * dS
                    loss = np.logaddexp(0.0, -margin)
                    wt = (1.0 / np.log2(1.0 + rpos)) if cfg.rank_weight else 1.0
                    total += loss * wt
                    n_pairs += 1

                    sig = 1.0 / (1.0 + np.exp(-margin))
                    d_loss_d_margin = (sig - 1.0) * wt
                    d_margin_dw = np.array([dL, gate * dI, dS])
                    d_loss_dw = d_loss_d_margin * d_margin_dw
                    J = np.diag(w) - np.outer(w, w)
                    d_loss_d_logits = J.T @ d_loss_dw
                    grad_W += np.outer(d_loss_d_logits, qi)

                    if grad_g is not None:
                        d_margin_d_gate = w[1] * dI
                        d_gate = gate * (1 - gate)
                        grad_g += d_loss_d_margin * d_margin_d_gate * d_gate * qi

            if n_pairs == 0:
                break
            grad_W /= n_pairs
            grad_W += cfg.l2 * self.W
            self.W -= lr * grad_W
            if grad_g is not None:
                grad_g /= n_pairs
                grad_g += cfg.l2 * self.gate_v
                self.gate_v -= lr * grad_g
            history.append(total / n_pairs)
            if (epoch + 1) % 15 == 0:
                lr *= 0.7
        return {"loss_history": history, "final_loss": history[-1] if history else None}
