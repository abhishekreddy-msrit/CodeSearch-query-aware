#!/usr/bin/env python
"""
Round 2 optimization — focused on stable ranking-oriented ideas.

Selection: VALIDATION MRR only. Test evaluated once after selection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.config import ARTIFACTS, RESULTS, SEED  # noqa: E402
from codesearch.data import load_functions  # noqa: E402
from codesearch.evaluate import label_matrix_from_split  # noqa: E402
from codesearch.features import FeatureArtifacts, FeaturePipeline  # noqa: E402
from codesearch.metrics import bootstrap_mrr_ci, metrics_from_matrices, mrr_from_matrices  # noqa: E402
from codesearch.models import fixed_hybrid_score, load_model  # noqa: E402
from codesearch.query_feats import batch_query_features, build_code_vocab  # noqa: E402
from codesearch.text import word_tokens as wt  # noqa: E402


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "optimize_round2.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ranks_desc(M: np.ndarray) -> np.ndarray:
    """1-based ranks; higher score => rank 1."""
    order = np.argsort(-M, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    rows = np.arange(M.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, M.shape[1] + 1)
    return ranks


def rrf_score(L, I, S, k: float = 60.0, w=(1.0, 1.0, 1.0)) -> np.ndarray:
    rL, rI, rS = ranks_desc(L), ranks_desc(I), ranks_desc(S)
    out = (w[0] / (k + rL.astype(np.float32)) + w[1] / (k + rI.astype(np.float32)) + w[2] / (k + rS.astype(np.float32)))
    return out.astype(np.float32)


def eval_m(scores, labels, name):
    m = metrics_from_matrices(scores, labels)
    mrr, lo, hi = bootstrap_mrr_ci(scores, labels, n_boot=300, seed=SEED)
    return {"system": name, "MRR": mrr, "P@5": m["P@5"], "R@5": m["R@5"], "MRR_ci95_lo": lo, "MRR_ci95_hi": hi}


def tune_hybrid(L, I, S, labels, allow_zero_i=True):
    best_w, best = (1 / 3, 1 / 3, 1 / 3), -1.0
    wis = (0.0, 0.1, 0.2, 0.3) if allow_zero_i else (0.1, 0.2, 0.3)
    for wl in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
        for wi in wis:
            ws = 1.0 - wl - wi
            if ws < 0.05:
                continue
            mrr = mrr_from_matrices(fixed_hybrid_score(L, I, S, (wl, wi, ws)), labels)
            if mrr > best:
                best, best_w = mrr, (wl, wi, ws)
    return best_w, best


def load_pipe():
    arts = FeatureArtifacts.load(ARTIFACTS / "feature_pipeline.joblib")
    pipe = FeaturePipeline(arts.sbert_name)
    pipe.tfidf = arts.tfidf
    pipe.scaler = arts.scaler
    pipe.code_emb = arts.code_emb
    pipe.func_ids = arts.func_ids
    pipe.load_side(ARTIFACTS / "feature_pipeline.joblib")
    return pipe


class QueryGatedMix:
    """score = (1-g(q))*hybrid + g(q)*S, g=sigmoid(v·q)."""

    def __init__(self, v, hybrid_w):
        self.v = np.asarray(v, dtype=np.float64)
        self.hybrid_w = hybrid_w

    def gate(self, Q):
        return 1.0 / (1.0 + np.exp(-Q @ self.v))

    def score(self, L, I, S, Q):
        hy = fixed_hybrid_score(L, I, S, self.hybrid_w)
        g = self.gate(Q)[:, None]
        return ((1 - g) * hy + g * S).astype(np.float32)


def fit_gated_mix(L, I, S, Q, labels, hybrid_w, epochs=80, lr=0.4, l2=1e-3, seed=42, max_q=2500):
    """Pairwise train gate to prefer hybrid vs S depending on query."""
    rng = np.random.default_rng(seed)
    n, m = L.shape
    v = np.zeros(Q.shape[1], dtype=np.float64)
    # mild prior: start near hybrid (negative bias on gate)
    v[0] = -1.0
    pos = labels.argmax(1)
    q_idx = np.arange(n)
    if n > max_q:
        q_idx = rng.choice(n, size=max_q, replace=False)

    hard = {}
    for i in q_idx:
        strength = np.maximum(L[i], S[i]).copy()
        strength[pos[i]] = -np.inf
        hard[int(i)] = np.argsort(-strength)[:40]

    for ep in range(epochs):
        grad = np.zeros_like(v)
        tot = 0.0
        npairs = 0
        for i in rng.permutation(q_idx):
            i = int(i)
            p = int(pos[i])
            qi = Q[i]
            g = 1.0 / (1.0 + np.exp(-v @ qi))
            hy = hybrid_w[0] * L[i] + hybrid_w[1] * I[i] + hybrid_w[2] * S[i]
            sc = (1 - g) * hy + g * S[i]
            # sample negs
            h = hard[i][:8]
            mask = np.ones(m, bool)
            mask[p] = False
            mask[h] = False
            pool = np.flatnonzero(mask)
            rnd = rng.choice(pool, size=8, replace=False)
            negs = np.concatenate([h, rnd])
            rpos = 1 + int(np.sum(sc > sc[p]))
            wt = 1.0 / np.log2(1.0 + rpos)
            for n_idx in negs:
                d = (sc[p] - sc[n_idx])
                # sc depends on g: dsc/dg = (S - hy)
                # margin = sc_p - sc_n = (1-g)(hy_p-hy_n) + g(S_p-S_n)
                dhy = hy[p] - hy[n_idx]
                dS = S[i, p] - S[i, n_idx]
                margin = (1 - g) * dhy + g * dS
                loss = np.logaddexp(0.0, -margin)
                tot += loss * wt
                npairs += 1
                sig = 1 / (1 + np.exp(-margin))
                dloss_dm = (sig - 1.0) * wt
                dm_dg = dS - dhy
                dg_dv = g * (1 - g) * qi
                grad += dloss_dm * dm_dg * dg_dv
        grad /= max(npairs, 1)
        grad += l2 * v
        v -= lr * grad
        if (ep + 1) % 20 == 0:
            lr *= 0.7
    return QueryGatedMix(v, hybrid_w)


class ClusterHybrid:
    def __init__(self, kmeans, weights_by_cluster, hybrid_default):
        self.kmeans = kmeans
        self.weights_by_cluster = weights_by_cluster
        self.hybrid_default = hybrid_default

    def score(self, L, I, S, Q):
        # drop bias col for clustering stability — use feats 1:
        X = Q[:, 1:]
        cl = self.kmeans.predict(X)
        out = np.zeros_like(L, dtype=np.float32)
        for c, w in self.weights_by_cluster.items():
            idx = np.where(cl == c)[0]
            if len(idx) == 0:
                continue
            out[idx] = fixed_hybrid_score(L[idx], I[idx], S[idx], w)
        return out


def fit_cluster_hybrid(L, I, S, Q, labels, n_clusters=3, seed=42):
    X = Q[:, 1:]
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cl = km.fit_predict(X)
    weights = {}
    for c in range(n_clusters):
        idx = np.where(cl == c)[0]
        if len(idx) < 30:
            weights[c] = (0.4, 0.2, 0.4)
            continue
        w, _ = tune_hybrid(L[idx], I[idx], S[idx], labels[idx])
        weights[c] = w
    return ClusterHybrid(km, weights, (0.4, 0.2, 0.4)), cl


def main():
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "optimize_round2.log").write_text("", encoding="utf-8")
    out_dir = ARTIFACTS / "models" / "optimize_r2"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=== Round 2 optimization (VAL selection) ===")
    pipe = load_pipe()
    fns = load_functions()
    train_fn = fns[fns.split == "train"].reset_index(drop=True)
    val_fn = fns[fns.split == "val"].reset_index(drop=True)
    test_fn = fns[fns.split == "test"].reset_index(drop=True)

    texts = [pipe._code_text[fid] for fid in train_fn.func_id]
    vocab = build_code_vocab(texts)
    df = {}
    for t in texts:
        for tok in set(wt(t)):
            df[tok] = df.get(tok, 0) + 1
    n_docs = len(texts)

    def mats(split_fn, split):
        return pipe.score_matrices_for_split(
            split_fn, query_emb_cache=ARTIFACTS / "embeddings" / f"{split}_query_emb_n{len(split_fn)}.npy"
        )

    log("Loading matrices…")
    tr = mats(train_fn, "train")
    va = mats(val_fn, "val")
    train_Q = batch_query_features(train_fn["query"].astype(str).tolist(), vocab, df, n_docs)
    val_Q = batch_query_features(val_fn["query"].astype(str).tolist(), vocab, df, n_docs)
    ytr = label_matrix_from_split(len(train_fn))
    yva = label_matrix_from_split(len(val_fn))

    hw = tuple(json.loads((ARTIFACTS / "models" / "hybrid_weights.json").read_text())["weights"])
    fusion_v1 = load_model(ARTIFACTS / "models" / "query_aware_fusion.joblib")

    Lt, It, St = tr["L"], tr["I"], tr["S"]
    Lv, Iv, Sv = va["L"], va["I"], va["S"]

    rows = []
    artifacts = {}

    def add(name, scores, split="val", **extra):
        row = eval_m(scores, yva if split == "val" else None, name)
        # fix if test later
        if split == "val":
            row = eval_m(scores, yva, name)
        row["split"] = split
        row.update(extra)
        rows.append(row)
        log(f"{split.upper()} {name:28s} MRR={row['MRR']:.4f} R@5={row['R@5']:.4f}")
        return row

    # baselines
    add("tfidf", Lv)
    add("sbert", Sv)
    add("fixed_hybrid", fixed_hybrid_score(Lv, Iv, Sv, hw))
    add("fusion_v1", fusion_v1.score(Lv, Iv, Sv, va["Qid"], va["Qlen"]))

    # Retune hybrid allowing wi=0 on TRAIN, apply to val (honest: tune on train)
    w_tr, mrr_tr = tune_hybrid(Lt, It, St, ytr, allow_zero_i=True)
    log(f"hybrid retuned on TRAIN: w={w_tr} train_MRR={mrr_tr:.4f}")
    add("hybrid_retune_train", fixed_hybrid_score(Lv, Iv, Sv, w_tr), weights=list(w_tr))
    artifacts["hybrid_retune_train"] = {"weights": w_tr}

    # RRF variants
    for k in (20, 40, 60):
        add(f"rrf_k{k}", rrf_score(Lv, Iv, Sv, k=k))
    # RRF weights tuned on a TRAIN subsample (full 5600² RRF is memory-heavy)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(train_fn), size=1500, replace=False)
    best_rrf, best_rrf_mrr = None, -1
    for k in (30, 60):
        for wS in (1.0, 1.5, 2.0):
            for wL in (1.0, 1.25):
                wI = 0.5
                sc_tr = rrf_score(Lt[sub][:, sub], It[sub][:, sub], St[sub][:, sub], k=k, w=(wL, wI, wS))
                # labels on subsample diagonal still hold if we take same indices for cand=query set
                y_sub = np.eye(len(sub), dtype=np.int8)
                m = mrr_from_matrices(sc_tr, y_sub)
                if m > best_rrf_mrr:
                    best_rrf_mrr, best_rrf = m, (k, (wL, wI, wS))
    add(
        "rrf_tuned_train",
        rrf_score(Lv, Iv, Sv, k=best_rrf[0], w=best_rrf[1]),
        rrf=list(best_rrf),
    )
    artifacts["rrf_tuned_train"] = {"k": best_rrf[0], "w": best_rrf[1]}
    log(f"best RRF on train-sub: {best_rrf} MRR={best_rrf_mrr:.4f}")

    # Cluster-specific hybrid (fit clusters+weights on TRAIN)
    log("Fitting cluster hybrid on TRAIN…")
    clus, _ = fit_cluster_hybrid(Lt, It, St, train_Q, ytr, n_clusters=3, seed=SEED)
    add("cluster_hybrid_k3", clus.score(Lv, Iv, Sv, val_Q), clusters=clus.weights_by_cluster)
    artifacts["cluster_hybrid_k3"] = clus
    log(f"cluster weights: {clus.weights_by_cluster}")

    clus4, _ = fit_cluster_hybrid(Lt, It, St, train_Q, ytr, n_clusters=4, seed=SEED)
    add("cluster_hybrid_k4", clus4.score(Lv, Iv, Sv, val_Q), clusters=clus4.weights_by_cluster)
    artifacts["cluster_hybrid_k4"] = clus4

    # Query-gated mix hybrid↔SBERT
    log("Training query-gated hybrid↔SBERT mix…")
    gated = fit_gated_mix(Lt, It, St, train_Q, ytr, hw, epochs=60, seed=SEED)
    add("gated_hybrid_sbert", gated.score(Lv, Iv, Sv, val_Q))
    artifacts["gated_hybrid_sbert"] = gated
    gval = gated.gate(val_Q)
    log(f"gate mean={gval.mean():.3f} std={gval.std():.3f} p10={np.percentile(gval,10):.3f} p90={np.percentile(gval,90):.3f}")

    # gated with train-retuned hybrid
    gated2 = fit_gated_mix(Lt, It, St, train_Q, ytr, w_tr, epochs=60, seed=SEED + 3)
    add("gated_retune_hybrid_sbert", gated2.score(Lv, Iv, Sv, val_Q))
    artifacts["gated_retune_hybrid_sbert"] = gated2

    # Simple: max(hybrid, sbert) / CombMNZ-style — diagnostic
    hy_v = fixed_hybrid_score(Lv, Iv, Sv, hw)
    add("max_hybrid_sbert", np.maximum(hy_v, Sv))
    add("combsum_hybrid_sbert", 0.5 * hy_v + 0.5 * Sv)

    # Linear residual toward oracle: learn α,β,γ additive on top of hybrid from TRAIN pairwise
    # score = hy + a*L + b*I + c*S with small a,b,c (equivalent to retuning) — skip if redundant

    val_df = pd.DataFrame(rows)
    val_df.to_csv(RESULTS / "optimize_r2_val.csv", index=False)

    # Select best deployable system
    ban = {"tfidf", "sbert", "max_hybrid_sbert"}  # keep combsum as candidate
    selectable = val_df[~val_df.system.isin(ban)]
    best = selectable.loc[selectable["MRR"].idxmax()]
    selected = best["system"]
    log(f"SELECTED on VAL: {selected} MRR={best['MRR']:.4f}")

    # Save selected
    payload = {
        "selected_name": selected,
        "hybrid_w": hw,
        "retune_w": w_tr,
        "vocab_meta": {"n_docs": n_docs},
    }
    if selected in artifacts:
        payload["artifact"] = artifacts[selected]
    joblib.dump(
        {"payload": payload, "artifacts_keys": list(artifacts.keys()), "code_vocab": vocab, "df": df, "n_docs": n_docs,
         "train_Q_dim": int(train_Q.shape[1]), "selected": selected},
        ARTIFACTS / "models" / "fusion_round2_selected.joblib",
    )
    # also dump individual artifacts
    joblib.dump(artifacts, out_dir / "all_artifacts.joblib")
    (ARTIFACTS / "models" / "selected_fusion.txt").write_text(f"round2:{selected}", encoding="utf-8")

    selection = {
        "selected": selected,
        "val_mrr": float(best["MRR"]),
        "val_r5": float(best["R@5"]),
        "val_leaderboard": selectable.sort_values("MRR", ascending=False)[
            ["system", "MRR", "P@5", "R@5"]
        ].head(12).to_dict(orient="records"),
        "test_not_used_for_selection": True,
        "prior_test_exposure_note": (
            "Earlier phases already viewed test metrics. This round selected on validation only; "
            "the following test numbers are a single post-selection readout."
        ),
    }
    (RESULTS / "optimize_r2_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ONE-SHOT test
    log("=== ONE-SHOT TEST ===")
    te = mats(test_fn, "test")
    test_Q = batch_query_features(test_fn["query"].astype(str).tolist(), vocab, df, n_docs)
    yte = label_matrix_from_split(len(test_fn))
    Lte, Ite, Ste = te["L"], te["I"], te["S"]

    def score_named(name):
        if name == "tfidf":
            return Lte
        if name == "sbert":
            return Ste
        if name == "fixed_hybrid":
            return fixed_hybrid_score(Lte, Ite, Ste, hw)
        if name == "fusion_v1":
            return fusion_v1.score(Lte, Ite, Ste, te["Qid"], te["Qlen"])
        if name == "hybrid_retune_train":
            return fixed_hybrid_score(Lte, Ite, Ste, w_tr)
        if name.startswith("rrf_k"):
            k = float(name.replace("rrf_k", ""))
            return rrf_score(Lte, Ite, Ste, k=k)
        if name == "rrf_tuned_train":
            return rrf_score(Lte, Ite, Ste, k=best_rrf[0], w=best_rrf[1])
        if name == "cluster_hybrid_k3":
            return clus.score(Lte, Ite, Ste, test_Q)
        if name == "cluster_hybrid_k4":
            return clus4.score(Lte, Ite, Ste, test_Q)
        if name == "gated_hybrid_sbert":
            return gated.score(Lte, Ite, Ste, test_Q)
        if name == "gated_retune_hybrid_sbert":
            return gated2.score(Lte, Ite, Ste, test_Q)
        if name == "combsum_hybrid_sbert":
            hy = fixed_hybrid_score(Lte, Ite, Ste, hw)
            return 0.5 * hy + 0.5 * Ste
        if name == "max_hybrid_sbert":
            hy = fixed_hybrid_score(Lte, Ite, Ste, hw)
            return np.maximum(hy, Ste)
        raise KeyError(name)

    test_rows = []
    # report baselines + selected + top val contenders
    report = list(
        dict.fromkeys(
            [
                "tfidf",
                "sbert",
                "fixed_hybrid",
                "fusion_v1",
                selected,
                "rrf_tuned_train",
                "cluster_hybrid_k3",
                "gated_hybrid_sbert",
                "hybrid_retune_train",
            ]
        )
    )
    for name in report:
        sc = score_named(name)
        row = eval_m(sc, yte, name)
        row["split"] = "test"
        row["selected"] = name == selected
        test_rows.append(row)
        mark = " <<SELECTED" if name == selected else ""
        log(f"TEST {name:28s} MRR={row['MRR']:.4f} P@5={row['P@5']:.4f} R@5={row['R@5']:.4f}{mark}")

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(RESULTS / "optimize_r2_test_oneshot.csv", index=False)

    hy = test_df[test_df.system == "fixed_hybrid"].iloc[0]
    sel = test_df[test_df.system == selected].iloc[0]
    delta = float(sel["MRR"] - hy["MRR"])
    summary = {
        **selection,
        "test_fixed_hybrid_mrr": float(hy["MRR"]),
        "test_selected_mrr": float(sel["MRR"]),
        "test_delta_mrr_vs_hybrid": delta,
        "target_delta": 0.010,
        "target_achieved": bool(delta >= 0.010 - 1e-9),
        "test_selected_metrics": {"MRR": float(sel["MRR"]), "P@5": float(sel["P@5"]), "R@5": float(sel["R@5"])},
        "test_fusion_v1_mrr": float(test_df[test_df.system == "fusion_v1"].iloc[0]["MRR"]),
    }
    (RESULTS / "optimize_r2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"TARGET check: hybrid={hy['MRR']:.4f} selected={sel['MRR']:.4f} delta={delta:+.4f} achieved={summary['target_achieved']}")
    log("DONE")


if __name__ == "__main__":
    main()
