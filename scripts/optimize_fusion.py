#!/usr/bin/env python
"""
Optimization sprint for query-aware fusion.

CRITICAL: model selection uses VALIDATION only.
Test is evaluated once at the end for the selected model (+ frozen baselines).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.config import ARTIFACTS, RESULTS, SEED  # noqa: E402
from codesearch.data import load_functions  # noqa: E402
from codesearch.evaluate import label_matrix_from_split  # noqa: E402
from codesearch.features import FeatureArtifacts, FeaturePipeline  # noqa: E402
from codesearch.fusion_v2 import FusionV2Config, QueryAwareFusionV2  # noqa: E402
from codesearch.metrics import bootstrap_mrr_ci, metrics_from_matrices  # noqa: E402
from codesearch.models import fixed_hybrid_score, load_model  # noqa: E402
from codesearch.query_feats import batch_query_features, build_code_vocab  # noqa: E402
from codesearch.text import word_tokens as wt  # noqa: E402


def log(msg: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "optimize_fusion.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_pipe() -> FeaturePipeline:
    arts = FeatureArtifacts.load(ARTIFACTS / "feature_pipeline.joblib")
    pipe = FeaturePipeline(arts.sbert_name)
    pipe.tfidf = arts.tfidf
    pipe.scaler = arts.scaler
    pipe.code_emb = arts.code_emb
    pipe.func_ids = arts.func_ids
    pipe.load_side(ARTIFACTS / "feature_pipeline.joblib")
    return pipe


def build_df_vocab(train_fn: pd.DataFrame, pipe: FeaturePipeline):
    texts = [pipe._code_text[fid] for fid in train_fn.func_id]
    vocab = build_code_vocab(texts)
    df: dict[str, int] = {}
    for t in texts:
        for tok in set(wt(t)):
            df[tok] = df.get(tok, 0) + 1
    return vocab, df, len(texts)


def get_mats(pipe, split_fn, split: str):
    return pipe.score_matrices_for_split(
        split_fn,
        query_emb_cache=ARTIFACTS / "embeddings" / f"{split}_query_emb_n{len(split_fn)}.npy",
    )


def eval_scores(scores, labels, name: str) -> dict:
    m = metrics_from_matrices(scores, labels)
    mrr, lo, hi = bootstrap_mrr_ci(scores, labels, n_boot=400, seed=SEED)
    return {"system": name, "MRR": mrr, "P@5": m["P@5"], "R@5": m["R@5"], "MRR_ci95_lo": lo, "MRR_ci95_hi": hi}


def main():
    np.random.seed(SEED)
    (RESULTS / "optimize_fusion.log").write_text("", encoding="utf-8")
    out_dir = ARTIFACTS / "models" / "optimize"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=== Optimization sprint (VAL selection only) ===")
    pipe = load_pipe()
    fns = load_functions()
    train_fn = fns[fns.split == "train"].reset_index(drop=True)
    val_fn = fns[fns.split == "val"].reset_index(drop=True)
    test_fn = fns[fns.split == "test"].reset_index(drop=True)

    code_vocab, df, n_docs = build_df_vocab(train_fn, pipe)
    log(f"code vocab size={len(code_vocab)} n_docs={n_docs}")

    log("Building train/val score matrices…")
    train_mats = get_mats(pipe, train_fn, "train")
    val_mats = get_mats(pipe, val_fn, "val")
    # NOTE: test matrices built only after selection

    train_Q = batch_query_features(train_fn["query"].astype(str).tolist(), code_vocab, df, n_docs)
    val_Q = batch_query_features(val_fn["query"].astype(str).tolist(), code_vocab, df, n_docs)
    log(f"query feat dim={train_Q.shape[1]} val Qid mean={val_Q[:,1].mean():.3f} Qspec mean={val_Q[:,3].mean():.3f}")

    y_tr = label_matrix_from_split(len(train_fn))
    y_va = label_matrix_from_split(len(val_fn))

    hw = json.loads((ARTIFACTS / "models" / "hybrid_weights.json").read_text())
    hybrid_w = tuple(hw["weights"])
    fusion_v1 = load_model(ARTIFACTS / "models" / "query_aware_fusion.joblib")

    Lt, It, St = train_mats["L"], train_mats["I"], train_mats["S"]
    Lv, Iv, Sv = val_mats["L"], val_mats["I"], val_mats["S"]

    val_rows = []

    # --- Baselines on VAL ---
    for name, sc in [
        ("tfidf", Lv),
        ("sbert", Sv),
        ("fixed_hybrid", fixed_hybrid_score(Lv, Iv, Sv, hybrid_w)),
        (
            "fusion_v1",
            fusion_v1.score(Lv, Iv, Sv, val_mats["Qid"], val_mats["Qlen"]),
        ),
    ]:
        row = eval_scores(sc, y_va, name)
        row["split"] = "val"
        val_rows.append(row)
        log(f"VAL {name:18s} MRR={row['MRR']:.4f} R@5={row['R@5']:.4f}")

    # --- Candidate configs ---
    candidates: list[tuple[str, FusionV2Config]] = [
        (
            "v2_zscore_only",
            FusionV2Config(
                name="v2_zscore_only",
                use_zscore=True,
                use_peakiness=False,
                use_i_gate=False,
                n_neg=12,
                epochs=50,
                seed=SEED,
            ),
        ),
        (
            "v2_full",
            FusionV2Config(
                name="v2_full",
                use_zscore=True,
                use_peakiness=True,
                use_i_gate=True,
                n_neg=16,
                hard_frac=0.5,
                epochs=60,
                lr=0.35,
                l2=5e-4,
                rank_weight=True,
                seed=SEED,
            ),
        ),
        (
            "v2_full_more_neg",
            FusionV2Config(
                name="v2_full_more_neg",
                use_zscore=True,
                use_peakiness=True,
                use_i_gate=True,
                n_neg=24,
                hard_frac=0.6,
                epochs=70,
                lr=0.3,
                l2=1e-3,
                rank_weight=True,
                seed=SEED,
            ),
        ),
        (
            "v2_no_I_peak",  # peakiness but no I gate; still uses I
            FusionV2Config(
                name="v2_no_I_peak",
                use_zscore=True,
                use_peakiness=True,
                use_i_gate=False,
                n_neg=16,
                epochs=60,
                seed=SEED,
            ),
        ),
        (
            "v2_semantic_prior",
            FusionV2Config(
                name="v2_semantic_prior",
                use_zscore=True,
                use_peakiness=True,
                use_i_gate=True,
                n_neg=16,
                epochs=60,
                lr=0.25,
                l2=1e-3,
                seed=SEED + 1,
            ),
        ),
    ]

    trained = {}
    for name, cfg in candidates:
        log(f"Training {name}…")
        feat_dim = 6 + (3 if cfg.use_peakiness else 0)
        model = QueryAwareFusionV2.create(cfg, feature_dim=feat_dim)
        hist = model.fit(Lt, It, St, train_Q, y_tr)
        scores = model.score_matrices(Lv, Iv, Sv, val_Q)
        row = eval_scores(scores, y_va, name)
        row["split"] = "val"
        row["final_loss"] = hist.get("final_loss")
        val_rows.append(row)
        trained[name] = model
        joblib.dump({"model": model, "cfg": cfg, "code_vocab": code_vocab, "df": df, "n_docs": n_docs}, out_dir / f"{name}.joblib")
        # inspect weight variation
        Lz, Iz, Sz, peaks = model.prepare_signals(Lv, Iv, Sv)
        Q = model.build_Q(val_Q, peaks)
        ws = model._weights(Q)
        log(
            f"VAL {name:18s} MRR={row['MRR']:.4f} R@5={row['R@5']:.4f} "
            f"w_std={ws.std(0).round(4).tolist()} loss={hist.get('final_loss'):.4f}"
        )

    # Extra ablation: v2_full but zero out I at inference (sanity)
    if "v2_full" in trained:
        m = trained["v2_full"]
        # temporary score without I: set I matrix to 0 after zscore path via cloning scores
        # Re-score with I * 0
        scores = m.score_matrices(Lv, np.zeros_like(Iv), Sv, val_Q)
        row = eval_scores(scores, y_va, "v2_full_I_zeroed")
        row["split"] = "val"
        val_rows.append(row)
        log(f"VAL {'v2_full_I_zeroed':18s} MRR={row['MRR']:.4f} R@5={row['R@5']:.4f}")

    # Retune fixed hybrid on VAL with z-scored signals? (fair "normalized hybrid")
    from codesearch.query_feats import zscore_rows

    Lvz, Ivz, Svz = zscore_rows(Lv), zscore_rows(Iv), zscore_rows(Sv)
    best_w, best_mrr = None, -1
    for wl in (0.2, 0.3, 0.4, 0.5, 0.6):
        for wi in (0.0, 0.1, 0.2, 0.3):
            ws = 1.0 - wl - wi
            if ws < 0.05:
                continue
            sc = fixed_hybrid_score(Lvz, Ivz, Svz, (wl, wi, ws))
            mrr = metrics_from_matrices(sc, y_va)["MRR"]
            if mrr > best_mrr:
                best_mrr, best_w = mrr, (wl, wi, ws)
    sc = fixed_hybrid_score(Lvz, Ivz, Svz, best_w)
    row = eval_scores(sc, y_va, "fixed_hybrid_zscore")
    row["split"] = "val"
    row["weights"] = list(best_w)
    val_rows.append(row)
    log(f"VAL fixed_hybrid_zscore w={best_w} MRR={row['MRR']:.4f} R@5={row['R@5']:.4f}")
    joblib.dump({"weights": best_w}, out_dir / "hybrid_zscore_weights.joblib")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(RESULTS / "optimize_val_comparison.csv", index=False)

    # Select best among NEW candidates + fusion_v1 + hybrids by VAL MRR
    selectable = val_df[~val_df.system.isin(["tfidf", "sbert", "v2_full_I_zeroed"])]
    # Prefer systems we could deploy: exclude pure diagnostics
    best_row = selectable.loc[selectable["MRR"].idxmax()]
    selected = best_row["system"]
    log(f"SELECTED on VAL: {selected} MRR={best_row['MRR']:.4f}")

    selection = {
        "selected": selected,
        "val_mrr": float(best_row["MRR"]),
        "val_r5": float(best_row["R@5"]),
        "val_p5": float(best_row["P@5"]),
        "selection_criterion": "max MRR on full-split validation",
        "test_not_used_for_selection": True,
        "prior_test_exposure_note": (
            "Earlier project phase already inspected test metrics for fusion_v1 vs hybrid. "
            "This sprint selected the new model using validation only; the final test numbers "
            "below are a single post-selection readout and should not be treated as a pristine holdout."
        ),
        "val_leaderboard": selectable.sort_values("MRR", ascending=False)[
            ["system", "MRR", "P@5", "R@5"]
        ].to_dict(orient="records"),
    }
    (RESULTS / "optimize_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # Persist selected artifact for ranker
    if selected in trained:
        joblib.dump(
            {
                "model": trained[selected],
                "code_vocab": code_vocab,
                "df": df,
                "n_docs": n_docs,
                "selected_name": selected,
            },
            ARTIFACTS / "models" / "query_aware_fusion_v2.joblib",
        )
        # also copy config marker
        (ARTIFACTS / "models" / "selected_fusion.txt").write_text(selected, encoding="utf-8")
    elif selected == "fixed_hybrid_zscore":
        joblib.dump(
            {"weights": best_w, "zscore": True, "selected_name": selected},
            ARTIFACTS / "models" / "query_aware_fusion_v2.joblib",
        )
        (ARTIFACTS / "models" / "selected_fusion.txt").write_text(selected, encoding="utf-8")
    elif selected in ("fixed_hybrid", "fusion_v1"):
        (ARTIFACTS / "models" / "selected_fusion.txt").write_text(selected, encoding="utf-8")
        joblib.dump({"selected_name": selected}, ARTIFACTS / "models" / "query_aware_fusion_v2.joblib")

    # --- ONE-SHOT test evaluation ---
    log("=== ONE-SHOT test evaluation (post-selection) ===")
    test_mats = get_mats(pipe, test_fn, "test")
    test_Q = batch_query_features(test_fn["query"].astype(str).tolist(), code_vocab, df, n_docs)
    y_te = label_matrix_from_split(len(test_fn))
    Lte, Ite, Ste = test_mats["L"], test_mats["I"], test_mats["S"]

    test_rows = []
    # always report frozen baselines + selected
    systems_test = {
        "tfidf": Lte,
        "sbert": Ste,
        "fixed_hybrid": fixed_hybrid_score(Lte, Ite, Ste, hybrid_w),
        "fusion_v1": fusion_v1.score(Lte, Ite, Ste, test_mats["Qid"], test_mats["Qlen"]),
        "fixed_hybrid_zscore": fixed_hybrid_score(
            zscore_rows(Lte), zscore_rows(Ite), zscore_rows(Ste), best_w
        ),
    }
    if selected in trained:
        systems_test[selected] = trained[selected].score_matrices(Lte, Ite, Ste, test_Q)
        # also report top-2 other v2 for transparency (NOT for selection — already selected)
        for alt in trained:
            if alt != selected:
                systems_test[alt] = trained[alt].score_matrices(Lte, Ite, Ste, test_Q)

    for name, sc in systems_test.items():
        row = eval_scores(sc, y_te, name)
        row["split"] = "test"
        row["selected"] = name == selected
        test_rows.append(row)
        mark = " <<SELECTED" if name == selected else ""
        log(f"TEST {name:22s} MRR={row['MRR']:.4f} P@5={row['P@5']:.4f} R@5={row['R@5']:.4f}{mark}")

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(RESULTS / "optimize_test_oneshot.csv", index=False)

    # Target check vs fixed_hybrid on TEST
    hy = test_df[test_df.system == "fixed_hybrid"].iloc[0]
    sel = test_df[test_df.system == selected].iloc[0]
    delta = float(sel["MRR"] - hy["MRR"])
    summary = {
        **selection,
        "test_fixed_hybrid_mrr": float(hy["MRR"]),
        "test_selected_mrr": float(sel["MRR"]),
        "test_delta_mrr_vs_hybrid": delta,
        "target_delta": 0.010,
        "target_achieved": delta >= 0.010 - 1e-9,
        "test_selected_metrics": {
            "MRR": float(sel["MRR"]),
            "P@5": float(sel["P@5"]),
            "R@5": float(sel["R@5"]),
        },
    }
    (RESULTS / "optimize_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        f"TARGET: hybrid={hy['MRR']:.4f} selected={sel['MRR']:.4f} "
        f"delta={delta:+.4f} achieved={summary['target_achieved']}"
    )
    log("DONE")


if __name__ == "__main__":
    main()
