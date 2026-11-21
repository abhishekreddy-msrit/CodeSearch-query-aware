#!/usr/bin/env python
"""
End-to-end ML pipeline: features → baselines → query-aware models → full-split eval → ablations.

Usage (from repo root):
  python scripts/run_pipeline.py --smoke          # tiny sanity check
  python scripts/run_pipeline.py                  # full val+test experiments
  python scripts/run_pipeline.py --skip-train     # reuse artifacts/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.config import (  # noqa: E402
    ABLATIONS,
    ALL_FEATURES,
    ARTIFACTS,
    BASE_FEATURES,
    DATA_DIR,
    LOGS,
    RESULTS,
    SEED,
)
from codesearch.data import check_scores, load_functions, load_pairs, load_pool  # noqa: E402
from codesearch.evaluate import (  # noqa: E402
    evaluate_matrix_system,
    label_matrix_from_split,
    materialize_scores,
    verify_against_pool,
)
from codesearch.features import FeaturePipeline  # noqa: E402
from codesearch.models import (  # noqa: E402
    explain_lr,
    fixed_hybrid_score,
    load_model,
    predict_proba_positive,
    save_model,
    select_feature_columns,
    train_logistic,
    train_random_forest,
    tune_hybrid_weights,
)
from codesearch.query_aware_fusion import QueryAwareFusion  # noqa: E402


def log(msg: str):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_seeds(seed: int = SEED):
    np.random.seed(seed)


def build_or_load_features(functions: pd.DataFrame, skip_train: bool) -> FeaturePipeline:
    art_path = ARTIFACTS / "feature_pipeline.joblib"
    pipe = FeaturePipeline()
    if skip_train and art_path.exists():
        log("Loading feature artifacts…")
        from codesearch.features import FeatureArtifacts

        arts = FeatureArtifacts.load(art_path)
        pipe.tfidf = arts.tfidf
        pipe.scaler = arts.scaler
        pipe.feature_names = arts.feature_names
        pipe.sbert_model_name = arts.sbert_name
        pipe.code_emb = arts.code_emb
        pipe.func_ids = arts.func_ids
        pipe.load_side(art_path)
        return pipe

    log("Fitting TF-IDF on train + encoding all function bodies with SBERT…")
    t0 = time.time()
    pipe.fit(functions)
    log(f"Feature fit done in {time.time() - t0:.1f}s")
    pipe.save(art_path)
    return pipe


def train_supervised(pipe: FeaturePipeline, skip_train: bool) -> dict:
    """Train non-adaptive + query-aware LR/RF on pairs_train; select on pairs_val (pool10)."""
    out = {}
    model_dir = ARTIFACTS / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "lr_aware": model_dir / "lr_query_aware.joblib",
        "lr_base": model_dir / "lr_nonadaptive.joblib",
        "rf_aware": model_dir / "rf_query_aware.joblib",
        "rf_base": model_dir / "rf_nonadaptive.joblib",
        "fusion": model_dir / "query_aware_fusion.joblib",
        "meta": model_dir / "train_meta.json",
    }

    if skip_train and all(
        paths[k].exists() for k in ("lr_aware", "lr_base", "rf_aware", "rf_base", "meta")
    ):
        log("Loading trained models…")
        for k in ("lr_aware", "lr_base", "rf_aware", "rf_base"):
            out[k] = load_model(paths[k])
        out["meta"] = json.loads(paths["meta"].read_text(encoding="utf-8"))
        from codesearch.features import FeatureArtifacts

        arts = FeatureArtifacts.load(ARTIFACTS / "feature_pipeline.joblib")
        pipe.scaler = arts.scaler
        if paths["fusion"].exists():
            out["fusion"] = load_model(paths["fusion"])
            return out
        log("Fusion model missing — will train it below using cached pair features…")
        # fall through to build features + fusion only
        skip_classifiers = True
    else:
        skip_classifiers = False

    log("Building training pair features (pairs_train)…")
    train_pairs = load_pairs("train")
    meta_tr, X_tr_raw, y_tr = pipe.pair_feature_frame(
        train_pairs, query_emb_cache=ARTIFACTS / "embeddings" / "train_query_emb.npy"
    )
    log(f"Train pairs: X={X_tr_raw.shape}, positives={int(y_tr.sum())}")

    if not skip_classifiers:
        pipe.fit_scaler(X_tr_raw)
        X_tr = pipe.transform(X_tr_raw)

        log("Building validation pair features (pairs_val, model selection)…")
        val_pairs = load_pairs("val")
        _, X_va_raw, y_va = pipe.pair_feature_frame(
            val_pairs, query_emb_cache=ARTIFACTS / "embeddings" / "val_pool10_query_emb.npy"
        )
        X_va = pipe.transform(X_va_raw)

        log("Training query-aware Logistic Regression…")
        lr_aware = train_logistic(X_tr, y_tr)
        X_tr_base = select_feature_columns(X_tr, ALL_FEATURES, BASE_FEATURES)
        X_va_base = select_feature_columns(X_va, ALL_FEATURES, BASE_FEATURES)
        log("Training non-adaptive Logistic Regression…")
        lr_base = train_logistic(X_tr_base, y_tr)

        log("Training query-aware Random Forest…")
        rf_aware = train_random_forest(X_tr, y_tr)
        log("Training non-adaptive Random Forest…")
        rf_base = train_random_forest(X_tr_base, y_tr)

        def pair_mrr(model, X, pairs):
            scores = predict_proba_positive(model, X)
            df = pairs[["query_id", "candidate_func_id", "label"]].copy()
            df["score"] = scores
            mrrs = []
            for _, g in df.groupby("query_id"):
                order = g.sort_values("score", ascending=False)
                ranks = np.where(order["label"].to_numpy() == 1)[0]
                if len(ranks):
                    mrrs.append(1.0 / (ranks[0] + 1))
            return float(np.mean(mrrs)) if mrrs else 0.0

        meta = {
            "seed": SEED,
            "n_train_pairs": int(len(y_tr)),
            "pos_rate": float(y_tr.mean()),
            "features_aware": ALL_FEATURES,
            "features_nonadaptive": BASE_FEATURES,
            "val_pool10_mrr": {
                "lr_query_aware": pair_mrr(lr_aware, X_va, val_pairs),
                "lr_nonadaptive": pair_mrr(lr_base, X_va_base, val_pairs),
                "rf_query_aware": pair_mrr(rf_aware, X_va, val_pairs),
                "rf_nonadaptive": pair_mrr(rf_base, X_va_base, val_pairs),
            },
            "lr_aware_coefs": explain_lr(lr_aware, ALL_FEATURES),
            "lr_base_coefs": explain_lr(lr_base, BASE_FEATURES),
        }
        log(f"Val pool10 MRR: {json.dumps(meta['val_pool10_mrr'], indent=2)}")

        save_model(lr_aware, paths["lr_aware"])
        save_model(lr_base, paths["lr_base"])
        save_model(rf_aware, paths["rf_aware"])
        save_model(rf_base, paths["rf_base"])
        pipe.save(ARTIFACTS / "feature_pipeline.joblib")
        out.update(
            {"lr_aware": lr_aware, "lr_base": lr_base, "rf_aware": rf_aware, "rf_base": rf_base}
        )
    else:
        meta = out.get("meta") or json.loads(paths["meta"].read_text(encoding="utf-8"))
        for k in ("lr_aware", "lr_base", "rf_aware", "rf_base"):
            out[k] = load_model(paths[k])

    log("Training query-aware fusion (pairwise ranking loss)…")
    fusion = QueryAwareFusion.from_uniform()
    hist = fusion.fit_pairwise(
        L=X_tr_raw[:, ALL_FEATURES.index("L")],
        I=X_tr_raw[:, ALL_FEATURES.index("I")],
        S=X_tr_raw[:, ALL_FEATURES.index("S")],
        Qid=X_tr_raw[:, ALL_FEATURES.index("Qid")],
        Qlen=X_tr_raw[:, ALL_FEATURES.index("Qlen")],
        y=y_tr,
        query_ids=meta_tr["query_id"].to_numpy(),
        epochs=35,
        seed=SEED,
    )
    save_model(fusion, paths["fusion"])
    meta["fusion_train_loss"] = hist.get("final_loss")
    meta["fusion_W"] = fusion.W.tolist()
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Fusion final pairwise loss={hist.get('final_loss')}")

    out.update({"fusion": fusion, "meta": meta})
    return out


def run_full_split(
    pipe: FeaturePipeline,
    models: dict,
    split: str,
    smoke_n: int | None = None,
) -> pd.DataFrame:
    """Evaluate all systems on full-split benchmark."""
    log(f"=== Full-split evaluation: {split} ===")
    functions = load_functions()
    split_fn = functions[functions["split"] == split].reset_index(drop=True)
    if smoke_n:
        split_fn = split_fn.iloc[:smoke_n].reset_index(drop=True)
        log(f"SMOKE mode: using first {smoke_n} functions only")

    mats = pipe.score_matrices_for_split(
        split_fn,
        query_emb_cache=ARTIFACTS / "embeddings" / f"{split}_query_emb_n{len(split_fn)}.npy",
    )
    func_ids = mats["func_ids"]
    n = len(func_ids)
    labels = label_matrix_from_split(n)

    # For smoke, pool check is skipped (pool is full 1200). For full, verify one system.
    pool = None
    if smoke_n is None:
        pool = load_pool("full", split)
        # Ensure our func order matches pool candidate set
        assert set(func_ids) == set(pool["candidate_func_id"].unique())

    L, I, S = mats["L"], mats["I"], mats["S"]

    # Tune hybrid weights on VAL only (never on test)
    hybrid_path = ARTIFACTS / "models" / "hybrid_weights.json"
    hybrid_path.parent.mkdir(parents=True, exist_ok=True)
    if split == "val":
        log("Tuning fixed hybrid weights on validation matrices…")
        w, best_mrr = tune_hybrid_weights(L, I, S, labels)
        hybrid_path.write_text(
            json.dumps({"weights": list(w), "val_mrr": best_mrr}, indent=2), encoding="utf-8"
        )
        log(f"Hybrid weights={w} val_MRR={best_mrr:.4f}")
    elif hybrid_path.exists():
        w = tuple(json.loads(hybrid_path.read_text(encoding="utf-8"))["weights"])
        log(f"Using hybrid weights from val: {w}")
    else:
        w = (0.4, 0.2, 0.4)
        log(f"WARNING: no hybrid weights file; using default {w}")

    systems = {
        "tfidf": L,
        "sbert": S,
        "fixed_hybrid": fixed_hybrid_score(L, I, S, w),
        "identifier_only": I,
    }

    # Supervised: scale the full 11-d vector (scaler was fit on ALL_FEATURES),
    # then select columns so non-adaptive models see the same transformed inputs
    # they were trained on.
    def supervised_scores(model, feat_names):
        shape = L.shape
        cols = [mats[n].ravel() for n in ALL_FEATURES]
        X_full = pipe.transform(np.column_stack(cols).astype(np.float32))
        X = select_feature_columns(X_full, ALL_FEATURES, feat_names)
        return predict_proba_positive(model, X).reshape(shape)

    systems["lr_nonadaptive"] = supervised_scores(models["lr_base"], BASE_FEATURES)
    systems["lr_query_aware"] = supervised_scores(models["lr_aware"], ALL_FEATURES)
    systems["rf_nonadaptive"] = supervised_scores(models["rf_base"], BASE_FEATURES)
    systems["rf_query_aware"] = supervised_scores(models["rf_aware"], ALL_FEATURES)

    if models.get("fusion") is not None:
        systems["query_aware_fusion"] = models["fusion"].score(
            L, I, S, mats["Qid"], mats["Qlen"]
        )

    # Ablations on query-aware LR feature subsets (retrain not needed: zero-out via
    # dedicated models trained in ablation section — here use coefficient masking
    # only for interactions ablation by scoring with BASE features model already done.
    # Full ablations with retraining happen in run_ablations().

    rows = []
    for name, scores in systems.items():
        row = evaluate_matrix_system(name, scores, labels, func_ids, bootstrap=(smoke_n is None))
        row["split"] = split
        row["benchmark"] = "full" if smoke_n is None else f"smoke_{smoke_n}"
        rows.append(row)
        log(
            f"{split:4s} {name:18s}  MRR={row['MRR']:.4f}  "
            f"P@5={row['P@5']:.4f}  R@5={row['R@5']:.4f}"
        )

    # Verify pool integrity for best supervised system on full benchmark
    if pool is not None:
        score_df = materialize_scores(systems["lr_query_aware"], func_ids)
        log("Running check_scores on lr_query_aware vs eval_full pool…")
        verify_against_pool(score_df, pool)
        log("check_scores OK")

    return pd.DataFrame(rows)


def run_ablations(pipe: FeaturePipeline, split: str = "val") -> pd.DataFrame:
    """Retrain LR on feature subsets; evaluate on full-split val (or smoke)."""
    log("=== Ablation study (LR feature subsets, full-split val) ===")
    train_pairs = load_pairs("train")
    _, X_tr_raw, y_tr = pipe.pair_feature_frame(
        train_pairs, query_emb_cache=ARTIFACTS / "embeddings" / "train_query_emb.npy"
    )
    if pipe.scaler is None:
        pipe.fit_scaler(X_tr_raw)
    X_tr_full = pipe.transform(X_tr_raw)

    functions = load_functions()
    split_fn = functions[functions["split"] == split].reset_index(drop=True)
    mats = pipe.score_matrices_for_split(
        split_fn,
        query_emb_cache=ARTIFACTS / "embeddings" / f"{split}_query_emb_n{len(split_fn)}.npy",
    )
    labels = label_matrix_from_split(len(mats["func_ids"]))

    rows = []
    for name, feat_names in ABLATIONS.items():
        log(f"Ablation: {name} features={feat_names}")
        X = select_feature_columns(X_tr_full, ALL_FEATURES, feat_names)
        clf = train_logistic(X, y_tr)

        shape = mats["L"].shape
        cols = [mats[n].ravel() for n in ALL_FEATURES]
        Xv = pipe.transform(np.column_stack(cols).astype(np.float32))
        Xv = select_feature_columns(Xv, ALL_FEATURES, feat_names)
        scores = predict_proba_positive(clf, Xv).reshape(shape)
        row = evaluate_matrix_system(f"ablation_{name}", scores, labels, mats["func_ids"], bootstrap=True)
        row["split"] = split
        row["ablation"] = name
        row["n_features"] = len(feat_names)
        rows.append(row)
        log(f"  MRR={row['MRR']:.4f}  R@5={row['R@5']:.4f}")
        save_model(clf, ARTIFACTS / "models" / f"ablation_{name}.joblib")

    return pd.DataFrame(rows)


def run_pool10_secondary(pipe: FeaturePipeline, models: dict, split: str) -> pd.DataFrame:
    """Secondary 10-candidate benchmark for all systems."""
    log(f"=== Secondary pool10 evaluation: {split} ===")
    pairs = load_pairs(split)
    pool = load_pool("pool10", split)
    _, X_raw, _ = pipe.pair_feature_frame(
        pairs, query_emb_cache=ARTIFACTS / "embeddings" / f"{split}_pool10_query_emb.npy"
    )
    X_full = pipe.transform(X_raw)

    # Unsupervised from stored pair-level reconstruction
    # Recompute L,I,S from feature columns
    feat_df = pd.DataFrame(X_raw, columns=ALL_FEATURES)  # unscaled raw for L,I,S
    # Actually X_raw is unscaled with interactions already — L,I,S are first three
    L = X_raw[:, ALL_FEATURES.index("L")]
    I = X_raw[:, ALL_FEATURES.index("I")]
    S = X_raw[:, ALL_FEATURES.index("S")]
    w = tuple(json.loads((ARTIFACTS / "models" / "hybrid_weights.json").read_text())["weights"])

    systems = {
        "tfidf": L,
        "sbert": S,
        "fixed_hybrid": fixed_hybrid_score(L, I, S, w),
        "lr_nonadaptive": predict_proba_positive(
            models["lr_base"], select_feature_columns(X_full, ALL_FEATURES, BASE_FEATURES)
        ),
        "lr_query_aware": predict_proba_positive(models["lr_aware"], X_full),
        "rf_nonadaptive": predict_proba_positive(
            models["rf_base"], select_feature_columns(X_full, ALL_FEATURES, BASE_FEATURES)
        ),
        "rf_query_aware": predict_proba_positive(models["rf_aware"], X_full),
    }
    if models.get("fusion") is not None:
        systems["query_aware_fusion"] = models["fusion"].score(
            L,
            I,
            S,
            X_raw[:, ALL_FEATURES.index("Qid")],
            X_raw[:, ALL_FEATURES.index("Qlen")],
        )

    rows = []
    for name, scores in systems.items():
        df = pairs[["query_id", "candidate_func_id", "label"]].copy()
        df["score"] = scores
        check_scores(df[["query_id", "candidate_func_id", "score"]], pool)
        # metrics via groupby
        mrrs, p5s, r5s = [], [], []
        for _, g in df.groupby("query_id"):
            g = g.sort_values("score", ascending=False)
            labels = g["label"].to_numpy()
            rank = int(np.where(labels == 1)[0][0]) + 1
            mrrs.append(1.0 / rank)
            top5 = labels[:5]
            p5s.append(top5.sum() / 5.0)
            r5s.append(top5.sum() / max(labels.sum(), 1))
        row = {
            "system": name,
            "split": split,
            "benchmark": "pool10",
            "MRR": float(np.mean(mrrs)),
            "P@5": float(np.mean(p5s)),
            "R@5": float(np.mean(r5s)),
            "n_queries": int(df["query_id"].nunique()),
        }
        rows.append(row)
        log(f"pool10 {split} {name:18s} MRR={row['MRR']:.4f} R@5={row['R@5']:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Tiny subset for sanity check")
    ap.add_argument("--skip-train", action="store_true", help="Reuse saved artifacts")
    ap.add_argument("--skip-test", action="store_true", help="Only run validation split")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    set_seeds()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (LOGS / "pipeline.log").write_text("", encoding="utf-8")

    log(f"DATA_DIR={DATA_DIR}")
    assert DATA_DIR.exists(), f"Missing data dir: {DATA_DIR}"

    functions = load_functions()
    log(f"Loaded functions: {len(functions)}  splits={functions['split'].value_counts().to_dict()}")

    pipe = build_or_load_features(functions, skip_train=args.skip_train)
    models = train_supervised(pipe, skip_train=args.skip_train)

    smoke_n = 50 if args.smoke else None
    all_rows = []

    val_rows = run_full_split(pipe, models, "val", smoke_n=smoke_n)
    all_rows.append(val_rows)

    if not args.smoke:
        # secondary pool10 on val
        all_rows.append(run_pool10_secondary(pipe, models, "val"))
        if not args.skip_ablation:
            all_rows.append(run_ablations(pipe, split="val"))
        if not args.skip_test:
            all_rows.append(run_full_split(pipe, models, "test", smoke_n=None))
            all_rows.append(run_pool10_secondary(pipe, models, "test"))

    results = pd.concat(all_rows, ignore_index=True)
    out_csv = RESULTS / ("metrics_smoke.csv" if args.smoke else "metrics_full.csv")
    results.to_csv(out_csv, index=False)
    results.to_json(out_csv.with_suffix(".json"), orient="records", indent=2)
    log(f"Wrote {out_csv}")

    # Summary pivot for report
    if not args.smoke:
        full = results[results["benchmark"] == "full"]
        if len(full):
            pivot = full.pivot_table(index="system", columns="split", values=["MRR", "P@5", "R@5"])
            pivot.to_csv(RESULTS / "comparison_table.csv")
            log("\n" + pivot.to_string())

    # Decision note for research question
    summary = {
        "hypothesis": (
            "Query-aware supervised feature interactions improve ranking over "
            "conventional retrieval and fixed-weight / non-adaptive supervised baselines."
        ),
        "primary_benchmark": "eval_full_{val,test} (1200 queries × 1200 candidates)",
        "embedding_model": pipe.sbert_model_name,
        "hybrid_weights_file": str(ARTIFACTS / "models" / "hybrid_weights.json"),
        "results_file": str(out_csv),
        "notes": [
            "Candidate text = func_name + docstring-stripped code (no query leakage).",
            "TF-IDF fitted on train queries+code only.",
            "Hybrid weights tuned on validation full-split only.",
            "Labels are silver (docstring–function pairs).",
        ],
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
