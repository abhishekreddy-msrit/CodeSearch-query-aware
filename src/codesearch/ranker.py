"""Inference API: rank candidate functions for a free-form query.

Design rules for the demo:
- Changing the model always triggers independent scoring (no label-only swaps).
- The primary `score` column is ALWAYS the value used to rank that system.
- Component L/I/S are diagnostic; they are never mislabeled as hybrid unless
  the active system is actually fixed_hybrid.
- Silent fallbacks are forbidden: unavailable systems raise or return an error flag.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import ARTIFACTS, DATA_DIR, RESULTS
from .data import load_functions
from .embeddings import code_repr
from .features import FeatureArtifacts, FeaturePipeline, identifier_overlap_pair, query_features
from .models import fixed_hybrid_score, load_model
from .text import tokenize_for_tfidf

SIG_RE = re.compile(r"^\s*(async\s+)?def\s+\w+\s*\([^)]*\)\s*(->\s*[^:]+)?:", re.M)

# Demo-facing catalog. Keys must match dispatch in score_system().
MODEL_CATALOG = {
    "tfidf": {
        "label": "Lexical Search (TF-IDF)",
        "short": "Lexical",
        "blurb": "Ranks by token/text overlap between the query and function code.",
        "kind": "single",
        "score_name": "Lexical Ranking Score",
        "signals_computed": ["L", "I", "S"],
        "signals_used": ["L"],
        "uses": ["L"],
    },
    "identifier": {
        "label": "Identifier Search",
        "short": "Identifier",
        "blurb": "Ranks by overlap between query tokens and function/code identifiers.",
        "kind": "single",
        "score_name": "Identifier Ranking Score",
        "signals_computed": ["L", "I", "S"],
        "signals_used": ["I"],
        "uses": ["I"],
    },
    "sbert": {
        "label": "Semantic Search (Sentence-BERT)",
        "short": "Semantic",
        "blurb": "Ranks by embedding cosine similarity (meaning, not exact wording).",
        "kind": "single",
        "score_name": "Semantic Ranking Score",
        "signals_computed": ["L", "I", "S"],
        "signals_used": ["S"],
        "uses": ["S"],
    },
    "fixed_hybrid": {
        "label": "Fixed Hybrid (validation-selected default)",
        "short": "Fixed Hybrid",
        "blurb": "Combines lexical, identifier, and semantic scores with fixed validation-tuned weights.",
        "kind": "hybrid",
        "score_name": "Hybrid Ranking Score",
        "signals_computed": ["L", "I", "S"],
        "signals_used": ["L", "I", "S"],
        "uses": ["L", "I", "S"],
    },
    "query_aware_fusion": {
        "label": "Query-Aware Fusion (experimental)",
        "short": "Query-Aware",
        "blurb": (
            "Query-conditioned fusion weights over L/I/S from query length and identifier density. "
            "Experimental — did not beat Fixed Hybrid on validation."
        ),
        "kind": "adaptive",
        "score_name": "Query-Aware Ranking Score",
        "signals_computed": ["L", "I", "S", "Qid", "Qlen"],
        "signals_used": ["L", "I", "S", "Qid", "Qlen"],
        "uses": ["L", "I", "S", "Qid", "Qlen"],
    },
}

# Faculty compare set (identifier remains available in single-model mode).
DEFAULT_COMPARE = ["tfidf", "sbert", "fixed_hybrid", "query_aware_fusion"]


def extract_signature(code: str, func_name: str = "") -> str:
    m = SIG_RE.search(code or "")
    if m:
        return " ".join(m.group(0).strip().rstrip(":").split())
    short = (func_name or "function").split(".")[-1]
    return f"def {short}(...)"


def load_offline_benchmarks() -> pd.DataFrame:
    """Load verified full-split metrics if present; empty frame otherwise."""
    path = RESULTS / "primary_fullsplit_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


class CodeSearchRanker:
    """Load frozen artifacts once; rank with explicitly selected systems."""

    def __init__(self, artifacts_dir: Path | str = ARTIFACTS, data_dir: Path | str = DATA_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        self.functions = load_functions(self.data_dir)
        art_path = self.artifacts_dir / "feature_pipeline.joblib"
        if not art_path.exists():
            raise FileNotFoundError(
                f"Missing feature artifacts at {art_path}. Run scripts/run_pipeline.py first."
            )
        arts = FeatureArtifacts.load(art_path)
        self.pipe = FeaturePipeline(arts.sbert_name)
        self.pipe.tfidf = arts.tfidf
        self.pipe.scaler = arts.scaler
        self.pipe.feature_names = arts.feature_names
        self.pipe.code_emb = arts.code_emb
        self.pipe.func_ids = arts.func_ids
        self.pipe.load_side(art_path)

        self.models = {}
        fusion_path = self.artifacts_dir / "models" / "query_aware_fusion.joblib"
        if fusion_path.exists():
            self.models["query_aware_fusion"] = load_model(fusion_path)

        hw = self.artifacts_dir / "models" / "hybrid_weights.json"
        self.hybrid_w = (
            tuple(json.loads(hw.read_text(encoding="utf-8"))["weights"]) if hw.exists() else (0.4, 0.2, 0.4)
        )
        self.default_system = "fixed_hybrid"

        ordered = [self.pipe._code_text[fid] for fid in self.pipe.func_ids]
        self._code_tfidf = self.pipe.tfidf.transform([tokenize_for_tfidf(c) for c in ordered])
        self._id_list = [self.pipe._id_sets[fid] for fid in self.pipe.func_ids]

        self.corpus_stats = {
            "n_functions": int(len(self.functions)),
            "n_repos": int(self.functions["repository_name"].nunique()),
            "splits": self.functions["split"].value_counts().to_dict(),
            "hybrid_weights": {
                "L": float(self.hybrid_w[0]),
                "I": float(self.hybrid_w[1]),
                "S": float(self.hybrid_w[2]),
            },
            "default_system": self.default_system,
        }
        self.benchmarks = load_offline_benchmarks()

    def available_systems(self) -> dict[str, dict]:
        """Return catalog entries that can actually run."""
        out = {}
        for key, meta in MODEL_CATALOG.items():
            if key in ("tfidf", "identifier", "sbert", "fixed_hybrid"):
                out[key] = meta
            elif key in self.models:
                out[key] = meta
        return out

    def hybrid_formula_text(self) -> str:
        wl, wi, ws = self.hybrid_w
        return f"{wl:.2f}·Lexical + {wi:.2f}·Identifier + {ws:.2f}·Semantic"

    def _candidate_indices(self, split: str | None) -> np.ndarray:
        if split is None or split == "all":
            return np.arange(len(self.pipe.func_ids), dtype=np.int64)
        split_ids = set(self.functions.loc[self.functions["split"] == split, "func_id"])
        return np.array([i for i, fid in enumerate(self.pipe.func_ids) if fid in split_ids], dtype=np.int64)

    def compute_signals(self, query: str, split: str | None = "all") -> dict:
        """Compute L/I/S once for a candidate pool. Shared by all systems."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Empty query")
        idxs = self._candidate_indices(split)
        fids = [self.pipe.func_ids[i] for i in idxs]

        qv = self.pipe.tfidf.transform([tokenize_for_tfidf(query)])
        L = cosine_similarity(qv, self._code_tfidf[idxs]).ravel().astype(np.float32)
        I = np.array([identifier_overlap_pair(query, self._id_list[i]) for i in idxs], dtype=np.float32)
        q_emb = self.pipe.sbert.encode([query], normalize_embeddings=True, show_progress_bar=False)
        S = (q_emb @ self.pipe.code_emb[idxs].T).ravel().astype(np.float32)
        Qid, Qlen = query_features([query])

        meta_cols = [
            c
            for c in [
                "func_name",
                "repository_name",
                "func_path_in_repository",
                "code",
                "func_code_url",
                "split",
                "code_n_lines",
            ]
            if c in self.functions.columns
        ]
        fmap = self.functions.drop_duplicates("func_id").set_index("func_id")
        base = fmap.loc[fids, meta_cols].reset_index()
        base["signature"] = [
            extract_signature(str(c), str(n)) for c, n in zip(base["code"], base["func_name"])
        ]
        return {
            "query": query,
            "split": split or "all",
            "n_candidates": int(len(fids)),
            "base": base,
            "L": L,
            "I": I,
            "S": S,
            "Qid": float(Qid[0]),
            "Qlen": float(Qlen[0]),
        }

    def score_system(self, system: str, signals: dict) -> tuple[np.ndarray, dict]:
        """
        Return (scores, meta) for one system.
        Raises KeyError if the system cannot run — never silently swaps models.
        """
        avail = self.available_systems()
        if system not in avail:
            raise KeyError(f"System '{system}' is not available. Available: {list(avail)}")

        L, I, S = signals["L"], signals["I"], signals["S"]
        Qid = np.full_like(L, signals["Qid"])
        Qlen = np.full_like(L, signals["Qlen"])
        info = {
            "system": system,
            "label": avail[system]["label"],
            "kind": avail[system]["kind"],
            "score_name": avail[system]["score_name"],
            "uses": list(avail[system]["uses"]),
            "formula": None,
            "adaptive_weights": None,
            "fallback": False,
        }

        info["signals_computed"] = list(avail[system]["signals_computed"])
        info["signals_used"] = list(avail[system]["signals_used"])

        if system == "tfidf":
            scores = L.copy()
            info["formula"] = "Lexical Ranking Score = TF-IDF cosine(query, code)"
        elif system == "identifier":
            scores = I.copy()
            info["formula"] = "Identifier Ranking Score = identifier overlap(query, code ids)"
        elif system == "sbert":
            scores = S.copy()
            info["formula"] = "Semantic Ranking Score = SBERT cosine(query, code)"
        elif system == "fixed_hybrid":
            scores = fixed_hybrid_score(L, I, S, self.hybrid_w)
            info["formula"] = f"Hybrid Ranking Score = {self.hybrid_formula_text()}"
        elif system == "query_aware_fusion":
            model = self.models["query_aware_fusion"]
            scores = model.score(L, I, S, Qid, Qlen)
            w = model.weights(np.array([signals["Qid"]]), np.array([signals["Qlen"]]))[0]
            wl, wi, ws = float(w[0]), float(w[1]), float(w[2])
            info["adaptive_weights"] = {"wL": wl, "wI": wi, "wS": ws}
            # Honesty check vs fixed hybrid — adaptation can be weak.
            fixed = np.array(self.hybrid_w, dtype=np.float64)
            adapt = np.array([wl, wi, ws], dtype=np.float64)
            max_shift = float(np.max(np.abs(adapt - fixed)))
            info["weight_shift_from_fixed"] = max_shift
            info["adaptation_note"] = (
                "Query-conditioned weights are close to Fixed Hybrid for this query "
                f"(max |Δw|={max_shift:.3f}). Adaptation is weak here."
                if max_shift < 0.08
                else f"Query-conditioned weights differ from Fixed Hybrid (max |Δw|={max_shift:.3f})."
            )
            info["formula"] = (
                f"Query-Aware Ranking Score = {wl:.3f}·L + {wi:.3f}·I + {ws:.3f}·S "
                f"(from Qid={signals['Qid']:.3f}, Qlen={signals['Qlen']:.3f})"
            )
        else:
            raise KeyError(f"No scoring path for '{system}'")

        return scores.astype(np.float32), info

    def rank(
        self,
        query: str,
        system: str | None = None,
        top_k: int = 5,
        split: str | None = "all",
    ) -> pd.DataFrame:
        system = system or self.default_system
        signals = self.compute_signals(query, split=split)
        scores, info = self.score_system(system, signals)
        return self._pack_hits(signals, scores, info, top_k)

    def rank_many(
        self,
        query: str,
        systems: list[str],
        top_k: int = 5,
        split: str | None = "all",
    ) -> dict[str, pd.DataFrame]:
        """Independently score each system on the same query and candidate pool."""
        signals = self.compute_signals(query, split=split)
        out = {}
        for sys in systems:
            scores, info = self.score_system(sys, signals)
            out[sys] = self._pack_hits(signals, scores, info, top_k)
        return out

    def _pack_hits(self, signals: dict, scores: np.ndarray, info: dict, top_k: int) -> pd.DataFrame:
        out = signals["base"].copy()
        out["score"] = scores  # PRIMARY ranking score for THIS system
        out["L"] = signals["L"]
        out["I"] = signals["I"]
        out["S"] = signals["S"]
        out["hybrid_score"] = fixed_hybrid_score(
            signals["L"], signals["I"], signals["S"], self.hybrid_w
        )
        out["Qid"] = signals["Qid"]
        out["Qlen"] = signals["Qlen"]
        out["system"] = info["system"]
        out["system_label"] = info["label"]
        out["score_name"] = info["score_name"]
        out["formula"] = info["formula"]
        out["kind"] = info["kind"]
        out["signals_computed"] = ",".join(info.get("signals_computed", []))
        out["signals_used"] = ",".join(info.get("signals_used", []))
        out["n_candidates"] = signals["n_candidates"]
        out["query"] = signals["query"]
        if info.get("adaptive_weights"):
            out["wL"] = info["adaptive_weights"]["wL"]
            out["wI"] = info["adaptive_weights"]["wI"]
            out["wS"] = info["adaptive_weights"]["wS"]
            out["weight_shift_from_fixed"] = info.get("weight_shift_from_fixed", 0.0)
            out["adaptation_note"] = info.get("adaptation_note", "")
        out = out.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def comparison_summary(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """One row per system: top-1 function and score for side-by-side demo."""
        rows = []
        for sys, df in results.items():
            if df.empty:
                continue
            top = df.iloc[0]
            rows.append(
                {
                    "system": sys,
                    "label": top["system_label"],
                    "rank1_function": top["func_name"],
                    "rank1_score": float(top["score"]),
                    "score_name": top["score_name"],
                    "formula": top["formula"],
                    "repo": top.get("repository_name", ""),
                }
            )
        return pd.DataFrame(rows)
