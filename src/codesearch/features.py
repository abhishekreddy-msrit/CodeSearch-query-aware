"""Feature extraction: lexical (TF-IDF), identifier overlap, SBERT, query cues, interactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES, ARTIFACTS, SEED
from .embeddings import cached_encode, code_repr, load_sbert
from .text import parse_identifier_list, query_id_density, query_len_norm, tokenize_for_tfidf, word_tokens


@dataclass
class FeatureArtifacts:
    tfidf: TfidfVectorizer
    scaler: StandardScaler | None = None
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FEATURES))
    sbert_name: str = ""
    code_emb: np.ndarray | None = None  # aligned to functions_all order
    func_ids: list[str] = field(default_factory=list)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tfidf": self.tfidf,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "sbert_name": self.sbert_name,
            "func_ids": self.func_ids,
        }
        joblib.dump(payload, path)
        if self.code_emb is not None:
            np.save(path.with_suffix(".code_emb.npy"), self.code_emb)

    @classmethod
    def load(cls, path: Path) -> "FeatureArtifacts":
        path = Path(path)
        payload = joblib.load(path)
        emb_path = path.with_suffix(".code_emb.npy")
        code_emb = np.load(emb_path) if emb_path.exists() else None
        return cls(
            tfidf=payload["tfidf"],
            scaler=payload.get("scaler"),
            feature_names=payload["feature_names"],
            sbert_name=payload.get("sbert_name", ""),
            code_emb=code_emb,
            func_ids=payload.get("func_ids", []),
        )


def fit_tfidf(train_functions: pd.DataFrame) -> TfidfVectorizer:
    """Fit TF-IDF on train queries + docstring-stripped code (no test leakage)."""
    docs = []
    for _, r in train_functions.iterrows():
        docs.append(tokenize_for_tfidf(str(r["query"])))
        docs.append(tokenize_for_tfidf(code_repr(r)))
    vec = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_features=50_000,
    )
    vec.fit(docs)
    return vec


def tfidf_cosine(vec: TfidfVectorizer, queries: list[str], codes: list[str]) -> np.ndarray:
    """Return (n_queries, n_codes) cosine similarities."""
    q = vec.transform([tokenize_for_tfidf(q) for q in queries])
    c = vec.transform([tokenize_for_tfidf(c) for c in codes])
    return cosine_similarity(q, c).astype(np.float32)


def identifier_overlap_matrix(queries: list[str], id_sets: list[set[str]]) -> np.ndarray:
    """
    I(q,c): mean of Jaccard(query_tokens, identifiers) and coverage |q∩ids|/|q|.

    Uses an inverted index so sparse overlaps stay fast on 1.2k×1.2k pools.
    """
    q_toks = [set(word_tokens(q)) for q in queries]
    n, m = len(q_toks), len(id_sets)
    out = np.zeros((n, m), dtype=np.float32)

    inv: dict[str, list[int]] = {}
    id_lens = np.array([len(s) for s in id_sets], dtype=np.int32)
    for j, ids in enumerate(id_sets):
        for t in ids:
            inv.setdefault(t, []).append(j)

    for i, qt in enumerate(q_toks):
        if not qt:
            continue
        inter_counts: dict[int, int] = {}
        for t in qt:
            for j in inv.get(t, ()):
                inter_counts[j] = inter_counts.get(j, 0) + 1
        qn = len(qt)
        for j, inter in inter_counts.items():
            union = qn + int(id_lens[j]) - inter
            jacc = inter / union if union else 0.0
            cov = inter / qn
            out[i, j] = 0.5 * (jacc + cov)
    return out


def identifier_overlap_pair(query: str, id_set: set[str]) -> float:
    qt = set(word_tokens(query))
    if not qt or not id_set:
        return 0.0
    inter = len(qt & id_set)
    if inter == 0:
        return 0.0
    jacc = inter / len(qt | id_set)
    cov = inter / len(qt)
    return 0.5 * (jacc + cov)


def query_features(queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qid = np.array([query_id_density(q) for q in queries], dtype=np.float32)
    qlen = np.array([query_len_norm(q) for q in queries], dtype=np.float32)
    return qid, qlen


def build_interaction_block(L, I, S, Qid, Qlen) -> dict[str, np.ndarray]:
    """Broadcast query cues against pair-level scores. Arrays may be 1D or 2D."""
    Qnld = 1.0 - Qid
    return {
        "L": L,
        "I": I,
        "S": S,
        "Qid": Qid,
        "Qlen": Qlen,
        "L_x_Qid": L * Qid,
        "I_x_Qid": I * Qid,
        "S_x_Qnld": S * Qnld,
        "L_x_Qlen": L * Qlen,
        "S_x_Qlen": S * Qlen,
        "I_x_Qlen": I * Qlen,
    }


def stack_features(feat_dict: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    cols = []
    for n in names:
        v = feat_dict[n]
        cols.append(v.reshape(-1, 1) if v.ndim == 1 else v.reshape(-1, 1))
    return np.hstack(cols).astype(np.float32)


class FeaturePipeline:
    """Fit-once feature pipeline used by training and full-split scoring."""

    def __init__(self, sbert_model_name: str | None = None):
        from .config import SBERT_MODEL

        self.sbert_model_name = sbert_model_name or SBERT_MODEL
        self.tfidf: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names = list(ALL_FEATURES)
        self._sbert = None
        self.func_ids: list[str] = []
        self.code_emb: np.ndarray | None = None
        self._id_sets: dict[str, set[str]] = {}
        self._code_text: dict[str, str] = {}
        self._query_text: dict[str, str] = {}

    @property
    def sbert(self):
        if self._sbert is None:
            self._sbert = load_sbert(self.sbert_model_name)
        return self._sbert

    def fit(self, functions: pd.DataFrame, emb_cache_dir: Path | None = None):
        """Fit TF-IDF on train; encode all function code representations once."""
        emb_cache_dir = Path(emb_cache_dir or ARTIFACTS / "embeddings")
        emb_cache_dir.mkdir(parents=True, exist_ok=True)

        train = functions[functions["split"] == "train"]
        self.tfidf = fit_tfidf(train)

        self.func_ids = functions["func_id"].tolist()
        self._code_text = {r["func_id"]: code_repr(r) for _, r in functions.iterrows()}
        self._query_text = {r["func_id"]: str(r["query"]) for _, r in functions.iterrows()}
        self._id_sets = {}
        for _, r in functions.iterrows():
            ids = parse_identifier_list(r["code_identifiers"])
            ids |= parse_identifier_list(r["func_name_tokens"])
            ids |= parse_identifier_list(str(r["func_name"]).split(".")[-1])
            self._id_sets[r["func_id"]] = ids

        texts = [self._code_text[fid] for fid in self.func_ids]
        self.code_emb = cached_encode(
            self.sbert,
            texts,
            emb_cache_dir / "all_code_emb.npy",
        )
        self.func_id_to_idx = {fid: i for i, fid in enumerate(self.func_ids)}
        return self

    def encode_queries(self, queries: list[str], cache_path: Path | None = None) -> np.ndarray:
        if cache_path is not None:
            return cached_encode(self.sbert, queries, cache_path)
        return cached_encode(self.sbert, queries, ARTIFACTS / "embeddings" / "_tmp_queries.npy")

    def pair_feature_frame(
        self,
        pairs: pd.DataFrame,
        query_emb_cache: Path | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Compute supervised feature matrix for a pairs_* dataframe."""
        assert self.tfidf is not None and self.code_emb is not None

        queries = pairs["query"].astype(str).tolist()
        # unique queries for SBERT
        qids = pairs["query_id"].tolist()
        uniq_q, inv = np.unique(qids, return_inverse=True)
        # map uniq query_id -> query text
        qid_to_text = pairs.drop_duplicates("query_id").set_index("query_id")["query"].astype(str).to_dict()
        uniq_texts = [qid_to_text[q] for q in uniq_q]
        cache = query_emb_cache or (ARTIFACTS / "embeddings" / f"query_emb_{len(uniq_texts)}.npy")
        # Don't reuse wrong-sized cache blindly — include split hint in caller
        q_emb_uniq = cached_encode(self.sbert, uniq_texts, cache)
        q_emb = q_emb_uniq[inv]

        c_idx = pairs["candidate_func_id"].map(self.func_id_to_idx).to_numpy()
        c_emb = self.code_emb[c_idx]
        S = np.sum(q_emb * c_emb, axis=1).astype(np.float32)  # already normalized

        # TF-IDF cosine per pair (batch via unique queries)
        code_texts = [self._code_text[fid] for fid in pairs["candidate_func_id"]]
        # Transform in chunks to limit memory
        L = self._pairwise_tfidf(queries, code_texts)

        I = np.array(
            [
                identifier_overlap_pair(q, self._id_sets[cid])
                for q, cid in zip(queries, pairs["candidate_func_id"])
            ],
            dtype=np.float32,
        )

        Qid, Qlen = query_features(queries)
        feats = build_interaction_block(L, I, S, Qid, Qlen)
        X = stack_features(feats, self.feature_names)
        y = pairs["label"].to_numpy().astype(np.int8)
        meta = pairs[["query_id", "candidate_func_id", "label"]].copy()
        if "ambiguous" in pairs.columns:
            meta["ambiguous"] = pairs["ambiguous"]
        if "pair_type" in pairs.columns:
            meta["pair_type"] = pairs["pair_type"]
        return meta, X, y

    def _pairwise_tfidf(self, queries: list[str], codes: list[str], chunk: int = 5000) -> np.ndarray:
        out = np.zeros(len(queries), dtype=np.float32)
        for start in range(0, len(queries), chunk):
            end = min(start + chunk, len(queries))
            q = self.tfidf.transform([tokenize_for_tfidf(t) for t in queries[start:end]])
            c = self.tfidf.transform([tokenize_for_tfidf(t) for t in codes[start:end]])
            # row-wise cosine for paired rows
            qn = np.sqrt(np.asarray(q.multiply(q).sum(axis=1))).ravel()
            cn = np.sqrt(np.asarray(c.multiply(c).sum(axis=1))).ravel()
            dot = np.asarray(q.multiply(c).sum(axis=1)).ravel()
            denom = np.maximum(qn * cn, 1e-12)
            out[start:end] = (dot / denom).astype(np.float32)
        return out

    def score_matrices_for_split(
        self,
        split_functions: pd.DataFrame,
        query_emb_cache: Path | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Build full (n_query, n_cand) score matrices for a split where every
        function contributes one query (its docstring summary) and is a candidate.
        Row/col order follows split_functions order.
        """
        assert self.tfidf is not None and self.code_emb is not None
        fids = split_functions["func_id"].tolist()
        queries = split_functions["query"].astype(str).tolist()
        codes = [self._code_text[fid] for fid in fids]
        id_sets = [self._id_sets[fid] for fid in fids]

        L = tfidf_cosine(self.tfidf, queries, codes)
        I = identifier_overlap_matrix(queries, id_sets)

        idxs = [self.func_id_to_idx[fid] for fid in fids]
        c_emb = self.code_emb[idxs]
        cache = query_emb_cache or (
            ARTIFACTS / "embeddings" / f"split_query_emb_{split_functions.iloc[0]['split']}.npy"
        )
        q_emb = cached_encode(self.sbert, queries, cache)
        S = (q_emb @ c_emb.T).astype(np.float32)

        Qid, Qlen = query_features(queries)
        # broadcast to matrices
        Qid_m = np.broadcast_to(Qid[:, None], L.shape).astype(np.float32).copy()
        Qlen_m = np.broadcast_to(Qlen[:, None], L.shape).astype(np.float32).copy()
        return {
            "func_ids": np.array(fids),
            "L": L,
            "I": I,
            "S": S,
            "Qid": Qid_m,
            "Qlen": Qlen_m,
            **{
                k: v
                for k, v in build_interaction_block(L, I, S, Qid_m, Qlen_m).items()
                if k not in ("L", "I", "S", "Qid", "Qlen")
            },
        }

    def matrix_to_feature_rows(
        self,
        mats: dict[str, np.ndarray],
        names: list[str] | None = None,
    ) -> np.ndarray:
        names = names or self.feature_names
        flat = {n: mats[n].ravel() for n in names}
        return stack_features(flat, names)

    def fit_scaler(self, X_train: np.ndarray):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return self.scaler.transform(X).astype(np.float32)

    def save(self, path: Path):
        arts = FeatureArtifacts(
            tfidf=self.tfidf,
            scaler=self.scaler,
            feature_names=self.feature_names,
            sbert_name=self.sbert_model_name,
            code_emb=self.code_emb,
            func_ids=self.func_ids,
        )
        arts.save(path)
        # also persist id sets / texts lightly
        joblib.dump(
            {"id_sets": self._id_sets, "code_text": self._code_text, "query_text": self._query_text},
            Path(path).with_suffix(".side.joblib"),
        )

    def load_side(self, path: Path):
        side = joblib.load(Path(path).with_suffix(".side.joblib"))
        self._id_sets = side["id_sets"]
        self._code_text = side["code_text"]
        self._query_text = side["query_text"]
        self.func_id_to_idx = {fid: i for i, fid in enumerate(self.func_ids)}
