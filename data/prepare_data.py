"""
prepare_data.py - Data / corpus pipeline for the AIML mini-project
"Smart Code Search Using Query-Aware Supervised Relevance Prediction".

Owner: Arnav Chatterjee (data / corpus lead)

Inputs (see README / data-quality report for provenance):
  raw/csn/csn_python_test.parquet         CodeSearchNet Python, original test partition (HF parquet mirror)
  raw/csn/csn_python_validation.parquet   CodeSearchNet Python, original valid partition (HF parquet mirror)
  raw/cosqa/*.json                        CoSQA (CoCLR repo copy of the CodeXGLUE release)

Outputs (out/):
  functions.parquet / functions.csv       cleaned, sampled function corpus with split assignment
  pairs_{train,val,test}.parquet / .csv   query-candidate pairs (1 positive + negatives per query)
  removed_records.csv                     every dropped record with the reason (audit trail)
  label_audit_sample.csv                  stratified sample for manual label checking
  cosqa_profile.json                      CoSQA inspection results
  stats.json                              every number used in the data-quality report

Run:  python prepare_data.py --raw raw --out out --seed 42 --target 8000
"""
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse, ast, hashlib, io, json, keyword, re, tokenize, random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------------------------------------------------------
# Parameters (all recorded in stats.json)
# ----------------------------------------------------------------------------
P = dict(
    seed=42,
    target_functions=8000,       # proposal: ~5,000-10,000 usable functions
    max_per_repo=40,             # cap so a few huge repos don't dominate the sample
    min_query_tokens=3,          # queries shorter than this are not usable search queries
    max_query_tokens=64,
    min_code_lines=3,            # after docstring removal
    split_ratio=(0.70, 0.15, 0.15),
    n_hard_neg=4,
    n_random_neg=5,
    bm25_top_k=50,               # hard negatives drawn from the top-k BM25 candidates
    amb_query_sim=0.80,          # candidate's own docstring ~= query  -> ambiguous
    amb_code_sim=0.90,           # candidate code ~= positive code    -> ambiguous
    xsplit_near_dup=0.95,        # cross-split near-duplicate threshold (TF-IDF cosine on code)
)

BOILERPLATE = re.compile(
    r"^(todo|fixme|xxx|deprecated|constructor|initializ(e|er)|init|wrapper|decorator|"
    r"helper( function)?|internal( function)?|private|not implemented|see above|"
    r"docstring|test|tests|override|overrides|getter|setter)\b[\s\.:!]*$", re.I)
PARAM_START = re.compile(r"^\s*(:param|:type|:return|@param|@return|args:|arguments:|parameters\b|returns?:|-{3,})", re.I)
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def first_paragraph(doc: str) -> str:
    """Summary part of a docstring = first paragraph, whitespace-normalised."""
    doc = (doc or "").strip()
    para = re.split(r"\n\s*\n", doc, maxsplit=1)[0]
    # stop at the first param/section marker inside the paragraph
    lines = []
    for ln in para.splitlines():
        if PARAM_START.match(ln):
            break
        lines.append(ln.strip())
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_identifier(tok: str):
    out = []
    for part in tok.split("_"):
        out += [p.lower() for p in CAMEL.split(part) if p]
    return out


def text_tokens(s: str):
    """lower-cased word tokens with snake/camel splitting (used for BM25 and checks)."""
    out = []
    for w in WORD.findall(s or ""):
        out += split_identifier(w)
    return [t for t in out if len(t) > 1]


def strip_docstring(code: str):
    """Remove the function's own docstring from its source. Returns (code, method)."""
    try:
        tree = ast.parse(code)
        fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if fn is not None and fn.body and isinstance(fn.body[0], ast.Expr) \
                and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
                and isinstance(fn.body[0].value.value, str):
            node = fn.body[0]
            lines = code.splitlines(keepends=True)
            start, end = node.lineno - 1, node.end_lineno  # 1-based inclusive -> slice
            # docstring on the same line as 'def ...:' (rare) -> regex fallback
            if start == fn.lineno - 1:
                raise SyntaxError
            new = "".join(lines[:start] + lines[end:])
            if len(fn.body) == 1:      # body was only a docstring -> keep it syntactically valid
                indent = re.match(r"\s*", lines[start]).group(0)
                new = "".join(lines[:start] + [indent + "pass\n"] + lines[end:])
            return new, "ast"
        return code, "ast_no_docstring"
    except (SyntaxError, ValueError, StopIteration):
        pass
    # Python-2 or otherwise unparsable code: remove the first triple-quoted string after the signature
    m = re.search(r'(:\s*\n)(\s*)[rRuUbB]{0,2}("""|\'\'\')(.*?)\3[^\n]*\n', code, re.S)
    if m:
        return code[:m.start(2)] + code[m.end():], "regex"
    return code, "none"


def py3_parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def normalised_code(code: str) -> str:
    """Code with comments, docstrings/strings and whitespace removed -> for normalised duplicate detection."""
    try:
        toks = []
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                          tokenize.ENDMARKER):
                continue
            toks.append("<S>" if t.type == tokenize.STRING else t.string)
        return " ".join(toks)
    except Exception:
        code = re.sub(r"#.*", "", code)
        return re.sub(r"\s+", " ", code).strip()


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def identifiers(code_tokens, func_name):
    ids = set()
    for t in list(code_tokens) + [func_name]:
        for w in WORD.findall(str(t)):
            if keyword.iskeyword(w) or w in ("self", "cls"):
                continue
            ids.add(w)
    return sorted(ids)


def ascii_ratio(s):
    return sum(c.isascii() for c in s) / max(1, len(s))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=P["seed"])
    ap.add_argument("--target", type=int, default=P["target_functions"])
    a = ap.parse_args()
    P["seed"], P["target_functions"] = a.seed, a.target
    rng = np.random.default_rng(P["seed"])
    random.seed(P["seed"])
    import os
    os.makedirs(a.out, exist_ok=True)
    S = {"params": P}

    # ---------------- 1. load ----------------
    frames = []
    for part in ("test", "validation"):
        f = pd.read_parquet(f"{a.raw}/csn/csn_python_{part}.parquet")
        frames.append(f)
    raw = pd.concat(frames, ignore_index=True)
    raw.insert(0, "func_id", [f"csn_py_{i:06d}" for i in range(len(raw))])
    S["raw"] = dict(records=len(raw), repos=int(raw.repository_name.nunique()),
                    by_original_partition=raw.split_name.value_counts().to_dict(),
                    columns=list(raw.columns))

    # missing / empty values per column (before anything is dropped)
    miss = {}
    for c in ["repository_name", "func_path_in_repository", "func_name", "whole_func_string",
              "func_code_string", "func_documentation_string", "func_code_url", "language"]:
        col = raw[c]
        miss[c] = dict(null=int(col.isna().sum()),
                       empty_or_whitespace=int((col.fillna("").astype(str).str.strip() == "").sum()))
    S["missing_values_raw"] = miss

    removed = []

    def drop(df, mask, reason):
        r = df[mask]
        removed.append(r[["func_id", "repository_name", "func_name", "func_code_url"]].assign(reason=reason))
        S.setdefault("cleaning_steps", []).append(dict(step=reason, removed=int(mask.sum()),
                                                       remaining=int((~mask).sum())))
        return df[~mask].copy()

    df = raw.copy()
    # ---------------- 2. cleaning ----------------
    req = ["repository_name", "func_path_in_repository", "func_name", "func_code_string",
           "func_documentation_string", "func_code_url"]
    m = np.zeros(len(df), bool)
    for c in req:
        m |= df[c].isna().values | (df[c].fillna("").astype(str).str.strip() == "").values
    df = drop(df, m, "missing/empty required field")

    df = drop(df, df.language.str.lower() != "python", "non-python language tag")

    # malformed: code must contain a def/async def header for the named function
    has_def = df.apply(lambda r: re.search(r"^\s*(async\s+)?def\s+" + re.escape(str(r.func_name).split(".")[-1]) + r"\s*\(",
                                           r.func_code_string, re.M) is not None, axis=1)
    df = drop(df, ~has_def, "malformed: no def header matching func_name")

    # query = first paragraph of docstring
    df["query"] = df.func_documentation_string.map(first_paragraph)
    df["query_n_tokens"] = df["query"].map(lambda s: len(s.split()))
    df = drop(df, df["query"].str.len() == 0, "docstring has no usable summary paragraph")
    df = drop(df, df["query"].map(ascii_ratio) < 0.9, "non-English/non-ASCII docstring summary")
    df = drop(df, df["query"].map(lambda s: bool(BOILERPLATE.match(s))), "boilerplate docstring (TODO/wrapper/...)")
    df = drop(df, df["query"].map(lambda s: len(text_tokens(s))) < P["min_query_tokens"],
              f"query too short (<{P['min_query_tokens']} word tokens)")
    df = drop(df, df["query_n_tokens"] > P["max_query_tokens"], f"query too long (>{P['max_query_tokens']} tokens)")
    df = drop(df, df["query"].str.match(r"^\s*(https?://\S+|\W+)\s*$"), "query is only a URL / punctuation")

    # docstring removal (leakage prevention)
    res = df.func_code_string.map(strip_docstring)
    df["code"] = [r[0] for r in res]
    df["docstring_removal_method"] = [r[1] for r in res]
    S["docstring_removal_methods"] = df.docstring_removal_method.value_counts().to_dict()
    df = drop(df, df.docstring_removal_method == "none", "docstring could not be removed from code")
    n_lines = df.code.map(lambda c: len([l for l in c.splitlines() if l.strip() and not l.strip().startswith("#")]))
    df = drop(df, n_lines < P["min_code_lines"], f"code too short after docstring removal (<{P['min_code_lines']} lines)")
    # residual leakage check: query text still verbatim in the stripped code (e.g. duplicated in a comment)
    q_low = df["query"].str.lower()
    leak = [q in c.lower() for q, c in zip(q_low, df.code)]
    df["query_in_code_verbatim"] = leak
    S["residual_verbatim_query_in_code"] = int(sum(leak))
    df = drop(df, df["query_in_code_verbatim"], "query text still present verbatim in code (comment/string)")

    df["py3_parseable"] = df.code.map(py3_parses)

    # duplicates
    df["code_hash"] = df.code.map(lambda c: sha(c))
    df["code_norm_hash"] = df.code.map(lambda c: sha(normalised_code(c)))
    df = df.sort_values("func_id")
    S["duplicates_detected"] = dict(
        exact_code_dup_records=int(df.duplicated("code_hash").sum()),
        normalised_code_dup_records=int(df.duplicated("code_norm_hash").sum()),
        exact_query_dup_records=int(df["query"].str.lower().duplicated().sum()),
        exact_query_and_code_dup=int(df.duplicated(["code_hash", "query"]).sum()),
    )
    df = drop(df, df.duplicated("code_hash"), "exact duplicate code")
    df = drop(df, df.duplicated("code_norm_hash"), "normalised duplicate code (whitespace/comments/strings)")
    # identical queries for different functions = ambiguous labels: keep only one per (lower-cased) query
    qn = df["query"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df["query_norm"] = qn
    qcount = qn.map(qn.value_counts())
    S["functions_sharing_identical_query_before_dedup"] = int((qcount > 1).sum())
    df = drop(df, qn.duplicated(), "duplicate query text (identical docstring summary for another function)")

    S["clean"] = dict(records=len(df), repos=int(df.repository_name.nunique()),
                      py3_parseable=int(df.py3_parseable.sum()),
                      py2_or_unparseable=int((~df.py3_parseable).sum()))

    # ---------------- 3. sampling ----------------
    # cap per repo, then random sample down to target
    df["_r"] = rng.random(len(df))
    df = df.sort_values(["repository_name", "_r"])
    df["_rank_in_repo"] = df.groupby("repository_name").cumcount()
    capped = df[df._rank_in_repo < P["max_per_repo"]]
    S["sampling"] = dict(after_repo_cap=len(capped), repos_after_cap=int(capped.repository_name.nunique()),
                         excluded_by_repo_cap=int(len(df) - len(capped)))
    if len(capped) > P["target_functions"]:
        samp = capped.sample(n=P["target_functions"], random_state=P["seed"])
    else:
        samp = capped
    samp = samp.drop(columns=["_r", "_rank_in_repo"]).sort_values("func_id").reset_index(drop=True)
    S["sampling"].update(sampled=len(samp), repos_sampled=int(samp.repository_name.nunique()))

    # ---------------- 4. repository-disjoint split ----------------
    repo_sizes = samp.repository_name.value_counts()
    repos = list(repo_sizes.index)
    rng.shuffle(repos)
    # largest repos first (in shuffled order within equal size) so greedy balancing works
    repos.sort(key=lambda r: -repo_sizes[r])
    names = ["train", "val", "test"]
    target = {n: r * len(samp) for n, r in zip(names, P["split_ratio"])}
    have = {n: 0 for n in names}
    assign = {}
    for r in repos:
        n = max(names, key=lambda k: (target[k] - have[k]) / target[k])
        assign[r] = n
        have[n] += repo_sizes[r]
    samp["split"] = samp.repository_name.map(assign)
    # verification
    rs = samp.groupby("repository_name").split.nunique()
    assert (rs == 1).all(), "repository leaks across splits!"
    sp = {}
    for n in names:
        sub = samp[samp.split == n]
        sp[n] = dict(functions=len(sub), pct_functions=round(100 * len(sub) / len(samp), 1),
                     repos=int(sub.repository_name.nunique()),
                     pct_repos=round(100 * sub.repository_name.nunique() / samp.repository_name.nunique(), 1),
                     largest_repo_share_pct=round(100 * sub.repository_name.value_counts().iloc[0] / len(sub), 1))
    S["splits"] = sp
    S["split_checks"] = dict(repos_in_more_than_one_split=int((rs > 1).sum()),
                             repo_overlap_train_val=len(set(samp[samp.split == "train"].repository_name) & set(samp[samp.split == "val"].repository_name)),
                             repo_overlap_train_test=len(set(samp[samp.split == "train"].repository_name) & set(samp[samp.split == "test"].repository_name)),
                             repo_overlap_val_test=len(set(samp[samp.split == "val"].repository_name) & set(samp[samp.split == "test"].repository_name)),
                             code_hash_overlap_across_splits=int(samp.groupby("code_hash").split.nunique().gt(1).sum()),
                             query_overlap_across_splits=int(samp.groupby("query_norm").split.nunique().gt(1).sum()))
    # cross-split near-duplicate check (TF-IDF over identifier-split code tokens)
    code_tok = samp.code.map(lambda c: " ".join(text_tokens(c)))
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
    X = vec.fit_transform(code_tok)
    tr = np.where(samp.split == "train")[0]
    nd = {}
    samp["max_sim_to_train_code"] = np.nan
    for n in ("val", "test"):
        idx = np.where(samp.split == n)[0]
        sims = linear_kernel(X[idx], X[tr]).max(axis=1)
        samp.loc[idx, "max_sim_to_train_code"] = sims
        nd[n] = dict(n_ge_threshold=int((sims >= P["xsplit_near_dup"]).sum()),
                     median_max_sim=round(float(np.median(sims)), 3))
    S["split_checks"]["cross_split_near_duplicates_vs_train"] = nd
    samp["near_dup_of_train"] = samp.max_sim_to_train_code.fillna(0) >= P["xsplit_near_dup"]

    # extra convenience columns for the modelling side
    samp["code_identifiers"] = [identifiers(t, fn) for t, fn in zip(samp.func_code_tokens, samp.func_name)]
    samp["func_name_tokens"] = samp.func_name.map(lambda s: split_identifier(s.split(".")[-1]))
    samp["code_n_lines"] = samp.code.map(lambda c: len(c.splitlines()))

    # ---------------- 5. pairs ----------------
    q_vec = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")
    Q = q_vec.fit_transform(samp["query"].map(lambda s: " ".join(text_tokens(s))))
    pair_rows = []
    hard_shortfall = 0
    for n in names:
        idx = np.where(samp.split == n)[0]
        sub = samp.iloc[idx]
        corpus = [text_tokens(f"{fn} {c}") for fn, c in zip(sub.func_name, sub.code)]
        bm = BM25Okapi(corpus)
        Qs, Xs = Q[idx], X[idx]
        qsim = linear_kernel(Qs, Qs)     # docstring-vs-docstring similarity
        csim = linear_kernel(Xs, Xs)     # code-vs-code similarity
        fname = sub.func_name.map(lambda s: s.split(".")[-1].lower()).values
        nhash = sub.code_norm_hash.values
        for i_local in range(len(idx)):
            qrow = sub.iloc[i_local]
            qid = f"q_{qrow.func_id}"
            scores = bm.get_scores(text_tokens(qrow["query"]))
            scores[i_local] = -np.inf
            order = np.argsort(-scores)
            rank_of = {j: r + 1 for r, j in enumerate(order[:P["bm25_top_k"]])}

            def flags(j):
                why = []
                if qsim[i_local, j] >= P["amb_query_sim"]:
                    why.append(f"candidate_docstring_similar_to_query({qsim[i_local, j]:.2f})")
                if csim[i_local, j] >= P["amb_code_sim"]:
                    why.append(f"code_near_duplicate_of_positive({csim[i_local, j]:.2f})")
                if fname[j] == fname[i_local]:
                    why.append("same_function_name")
                return why

            chosen = {i_local}
            cands = [(i_local, 1, "positive", None, [])]
            # hard negatives: highest BM25 among top-k, skipping exact/normalised duplicates of the positive
            for j in order[:P["bm25_top_k"]]:
                if len([c for c in cands if c[2] == "hard"]) >= P["n_hard_neg"]:
                    break
                if j in chosen or nhash[j] == nhash[i_local] or scores[j] <= 0:
                    continue
                chosen.add(j)
                cands.append((j, 0, "hard", rank_of[j], flags(j)))
            hard_shortfall += P["n_hard_neg"] - len([c for c in cands if c[2] == "hard"])
            # random negatives: uniform from the same split
            pool = [j for j in range(len(idx)) if j not in chosen]
            for j in rng.choice(pool, size=P["n_random_neg"], replace=False):
                chosen.add(int(j))
                cands.append((int(j), 0, "random", rank_of.get(int(j)), flags(int(j))))
            for j, lab, typ, rk, why in cands:
                c = sub.iloc[j]
                pair_rows.append(dict(
                    split=n, query_id=qid, query=qrow["query"], query_func_id=qrow.func_id,
                    candidate_func_id=c.func_id, label=lab, pair_type=typ,
                    bm25_score=float(scores[j]) if j != i_local else float(bm.get_scores(text_tokens(qrow["query"]))[j]),
                    bm25_rank_in_split=rk, same_repo=bool(c.repository_name == qrow.repository_name),
                    ambiguous=bool(why), ambiguity_reasons=";".join(why),
                ))
    pairs = pd.DataFrame(pair_rows)
    S["pairs"] = dict(
        per_query=dict(positives=1, hard_negatives=P["n_hard_neg"], random_negatives=P["n_random_neg"]),
        hard_negative_shortfall=int(hard_shortfall),
        by_split={n: dict(queries=int(pairs[pairs.split == n].query_id.nunique()),
                          pairs=int((pairs.split == n).sum()),
                          positives=int(((pairs.split == n) & (pairs.label == 1)).sum()),
                          negatives=int(((pairs.split == n) & (pairs.label == 0)).sum()))
                  for n in names},
        by_type=pairs.pair_type.value_counts().to_dict(),
        ambiguous_by_type=pairs[pairs.ambiguous].pair_type.value_counts().to_dict(),
        ambiguity_reason_counts=dict(Counter(r.split("(")[0] for rs in pairs.ambiguity_reasons if rs for r in rs.split(";"))),
        hard_neg_same_repo_pct=round(100 * pairs[pairs.pair_type == "hard"].same_repo.mean(), 1),
        random_neg_same_repo_pct=round(100 * pairs[pairs.pair_type == "random"].same_repo.mean(), 1),
        hard_neg_mean_bm25=round(float(pairs[pairs.pair_type == "hard"].bm25_score.mean()), 2),
        random_neg_mean_bm25=round(float(pairs[pairs.pair_type == "random"].bm25_score.mean()), 2),
        positive_mean_bm25=round(float(pairs[pairs.pair_type == "positive"].bm25_score.mean()), 2),
        positive_is_bm25_top1_pct=round(100 * float(np.mean([
            g[g.label == 1].bm25_score.iloc[0] >= g.bm25_score.max() for _, g in pairs.groupby("query_id")])), 1),
    )

    # ---------------- 6. audit sample ----------------
    aud = []
    for typ, k in (("positive", 20), ("hard", 20), ("random", 10)):
        aud.append(pairs[(pairs.pair_type == typ) & (~pairs.ambiguous)].sample(k, random_state=P["seed"]))
    aud.append(pairs[pairs.ambiguous].sample(min(20, int(pairs.ambiguous.sum())), random_state=P["seed"]))
    aud = pd.concat(aud).merge(samp[["func_id", "func_name", "repository_name", "code", "func_code_url"]],
                               left_on="candidate_func_id", right_on="func_id", how="left")
    aud = aud[["split", "query_id", "query", "pair_type", "label", "ambiguous", "ambiguity_reasons",
               "candidate_func_id", "func_name", "repository_name", "func_code_url", "code"]]
    aud["human_judgement(relevant/not/unsure)"] = ""
    aud["reviewer"] = ""
    aud["notes"] = ""

    # ---------------- 7. CoSQA profile ----------------
    cq = {}
    try:
        ctr = json.load(open(f"{a.raw}/cosqa/cosqa-train.json"))
        cdv = json.load(open(f"{a.raw}/cosqa/cosqa-dev.json"))
        rte = json.load(open(f"{a.raw}/cosqa/cosqa-retrieval-test-500.json"))
        rdv = json.load(open(f"{a.raw}/cosqa/cosqa-retrieval-dev-500.json"))
        cmap = json.load(open(f"{a.raw}/cosqa/code_idx_map.txt"))
        allq = ctr + cdv
        cq["pair_files"] = dict(train=len(ctr), dev=len(cdv), total=len(allq),
                                label_counts=dict(Counter(int(x["label"]) for x in allq)),
                                unique_queries=len(set(x["doc"] for x in allq)),
                                unique_codes=len(set(x["code"] for x in allq)),
                                fields=list(ctr[0].keys()),
                                has_repo_metadata=False)
        ql = [len(x["doc"].split()) for x in allq]
        cq["query_length_tokens"] = dict(mean=round(float(np.mean(ql)), 1), median=float(np.median(ql)))
        cq["retrieval_files"] = dict(dev=len(rdv), test=len(rte), code_pool_size=len(cmap),
                                     test_labels=dict(Counter(int(x["label"]) for x in rte)),
                                     positives_per_test_query=dict(Counter(Counter(x["doc"] for x in rte).values())))
        # overlap of CoSQA code with our sampled CSN corpus (whitespace-normalised)
        norm = lambda s: re.sub(r"\s+", "", s)
        ours_whole = set(samp.whole_func_string.map(norm))
        ours_all = set(raw.whole_func_string.map(norm))
        cos_codes = set(norm(x["code"]) for x in allq) | set(norm(k) for k in cmap)
        samp["in_cosqa_code_pool"] = samp.whole_func_string.map(norm).isin(cos_codes)
        cq["code_overlap"] = dict(cosqa_unique_codes=len(cos_codes),
                                  in_our_8k_sample=len(cos_codes & ours_whole),
                                  in_our_sample_by_split=samp.groupby("split").in_cosqa_code_pool.sum().astype(int).to_dict(),
                                  in_csn_val_test_raw=len(cos_codes & ours_all))
        # do CoSQA test queries also appear in CoSQA train?
        trq = set(x["doc"].lower() for x in ctr)
        cq["retrieval_test_queries_also_in_pair_train"] = sum(x["doc"].lower() in trq for x in rte)
    except FileNotFoundError as e:
        cq["error"] = str(e)
    S["cosqa"] = cq

    # ---------------- 7b. full-split evaluation benchmark (PRIMARY for val/test) ----------------
    # Every val/test query is ranked against ALL functions in its own split (its positive + every other
    # function as a negative). Pools are materialised so every system scores exactly the same pairs.
    in10 = pairs.set_index(["query_id", "candidate_func_id"]).pair_type
    full = {}
    S["full_split_benchmark"] = {}
    for n in ("val", "test"):
        idx = np.where(samp.split == n)[0]
        sub = samp.iloc[idx]
        fids = sub.func_id.values
        qids = np.array([f"q_{f}" for f in fids])
        m = len(idx)
        qsim = linear_kernel(Q[idx], Q[idx])
        csim = linear_kernel(X[idx], X[idx])
        fname = sub.func_name.map(lambda s: s.split(".")[-1].lower()).values
        repo = sub.repository_name.values
        qi = np.repeat(np.arange(m), m)
        ci = np.tile(np.arange(m), m)
        lab = (qi == ci).astype(np.int8)
        a_q = (qsim[qi, ci] >= P["amb_query_sim"]) & (lab == 0)
        a_c = (csim[qi, ci] >= P["amb_code_sim"]) & (lab == 0)
        a_n = (fname[qi] == fname[ci]) & (lab == 0)
        reasons = np.full(m * m, "", dtype=object)
        for mask, tag, sims in ((a_q, "candidate_docstring_similar_to_query", qsim), (a_c, "code_near_duplicate_of_positive", csim),
                                (a_n, "same_function_name", None)):
            for k in np.where(mask)[0]:
                r = f"{tag}({sims[qi[k], ci[k]]:.2f})" if sims is not None else tag
                reasons[k] = r if not reasons[k] else reasons[k] + ";" + r
        fp = pd.DataFrame(dict(split=n, query_id=qids[qi], query_func_id=fids[qi], candidate_func_id=fids[ci],
                               label=lab, same_repo=repo[qi] == repo[ci],
                               ambiguous=a_q | a_c | a_n, ambiguity_reasons=reasons))
        key = pd.MultiIndex.from_arrays([fp.query_id, fp.candidate_func_id])
        fp["in_10_pool_as"] = in10.reindex(key).fillna("").values
        full[n] = fp
        g = fp.groupby("query_id")
        S["full_split_benchmark"][n] = dict(
            queries=int(fp.query_id.nunique()), candidates_per_query=int(m), pairs=int(len(fp)),
            positives=int(lab.sum()), positives_per_query_all_equal_1=bool((g.label.sum() == 1).all()),
            pool_identical_for_every_query=bool(g.candidate_func_id.apply(frozenset).nunique() == 1),
            ambiguous_negatives=int(fp.ambiguous.sum()),
            queries_with_any_ambiguous_negative=int(g.ambiguous.any().sum()),
            ambiguity_reason_counts=dict(Counter(r.split("(")[0] for rs in fp.ambiguity_reasons if rs for r in rs.split(";"))),
            ten_pool_pairs_contained=int((fp.in_10_pool_as != "").sum()),
            ten_pool_pairs_total=int((pairs.split == n).sum()),
        )
        assert S["full_split_benchmark"][n]["ten_pool_pairs_contained"] == S["full_split_benchmark"][n]["ten_pool_pairs_total"], \
            "10-candidate pool is not a subset of the full-split pool"

    # ---------------- 8. save ----------------
    for n, fp in full.items():
        fp.to_parquet(f"{a.out}/eval_full_{n}.parquet", index=False)
    keep = ["func_id", "split", "repository_name", "func_path_in_repository", "func_name", "func_code_url",
            "split_name", "query", "query_n_tokens", "func_documentation_string", "code", "whole_func_string",
            "code_identifiers", "func_name_tokens", "code_n_lines", "py3_parseable", "docstring_removal_method",
            "code_hash", "code_norm_hash", "max_sim_to_train_code", "near_dup_of_train", "in_cosqa_code_pool"]
    out_f = samp[keep].rename(columns={"split_name": "csn_original_partition",
                                       "func_documentation_string": "docstring_full",
                                       "whole_func_string": "original_source_with_docstring"})
    out_f.to_parquet(f"{a.out}/functions.parquet", index=False)
    o2 = out_f.copy()
    o2["code_identifiers"] = o2.code_identifiers.map(" ".join)
    o2["func_name_tokens"] = o2.func_name_tokens.map(" ".join)
    o2.to_csv(f"{a.out}/functions.csv", index=False)
    for n in names:
        p = pairs[pairs.split == n].reset_index(drop=True)
        p.to_parquet(f"{a.out}/pairs_{n}.parquet", index=False)
        p.to_csv(f"{a.out}/pairs_{n}.csv", index=False)
    pd.concat(removed).to_csv(f"{a.out}/removed_records.csv", index=False)
    aud.to_csv(f"{a.out}/label_audit_sample.csv", index=False)

    # descriptive stats of final corpus
    S["final_corpus"] = dict(
        functions=len(out_f), repos=int(out_f.repository_name.nunique()),
        query_tokens=dict(mean=round(float(out_f.query_n_tokens.mean()), 1), median=float(out_f.query_n_tokens.median()),
                          p95=float(out_f.query_n_tokens.quantile(.95))),
        code_lines=dict(mean=round(float(out_f.code_n_lines.mean()), 1), median=float(out_f.code_n_lines.median()),
                        p95=float(out_f.code_n_lines.quantile(.95))),
        functions_per_repo=dict(mean=round(float(out_f.repository_name.value_counts().mean()), 1),
                                median=float(out_f.repository_name.value_counts().median()),
                                max=int(out_f.repository_name.value_counts().max())),
        py3_parseable_pct=round(100 * float(out_f.py3_parseable.mean()), 1),
        methods_pct=round(100 * float(out_f.func_name.str.contains(r"\.").mean()), 1),
        missing_values=out_f.drop(columns=["max_sim_to_train_code"]).isna().sum().astype(int).to_dict(),
    )
    S["removed_total"] = int(sum(s["removed"] for s in S["cleaning_steps"]))
    json.dump(S, open(f"{a.out}/stats.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
