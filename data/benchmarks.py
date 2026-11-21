"""
benchmarks.py - load the evaluation pools and check that a system scored exactly the right pairs.

Two benchmarks, same queries, same labels:
  full  (PRIMARY)   eval_full_{val,test}.parquet : every query vs ALL functions in its split (1,200 x 1,200)
  pool10 (SECONDARY) pairs_{val,test}.parquet     : every query vs its fixed 10 candidates (1 pos + 4 hard + 5 random)

Every baseline (TF-IDF, SBERT, fixed hybrid, LR, RF) and the proposed model must be scored on the SAME pool.
Usage:
    from benchmarks import load_pool, check_scores
    pool = load_pool("full", "test", data_dir="aiml_code_search_data_v1")
    scores = pool[["query_id", "candidate_func_id"]].copy(); scores["score"] = my_model(...)
    check_scores(scores, pool)          # raises if any pair is missing, extra or duplicated
"""
import pandas as pd

COLS = ["split", "query_id", "query_func_id", "candidate_func_id", "label", "same_repo", "ambiguous", "ambiguity_reasons"]


def load_pool(benchmark: str, split: str, data_dir: str = ".") -> pd.DataFrame:
    if split not in ("val", "test"):
        raise ValueError("benchmarks are defined for val and test only; train uses pairs_train.parquet")
    if benchmark == "full":
        df = pd.read_parquet(f"{data_dir}/eval_full_{split}.parquet")
    elif benchmark == "pool10":
        df = pd.read_parquet(f"{data_dir}/pairs_{split}.parquet")
    else:
        raise ValueError("benchmark must be 'full' or 'pool10'")
    return df.sort_values(["query_id", "candidate_func_id"]).reset_index(drop=True)


def check_scores(scores: pd.DataFrame, pool: pd.DataFrame) -> None:
    """scores needs columns query_id, candidate_func_id, score. Raises AssertionError on any mismatch."""
    k = ["query_id", "candidate_func_id"]
    assert not scores.duplicated(k).any(), "duplicate (query_id, candidate_func_id) rows in scores"
    assert scores["score"].notna().all(), "missing scores"
    got = set(map(tuple, scores[k].values))
    want = set(map(tuple, pool[k].values))
    missing, extra = want - got, got - want
    assert not missing and not extra, f"pool mismatch: {len(missing)} missing pairs, {len(extra)} extra pairs"


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    for b in ("pool10", "full"):
        for s in ("val", "test"):
            p = load_pool(b, s, d)
            print(f"{b:6s} {s:4s}: {p.query_id.nunique():5d} queries, {len(p):9,d} pairs, "
                  f"{int(p.label.sum())} positives, {int(p.ambiguous.sum())} ambiguous negatives")
