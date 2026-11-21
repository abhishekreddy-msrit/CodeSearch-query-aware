#!/usr/bin/env python
"""Quick import + data-load smoke test (no SBERT download if already cached)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.data import check_scores, load_functions, load_pairs, load_pool
from codesearch.text import query_id_density, tokenize_for_tfidf


def main():
    f = load_functions()
    assert len(f) == 8000
    assert set(f["split"]) == {"train", "val", "test"}
    pt = load_pairs("train")
    assert len(pt) == 55989
    pool = load_pool("pool10", "val")
    scores = pool[["query_id", "candidate_func_id"]].copy()
    scores["score"] = 1.0
    check_scores(scores, pool)
    assert query_id_density("parse JSON from snake_case_path") > 0
    assert len(tokenize_for_tfidf("HelloWorld_foo")) >= 2
    # full eval load (duckdb)
    ev = load_pool("full", "val")
    assert len(ev) == 1_440_000
    print("SMOKE OK", {"functions": len(f), "train_pairs": len(pt), "eval_full_val": len(ev)})


if __name__ == "__main__":
    main()
