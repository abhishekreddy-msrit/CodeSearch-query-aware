"""Test log: each model ranks independently; compare uses one shared candidate pool."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.ranker import DEFAULT_COMPARE, CodeSearchRanker


def main() -> int:
    q = "Parse JSON from a string"
    top_k = 5
    split = "all"
    ranker = CodeSearchRanker()
    systems = [s for s in DEFAULT_COMPARE if s in ranker.available_systems()]
    assert "tfidf" in systems and "sbert" in systems and "fixed_hybrid" in systems

    print("=" * 72)
    print("TEST LOG — independent ranking + shared compare pool")
    print("=" * 72)
    print(f"Query: {q!r}")
    print(f"Top-K: {top_k}")
    print(f"Corpus: {split}")
    print(f"Systems: {systems}")
    print()

    # Shared signal computation once (what rank_many does internally)
    signals = ranker.compute_signals(q, split=split)
    n_cand = signals["n_candidates"]
    print(f"[pool] n_candidates = {n_cand} (must be identical for every system)")
    assert n_cand == 8000, f"Expected full corpus 8000, got {n_cand}"

    results = ranker.rank_many(q, systems, top_k=top_k, split=split)

    # Every result frame must report the same candidate pool + query
    for sys, df in results.items():
        assert int(df["n_candidates"].iloc[0]) == n_cand
        assert df["query"].iloc[0] == q
        assert df["system"].iloc[0] == sys
        print(f"[ok] {sys}: n_candidates={n_cand}, labeled system={sys}")

    print()
    tops = {}
    for sys, df in results.items():
        top = df.iloc[0]
        tops[sys] = top["func_name"]
        print(f"[{sys}]")
        print(f"  score_name : {top['score_name']}")
        print(f"  formula    : {top['formula']}")
        print(f"  signals_used: {top['signals_used']}")
        print(f"  #1 {top['func_name']}  ranking_score={float(top['score']):.4f}")
        print(f"     L={float(top['L']):.4f} I={float(top['I']):.4f} S={float(top['S']):.4f}")

        if sys == "tfidf":
            assert abs(float(top["score"]) - float(top["L"])) < 1e-6
            assert "Hybrid" not in str(top["score_name"])
        elif sys == "sbert":
            assert abs(float(top["score"]) - float(top["S"])) < 1e-6
            assert "Hybrid" not in str(top["score_name"])
        elif sys == "fixed_hybrid":
            wl, wi, ws = ranker.hybrid_w
            expected = wl * float(top["L"]) + wi * float(top["I"]) + ws * float(top["S"])
            assert abs(float(top["score"]) - expected) < 1e-5, (top["score"], expected)
            print(f"  hybrid check: {wl}*L+{wi}*I+{ws}*S = {expected:.4f}")
        elif sys == "query_aware_fusion":
            assert "wL" in top.index
            print(
                f"  adaptive w = ({float(top['wL']):.3f}, {float(top['wI']):.3f}, {float(top['wS']):.3f}) "
                f"shift={float(top['weight_shift_from_fixed']):.3f}"
            )
            print(f"  note: {top['adaptation_note']}")
        print()

    # Distinct score vectors ⇒ independent ranking, not label swap
    score_vecs = {sys: tuple(results[sys]["score"].round(6)) for sys in systems}
    assert len(set(score_vecs.values())) > 1, "All systems produced identical scores"

    # Top-1 should differ for at least one pair on this known demo query
    assert len(set(tops.values())) > 1, f"Unexpected: all systems same #1 {tops}"

    print("Top-1 functions by system:")
    for sys, name in tops.items():
        print(f"  {sys:22s} → {name}")
    print()
    print("PASS — independent executable models; compare uses identical query/candidate pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
