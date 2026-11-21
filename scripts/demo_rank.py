#!/usr/bin/env python
"""Quick ranking demo for a natural-language query."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.ranker import CodeSearchRanker


def main():
    q = " ".join(sys.argv[1:]) or "parse json from a string and return a dictionary"
    ranker = CodeSearchRanker()
    system = ranker.default_system  # validation-selected (fixed_hybrid after optimization sprint)
    hits = ranker.rank(q, split="test", system=system, top_k=5)
    print(f"Query: {q}")
    print(f"System: {system}\n")
    for i, r in hits.iterrows():
        print(f"{i+1}. [{r.score:.4f}] {r.func_name}  ({r.repository_name})")
        print(f"    L={r.L:.3f} I={r.I:.3f} S={r.S:.3f}")
        snippet = str(r.code).strip().splitlines()[:3]
        for ln in snippet:
            print(f"    {ln[:100]}")
        print()


if __name__ == "__main__":
    main()
