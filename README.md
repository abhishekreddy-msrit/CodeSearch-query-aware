# Code Search — Query-Aware Function Ranking

Natural-language search over Python functions. Compares lexical (TF-IDF), Sentence-BERT semantic, fixed-weight hybrid, and query-aware fusion ranking.

## Faculty demo

```bat
cd /d "D:\RIT\Third Year\AI&ML\CodeSearch"
python -m streamlit run app\streamlit_app.py
```

Open **http://localhost:8501**.

### Demo workflow (≈4–5 minutes)

1. **Query** — click `Parse JSON from a string` (fills the box only; does not run a model).
2. **Single model → Lexical Search (TF-IDF)** → **Run Search**. Note Rank #1 and the Lexical Ranking Score.
3. Switch to **Semantic Search (Sentence-BERT)** → **Run Search** again (same query). Rankings change.
4. Switch to **Fixed Hybrid** → **Run Search**. Score = `0.40·L + 0.20·I + 0.40·S`.
5. Switch to **Query-Aware Fusion (experimental)** → **Run Search**. Inspect query-conditioned weights (honest adaptation note if weak).
6. Switch workflow to **Compare models** → keep Lexical / Semantic / Fixed Hybrid / Query-Aware → **Compare Models**.
   - Top-1 table, rank matrix, agreement / unique hits, per-model tabs.

**Defaults:** Fixed Hybrid (validation-selected) · Top-K = 5 · Full Corpus (8,000).

**Important:** Changing the model hides previous results until you click **Run Search** again. The result banner always names the model that produced the displayed ranking.

### What the numbers mean

| Concept | Meaning |
|---------|---------|
| Live retrieval | Ranked functions for the query you just ran |
| Ranking score | Similarity used to order candidates (not a probability) |
| Top-K Results | How many highest-ranked functions to show — **not** accuracy |
| Candidate corpus | Which functions are searchable (Full / Train / Val / Test) |
| Benchmark MRR / P@5 / R@5 | Offline metrics on a labeled eval set |

Model selection used **validation** only. Fixed Hybrid won. Query-Aware Fusion is experimental and available for comparison — it is **not** marketed as the winner.

## Verify ranking independence

```bat
python scripts\test_ranker_independence.py
```

## ML pipeline

```bat
pip install -r requirements.txt
python scripts\smoke_test.py
python scripts\run_pipeline.py
python scripts\demo_rank.py "parse json from a string"
```

## Layout

| Path | Purpose |
|------|---------|
| `app/` | Streamlit faculty demo |
| `data/` | Frozen CodeSearchNet Python subset |
| `src/codesearch/` | Features, models, ranker |
| `scripts/` | Train / eval / optimize / launch / tests |
| `artifacts/` | Embeddings + trained models |
| `results/` | Benchmark tables + optimize summaries |

## Evaluation

Primary: full-split `eval_full_{val,test}` (1200×1200). Metrics: MRR, P@5, R@5.
Artifacts: `results/primary_fullsplit_metrics.csv`, `results/optimize_r2_summary.json`.
