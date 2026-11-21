# CodeSearch-query-aware

Natural-language code search over Python functions. The system ranks candidates using complementary retrieval signals—**lexical (TF-IDF)**, **identifier overlap**, and **semantic (Sentence-BERT)**—then compares a **Fixed Hybrid** baseline with **Query-Aware Fusion**, a research method that adapts fusion weights from query characteristics.

Repository: [abhishekreddy-msrit/CodeSearch-query-aware](https://github.com/abhishekreddy-msrit/CodeSearch-query-aware)

---

## What this project does

Given a programming query such as *“Parse JSON from a string”*, the app scores every function in a selected candidate pool and returns the Top-K ranked functions with model-specific scores.

| Ranking model | Role | Score |
|---------------|------|-------|
| Lexical (TF-IDF) | Baseline | `L` |
| Identifier | Baseline | `I` |
| Semantic (SBERT) | Baseline | `S` |
| **Fixed Hybrid** | **Validation-selected default** | `0.40·L + 0.20·I + 0.40·S` |
| Query-Aware Fusion | Research / experimental | `w = softmax(W·[1, Qid, Qlen])` then `wL·L + wI·I + wS·S` |

**Honest selection result:** Fixed Hybrid won on validation full-split MRR (~0.817). Query-Aware Fusion was close (~0.813) but did **not** meet the +0.010 target and is **not** presented as the winner. Offline LR/RF models were investigated during development and are **not** wired into the Streamlit runtime.

Runtime scoring is **exhaustive** over the chosen pool (Full / Train / Val / Test). There is **no ANN index** and **no vector database**.

---

## Dataset

- **Source:** CodeSearchNet Python (Hugging Face parquet)
- **Final corpus:** 8,000 functions · 1,147 repositories
- **Splits:** repository-disjoint 70 / 15 / 15 → Train 5,600 · Val 1,200 · Test 1,200 (seed 42)
- **Primary benchmark:** full-split ranking (1,200 queries × 1,200 candidates per split)
- **Metrics:** MRR, Precision@5, Recall@5

Key artifacts: `data/`, `results/primary_fullsplit_metrics.csv`, `artifacts/models/hybrid_weights.json`

---

## Quick start — Streamlit demo

```bat
cd /d "D:\RIT\Third Year\AI&ML\CodeSearch"
pip install -r requirements.txt
python -m streamlit run app\streamlit_app.py
```

Open **http://localhost:8501**.

### Demo workflow (~4–5 minutes)

1. **Query** — click `Parse JSON from a string` (fills the box only; does not run a search).
2. **Single model → Lexical Search (TF-IDF)** → **Run Search**. Note Rank #1 and the Lexical Ranking Score.
3. Switch to **Semantic Search (Sentence-BERT)** → **Run Search** again (same query). Rankings change.
4. Switch to **Fixed Hybrid** → **Run Search**. Score = `0.40·L + 0.20·I + 0.40·S`.
5. Switch to **Query-Aware Fusion (experimental)** → **Run Search**. Inspect query-conditioned weights.
6. Switch workflow to **Compare models** → keep Lexical / Semantic / Fixed Hybrid / Query-Aware → **Compare Models**.
   - Top-1 table, rank matrix, agreement / unique hits, per-model tabs.

**Defaults:** Fixed Hybrid · Top-K = 5 · Full Corpus (8,000).

Changing the model hides previous results until you click **Run Search** again. The result banner names the model that produced the displayed ranking.

### What the numbers mean

| Concept | Meaning |
|---------|---------|
| Live retrieval | Ranked functions for the query you just ran |
| Ranking score | Similarity used to order candidates (not a probability) |
| Top-K Results | How many highest-ranked functions to show — **not** accuracy |
| Candidate corpus | Which functions are searchable (Full / Train / Val / Test) |
| Benchmark MRR / P@5 / R@5 | Offline metrics on a labeled eval set |

---

## Architecture (offline vs runtime)

**Offline (research / training / evaluation)**  
Data preparation → repository-disjoint splits → feature preparation (TF-IDF, identifiers, SBERT embeddings) → baselines + Fixed Hybrid + Query-Aware Fusion → full-split evaluation → **frozen artifacts**.

**Runtime (Streamlit)**  
User query + model + corpus + Top-K → candidate pool → shared signals `L / I / S` (+ `Qid`, `Qlen` for fusion) → selected ranker → Top-K results with scores.

Artifacts are loaded once at startup. Offline LR/RF stay offline-only.

---

## ML pipeline (offline)

```bat
pip install -r requirements.txt
python scripts\smoke_test.py
python scripts\run_pipeline.py
python scripts\demo_rank.py "parse json from a string"
python scripts\test_ranker_independence.py
```

Optional fusion optimization scripts: `scripts/optimize_fusion.py`, `scripts/optimize_fusion_round2.py`.

---

## Project layout

| Path | Purpose |
|------|---------|
| `app/` | Streamlit faculty demo |
| `data/` | Frozen CodeSearchNet Python subset + benchmarks |
| `src/codesearch/` | Features, embeddings, models, ranker, fusion |
| `scripts/` | Train / eval / optimize / launch / independence test |
| `artifacts/` | Embeddings + fitted pipelines + model weights |
| `results/` | Benchmark tables and optimization summaries |
| `requirements.txt` | Python dependencies |

---

## Evaluation summary (full-split)

| System | Val MRR | Val P@5 | Val R@5 | Test MRR | Test P@5 | Test R@5 |
|--------|---------|---------|---------|----------|----------|----------|
| Lexical (TF-IDF) | 0.624 | 0.153 | 0.765 | 0.580 | 0.146 | 0.732 |
| Identifier | 0.465 | 0.114 | 0.570 | 0.400 | 0.099 | 0.496 |
| Semantic (SBERT) | 0.761 | 0.174 | 0.871 | 0.748 | 0.174 | 0.868 |
| **Fixed Hybrid ★** | **0.817** | **0.182** | **0.912** | **0.786** | **0.179** | **0.894** |
| Query-Aware Fusion | 0.813 | 0.183 | 0.915 | 0.790 | 0.179 | 0.896 |

Selection used **validation only**. Test is a post-selection readout (prior exposure disclosed in the project report).

---

## Requirements

Python 3.10+ recommended. See `requirements.txt` for:

- pandas, numpy, scikit-learn, pyarrow, duckdb, joblib
- sentence-transformers (`all-MiniLM-L6-v2`)
- streamlit, tqdm

First SBERT run downloads the MiniLM model from Hugging Face.

---

## License / academic use

Course project for **Artificial Intelligence and Machine Learning (24CS52)**, Department of Computer Science and Engineering, Ramaiah Institute of Technology.
