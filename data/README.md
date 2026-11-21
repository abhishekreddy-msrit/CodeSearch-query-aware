# AIML Code Search — dataset extract v2 (data/corpus lead: Arnav)

Full details, counts and data dictionary: see the "AIML Code Search — Data Quality Report (v1)" doc.

## Files
- `functions.parquet` / `.csv` — 8,000 cleaned Python functions (CodeSearchNet), with `split` (train/val/test, repository-disjoint 70/15/15).
  Use the `code` column for features (docstring removed). `original_source_with_docstring` is for display only.
- `pairs_train|val|test.parquet` / `.csv` — 79,982 query-candidate pairs: per query 1 positive + 4 BM25 hard negatives + 5 random negatives.
  Group by `query_id` for MRR / P@5 / R@5. Join `candidate_func_id` -> `functions.func_id`.
  `ambiguous=True` marks negatives that may actually be relevant (see `ambiguity_reasons`).
- `eval_full_val|test.parquet` — PRIMARY benchmark: every val/test query vs all 1,200 functions in its split (1.44M pairs each).
  `in_10_pool_as` marks pairs that are also in the secondary 10-candidate pool.
- `pairs_val|test` double as the SECONDARY benchmark (fixed 10-candidate pool). Training always uses `pairs_train`.
- `benchmarks.py` — `load_pool("full"|"pool10", "val"|"test")` and `check_scores(scores, pool)`; run the check for every
  system so all baselines and the proposed model are scored on exactly the same pairs.
- `label_audit_sample.csv` — 70 stratified pairs for manual label checking (fill the blank columns).
- `removed_records.csv` — every dropped record with its reason.
- `stats.json` — every number in the report + all parameters.
- `prepare_data.py` — rebuilds everything (seed 42; identical on re-run).

## Quick load
```python
import pandas as pd
f = pd.read_parquet("functions.parquet")
p = pd.read_parquet("pairs_train.parquet").merge(f[["func_id", "code", "code_identifiers"]],
                                                 left_on="candidate_func_id", right_on="func_id")
```

## Rebuild
```
pip install pandas pyarrow scikit-learn rank_bm25
raw/csn/csn_python_test.parquet, raw/csn/csn_python_validation.parquet   # HF code-search-net/code_search_net, config python, refs/convert/parquet
raw/cosqa/cosqa-train.json, cosqa-dev.json, cosqa-retrieval-dev-500.json, cosqa-retrieval-test-500.json, code_idx_map.txt
                                                                       # github.com/Jun-jie-Huang/CoCLR  data/qa and data/search
python prepare_data.py --raw raw --out out --seed 42 --target 8000
```
