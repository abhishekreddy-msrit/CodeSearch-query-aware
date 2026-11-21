"""Dataset loading — CSV for tables, DuckDB for eval_full parquet (pyarrow 19 incompatible)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import DATA_DIR


def _conn():
    return duckdb.connect()


def load_functions(data_dir: Path | str = DATA_DIR, columns: list[str] | None = None) -> pd.DataFrame:
    """Load function corpus. Prefer CSV (reliable); parquet via DuckDB if needed."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "functions.csv"
    if csv_path.exists():
        usecols = columns
        df = pd.read_csv(csv_path, usecols=usecols)
    else:
        path = (data_dir / "functions.parquet").as_posix()
        cols = ", ".join(columns) if columns else "*"
        df = _conn().execute(f"SELECT {cols} FROM read_parquet('{path}')").df()
    return df


def load_pairs(split: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    assert split in ("train", "val", "test")
    data_dir = Path(data_dir)
    csv_path = data_dir / f"pairs_{split}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    path = (data_dir / f"pairs_{split}.parquet").as_posix()
    return _conn().execute(f"SELECT * FROM read_parquet('{path}')").df()


def load_eval_full(split: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Primary benchmark: every query vs all functions in the split."""
    assert split in ("val", "test")
    data_dir = Path(data_dir)
    path = (data_dir / f"eval_full_{split}.parquet").as_posix()
    # Keep only columns needed for scoring / metrics
    cols = (
        "split, query_id, query_func_id, candidate_func_id, "
        "label, same_repo, ambiguous, ambiguity_reasons, in_10_pool_as"
    )
    return _conn().execute(f"SELECT {cols} FROM read_parquet('{path}')").df()


def load_pool(benchmark: str, split: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Mirror data/benchmarks.py with environment-safe loaders."""
    if split not in ("val", "test"):
        raise ValueError("benchmarks are for val/test only")
    if benchmark == "full":
        df = load_eval_full(split, data_dir)
    elif benchmark == "pool10":
        df = load_pairs(split, data_dir)
    else:
        raise ValueError("benchmark must be 'full' or 'pool10'")
    return df.sort_values(["query_id", "candidate_func_id"]).reset_index(drop=True)


def check_scores(scores: pd.DataFrame, pool: pd.DataFrame) -> None:
    """Raise if scores miss/extra/duplicate pool pairs (same contract as Arnav's utility)."""
    k = ["query_id", "candidate_func_id"]
    assert not scores.duplicated(k).any(), "duplicate (query_id, candidate_func_id) in scores"
    assert scores["score"].notna().all(), "missing scores"
    got = set(map(tuple, scores[k].to_numpy()))
    want = set(map(tuple, pool[k].to_numpy()))
    missing, extra = want - got, got - want
    assert not missing and not extra, (
        f"pool mismatch: {len(missing)} missing, {len(extra)} extra pairs"
    )


def functions_by_split(functions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for s in ("train", "val", "test"):
        out[s] = functions[functions["split"] == s].reset_index(drop=True)
    return out
