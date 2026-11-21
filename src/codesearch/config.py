"""Paths, seeds, and experiment knobs."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

SEED = 42
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH = 64

# Feature names used by supervised models
BASE_FEATURES = ["L", "I", "S", "Qid", "Qlen"]
INTERACTION_FEATURES = [
    "L_x_Qid",
    "I_x_Qid",
    "S_x_Qnld",  # semantic × natural-language density (1 - Qid)
    "L_x_Qlen",
    "S_x_Qlen",
    "I_x_Qlen",
]
ALL_FEATURES = BASE_FEATURES + INTERACTION_FEATURES

# Ablation feature sets
ABLATIONS = {
    "full_query_aware": ALL_FEATURES,
    "no_interactions": BASE_FEATURES,  # non-adaptive supervised
    "no_semantic": [f for f in ALL_FEATURES if not f.startswith("S")],
    "no_identifier": [f for f in ALL_FEATURES if not f.startswith("I")],
    "lexical_only": ["L", "Qid", "Qlen", "L_x_Qid", "L_x_Qlen"],
    "lexical_id_only": [f for f in ALL_FEATURES if not f.startswith("S")],
}
