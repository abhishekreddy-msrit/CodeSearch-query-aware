"""Tokenization helpers shared by lexical / identifier features."""
from __future__ import annotations

import keyword
import re

CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Identifier-shaped tokens: snake, camel, dotted, backtick, digits mixed in
ID_SHAPED = re.compile(
    r"`[^`]+`"
    r"|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"
    r"|[A-Za-z]+_[A-Za-z0-9_]+"
    r"|[a-z]+[A-Z][A-Za-z0-9]*"
    r"|[A-Za-z]*\d+[A-Za-z0-9_]*"
)


def split_identifier(tok: str) -> list[str]:
    out: list[str] = []
    for part in tok.split("_"):
        out.extend(p.lower() for p in CAMEL.split(part) if p)
    return out


def word_tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in WORD.findall(text or ""):
        out.extend(split_identifier(w))
    return [t for t in out if len(t) > 1]


def tokenize_for_tfidf(text: str) -> str:
    """Space-joined identifier-split tokens for TF-IDF / BM25-style matching."""
    return " ".join(word_tokens(text))


def parse_identifier_list(value) -> set[str]:
    """Normalize code_identifiers from list (parquet) or space-joined CSV string."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        s = str(value).strip()
        if not s or s == "nan":
            return set()
        raw = s.split()
    ids: set[str] = set()
    for tok in raw:
        if not tok or keyword.iskeyword(tok) or tok in ("self", "cls"):
            continue
        ids.add(tok)
        ids.update(split_identifier(tok))
    return {t.lower() for t in ids if len(t) > 1}


def query_id_density(query: str) -> float:
    """Fraction of whitespace tokens that look code/identifier-like (Qid)."""
    q = (query or "").strip()
    if not q:
        return 0.0
    toks = q.split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if ID_SHAPED.fullmatch(t.strip(".,;:()[]{}\"'")) or ID_SHAPED.search(t))
    # Also count snake/camel pieces inside the whole string relative to word tokens
    words = WORD.findall(q)
    if not words:
        return float(hits) / len(toks)
    shaped = 0
    for w in words:
        if "_" in w or CAMEL.search(w) or any(ch.isdigit() for ch in w):
            shaped += 1
    return max(hits / len(toks), shaped / len(words))


def query_len_norm(query: str, cap: float = 40.0) -> float:
    n = len((query or "").split())
    return min(n / cap, 1.0)
