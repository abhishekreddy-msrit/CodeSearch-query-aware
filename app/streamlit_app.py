"""
Code Search — faculty demo for NL → Python function retrieval.

Primary workflow:
  1. Enter a natural-language programming query
  2. Select one ranking model OR Compare Models
  3. Confirm candidate corpus (default: full 8,000)
  4. Click Run Search
  5. Inspect ranked results — labeled with the model that produced them

Live retrieval ≠ offline benchmark metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codesearch.ranker import DEFAULT_COMPARE, MODEL_CATALOG, CodeSearchRanker  # noqa: E402

EXAMPLES = [
    "Parse JSON from a string",
    "Read a file from disk",
    "Convert degrees to radians",
    "Compute cosine similarity between vectors",
    "Sort a list of dictionaries by a key",
]

# Single-model picker order (identifier is optional / secondary).
DEMO_SYSTEMS = ["tfidf", "sbert", "fixed_hybrid", "query_aware_fusion", "identifier"]


def inject_css() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading corpus + frozen models…")
def get_ranker() -> CodeSearchRanker:
    return CodeSearchRanker()


def bar(label: str, value: float, kind: str = "L", used: bool = True) -> str:
    pct = max(0.0, min(1.0, float(value))) * 100.0
    dim = "" if used else " dim"
    tag = "" if used else " · diagnostic only"
    return (
        f'<div class="signal-row{dim}">'
        f'<div class="signal-label">{label}{tag}</div>'
        f'<div class="signal-track"><div class="signal-fill {kind}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="signal-val">{value:.3f}</div>'
        f"</div>"
    )


def corpus_label(split: str, n: int, splits: dict) -> str:
    if split == "all":
        return f"Full Corpus — {n:,} candidate functions"
    counts = {
        "train": splits.get("train", 5600),
        "val": splits.get("val", 1200),
        "test": splits.get("test", 1200),
    }
    names = {"train": "Train", "val": "Validation", "test": "Test"}
    return f"{names[split]} split — {counts[split]:,} candidate functions"


def results_are_current(last: dict | None, query: str, systems: list[str], top_k: int, split: str) -> bool:
    if not last:
        return False
    return (
        last["query"] == query
        and list(last["systems"]) == list(systems)
        and int(last["top_k"]) == int(top_k)
        and last["split"] == split
    )


def render_score_panel(row: pd.Series, system: str, hybrid_w: tuple[float, float, float]) -> None:
    """Model-aware score visualization. Distinguishes computed vs used vs ranking score."""
    kind = str(row.get("kind") or "single")
    score_name = str(row.get("score_name") or "Ranking Score")
    primary = float(row["score"])
    L, I, S = float(row["L"]), float(row["I"]), float(row["S"])
    used = set(str(row.get("signals_used") or "").split(","))

    st.markdown(f"**{score_name}:** `{primary:.4f}` *(similarity / ranking score — not a probability)*")
    st.caption(str(row.get("formula") or ""))

    if kind == "single":
        # Only the ranking signal is primary; others are diagnostic.
        primary_kind = {"tfidf": ("Lexical", L, "L"), "identifier": ("Identifier", I, "I"), "sbert": ("Semantic", S, "S")}
        name, val, css = primary_kind[system]
        html = bar(f"{name} (used to rank)", val, css, used=True)
        # Diagnostics collapsed by default via expander
        st.markdown(html, unsafe_allow_html=True)
        with st.expander("Other signals (diagnostic — did not rank this result)", expanded=False):
            extra = []
            if system != "tfidf":
                extra.append(bar("Lexical", L, "L", used=False))
            if system != "identifier":
                extra.append(bar("Identifier", I, "I", used=False))
            if system != "sbert":
                extra.append(bar("Semantic", S, "S", used=False))
            st.markdown("".join(extra), unsafe_allow_html=True)
            st.caption("These bars are for inspection only. They did not contribute to this model's ranking.")

    elif kind == "hybrid":
        wl, wi, ws = hybrid_w
        expected = wl * L + wi * I + ws * S
        st.markdown(
            bar("Lexical (used)", L, "L", True)
            + bar("Identifier (used)", I, "I", True)
            + bar("Semantic (used)", S, "S", True)
            + bar("Hybrid Ranking Score", primary, "H", True),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Check: {wl:.2f}×{L:.3f} + {wi:.2f}×{I:.3f} + {ws:.2f}×{S:.3f} = {expected:.4f} "
            f"(displayed {primary:.4f})"
        )

    elif kind == "adaptive":
        wl = float(row.get("wL", 0))
        wi = float(row.get("wI", 0))
        ws = float(row.get("wS", 0))
        note = str(row.get("adaptation_note") or "")
        st.markdown(
            f"**Query-conditioned weights:** wL=`{wl:.3f}` · wI=`{wi:.3f}` · wS=`{ws:.3f}`  \n"
            f"Fixed Hybrid reference: `{hybrid_w[0]:.2f}` / `{hybrid_w[1]:.2f}` / `{hybrid_w[2]:.2f}`"
        )
        if note:
            st.info(note)
        st.markdown(
            bar("Lexical (used)", L, "L", True)
            + bar("Identifier (used)", I, "I", True)
            + bar("Semantic (used)", S, "S", True)
            + bar("Query-Aware Ranking Score", primary, "H", True),
            unsafe_allow_html=True,
        )


def render_hit(row: pd.Series, system: str, hybrid_w: tuple[float, float, float], expand_code: bool) -> None:
    code = str(row.get("code") or "")
    url = str(row.get("func_code_url") or "")
    path = str(row.get("func_path_in_repository") or "")
    repo = str(row.get("repository_name") or "")
    fname = str(row.get("func_name") or "")
    sig = str(row.get("signature") or "")
    score_name = str(row.get("score_name") or "Ranking Score")
    rank = int(row["rank"])
    primary = float(row["score"])
    link = f' · <a href="{url}" target="_blank" rel="noopener">GitHub source</a>' if url.startswith("http") else ""

    st.markdown(
        f"""
        <div class="cs-card">
          <div class="cs-rank">Rank #{rank}</div>
          <div class="cs-fname">{fname}</div>
          <div class="cs-meta">{repo} · {path}{link}</div>
          <div class="cs-sig">{sig}</div>
          <div class="cs-score">{score_name}: {primary:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_score_panel(row, system, hybrid_w)
    with st.expander("Source code", expanded=expand_code):
        st.code(code[:4500] + ("\n# … truncated …" if len(code) > 4500 else ""), language="python")


def offline_benchmark_table(ranker: CodeSearchRanker) -> pd.DataFrame:
    df = ranker.benchmarks
    if df.empty:
        return pd.DataFrame()
    keep = ["tfidf", "identifier_only", "sbert", "fixed_hybrid", "query_aware_fusion"]
    label_map = {
        "tfidf": "Lexical (TF-IDF)",
        "identifier_only": "Identifier",
        "sbert": "Semantic (SBERT)",
        "fixed_hybrid": "Fixed Hybrid ★ selected",
        "query_aware_fusion": "Query-Aware Fusion (experimental)",
    }
    rows = []
    for split in ("val", "test"):
        sub = df[df["split"] == split]
        for sys in keep:
            hit = sub[sub["system"] == sys]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "Split": "Validation (model selection)" if split == "val" else "Test (post-selection readout)",
                    "Model": label_map.get(sys, sys),
                    "MRR": round(float(r["MRR"]), 3),
                    "P@5": round(float(r["P@5"]), 3),
                    "R@5": round(float(r["R@5"]), 3),
                }
            )
    return pd.DataFrame(rows)


def build_rank_matrix(results: dict[str, pd.DataFrame], top_k: int) -> pd.DataFrame:
    order_sys = "fixed_hybrid" if "fixed_hybrid" in results else next(iter(results))
    names = list(results[order_sys]["func_name"].head(top_k))
    for sys, df in results.items():
        for n in df["func_name"].head(top_k):
            if n not in names:
                names.append(n)

    cols: dict = {"Function": names}
    for sys, df in results.items():
        short = MODEL_CATALOG.get(sys, {}).get("short", sys)
        rank_map = {r["func_name"]: int(r["rank"]) for _, r in df.iterrows()}
        score_map = {r["func_name"]: float(r["score"]) for _, r in df.iterrows()}
        cols[f"{short} rank"] = [rank_map.get(n, "—") for n in names]
        cols[f"{short} score"] = [f"{score_map[n]:.3f}" if n in score_map else "—" for n in names]

    # Agreement: how many systems retrieved this function in their top-k
    agreement = []
    for n in names:
        hits = sum(1 for sys, df in results.items() if n in set(df["func_name"]))
        agreement.append(f"{hits}/{len(results)}")
    cols["Models retrieving"] = agreement
    return pd.DataFrame(cols)


def common_and_unique(results: dict[str, pd.DataFrame]) -> tuple[list[str], dict[str, list[str]]]:
    sets = {sys: set(df["func_name"]) for sys, df in results.items()}
    if not sets:
        return [], {}
    common = sorted(set.intersection(*sets.values())) if len(sets) > 1 else sorted(next(iter(sets.values())))
    unique = {}
    for sys, s in sets.items():
        others = set.union(*(v for k, v in sets.items() if k != sys)) if len(sets) > 1 else set()
        unique[sys] = sorted(s - others)
    return common, unique


def main() -> None:
    st.set_page_config(
        page_title="Code Search — Retrieval Demo",
        page_icon="⌕",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    try:
        ranker = get_ranker()
    except Exception as e:
        st.error(f"Could not load models/artifacts: {e}")
        st.info("Run `python scripts/run_pipeline.py` once, then restart this app.")
        return

    available = ranker.available_systems()
    demo_keys = [k for k in DEMO_SYSTEMS if k in available]
    n_funcs = ranker.corpus_stats["n_functions"]
    n_repos = ranker.corpus_stats["n_repos"]
    hw = ranker.corpus_stats["hybrid_weights"]
    hybrid_w = (hw["L"], hw["I"], hw["S"])
    splits = ranker.corpus_stats["splits"]

    # ── Session defaults ────────────────────────────────────────────────────
    # Example chips only fill the query — they do NOT auto-run a model.
    if "pending_query" in st.session_state:
        st.session_state.query = st.session_state.pop("pending_query")
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "run_token" not in st.session_state:
        st.session_state.run_token = 0
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "mode" not in st.session_state:
        st.session_state.mode = "Single model"

    # ── Header with live active-model pill ──────────────────────────────────
    # Resolve selected systems early for the pill (widgets below update state).
    st.markdown('<p class="cs-brand">Code Search</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="cs-tagline">'
        "Natural-language retrieval over Python functions. "
        "Run one ranking model, or compare several on the same query and candidate pool."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── 1. Query ────────────────────────────────────────────────────────────
    st.markdown("### 1 · Query")
    st.text_input(
        "Natural-language programming query",
        key="query",
        placeholder='e.g. "Parse JSON from a string"',
        label_visibility="collapsed",
    )
    st.caption("Example queries fill the box only — they do not run a search. Click **Run Search** when ready.")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = ex
            st.rerun()

    # ── 2. Model / Compare ──────────────────────────────────────────────────
    st.markdown("### 2 · Ranking model")
    mode = st.radio(
        "Workflow",
        ["Single model", "Compare models"],
        horizontal=True,
        key="mode",
        help="Compare Models runs each selected system independently on the identical query and candidate pool.",
    )

    if mode == "Single model":
        # Primary research models first; identifier under "more"
        primary = [k for k in ["tfidf", "sbert", "fixed_hybrid", "query_aware_fusion"] if k in available]
        extra = [k for k in ["identifier"] if k in available]
        options = primary + extra
        labels = [available[k]["label"] for k in options]
        default_ix = options.index("fixed_hybrid") if "fixed_hybrid" in options else 0
        chosen = st.radio("Select ranking model", labels, index=default_ix, key="single_model_radio")
        selected = [options[labels.index(chosen)]]
        st.markdown(f'<div class="cs-blurb">{available[selected[0]]["blurb"]}</div>', unsafe_allow_html=True)
    else:
        label_to_key = {available[k]["label"]: k for k in demo_keys if k in DEFAULT_COMPARE or k in available}
        # Prefer DEFAULT_COMPARE order for compare defaults
        compare_opts = [k for k in DEFAULT_COMPARE if k in available]
        # Also allow identifier if user wants it
        for k in demo_keys:
            if k not in compare_opts:
                compare_opts.append(k)
        label_to_key = {available[k]["label"]: k for k in compare_opts}
        picked = st.multiselect(
            "Models to compare (each scored independently)",
            options=list(label_to_key.keys()),
            default=[available[k]["label"] for k in DEFAULT_COMPARE if k in available],
            key="compare_multiselect",
        )
        selected = [label_to_key[lb] for lb in picked]
        if len(selected) < 2:
            st.warning("Select at least two models for a meaningful comparison.")
        for k in selected:
            st.caption(f"**{available[k]['short']}** — signals used to rank: `{', '.join(available[k]['signals_used'])}`")

    # Active model pill (selected controls — not hardcoded Fixed Hybrid)
    if mode == "Compare models" and selected:
        pill = "Compare · " + " · ".join(available[k]["short"] for k in selected)
    elif selected:
        pill = f"Active model · {available[selected[0]]['label']}"
    else:
        pill = "No model selected"
    st.markdown(f'<div class="cs-pill"><strong>{pill}</strong></div>', unsafe_allow_html=True)

    # ── 3. Top-K + corpus ───────────────────────────────────────────────────
    st.markdown("### 3 · Result count & corpus")
    c1, c2 = st.columns([1, 1])
    with c1:
        top_k = st.slider(
            "Top-K Results",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of highest-ranked functions to display. This is result count, not accuracy or model quality.",
        )
        st.caption("Top-K = how many highest-ranked functions to show (default 5). Not accuracy.")
    with c2:
        split_choice = st.selectbox(
            "Candidate corpus",
            options=["all", "test", "val", "train"],
            index=0,
            format_func=lambda s: {
                "all": f"Full Corpus ({n_funcs:,} functions) — demo default",
                "test": f"Test ({splits.get('test', 1200):,} functions)",
                "val": f"Validation ({splits.get('val', 1200):,} functions)",
                "train": f"Train ({splits.get('train', 5600):,} functions)",
            }[s],
            key="corpus_split",
            help=(
                "Changes the candidate universe the models search over — not the ranking model. "
                "Full Corpus = all 8,000 functions. Splits exist for research checks."
            ),
        )
        st.caption("Pool selection changes candidates searched, not which ranking model runs.")

    with st.expander("Advanced: what do these corpus options mean?", expanded=False):
        st.markdown(
            f"""
| Option | Candidates | When to use |
|--------|------------|-------------|
| **Full Corpus** | **{n_funcs:,}** | Faculty demo (default) |
| **Train** | **{splits.get('train', 5600):,}** | Research / debugging only |
| **Validation** | **{splits.get('val', 1200):,}** | Research checks aligned with model selection |
| **Test** | **{splits.get('test', 1200):,}** | Held-out research checks (not for model selection) |

Corpus size: **{n_funcs:,}** functions across **{n_repos}** repositories.
Fixed Hybrid weights (validation-tuned): **{hw['L']:.2f}·L + {hw['I']:.2f}·I + {hw['S']:.2f}·S**.
            """
        )

    # ── Run Search ──────────────────────────────────────────────────────────
    q = (st.session_state.query or "").strip()
    can_run = bool(q) and bool(selected) and (mode != "Compare models" or len(selected) >= 1)

    b1, b2, _ = st.columns([1, 1, 2])
    run_clicked = b1.button("Run Search", type="primary", use_container_width=True, disabled=not can_run)
    if mode == "Compare models":
        compare_clicked = b2.button(
            "Compare Models",
            use_container_width=True,
            disabled=not (q and len(selected) >= 2),
            help="Same as Run Search in Compare mode — scores every selected model independently.",
        )
    else:
        compare_clicked = False
        b2.button("Compare Models", use_container_width=True, disabled=True, help="Switch workflow to Compare models first.")

    # Execute ONLY on explicit button click — never on model/query/corpus widget changes alone.
    if (run_clicked or compare_clicked) and q and selected:
        st.session_state.run_token += 1
        label = (
            f"Comparing {len(selected)} models on {corpus_label(split_choice, n_funcs, splits)}…"
            if len(selected) > 1
            else f"Running {available[selected[0]]['short']} on {corpus_label(split_choice, n_funcs, splits)}…"
        )
        with st.spinner(label):
            try:
                results = ranker.rank_many(q, selected, top_k=top_k, split=split_choice)
                st.session_state.last_run = {
                    "run_token": st.session_state.run_token,
                    "query": q,
                    "systems": list(selected),
                    "top_k": top_k,
                    "split": split_choice,
                    "n_candidates": int(next(iter(results.values()))["n_candidates"].iloc[0]),
                    "mode": mode,
                    "results": results,
                }
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.session_state.last_run = None

    last = st.session_state.last_run
    current = results_are_current(last, q, selected, top_k, split_choice)

    # ── 4. Results ──────────────────────────────────────────────────────────
    st.markdown("### 4 · Live retrieval results")
    st.caption(
        "Live rankings for the query you ran. "
        "These are not offline MRR / P@5 / R@5 benchmark numbers."
    )

    if last and not current:
        produced = (
            "Compare · " + " · ".join(available[s]["short"] for s in last["systems"] if s in available)
            if len(last["systems"]) > 1
            else available.get(last["systems"][0], {}).get("label", last["systems"][0])
        )
        st.warning(
            f"Controls changed. Last run used **{produced}** on query `{last['query']}`. "
            "Those results are hidden until you click **Run Search** again with the current settings."
        )
        # Do not show stale results — prevents model/result mismatch.
    elif not last:
        st.markdown(
            '<div class="cs-empty">'
            "Enter a query, select a model (or Compare Models), then click <b>Run Search</b>."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        n_cand = last["n_candidates"]
        pool = corpus_label(last["split"], n_cand if last["split"] == "all" else n_cand, splits)
        # Prefer actual n_candidates from the run
        pool = (
            f"Full Corpus — {n_cand:,} candidate functions"
            if last["split"] == "all"
            else f"{last['split'].title()} split — {n_cand:,} candidate functions"
        )

        if len(last["systems"]) == 1:
            sys = last["systems"][0]
            meta = available[sys]
            st.markdown(
                f"""
                <div class="cs-result-banner">
                  <div><strong>Model that produced these results:</strong> {meta['label']}</div>
                  <div><strong>Query:</strong> <code>{last['query']}</code></div>
                  <div><strong>Corpus searched:</strong> {pool}</div>
                  <div><strong>Top-K:</strong> {last['top_k']} · <strong>Ranking score:</strong> {meta['score_name']}</div>
                  <div><strong>Signals used to rank:</strong> {', '.join(meta['signals_used'])}
                      &nbsp;|&nbsp; <strong>Signals computed:</strong> {', '.join(meta['signals_computed'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            df = last["results"][sys]
            for _, row in df.iterrows():
                render_hit(row, sys, hybrid_w, expand_code=(int(row["rank"]) == 1))
        else:
            st.markdown(
                f"""
                <div class="cs-result-banner">
                  <div><strong>Comparison run</strong> — each model scored independently</div>
                  <div><strong>Query:</strong> <code>{last['query']}</code></div>
                  <div><strong>Corpus searched:</strong> {pool} (identical for every model)</div>
                  <div><strong>Top-K:</strong> {last['top_k']} · <strong>Models:</strong>
                    {', '.join(available[s]['short'] for s in last['systems'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            summary = ranker.comparison_summary(last["results"])
            st.markdown("#### Top-1 by model")
            st.dataframe(
                summary.rename(
                    columns={
                        "label": "Model",
                        "rank1_function": "#1 function",
                        "rank1_score": "#1 ranking score",
                        "score_name": "Score meaning",
                        "formula": "Formula",
                    }
                )[["Model", "#1 function", "#1 ranking score", "Score meaning", "Formula"]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Rank matrix (same functions across models)")
            st.caption("Each rank/score comes from that model's own ranking pass — not relabeled.")
            matrix = build_rank_matrix(last["results"], last["top_k"])
            st.dataframe(matrix, use_container_width=True, hide_index=True)

            common, unique = common_and_unique(last["results"])
            st.markdown("#### Agreement")
            if common:
                st.success("Retrieved by **all** compared models: " + ", ".join(f"`{c}`" for c in common))
            else:
                st.info("No function appears in every model's top-K for this query.")
            for sys, uniq in unique.items():
                if uniq:
                    st.caption(
                        f"**Unique to {available[sys]['short']}:** "
                        + ", ".join(f"`{u}`" for u in uniq)
                    )

            tabs = st.tabs([available[s]["short"] for s in last["systems"]])
            for tab, sys in zip(tabs, last["systems"]):
                with tab:
                    meta = available[sys]
                    st.markdown(
                        f"**{meta['label']}** — ranking score = **{meta['score_name']}**  \n"
                        f"Signals used to rank: `{', '.join(meta['signals_used'])}`"
                    )
                    df = last["results"][sys]
                    if not df.empty:
                        st.caption(str(df.iloc[0]["formula"]))
                    for _, row in df.iterrows():
                        render_hit(row, sys, hybrid_w, expand_code=(int(row["rank"]) == 1))

    # ── 5. Offline research ─────────────────────────────────────────────────
    st.markdown("### 5 · Offline benchmark (dataset-level — not this query)")
    st.markdown(
        """
<div class="cs-card">
  <p style="margin:0; color:#8b9bb0; font-size:0.9rem; line-height:1.55;">
  <b style="color:#e8eef7;">Keep these distinct</b><br/>
  • <b>Live retrieval</b> — ranked functions for the query above<br/>
  • <b>Ranking score</b> — similarity assigned to one candidate (not a probability)<br/>
  • <b>Benchmark metrics</b> — MRR / P@5 / R@5 over a labeled eval set<br/>
  • <b>Model selection</b> — Fixed Hybrid won on <i>validation</i>; Query-Aware Fusion did not beat it
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
    bench = offline_benchmark_table(ranker)
    if bench.empty:
        st.warning("No `results/primary_fullsplit_metrics.csv` found.")
    else:
        st.dataframe(bench, use_container_width=True, hide_index=True)
        st.markdown(
            f"""
- **MRR** — mean reciprocal rank of the gold function (higher is better).
- **Validation** selected **Fixed Hybrid** (MRR ≈ 0.817). Query-Aware Fusion ≈ 0.813 — **not** selected.
- **Test** was viewed in earlier phases (prior exposure). Post-selection readout only:
  Fixed Hybrid ≈ 0.786 · Query-Aware ≈ 0.790. That tiny test edge was **not** used for selection.
- Target +0.010 MRR over Fixed Hybrid on validation: **not achieved**.
- Inference hybrid formula: `{ranker.hybrid_formula_text()}`.
            """
        )

    st.markdown(
        '<p class="cs-footnote">'
        "Code Search · CodeSearchNet Python subset · Sentence-BERT all-MiniLM-L6-v2 · "
        "frozen artifacts · validation-driven selection · Fixed Hybrid is the default."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
