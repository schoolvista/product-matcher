"""
Streamlit UI for replacement_llm.run_single_lookup.

Each widget maps to the same arguments as ``replacement_llm.py`` CLI; values are passed
directly to ``run_single_lookup()`` (no subprocess). Use the sidebar “Equivalent CLI”
block to copy the matching command line.

Local:  streamlit run app.py
Cloud:  Secrets — ANTHROPIC_API_KEY, TAVILY_API_KEY
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import streamlit as st

# Local .env (optional; Cloud uses st.secrets)
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env")
    load_dotenv(Path.cwd() / ".env", override=True)
except ImportError:
    pass


def _apply_streamlit_secrets() -> None:
    try:
        for key in (
            "ANTHROPIC_API_KEY",
            "TAVILY_API_KEY",
            "ANTHROPIC_MODEL",
            "TAVILY_SEARCH_DEPTH",
            "TAVILY_TARGET_DOMAINS",
        ):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


# Must be the first Streamlit command — accessing st.secrets before this causes a blank/black UI.
st.set_page_config(
    page_title="Product substitute lookup",
    page_icon="🔍",
    layout="wide",
)

_apply_streamlit_secrets()

import replacement_llm as m

st.title("Product substitute lookup")
st.caption(
    "Form fields mirror `replacement_llm.py` flags. The app calls the same Python API "
    "the CLI uses — see **Equivalent CLI** in the sidebar to copy a command line."
)


def build_equivalent_cli_command(
    *,
    source_brand: str,
    product_code: str,
    target_brand: str,
    description: str,
    legacy: bool,
    tavily_depth: str | None,
    top_search_results: int,
    max_substitutes: int,
    max_target_queries: int,
    llm_expand: bool,
    max_llm_search_queries: int,
) -> str:
    """Single line for display (POSIX-style quoting; fine for Git Bash / WSL / docs)."""
    parts = [
        "python",
        "replacement_llm.py",
        "--no-prompt",
        "--source-brand",
        shlex.quote(source_brand.strip()),
        "-c",
        shlex.quote(product_code.strip()),
        "--target-brand",
        shlex.quote(target_brand.strip()),
    ]
    if (description or "").strip():
        parts += ["-d", shlex.quote(description.strip())]
    if tavily_depth:
        parts += ["--tavily-depth", shlex.quote(tavily_depth)]
    parts += [
        "--top-search-results",
        str(int(top_search_results)),
        "--max-substitutes",
        str(int(max_substitutes)),
        "--max-target-queries",
        str(int(max_target_queries)),
    ]
    if llm_expand:
        parts.append("--llm-expand-search")
        parts += ["--max-llm-search-queries", str(int(max_llm_search_queries))]
    if legacy:
        parts.append("--legacy-single-pass")
    parts.append("-q")
    return " ".join(parts)


anthropic_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
tavily_ok = bool(os.environ.get("TAVILY_API_KEY"))
if not anthropic_ok or not tavily_ok:
    st.error(
        "Missing API keys. For **Streamlit Cloud**, add `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` "
        "under app **Secrets**. For **local** runs, use a `.env` file (see `env.example`)."
    )
    st.stop()

with st.sidebar:
    st.header("CLI-linked options")
    legacy = st.checkbox(
        "`--legacy-single-pass`",
        value=False,
        help="Legacy: one merged Tavily pass + single match (no attribute pipeline).",
    )
    tavily_depth_sel = st.selectbox(
        "`--tavily-depth`",
        options=["(env default)", "basic", "fast", "ultra-fast", "advanced"],
        index=0,
        help="Tavily search_depth; advanced uses 2 API credits per search.",
    )
    depth_val = None if tavily_depth_sel == "(env default)" else tavily_depth_sel
    top_results = st.slider(
        "`--top-search-results`",
        min_value=3,
        max_value=15,
        value=8,
        help="Max results returned per Tavily query.",
    )
    max_subs = st.slider(
        "`--max-substitutes`",
        min_value=1,
        max_value=10,
        value=5,
        help="Max substitute rows from Claude.",
    )
    max_target_q = st.slider(
        "`--max-target-queries`",
        min_value=3,
        max_value=12,
        value=8,
        help="Attribute pipeline: cap on target-brand Tavily queries.",
    )
    llm_expand = st.checkbox(
        "`--llm-expand-search`",
        value=False,
        help="Legacy only: Claude proposes extra search queries.",
    )
    max_llm_q = st.slider(
        "`--max-llm-search-queries`",
        min_value=1,
        max_value=8,
        value=3,
        help="Used only when --llm-expand-search is on.",
        disabled=not llm_expand,
    )

    st.divider()
    st.markdown("**Widget → flag** (main form)")
    st.markdown(
        "| Widget | CLI |\n|--------|-----|\n"
        "| Source brand | `--source-brand` |\n"
        "| Product code | `-c` / `--product-code` |\n"
        "| Target brand | `--target-brand` |\n"
        "| Description | `-d` / `--description` |\n"
        "| (Streamlit always supplies values) | `--no-prompt` |\n"
        "| (no stderr spam) | `-q` / `--quiet` |\n"
    )

st.subheader("Inputs (same as CLI)")
c1, c2, c3 = st.columns(3)
with c1:
    source_brand = st.text_input(
        "Source brand",
        value="SAL",
        placeholder="e.g. SAL",
        help="CLI: `--source-brand`",
    )
with c2:
    product_code = st.text_input(
        "Product code",
        placeholder="e.g. S9065TCWH",
        help="CLI: `-c` / `--product-code`",
    )
with c3:
    target_brand = st.text_input(
        "Target brand",
        value="Haneco",
        placeholder="e.g. Haneco",
        help="CLI: `--target-brand`",
    )

description = st.text_area(
    "Description (optional)",
    placeholder="e.g. 9W tri-colour dimmable downlight 92mm cutout IP44",
    height=80,
    help="CLI: `-d` / `--description`",
)

cli_preview = build_equivalent_cli_command(
    source_brand=source_brand or "",
    product_code=product_code or "",
    target_brand=target_brand or "",
    description=description or "",
    legacy=legacy,
    tavily_depth=depth_val,
    top_search_results=top_results,
    max_substitutes=max_subs,
    max_target_queries=max_target_q,
    llm_expand=llm_expand,
    max_llm_search_queries=max_llm_q,
)
with st.sidebar:
    with st.expander("Equivalent CLI (copy)", expanded=False):
        st.code(cli_preview, language="bash")

run = st.button("Run lookup", type="primary", use_container_width=True)

if run:
    if not (source_brand or "").strip() or not (product_code or "").strip() or not (target_brand or "").strip():
        st.warning("Fill in source brand, product code, and target brand.")
    else:
        with st.status("Running Tavily + Claude (may take 30–90 seconds)…", expanded=True) as status:
            status.write("Calling `run_single_lookup()` with the same arguments as the CLI above…")
            try:
                payload = m.run_single_lookup(
                    source_brand=source_brand.strip(),
                    product_code=product_code.strip(),
                    target_brand=target_brand.strip(),
                    description=(description or "").strip(),
                    tavily_api_key=os.environ["TAVILY_API_KEY"],
                    anthropic_key=os.environ["ANTHROPIC_API_KEY"],
                    top_search_results=int(top_results),
                    max_substitutes=int(max_subs),
                    llm_expand_search=llm_expand,
                    max_llm_search_queries=int(max_llm_q),
                    tavily_search_depth=depth_val,
                    legacy_single_pass=legacy,
                    max_target_queries=int(max_target_q),
                    verbose=False,
                )
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Failed", state="error", expanded=True)
                st.exception(e)
                st.stop()

        st.success(f"Pipeline: **{payload.get('pipeline', '')}**")
        st.caption("Equivalent command used for the same logic:")
        st.code(cli_preview, language="bash")

        prof = payload.get("source_profile")
        if isinstance(prof, dict) and prof.get("summary"):
            with st.expander("Extracted source profile", expanded=True):
                st.json(prof)

        llm = payload.get("llm") or {}
        subs = llm.get("substitutes") or []
        if subs:
            st.subheader("Suggested matches")
            rows = []
            for s in subs:
                if not isinstance(s, dict):
                    continue
                rows.append(
                    {
                        "SKU / model": s.get("suggested_identifier", ""),
                        "Confidence": s.get("confidence", ""),
                        "Reasoning": s.get("reasoning", ""),
                        "URLs": "; ".join(s.get("cited_urls") or []),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No substitutes returned (check evidence / try description or legacy mode).")

        notes = llm.get("overall_notes", "")
        if notes:
            st.markdown("**Notes**")
            st.write(notes)

        errs = payload.get("search_errors") or []
        if errs:
            with st.expander("Search warnings / errors"):
                for e in errs:
                    st.code(str(e), language=None)

        with st.expander("Raw JSON response"):
            st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
