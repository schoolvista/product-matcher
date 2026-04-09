"""
Streamlit UI for replacement_llm.run_single_lookup.

Local:  streamlit run app.py
Cloud:  set main file to app.py; add secrets (ANTHROPIC_API_KEY, TAVILY_API_KEY) in Streamlit Cloud.
"""

from __future__ import annotations

import json
import os
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


_apply_streamlit_secrets()

import replacement_llm as m

st.set_page_config(
    page_title="Product substitute lookup",
    page_icon="🔍",
    layout="wide",
)

st.title("Product substitute lookup")
st.caption("Tavily + Claude — default pipeline extracts source specs, then matches target-brand SKUs from web evidence.")

anthropic_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
tavily_ok = bool(os.environ.get("TAVILY_API_KEY"))
if not anthropic_ok or not tavily_ok:
    st.error(
        "Missing API keys. For **Streamlit Cloud**, add `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` "
        "under app **Secrets**. For **local** runs, use a `.env` file (see `env.example`)."
    )
    st.stop()

with st.sidebar:
    st.header("Options")
    legacy = st.checkbox("Legacy single-pass search", value=False, help="Skip spec extraction; one merged Tavily pass.")
    tavily_depth = st.selectbox(
        "Tavily depth",
        options=["(env default)", "basic", "fast", "ultra-fast", "advanced"],
        index=0,
    )
    depth_val = None if tavily_depth == "(env default)" else tavily_depth
    top_results = st.slider("Results per Tavily query", 3, 15, 8)
    max_subs = st.slider("Max substitutes", 1, 10, 5)
    max_target_q = st.slider("Max target queries (attribute pipeline)", 3, 12, 8)
    llm_expand = st.checkbox(
        "LLM expand (legacy only)",
        value=False,
        help="Extra search queries from Claude; only applies in legacy mode.",
    )

st.subheader("Product")
c1, c2, c3 = st.columns(3)
with c1:
    source_brand = st.text_input("Source brand", value="SAL", placeholder="e.g. SAL")
with c2:
    product_code = st.text_input("Product code", placeholder="e.g. S9065TCWH")
with c3:
    target_brand = st.text_input("Target brand", value="Haneco", placeholder="e.g. Haneco")

description = st.text_area(
    "Description (optional, recommended)",
    placeholder="e.g. 9W tri-colour dimmable downlight 92mm cutout IP44",
    height=80,
)

run = st.button("Run lookup", type="primary", use_container_width=True)

if run:
    if not (source_brand or "").strip() or not (product_code or "").strip() or not (target_brand or "").strip():
        st.warning("Fill in source brand, product code, and target brand.")
    else:
        with st.status("Running Tavily + Claude (may take 30–90 seconds)…", expanded=True) as status:
            status.write("Searching and matching…")
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
