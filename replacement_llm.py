"""
Find a target-brand match for a source product using Tavily Search + Anthropic Claude.

API keys (any one of):
  - A ``.env`` file next to this script or in the current working directory
  - Real environment variables (they override .env if set)
  - ``--env-file PATH`` to load a specific .env

Variables: ANTHROPIC_API_KEY, TAVILY_API_KEY
Optional: ANTHROPIC_MODEL, TAVILY_SEARCH_DEPTH, TAVILY_TARGET_DOMAINS (comma-separated domains
for target-brand searches only)

Single-product mode (default): prompts or flags for source brand, product code, target brand.
Batch mode: pass --input path to Excel/CSV (see module docstring for columns).

Default flow (attribute pipeline):
  1) Search for the **source** product and have the LLM extract specs (wattage, cutout, IP,
     colour temp, dimmable, product type, etc.).
  2) Run **target-brand** searches built from those specs (not the source SKU on target sites).
  3) Match **target product codes** (e.g. Haneco VIVA110) that appear in evidence to the
     extracted profile. SKUs must still appear in snippets — the model does not invent them.

Optional: ``--legacy-single-pass`` uses the older one-shot search + match. Optional env
``TAVILY_TARGET_DOMAINS`` (comma-separated, e.g. ``haneco.com.au``) limits target-phase searches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore


def _preload_dotenv() -> None:
    """Load ``.env`` from this script's folder, then CWD (CWD wins on duplicate keys)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")
    load_dotenv(Path.cwd() / ".env", override=True)


_preload_dotenv()


def _progress(msg: str, *, prefix: str = "") -> None:
    """Status line to stderr so stdout stays clean for piping (e.g. --json-out)."""
    line = f"{prefix}{msg}" if prefix else msg
    print(line, file=sys.stderr, flush=True)


TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Max characters of Tavily ``content`` per result passed to the LLM (avoid huge context).
TAVILY_SNIPPET_MAX_CHARS = 2500

_FALLBACK_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def _anthropic_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", _FALLBACK_ANTHROPIC_MODEL)


def load_env_files(extra_env_path: Path | None = None) -> None:
    """Reload default .env locations, then optionally ``--env-file`` (highest priority)."""
    _preload_dotenv()
    if extra_env_path is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if extra_env_path.is_file():
        load_dotenv(extra_env_path, override=True)


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _short_desc_for_query(description: str, max_len: int = 100) -> str:
    if not description or not str(description).strip():
        return ""
    text = re.sub(r"\s+", " ", str(description).strip())
    if len(text) > max_len:
        text = text[: max_len - 1].rsplit(" ", 1)[0]
    return text


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in lower.items()}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in inv:
                return inv[n]
        return None

    colmap = {}
    pc = pick("product_code", "sku", "code", "sal product code")
    desc = pick("description", "sal description", "product description")
    sup = pick("supplier", "brand", "source supplier")
    if pc:
        colmap[pc] = "product_code"
    if desc:
        colmap[desc] = "description"
    if sup:
        colmap[sup] = "supplier"
    df = df.rename(columns=colmap)

    if "product_code" not in df.columns:
        raise ValueError(
            "Input must include a product code column "
            "(e.g. product_code, sku, or 'SAL Product Code')."
        )
    if "description" not in df.columns:
        df["description"] = ""
    if "supplier" not in df.columns:
        df["supplier"] = ""

    df["product_code"] = df["product_code"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["supplier"] = df["supplier"].fillna("").astype(str).str.strip()

    return df


def _tavily_include_domains() -> list[str] | None:
    raw = (os.environ.get("TAVILY_TARGET_DOMAINS") or "").strip()
    if not raw:
        return None
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return parts or None


def tavily_search(
    query: str,
    api_key: str,
    num_results: int = 8,
    search_depth: str | None = None,
    include_domains: list[str] | None = None,
) -> list[dict[str, str]]:
    """Call Tavily Search API; map results to title/link/snippet for the rest of the pipeline."""
    num_results = max(1, min(num_results, 20))
    depth = (search_depth or os.environ.get("TAVILY_SEARCH_DEPTH") or "basic").strip().lower()
    if depth not in ("basic", "fast", "ultra-fast", "advanced"):
        depth = "basic"

    body: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": num_results,
        "search_depth": depth,
        "include_answer": False,
    }
    if include_domains:
        body["include_domains"] = include_domains
    r = requests.post(TAVILY_SEARCH_URL, json=body, timeout=60)
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        r.raise_for_status()
        raise RuntimeError(f"Tavily invalid JSON: {e}") from e

    if r.status_code >= 400:
        msg = data.get("detail") or data.get("error") or data.get("message") or r.text
        raise RuntimeError(str(msg))

    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    out: list[dict[str, str]] = []
    for item in data.get("results") or []:
        url = item.get("url") or ""
        title = item.get("title") or ""
        content = (item.get("content") or "").strip()
        if len(content) > TAVILY_SNIPPET_MAX_CHARS:
            content = content[: TAVILY_SNIPPET_MAX_CHARS - 1] + "…"
        out.append({"title": title, "link": url, "snippet": content})
    return out


def merge_search_results(
    *result_groups: list[dict[str, str]],
) -> list[dict[str, str]]:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for group in result_groups:
        for item in group:
            link = item.get("link") or ""
            if not link or link in seen:
                continue
            seen.add(link)
            merged.append(item)
    return merged


def _format_search_context(results: list[dict[str, str]]) -> str:
    if not results:
        return "(No search results returned.)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    URL: {r['link']}\n    {r['snippet']}")
    return "\n\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def claude_suggest_substitutes(
    client: Anthropic,
    model: str,
    source_brand: str,
    target_brand: str,
    product_code: str,
    description: str,
    search_context: str,
    max_substitutes: int,
) -> dict[str, Any]:
    system = (
        "You are a careful lighting and electrical procurement assistant. "
        "You only use the web search snippets provided; do not invent specific "
        "product codes or URLs that do not appear in the evidence. "
        "If evidence is weak, say so and lower confidence. "
        "Respond with a single JSON object only (no markdown outside JSON)."
    )
    user = f"""Source brand / manufacturer (existing product): {source_brand}
Manufacturer product code: {product_code}
Product description (optional): {description or "(none provided)"}

Target brand to find a matching product from: {target_brand}

Web search results (snippets only; may be incomplete or wrong):
{search_context}

Return JSON with this exact shape:
{{
  "substitutes": [
    {{
      "suggested_identifier": "string — model/SKU or product name as shown in evidence",
      "confidence": "low|medium|high",
      "reasoning": "short explanation tied to snippets",
      "cited_urls": ["urls from the search results that support this suggestion"]
    }}
  ],
  "overall_notes": "string — limitations, missing data, or if no good match found"
}}

Rules:
- At most {max_substitutes} items in substitutes; fewer is fine if evidence is weak.
- Every cited_urls entry must be copied exactly from the URLs shown in the search results above.
- If nothing relevant to {target_brand} appears in the snippets, substitutes may be empty."""

    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    block = msg.content[0]
    if block.type != "text":
        raise RuntimeError(f"Unexpected response block type: {block.type}")
    return _extract_json_object(block.text)


def claude_extract_source_product_profile(
    client: Anthropic,
    model: str,
    source_brand: str,
    product_code: str,
    user_description: str,
    source_search_context: str,
) -> dict[str, Any]:
    """Parse source-product evidence into structured specs for cross-brand matching."""
    system = (
        "You extract structured product facts for lighting/electrical procurement. "
        "Use ONLY the search evidence and the user description; use null if unknown. "
        "Respond with a single JSON object only (no markdown)."
    )
    user = f"""Source brand: {source_brand}
Source manufacturer code: {product_code}
User-provided description (may be empty): {user_description or "(none)"}

Search evidence (retailer pages, datasheets, etc.):
{source_search_context}

Return JSON with this exact shape:
{{
  "summary": "one concise line describing the fitting",
  "product_type": "e.g. recessed LED downlight, batten, floodlight",
  "wattage_w": null or number,
  "cutout_or_hole_mm": null or number,
  "overall_diameter_mm": null or number,
  "ip_rating": null or string like IP44,
  "colour_temperature": null or string e.g. tri-colour 3000/4000/5700K selectable,
  "dimmable": null or true or false,
  "body_finish": null or string e.g. white,
  "ic_or_insulation_rating_note": null or string,
  "voltage_v": null or number,
  "series_or_family": null or string,
  "tavily_queries_for_target_brand": [
    "up to 5 short web search queries that INCLUDE the target brand name once we search for substitutes — you do NOT know the target brand here, so use the literal placeholder TARGET_BRAND in each query string"
  ]
}}

Rules for tavily_queries_for_target_brand:
- Exactly use the substring TARGET_BRAND in each query (we will replace it with the real brand).
- Focus on specs above (wattage, cutout, IP, CCT, dimmable, product type) — NOT the competitor SKU.
- Each query under 100 characters if possible."""

    msg = client.messages.create(
        model=model,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    block = msg.content[0]
    if block.type != "text":
        return {}
    try:
        return _extract_json_object(block.text)
    except (json.JSONDecodeError, ValueError):
        return {}


def _substitute_target_brand_in_queries(
    queries: list[Any], target_brand: str
) -> list[str]:
    tb = target_brand.strip()
    out: list[str] = []
    for q in queries:
        if not isinstance(q, str) or not q.strip():
            continue
        out.append(q.strip().replace("TARGET_BRAND", tb))
    return out


def build_target_queries_from_profile(
    target_brand: str,
    profile: dict[str, Any],
    product_code: str,
    user_description: str,
    max_queries: int,
) -> list[str]:
    """Combine LLM-suggested queries with deterministic spec-based queries."""
    tb = target_brand.strip()
    out: list[str] = []

    raw_llm = profile.get("tavily_queries_for_target_brand") or profile.get(
        "target_brand_search_queries"
    )
    if isinstance(raw_llm, list):
        out.extend(_substitute_target_brand_in_queries(raw_llm, tb))

    watt = profile.get("wattage_w")
    cut = profile.get("cutout_or_hole_mm")
    ip = profile.get("ip_rating")
    pt = (profile.get("product_type") or "").strip()
    cct = (profile.get("colour_temperature") or "").strip()
    dim = profile.get("dimmable")

    spec_parts = [tb]
    if isinstance(watt, (int, float)):
        spec_parts.append(f"{int(watt)}W")
    if isinstance(cut, (int, float)):
        spec_parts.append(f"{int(cut)}mm cutout")
    if ip:
        spec_parts.append(str(ip))
    if cct and len(cct) < 80:
        spec_parts.append(cct)
    if dim is True:
        spec_parts.append("dimmable")
    if pt:
        spec_parts.append(pt)
    out.append(" ".join(spec_parts))

    out.append(f"{tb} LED downlight product code datasheet")
    out.append(f"{tb} downlight range catalogue")

    if user_description.strip():
        short = _short_desc_for_query(user_description, 90)
        if short:
            out.append(f"{tb} {short}")

    return _dedupe_queries(out)[:max(1, max_queries)]


def claude_match_target_by_profile(
    client: Anthropic,
    model: str,
    source_brand: str,
    source_product_code: str,
    target_brand: str,
    profile: dict[str, Any],
    target_search_context: str,
    max_substitutes: int,
) -> dict[str, Any]:
    """Pick target-brand SKU(s) from evidence that align with the extracted source profile."""
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
    system = (
        "You match a target supplier's product codes to a source product's technical profile. "
        "You may ONLY name target product codes/SKUs that literally appear in the target-brand "
        "search evidence (titles, snippets, URLs). Do not invent Haneco/SAL codes. "
        "If no target SKU in evidence fits reasonably, return an empty candidates list. "
        "Respond with a single JSON object only (no markdown)."
    )
    user = f"""Source: {source_brand} / code {source_product_code}
Target brand to buy from: {target_brand}

Extracted source product profile (from web evidence):
{profile_json}

Target-brand search results (only valid evidence for target SKUs):
{target_search_context}

Return JSON:
{{
  "substitutes": [
    {{
      "suggested_identifier": "exact target product code or model as it appears in evidence e.g. VIVA110-MULTI",
      "confidence": "low|medium|high",
      "attribute_match": "which profile fields align or differ vs this SKU's description in evidence",
      "reasoning": "brief",
      "cited_urls": ["urls from target search results that mention this SKU"]
    }}
  ],
  "overall_notes": "string"
}}

Rules:
- At most {max_substitutes} substitutes; fewer is OK.
- suggested_identifier must be copied from text in the evidence, not guessed.
- cited_urls must be copied exactly from the target search results above."""

    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    block = msg.content[0]
    if block.type != "text":
        raise RuntimeError(f"Unexpected response block type: {block.type}")
    data = _extract_json_object(block.text)
    # Normalise optional LLM key "candidates" -> substitutes
    if "substitutes" not in data and "candidates" in data:
        subs = []
        for c in data.get("candidates") or []:
            if not isinstance(c, dict):
                continue
            subs.append(
                {
                    "suggested_identifier": c.get("target_product_code")
                    or c.get("suggested_identifier")
                    or "",
                    "confidence": c.get("confidence", ""),
                    "reasoning": (c.get("attribute_match") or "")
                    + " "
                    + (c.get("reasoning") or ""),
                    "cited_urls": c.get("cited_urls") or [],
                }
            )
        data["substitutes"] = subs
    # Merge attribute_match into reasoning if present
    for s in data.get("substitutes") or []:
        if isinstance(s, dict) and s.get("attribute_match"):
            am = str(s.pop("attribute_match", "")).strip()
            if not am:
                continue
            rs = str(s.get("reasoning") or "").strip()
            s["reasoning"] = f"{am} | {rs}" if rs else am
    return data


def claude_propose_target_brand_queries(
    client: Anthropic,
    model: str,
    source_brand: str,
    target_brand: str,
    product_code: str,
    description: str,
    source_product_snippets: str,
    max_queries: int,
) -> list[str]:
    """Use LLM to suggest search queries that surface *target* brand hits using specs, not only competitor SKU."""
    system = (
        "You help procurement researchers write short web search queries. "
        "Respond with a single JSON object only (no markdown)."
    )
    user = f"""We need to find **{target_brand}** products that could replace this source item:
- Source brand: {source_brand}
- Product code: {product_code}
- Description: {description or "(none)"}

The snippets below are from a search for the source product. Other brands rarely use
{source_brand}'s product code on their own pages, so we need new queries that will surface
**{target_brand}** catalogues, datasheets, or shops using **product type and specs**
(wattage, cut-out, IP, CCT, beam angle, dimming, shape, series names) visible in the text.

Snippets:
{source_product_snippets}

Return JSON exactly:
{{"follow_up_search_queries": ["query1", "query2", ...]}}

Rules:
- At most {max_queries} queries.
- Every query MUST include the word **{target_brand}** (or obvious spelling) so results skew to that brand.
- Prefer concrete attributes from the snippets/description over repeating only "{product_code}".
- Each query under 120 characters."""

    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    block = msg.content[0]
    if block.type != "text":
        return []
    try:
        data = _extract_json_object(block.text)
    except (json.JSONDecodeError, ValueError):
        return []
    raw = (
        data.get("follow_up_search_queries")
        or data.get("follow_up_google_queries")
        or data.get("queries")
        or []
    )
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[:max_queries]:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def default_search_queries(
    source_brand: str,
    target_brand: str,
    product_code: str,
    description: str,
) -> list[str]:
    """Mix source identification, cross-reference style, and target-brand + attribute queries."""
    sb = source_brand.strip()
    tb = target_brand.strip()
    code = product_code.strip()
    desc = (description or "").strip()

    queries: list[str] = []

    # 1) Source product (specs, distributors — good context, rarely lists other brands' SKUs)
    src_q = f'"{sb}" {code}'
    if desc:
        src_q += f" {desc}"
    queries.append(src_q)

    # 2) Cross-reference / equivalent (sometimes wholesalers publish matrices)
    queries.append(
        f"{tb} (equivalent OR substitute OR cross reference OR comparable OR compatible) "
        f"{sb} {code}"
    )

    # 3) Target brand + generic product terms (competitor code often absent on target sites)
    short = _short_desc_for_query(desc, 110)
    if short:
        queries.append(f"{tb} {short}")
    else:
        queries.append(f"{tb} LED lighting fitting {code}")

    # 4) Target brand catalogue / product discovery
    queries.append(f"{tb} (catalogue OR catalog OR products OR range OR series) LED")

    return _dedupe_queries(queries)


def run_tavily_query_list(
    queries: list[str],
    tavily_api_key: str,
    top_search_results: int,
    search_depth: str | None = None,
    include_domains: list[str] | None = None,
    verbose: bool = False,
    progress_prefix: str = "",
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Returns (merged_results, errors_per_query, queries_executed_in_order)."""
    all_groups: list[list[dict[str, str]]] = []
    errors: list[str] = []
    n = len(queries)
    for i, q in enumerate(queries, 1):
        if verbose:
            qshort = (q[:72] + "…") if len(q) > 72 else q
            _progress(f"  Tavily {i}/{n}: {qshort}", prefix=progress_prefix)
        try:
            all_groups.append(
                tavily_search(
                    q,
                    tavily_api_key,
                    num_results=top_search_results,
                    search_depth=search_depth,
                    include_domains=include_domains,
                )
            )
        except Exception as e:
            all_groups.append([])
            errors.append(f"{q!r}: {e}")
            if verbose:
                _progress(f"  ! error: {e}", prefix=progress_prefix)
        else:
            if verbose:
                nh = len(all_groups[-1])
                _progress(f"  ← {nh} hit(s)", prefix=progress_prefix)
    merged = merge_search_results(*all_groups)
    return merged, errors, queries


def run_single_lookup(
    source_brand: str,
    product_code: str,
    target_brand: str,
    description: str,
    tavily_api_key: str,
    anthropic_key: str,
    top_search_results: int,
    max_substitutes: int,
    queries: list[str] | None = None,
    llm_expand_search: bool = False,
    max_llm_search_queries: int = 3,
    tavily_search_depth: str | None = None,
    legacy_single_pass: bool = False,
    max_target_queries: int = 8,
    anthropic_client: Anthropic | None = None,
    verbose: bool = True,
    progress_prefix: str = "",
) -> dict[str, Any]:
    if Anthropic is None:
        raise RuntimeError("Install anthropic: pip install anthropic")

    client = anthropic_client or Anthropic(api_key=anthropic_key)
    pf = progress_prefix

    if legacy_single_pass or queries is not None:
        base_queries = (
            queries
            if queries is not None
            else default_search_queries(source_brand, target_brand, product_code, description)
        )

        llm_proposed: list[str] = []
        if llm_expand_search and queries is None and base_queries:
            if verbose:
                _progress("Legacy: extra Tavily + LLM for follow-up queries…", prefix=pf)
            early_hits, _, _ = run_tavily_query_list(
                [base_queries[0]],
                tavily_api_key,
                top_search_results,
                search_depth=tavily_search_depth,
                verbose=verbose,
                progress_prefix=pf,
            )
            snippet_text = _format_search_context(early_hits)
            try:
                llm_proposed = claude_propose_target_brand_queries(
                    client=client,
                    model=_anthropic_model(),
                    source_brand=source_brand,
                    target_brand=target_brand,
                    product_code=product_code,
                    description=description,
                    source_product_snippets=snippet_text,
                    max_queries=max_llm_search_queries,
                )
            except Exception:
                llm_proposed = []

        qlist = _dedupe_queries(list(base_queries) + llm_proposed)
        if verbose:
            _progress(
                f"Legacy: running {len(qlist)} Tavily search(es), then Claude match…",
                prefix=pf,
            )
        merged, errors, executed = run_tavily_query_list(
            qlist,
            tavily_api_key,
            top_search_results,
            search_depth=tavily_search_depth,
            verbose=verbose,
            progress_prefix=pf,
        )
        if verbose:
            _progress(
                f"  Merged {len(merged)} unique page(s). Calling Claude…",
                prefix=pf,
            )
        ctx = _format_search_context(merged)

        try:
            parsed = claude_suggest_substitutes(
                client=client,
                model=_anthropic_model(),
                source_brand=source_brand,
                target_brand=target_brand,
                product_code=product_code,
                description=description,
                search_context=ctx,
                max_substitutes=max_substitutes,
            )
        except Exception as e:
            parsed = {"substitutes": [], "overall_notes": f"LLM error: {e}"}

        if verbose:
            ns = len(parsed.get("substitutes") or [])
            _progress(f"Done (legacy). {ns} substitute(s) suggested.", prefix=pf)

        return {
            "source_brand": source_brand,
            "product_code": product_code,
            "target_brand": target_brand,
            "description": description,
            "source_profile": None,
            "source_search_queries": [],
            "target_search_queries": [],
            "search_queries": executed,
            "llm_proposed_queries": llm_proposed,
            "search_errors": errors,
            "source_result_count": 0,
            "target_result_count": len(merged),
            "result_count": len(merged),
            "llm": parsed,
            "pipeline": "legacy",
        }

    # --- Spec extraction + target-brand search + attribute match (default) ---
    if verbose:
        _progress(
            f"Attribute pipeline: {source_brand} {product_code} → {target_brand}",
            prefix=pf,
        )
        _progress("Step 1/4: Tavily — source product…", prefix=pf)

    src_q = f'"{source_brand.strip()}" {product_code.strip()}'
    if description.strip():
        src_q += f" {description.strip()}"

    src_hits, src_err, src_exec = run_tavily_query_list(
        [src_q],
        tavily_api_key,
        top_search_results,
        search_depth=tavily_search_depth,
        include_domains=None,
        verbose=verbose,
        progress_prefix=pf,
    )
    src_ctx = _format_search_context(src_hits)

    if verbose:
        _progress(
            f"Step 2/4: Claude — extract specs ({len(src_hits)} page(s))…",
            prefix=pf,
        )

    try:
        profile = claude_extract_source_product_profile(
            client=client,
            model=_anthropic_model(),
            source_brand=source_brand,
            product_code=product_code,
            user_description=description,
            source_search_context=src_ctx,
        )
    except Exception:
        profile = {}
    if not isinstance(profile, dict):
        profile = {}

    doms = _tavily_include_domains()
    tqueries = build_target_queries_from_profile(
        target_brand,
        profile,
        product_code,
        description,
        max_target_queries,
    )
    if not tqueries:
        tb = target_brand.strip()
        tqueries = [f"{tb} LED downlight", f"{tb} products downlight"]

    if verbose:
        dom_note = f" [domains: {', '.join(doms)}]" if doms else ""
        _progress(
            f"Step 3/4: Tavily — {len(tqueries)} target-brand search(es){dom_note}…",
            prefix=pf,
        )

    tgt_hits, tgt_err, tgt_exec = run_tavily_query_list(
        tqueries,
        tavily_api_key,
        top_search_results,
        search_depth=tavily_search_depth,
        include_domains=doms,
        verbose=verbose,
        progress_prefix=pf,
    )
    tgt_ctx = _format_search_context(tgt_hits)

    if verbose:
        _progress(
            f"Step 4/4: Claude — match target SKU(s); {len(tgt_hits)} target page(s)…",
            prefix=pf,
        )

    try:
        parsed = claude_match_target_by_profile(
            client=client,
            model=_anthropic_model(),
            source_brand=source_brand,
            source_product_code=product_code,
            target_brand=target_brand,
            profile=profile,
            target_search_context=tgt_ctx,
            max_substitutes=max_substitutes,
        )
    except Exception as e:
        parsed = {"substitutes": [], "overall_notes": f"LLM error: {e}"}

    if verbose:
        ns = len(parsed.get("substitutes") or [])
        _progress(f"Done. {ns} substitute(s) suggested.", prefix=pf)

    return {
        "source_brand": source_brand,
        "product_code": product_code,
        "target_brand": target_brand,
        "description": description,
        "source_profile": profile,
        "source_search_queries": src_exec,
        "target_search_queries": tgt_exec,
        "search_queries": list(src_exec) + list(tgt_exec),
        "llm_proposed_queries": [],
        "search_errors": src_err + tgt_err,
        "source_result_count": len(src_hits),
        "target_result_count": len(tgt_hits),
        "result_count": len(tgt_hits),
        "llm": parsed,
        "pipeline": "attribute",
    }


def print_single_report(payload: dict[str, Any]) -> None:
    pipe = payload.get("pipeline", "legacy")
    print(f"\n--- Pipeline: {pipe} ---")
    prof = payload.get("source_profile")
    if prof and isinstance(prof, dict) and prof.get("summary"):
        print("\n--- Extracted source profile ---")
        print(f"  {prof.get('summary', '')}")
        w = prof.get("wattage_w")
        c = prof.get("cutout_or_hole_mm")
        if w is not None or c is not None:
            print(f"  wattage_w={w!s}  cutout_mm={c!s}  IP={prof.get('ip_rating')!s}")
        print(f"  colour_temp: {prof.get('colour_temperature')!s}")
        print(f"  dimmable: {prof.get('dimmable')!s}")

    print("\n--- Search (Tavily) ---")
    ssq = payload.get("source_search_queries") or []
    tsq = payload.get("target_search_queries") or []
    if ssq:
        print("  Source queries:")
        for q in ssq:
            print(f"    {q}")
    if tsq:
        print("  Target-brand queries:")
        for q in tsq:
            print(f"    {q}")
    if not ssq and not tsq:
        for q in payload["search_queries"]:
            print(f"  Query: {q}")
    extra = payload.get("llm_proposed_queries") or []
    if extra:
        print(
            f"  ({len(extra)} query/queries were suggested by the LLM from "
            "source-product snippets.)"
        )
    if payload["search_errors"]:
        print("  Errors:")
        for e in payload["search_errors"]:
            print(f"    {e}")
    src_n = payload.get("source_result_count")
    tgt_n = payload.get("target_result_count")
    if src_n is not None and tgt_n is not None and pipe == "attribute":
        print(f"  Source hits: {src_n}  Target hits (merged): {tgt_n}")
    else:
        print(f"  Unique result pages merged: {payload['result_count']}")

    print("\n--- Suggested matches ({}) ---".format(payload["target_brand"]))
    llm = payload["llm"]
    subs = llm.get("substitutes") or []
    if not subs:
        print("  (none)")
    for i, s in enumerate(subs, 1):
        print(f"\n  [{i}] {s.get('suggested_identifier', '')}")
        print(f"      Confidence: {s.get('confidence', '')}")
        print(f"      Reasoning: {s.get('reasoning', '')}")
        urls = s.get("cited_urls") or []
        if urls:
            print("      URLs:")
            for u in urls:
                print(f"        {u}")
    notes = llm.get("overall_notes", "")
    if notes:
        print(f"\n--- Notes ---\n{notes}")


def prompt_line(label: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    line = input(f"{label}{hint}: ").strip()
    return line if line else default


def resolve_single_args(
    source_brand: str,
    product_code: str,
    target_brand: str,
    description: str,
    no_prompt: bool,
) -> tuple[str, str, str, str]:
    if no_prompt:
        missing = [
            n
            for n, v in [
                ("--source-brand", source_brand),
                ("--product-code", product_code),
                ("--target-brand", target_brand),
            ]
            if not (v or "").strip()
        ]
        if missing:
            raise SystemExit(
                "Missing required arguments (use --no-prompt with all of): "
                "--source-brand, --product-code, --target-brand"
            )
        return source_brand.strip(), product_code.strip(), target_brand.strip(), description.strip()

    print("Product match lookup (Tavily + Claude). Leave blank to type.\n")
    sb = (source_brand or "").strip() or prompt_line("Source brand / manufacturer")
    pc = (product_code or "").strip() or prompt_line("Manufacturer product code")
    tb = (target_brand or "").strip() or prompt_line("Target brand (match to find)")
    desc = (description or "").strip()
    if not desc:
        desc = prompt_line("Product description (optional)", "")
    return sb.strip(), pc.strip(), tb.strip(), desc.strip()


def run_batch(
    input_path: Path,
    output_path: Path,
    source_supplier: str,
    target_supplier: str,
    search_query_template: str,
    top_search_results: int,
    max_substitutes: int,
    delay_seconds: float,
    limit_rows: int | None,
    tavily_api_key: str,
    api_key: str,
    llm_expand_search: bool = False,
    max_llm_search_queries: int = 3,
    tavily_search_depth: str | None = None,
    legacy_single_pass: bool = False,
    max_target_queries: int = 8,
    verbose: bool = True,
) -> None:
    suffix = input_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be .xlsx, .xls, or .csv")

    df = _normalise_columns(df)
    if limit_rows is not None:
        df = df.head(limit_rows)

    nrows = len(df)
    if verbose:
        pipe = "legacy single-pass" if legacy_single_pass else "attribute"
        _progress(f"Batch: {nrows} row(s), {pipe} pipeline → {output_path.name}")

    client = Anthropic(api_key=api_key)
    rows_out: list[dict[str, Any]] = []

    for num, (idx, row) in enumerate(df.iterrows(), 1):
        code = row["product_code"]
        desc = row["description"]
        sup = row["supplier"] or source_supplier

        seed = search_query_template.format(
            source_supplier=source_supplier,
            target_supplier=target_supplier,
            product_code=code,
            description=desc,
            supplier=sup,
        )

        if legacy_single_pass:
            legacy_queries = _dedupe_queries(
                [seed]
                + default_search_queries(
                    source_supplier, target_supplier, str(code), str(desc)
                )
            )
        else:
            legacy_queries = None

        if verbose:
            _progress(f"━━ Row {num}/{nrows}: {code} ━━")

        payload = run_single_lookup(
            source_brand=sup,
            product_code=str(code),
            target_brand=target_supplier,
            description=str(desc),
            tavily_api_key=tavily_api_key,
            anthropic_key=api_key,
            top_search_results=top_search_results,
            max_substitutes=max_substitutes,
            queries=legacy_queries,
            llm_expand_search=llm_expand_search,
            max_llm_search_queries=max_llm_search_queries,
            tavily_search_depth=tavily_search_depth,
            legacy_single_pass=legacy_single_pass,
            max_target_queries=max_target_queries,
            anthropic_client=client,
            verbose=verbose,
            progress_prefix=f"[{num}/{nrows}] " if verbose else "",
        )

        parsed = payload.get("llm") or {}
        raw_json = json.dumps(parsed, ensure_ascii=False)
        search_error = "; ".join(payload.get("search_errors") or [])
        all_q = " | ".join(payload.get("search_queries") or [])
        profile_json = ""
        prof = payload.get("source_profile")
        if isinstance(prof, dict) and prof:
            profile_json = json.dumps(prof, ensure_ascii=False)

        subs = parsed.get("substitutes") or []
        base = {
            "row_index": idx,
            "source_supplier": sup,
            "source_product_code": code,
            "source_description": desc,
            "pipeline": payload.get("pipeline", ""),
            "seed_query": seed,
            "all_queries": all_q,
            "source_profile_json": profile_json,
            "search_error": search_error,
            "overall_notes": parsed.get("overall_notes", ""),
            "llm_response_json": raw_json,
        }
        for j, sub in enumerate(subs):
            rows_out.append(
                {
                    **base,
                    "substitute_rank": j + 1,
                    "suggested_identifier": sub.get("suggested_identifier", ""),
                    "confidence": sub.get("confidence", ""),
                    "reasoning": sub.get("reasoning", ""),
                    "cited_urls": "; ".join(sub.get("cited_urls") or []),
                    "llm_response_json": raw_json if j == 0 else "",
                }
            )
        if not subs:
            rows_out.append(
                {
                    **base,
                    "substitute_rank": "",
                    "suggested_identifier": "",
                    "confidence": "",
                    "reasoning": "",
                    "cited_urls": "",
                }
            )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    out_df = pd.DataFrame(rows_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False, engine="openpyxl")
    if verbose:
        _progress(f"Batch finished. Wrote {len(out_df)} row(s) → {output_path}")
    else:
        print(f"Wrote {len(out_df)} rows to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Find target-brand product matches using Tavily + Anthropic. "
            "Default: extract specs from source product web results, search the target brand "
            "by attributes, then match a target SKU from evidence. "
            "Use --legacy-single-pass for the older single merged search."
        )
    )
    p.add_argument(
        "--source-brand",
        "--source-supplier",
        dest="source_brand",
        default="",
        help="Brand/manufacturer of the existing product",
    )
    p.add_argument(
        "--product-code",
        "-c",
        default="",
        help="Manufacturer product / SKU code",
    )
    p.add_argument(
        "--target-brand",
        "--target-supplier",
        dest="target_brand",
        default="",
        help="Brand to find a matching product from",
    )
    p.add_argument(
        "--description",
        "-d",
        default="",
        help="Optional product description for better search",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt; exit if required flags are missing",
    )
    p.add_argument(
        "--json-out",
        action="store_true",
        help="Print full result as JSON to stdout (single-product mode)",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Write full result to this JSON file (single-product mode)",
    )

    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Batch mode: Excel or CSV with product_code (+ description columns)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("product_substitution_llm.xlsx"),
        help="Batch mode: output Excel path",
    )
    p.add_argument(
        "--query-template",
        default=(
            "{target_supplier} equivalent OR substitute OR alternative "
            "{source_supplier} {product_code} {description}"
        ),
        help="Batch mode: extra search query template (format keys as shown)",
    )
    p.add_argument("--top-search-results", type=int, default=8)
    p.add_argument("--max-substitutes", type=int, default=5)
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Batch only: seconds between rows",
    )
    p.add_argument("--limit", type=int, default=None, help="Batch only: first N rows")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Load API keys from this .env file (overrides .env next to script / in CWD)",
    )
    p.add_argument(
        "--llm-expand-search",
        action="store_true",
        help=(
            "Legacy mode only: extra search queries from LLM after source snippets "
            "(uses more API calls). Ignored for the default attribute pipeline."
        ),
    )
    p.add_argument(
        "--max-llm-search-queries",
        type=int,
        default=3,
        help="Max extra search queries proposed by the LLM when --llm-expand-search is set",
    )
    p.add_argument(
        "--tavily-depth",
        default=None,
        choices=["basic", "fast", "ultra-fast", "advanced"],
        help="Tavily search_depth (default: env TAVILY_SEARCH_DEPTH or basic). "
        "advanced uses 2 API credits per search.",
    )
    p.add_argument(
        "--legacy-single-pass",
        action="store_true",
        help="Old behaviour: one merged Tavily pass + single match (no spec extraction step).",
    )
    p.add_argument(
        "--max-target-queries",
        type=int,
        default=8,
        help="Attribute pipeline: max Tavily queries for the target brand (default 8).",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="No progress on stderr (useful when piping stdout, e.g. --json-out).",
    )
    args = p.parse_args()

    load_env_files(extra_env_path=args.env_file)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print(
            "Missing ANTHROPIC_API_KEY — set it in the environment or in a .env file "
            "(see env.example). Optional: pip install python-dotenv",
            file=sys.stderr,
        )
        sys.exit(1)
    if not tavily_key:
        print(
            "Missing TAVILY_API_KEY — get a key at https://tavily.com — set in environment or .env "
            "(see env.example). Optional: pip install python-dotenv",
            file=sys.stderr,
        )
        sys.exit(1)

    tavily_depth = args.tavily_depth
    show_progress = not args.quiet

    if args.input is not None:
        if not args.input.exists():
            print(f"Input not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        run_batch(
            input_path=args.input,
            output_path=args.output,
            source_supplier=args.source_brand or "SAL",
            target_supplier=args.target_brand or "Haneco",
            search_query_template=args.query_template,
            top_search_results=args.top_search_results,
            max_substitutes=args.max_substitutes,
            delay_seconds=args.delay,
            limit_rows=args.limit,
            tavily_api_key=tavily_key,
            api_key=api_key,
            llm_expand_search=args.llm_expand_search,
            max_llm_search_queries=args.max_llm_search_queries,
            tavily_search_depth=tavily_depth,
            legacy_single_pass=args.legacy_single_pass,
            max_target_queries=args.max_target_queries,
            verbose=show_progress,
        )
        return

    sb, pc, tb, desc = resolve_single_args(
        args.source_brand,
        args.product_code,
        args.target_brand,
        args.description,
        args.no_prompt,
    )

    payload = run_single_lookup(
        source_brand=sb,
        product_code=pc,
        target_brand=tb,
        description=desc,
        tavily_api_key=tavily_key,
        anthropic_key=api_key,
        top_search_results=args.top_search_results,
        max_substitutes=args.max_substitutes,
        llm_expand_search=args.llm_expand_search,
        max_llm_search_queries=args.max_llm_search_queries,
        tavily_search_depth=tavily_depth,
        legacy_single_pass=args.legacy_single_pass,
        max_target_queries=args.max_target_queries,
        verbose=show_progress,
    )

    if args.json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_single_report(payload)

    if args.save_json:
        args.save_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved JSON to {args.save_json}")


if __name__ == "__main__":
    main()
