"""
Print raw Tavily hits for the same query strategy as replacement_llm.py.

Usage:
  python debug_tavily.py
  python debug_tavily.py --source-brand SAL --code SYAS9065TCWH --target-brand Haneco
  python debug_tavily.py -d "9W tri colour downlight 92mm"

Loads TAVILY_API_KEY from .env (same rules as replacement_llm).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Debug Tavily results per query")
    p.add_argument("--source-brand", default="SAL")
    p.add_argument("--code", default="SYAS9065TCWH", help="Product / SKU code")
    p.add_argument("--target-brand", default="Haneco")
    p.add_argument("-d", "--description", default="")
    p.add_argument("--max-results", type=int, default=8)
    p.add_argument(
        "--tavily-depth",
        default=None,
        choices=["basic", "fast", "ultra-fast", "advanced"],
    )
    p.add_argument("--needle", default="viva110", help="Substring to flag in hits (lower case)")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replacement_llm as m

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore
    if load_dotenv:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        load_dotenv(Path.cwd() / ".env", override=True)

    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        print("Set TAVILY_API_KEY or add it to .env", file=sys.stderr)
        sys.exit(1)

    sb = args.source_brand.strip()
    tb = (args.target_brand or "").strip()
    code = args.code.strip()
    desc = (args.description or "").strip()
    queries = [f'"{sb}" {code}' + (f" {desc}" if desc else "")]
    if tb:
        queries.append(
            f"{tb} (equivalent OR substitute OR cross reference) {sb} {code}"
        )
    else:
        queries.append(f"(equivalent OR substitute) {sb} {code} LED")
    needle = (args.needle or "").lower()

    # UTF-8 on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"Queries: {len(queries)}\n")
    for q in queries:
        print("=" * 72)
        print("QUERY:", q)
        print("-" * 72)
        try:
            rows = m.tavily_search(
                q,
                key,
                num_results=args.max_results,
                search_depth=args.tavily_depth,
            )
        except Exception as e:
            print("ERROR:", e)
            continue
        print("hits:", len(rows))
        for j, r in enumerate(rows, 1):
            url = r.get("link") or ""
            snip = r.get("snippet") or ""
            blob = (url + snip).lower()
            print(f" [{j}] {r.get('title', '')[:100]}")
            print(f"     {url[:120]}")
            print(f"     needle {needle!r}: {needle in blob}")
            preview = snip.replace("\n", " ")[:380]
            print(f"     {preview}{'…' if len(snip) > 380 else ''}")
        print()


if __name__ == "__main__":
    main()
