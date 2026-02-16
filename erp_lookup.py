#!/usr/bin/env python3
"""
ERP lookup helper for coop account lists.

Reads account names from CSV/XLSX, searches the web for each account,
then scores mentions of target ERP systems in search snippets.
Outputs a review-friendly summary in CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_ERP_ALIASES = {
    "AGRIS": ["agris", "agris erp", "agris business", "agri business software"],
    "Agvance": ["agvance", "agvance software", "agvance agri-business"],
    "Merchant Ag": ["merchant ag", "merchantag", "merchant agriculture"],
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


@dataclass
class AccountResult:
    account: str
    top_matches: list[tuple[str, int]]
    best_guess: str
    confidence: float
    evidence: list[dict]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find likely ERP systems for a list of coop accounts."
    )
    parser.add_argument("input_file", help="Path to CSV or XLSX file.")
    parser.add_argument(
        "--account-column",
        default="account",
        help="Column containing account names (default: account).",
    )
    parser.add_argument(
        "--erp",
        nargs="+",
        default=None,
        help=(
            "Optional custom ERP names (example: --erp AGRIS Agvance \"Merchant Ag\"). "
            "If omitted, defaults to AGRIS, Agvance, Merchant Ag."
        ),
    )
    parser.add_argument(
        "--queries-per-account",
        type=int,
        default=6,
        help="Number of web results to inspect per account (default: 6).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Delay between account searches to reduce throttling risk (default: 1.5).",
    )
    parser.add_argument(
        "--output-prefix",
        default="erp_summary",
        help="Prefix for output files (default: erp_summary).",
    )
    parser.add_argument(
        "--query-template",
        default='"{account}" (ERP OR "enterprise resource planning" OR AGRIS OR Agvance OR "Merchant Ag")',
        help="Search query template. Must include {account}.",
    )
    parser.add_argument(
        "--engine",
        choices=["serpapi", "ddg"],
        default="serpapi",
        help="Search engine backend (default: serpapi).",
    )
    parser.add_argument(
        "--serpapi-api-key",
        default=None,
        help="SerpApi API key. If omitted, reads SERPAPI_API_KEY environment variable.",
    )
    return parser.parse_args()


def load_accounts(input_file: Path, account_column: str) -> list[str]:
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        return load_accounts_csv(input_file, account_column)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return load_accounts_excel(input_file, account_column)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or XLSX.")


def load_accounts_csv(path: Path, account_column: str) -> list[str]:
    accounts: list[str] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        field_map = {normalize_text(name): name for name in reader.fieldnames}
        key = normalize_text(account_column)
        if key not in field_map:
            raise ValueError(
                f"Column '{account_column}' not found. Available columns: {reader.fieldnames}"
            )

        source_col = field_map[key]
        for row in reader:
            value = (row.get(source_col) or "").strip()
            if value:
                accounts.append(value)

    return dedupe_keep_order(accounts)


def load_accounts_excel(path: Path, account_column: str) -> list[str]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading Excel requires pandas and openpyxl. "
            "Install with: pip install pandas openpyxl"
        ) from exc

    df = pd.read_excel(path)
    normalized = {normalize_text(col): col for col in df.columns.astype(str)}
    key = normalize_text(account_column)
    if key not in normalized:
        raise ValueError(
            f"Column '{account_column}' not found. Available columns: {list(df.columns)}"
        )

    col = normalized[key]
    values = [str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()]
    return dedupe_keep_order(values)


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def build_erp_aliases(custom_erps: list[str] | None) -> dict[str, list[str]]:
    if not custom_erps:
        return DEFAULT_ERP_ALIASES

    aliases: dict[str, list[str]] = {}
    for erp_name in custom_erps:
        norm = erp_name.strip()
        if not norm:
            continue
        name_l = norm.lower()
        compact = re.sub(r"\s+", "", name_l)
        aliases[norm] = [name_l, compact]
    return aliases


def search_web_ddg(query: str, max_results: int) -> list[SearchHit]:
    """Search with DuckDuckGo via duckduckgo_search package."""
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: duckduckgo-search. Install with: pip install duckduckgo-search"
        ) from exc

    hits: list[SearchHit] = []
    with DDGS() as ddgs:
        for row in ddgs.text(query, max_results=max_results):
            hits.append(
                SearchHit(
                    title=row.get("title", ""),
                    url=row.get("href", ""),
                    snippet=row.get("body", ""),
                )
            )
    return hits


def search_web_serpapi(query: str, max_results: int, api_key: str) -> list[SearchHit]:
    """Search Google results through SerpApi JSON endpoint."""
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": api_key,
    }
    url = f"https://serpapi.com/search.json?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "error" in payload:
        raise RuntimeError(f"SerpApi error: {payload['error']}")

    hits: list[SearchHit] = []
    for row in payload.get("organic_results", []):
        hits.append(
            SearchHit(
                title=row.get("title", ""),
                url=row.get("link", ""),
                snippet=row.get("snippet", ""),
            )
        )
    return hits


def search_web(
    query: str,
    max_results: int,
    engine: str,
    serpapi_api_key: str | None,
) -> list[SearchHit]:
    if engine == "ddg":
        return search_web_ddg(query, max_results)
    if engine == "serpapi":
        if not serpapi_api_key:
            raise ValueError(
                "SerpApi key is required for engine=serpapi. "
                "Use --serpapi-api-key or set SERPAPI_API_KEY."
            )
        return search_web_serpapi(query, max_results, serpapi_api_key)
    raise ValueError(f"Unsupported engine: {engine}")


def score_hits(
    account: str,
    hits: list[SearchHit],
    erp_aliases: dict[str, list[str]],
) -> AccountResult:
    score_map = {erp: 0 for erp in erp_aliases}
    evidence_map: dict[str, list[dict]] = {erp: [] for erp in erp_aliases}

    for hit in hits:
        text = normalize_text(f"{hit.title} {hit.snippet}")
        for erp_name, aliases in erp_aliases.items():
            match_count = sum(1 for alias in aliases if alias and alias in text)
            if match_count > 0:
                score_map[erp_name] += match_count
                if len(evidence_map[erp_name]) < 3:
                    evidence_map[erp_name].append(
                        {
                            "title": hit.title,
                            "url": hit.url,
                            "snippet": hit.snippet,
                            "matches": match_count,
                        }
                    )

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1] if ranked else 0
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    best_guess = ranked[0][0] if ranked and top_score > 0 else "Unknown"

    # Basic confidence heuristic from score magnitude + separation from runner-up.
    if top_score == 0:
        confidence = 0.0
    else:
        confidence = min(1.0, (top_score / max(1.0, len(hits))) * 0.6 + (top_score - second_score) * 0.15)

    evidence = evidence_map.get(best_guess, []) if best_guess != "Unknown" else []

    return AccountResult(
        account=account,
        top_matches=ranked,
        best_guess=best_guess,
        confidence=round(confidence, 3),
        evidence=evidence,
    )


def write_outputs(results: list[AccountResult], output_prefix: str) -> tuple[Path, Path]:
    csv_path = Path(f"{output_prefix}.csv")
    json_path = Path(f"{output_prefix}.json")

    all_erps = sorted({name for result in results for name, _ in result.top_matches})

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["account", "best_guess", "confidence", *all_erps, "evidence_urls"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "account": r.account,
                "best_guess": r.best_guess,
                "confidence": r.confidence,
                "evidence_urls": " | ".join(item.get("url", "") for item in r.evidence),
            }
            for erp, score in r.top_matches:
                row[erp] = score
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    return csv_path, json_path


def main() -> int:
    args = parse_args()

    if "{account}" not in args.query_template:
        print("Error: --query-template must contain {account} placeholder.", file=sys.stderr)
        return 2

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        accounts = load_accounts(input_path, args.account_column)
    except Exception as exc:
        print(f"Failed to load accounts: {exc}", file=sys.stderr)
        return 2

    if not accounts:
        print("No accounts found in input file.", file=sys.stderr)
        return 2

    serpapi_api_key = args.serpapi_api_key or os.getenv("SERPAPI_API_KEY")
    if args.engine == "serpapi" and not serpapi_api_key:
        print(
            "Error: SerpApi key missing. Set SERPAPI_API_KEY or pass --serpapi-api-key.",
            file=sys.stderr,
        )
        return 2

    erp_aliases = build_erp_aliases(args.erp)
    print(f"Loaded {len(accounts)} unique accounts.")
    print(f"Tracking ERPs: {', '.join(erp_aliases.keys())}")
    print(f"Search engine: {args.engine}")

    results: list[AccountResult] = []
    for i, account in enumerate(accounts, start=1):
        query = args.query_template.format(account=account)
        print(f"[{i}/{len(accounts)}] Searching: {account}")

        try:
            hits = search_web(
                query,
                args.queries_per_account,
                args.engine,
                serpapi_api_key,
            )
        except Exception as exc:
            print(f"  Search failed for '{account}': {exc}", file=sys.stderr)
            results.append(
                AccountResult(
                    account=account,
                    top_matches=[(name, 0) for name in erp_aliases],
                    best_guess="Unknown",
                    confidence=0.0,
                    evidence=[],
                )
            )
            continue

        result = score_hits(account, hits, erp_aliases)
        print(f"  -> Best guess: {result.best_guess} (confidence={result.confidence})")
        results.append(result)

        if i < len(accounts) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    csv_path, json_path = write_outputs(results, args.output_prefix)
    print(f"Done. Wrote summary: {csv_path}")
    print(f"Done. Wrote details: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
