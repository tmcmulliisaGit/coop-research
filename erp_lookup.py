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
from urllib.request import Request, urlopen


DEFAULT_ERP_ALIASES = {
    "AGRIS": ["agris", "agris erp", "agris business", "agri business software"],
    "Agvance": ["agvance", "agvance software", "agvance agri-business"],
    "Merchant Ag": ["merchant ag", "merchantag", "merchant agriculture"],
    "AgTrax": ["agtrax", "ag trax", "agtrax erp", "agtrax software"],
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


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        # Keep process env as highest precedence.
        os.environ.setdefault(key, value)


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
            "If omitted, defaults to AGRIS, Agvance, Merchant Ag, AgTrax."
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
        default='"{account}" (ERP OR "enterprise resource planning" OR AGRIS OR Agvance OR "Merchant Ag" OR AgTrax)',
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
        help="SerpApi API key. If omitted, reads SERPAPI_API_KEY from env/.env.",
    )
    parser.add_argument(
        "--scorer",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Scoring backend for ERP prediction (default: heuristic).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for --scorer llm. If omitted, reads OPENAI_API_KEY from env/.env.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="Model used for --scorer llm (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible API (default: https://api.openai.com/v1).",
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


def parse_json_object(text: str) -> dict:
    """Extract first JSON object from a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Model response did not contain a valid JSON object.")


def extract_responses_output_text(raw: dict) -> str:
    """Extract plain text from a /v1/responses payload."""
    output_text = raw.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in raw.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if content_type in {"output_text", "text"}:
                text = content.get("text", "")
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def llm_score_hits(
    account: str,
    hits: list[SearchHit],
    erp_aliases: dict[str, list[str]],
    openai_api_key: str,
    llm_model: str,
    openai_base_url: str,
) -> AccountResult:
    erp_names = list(erp_aliases.keys())
    search_payload = [
        {
            "id": idx + 1,
            "title": hit.title,
            "url": hit.url,
            "snippet": hit.snippet,
        }
        for idx, hit in enumerate(hits)
    ]

    system_prompt = (
        "You are an ERP classification assistant. "
        "Given search results for one company and a fixed ERP candidate list, "
        "estimate which ERP is most likely in use. "
        "Use only the provided results. "
        "Return strict JSON only."
    )

    user_payload = {
        "account": account,
        "erp_candidates": erp_names,
        "search_results": search_payload,
        "instructions": {
            "output_schema": {
                "best_guess": "string, one of ERP candidates or Unknown",
                "confidence": "number 0.0 to 1.0",
                "scores": "object with each ERP candidate as key and integer score 0-100",
                "evidence_ids": "array of result ids (1-based) that support the best guess",
            }
        },
    }

    request_body = {
        "model": llm_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "text": {"format": {"type": "json_object"}},
    }

    endpoint = openai_base_url.rstrip("/") + "/responses"
    req = Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=45) as response:
        raw = json.loads(response.read().decode("utf-8"))

    content = extract_responses_output_text(raw)
    if not content:
        raise RuntimeError("LLM response text was empty.")

    parsed = parse_json_object(content)
    raw_best_guess = str(parsed.get("best_guess", "Unknown")).strip()
    best_guess = raw_best_guess if raw_best_guess in erp_names else "Unknown"

    score_map: dict[str, int] = {}
    raw_scores = parsed.get("scores", {})
    for erp in erp_names:
        score_val = raw_scores.get(erp, 0) if isinstance(raw_scores, dict) else 0
        try:
            score_map[erp] = max(0, int(round(float(score_val))))
        except Exception:
            score_map[erp] = 0

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    if ranked and ranked[0][1] == 0:
        best_guess = "Unknown"

    raw_confidence = parsed.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except Exception:
        confidence = 0.0

    evidence: list[dict] = []
    raw_evidence_ids = parsed.get("evidence_ids", [])
    if isinstance(raw_evidence_ids, list):
        for evidence_id in raw_evidence_ids[:3]:
            if not isinstance(evidence_id, int):
                continue
            idx = evidence_id - 1
            if 0 <= idx < len(hits):
                hit = hits[idx]
                evidence.append(
                    {
                        "title": hit.title,
                        "url": hit.url,
                        "snippet": hit.snippet,
                        "matches": "llm_evidence",
                    }
                )

    return AccountResult(
        account=account,
        top_matches=ranked,
        best_guess=best_guess,
        confidence=round(confidence, 3),
        evidence=evidence,
    )


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
    load_dotenv()
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
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if args.scorer == "llm" and not openai_api_key:
        print(
            "Error: OpenAI key missing. Set OPENAI_API_KEY or pass --openai-api-key for --scorer llm.",
            file=sys.stderr,
        )
        return 2

    erp_aliases = build_erp_aliases(args.erp)
    print(f"Loaded {len(accounts)} unique accounts.")
    print(f"Tracking ERPs: {', '.join(erp_aliases.keys())}")
    print(f"Search engine: {args.engine}")
    print(f"Scorer: {args.scorer}")

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

        try:
            if args.scorer == "llm":
                result = llm_score_hits(
                    account=account,
                    hits=hits,
                    erp_aliases=erp_aliases,
                    openai_api_key=openai_api_key or "",
                    llm_model=args.llm_model,
                    openai_base_url=args.openai_base_url,
                )
            else:
                result = score_hits(account, hits, erp_aliases)
        except Exception as exc:
            print(
                f"  Scoring failed for '{account}' with scorer={args.scorer}: {exc}. Falling back to heuristic.",
                file=sys.stderr,
            )
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
