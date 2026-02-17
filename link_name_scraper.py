#!/usr/bin/env python3
"""
Extract names from links on a webpage.

Examples:
  python link_name_scraper.py "https://example.com/members"
  python link_name_scraper.py "https://example.com/members" --contains /member/ --name-source href-last-segment
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0 (compatible; CoopNameScraper/1.0)"


@dataclass
class LinkRecord:
    name: str
    text: str
    href: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract names from links on a webpage.")
    parser.add_argument("url", help="Page URL to scrape.")
    parser.add_argument(
        "--selector",
        default="a",
        help="CSS selector for links (default: a).",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Optional substring that href must contain (example: /member/).",
    )
    parser.add_argument(
        "--same-domain",
        action="store_true",
        help="Keep only links on the same domain as the page URL.",
    )
    parser.add_argument(
        "--name-source",
        choices=["text", "href-last-segment", "text-or-href"],
        default="text-or-href",
        help="How to derive name values (default: text-or-href).",
    )
    parser.add_argument(
        "--output-prefix",
        default="link_names",
        help="Prefix for output files (default: link_names).",
    )
    return parser.parse_args()


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slug_to_name(href: str) -> str:
    path = urlparse(href).path.strip("/")
    if not path:
        return ""
    segment = path.split("/")[-1]
    segment = re.sub(r"[-_]+", " ", segment)
    return normalize_spaces(segment)


def derive_name(text: str, href: str, mode: str) -> str:
    clean_text = normalize_spaces(text)
    from_href = slug_to_name(href)

    if mode == "text":
        return clean_text
    if mode == "href-last-segment":
        return from_href
    # text-or-href
    return clean_text or from_href


def extract_links(
    page_url: str,
    html: str,
    selector: str,
    contains: str | None,
    same_domain: bool,
    name_source: str,
) -> list[LinkRecord]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: beautifulsoup4. Install with: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    source_domain = urlparse(page_url).netloc.lower()

    out: list[LinkRecord] = []
    seen: set[tuple[str, str]] = set()

    for el in soup.select(selector):
        if el.name != "a":
            continue

        raw_href = (el.get("href") or "").strip()
        if not raw_href:
            continue

        href = urljoin(page_url, raw_href)
        text = normalize_spaces(el.get_text(" ", strip=True))

        if contains and contains not in href:
            continue

        if same_domain and urlparse(href).netloc.lower() != source_domain:
            continue

        name = derive_name(text, href, name_source)
        if not name:
            continue

        key = (name.lower(), href)
        if key in seen:
            continue
        seen.add(key)

        out.append(LinkRecord(name=name, text=text, href=href))

    return out


def write_outputs(records: list[LinkRecord], output_prefix: str) -> tuple[Path, Path]:
    csv_path = Path(f"{output_prefix}.csv")
    json_path = Path(f"{output_prefix}.json")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "text", "href"])
        writer.writeheader()
        for row in records:
            writer.writerow({"name": row.name, "text": row.text, "href": row.href})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([row.__dict__ for row in records], f, indent=2)

    return csv_path, json_path


def main() -> int:
    args = parse_args()

    try:
        html = fetch_html(args.url)
        records = extract_links(
            page_url=args.url,
            html=html,
            selector=args.selector,
            contains=args.contains,
            same_domain=args.same_domain,
            name_source=args.name_source,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    csv_path, json_path = write_outputs(records, args.output_prefix)
    print(f"Extracted {len(records)} names.")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
