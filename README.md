# ERP Lookup Script

This project helps you avoid manual one-by-one web searching.

It reads coop account names from CSV or Excel, searches the web for each account, and estimates which ERP system is most likely in use (from three ERP targets by default).

## What it does

- Input: CSV or XLSX with an account-name column
- Search: SerpApi (Google results) per account by default
- Scoring: Looks for ERP mentions in search result titles/snippets
- Output:
  - `erp_summary.csv` (quick review table)
  - `erp_summary.json` (full evidence details)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Create a local `.env` file for keys:

```bash
SERPAPI_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
```

Use CSV with SerpApi (default engine):

```bash
python erp_lookup.py sample_accounts.csv --account-column account
```

Use Excel:

```bash
python erp_lookup.py your_accounts.xlsx --account-column "Account Name"
```

Use DuckDuckGo fallback:

```bash
python erp_lookup.py sample_accounts.csv --account-column account --engine ddg
```

Use LLM scoring on top of search results:

```bash
python erp_lookup.py sample_accounts.csv --account-column account --scorer llm
```

Notes:

- `.env` is auto-loaded by `erp_lookup.py`.
- CLI flags still override `.env` values (`--serpapi-api-key`, `--openai-api-key`).
- LLM scoring now uses OpenAI **Responses API** (`/v1/responses`), which supports newer models.

## Default ERP systems tracked

- AGRIS
- Agvance
- Merchant Ag

## Override ERP systems

```bash
python erp_lookup.py sample_accounts.csv --erp SAP Infor Epicor
```

## Suggested workflow

1. Run the script against your full account list.
2. Sort by `confidence` in `erp_summary.csv`.
3. Manually verify low-confidence rows first.
4. Keep the JSON file as your audit trail.

## Notes

- Direct Google scraping is brittle and can violate terms. This script uses `duckduckgo-search` as a practical starting point.
- Confidence is heuristic and intended to prioritize review, not guarantee correctness.

## Link Name Scraper

Use this when you need to pull names from links on a webpage.

Basic run:

```bash
python link_name_scraper.py "https://example.com/members"
```

Only keep links that contain a pattern and stay on the same domain:

```bash
python link_name_scraper.py "https://example.com/members" --contains /member/ --same-domain
```

If names are in URL slugs instead of link text:

```bash
python link_name_scraper.py "https://example.com/members" --name-source href-last-segment
```

Output files:

- `link_names.csv`
- `link_names.json`
- `link_names_accounts.csv` (parsed from `City: Co-op Name` text)

For pages like the Kansas co-op list, use `*_accounts.csv` with ERP lookup. It includes:

- `account`: combined value (`City Co-op Name`) for better search context
- `city`
- `coop_name`
- `source_text`
- `href`
