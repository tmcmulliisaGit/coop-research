# ERP Lookup Script

This project helps you avoid manual one-by-one web searching.

It reads coop account names from CSV or Excel, searches the web for each account, and estimates which ERP system is most likely in use (from four ERP targets by default).

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

## Command-line arguments (`erp_lookup.py`)

Positional:

- `input_file` (required): Path to input `.csv`, `.xlsx`, `.xlsm`, or `.xls`.

Options:

- `--account-column` (default: `account`): Column name containing account names.
- `--erp` (default: built-in list): One or more ERP names to track.  
  Example: `--erp AGRIS Agvance "Merchant Ag" AgTrax`
- `--queries-per-account` (default: `6`): Number of web results to inspect per account.
- `--sleep-seconds` (default: `1.5`): Delay between account searches.
- `--output-prefix` (default: `erp_summary`): Output prefix; writes `<prefix>.csv` and `<prefix>.json`.
- `--query-template` (default includes `{account}` and ERP terms): Search query template. Must contain `{account}`.
- `--engine` (default: `serpapi`, choices: `serpapi`, `ddg`): Search backend.
- `--serpapi-api-key` (default: unset): SerpApi key. If omitted, reads `SERPAPI_API_KEY` from environment or `.env`.
- `--scorer` (default: `heuristic`, choices: `heuristic`, `llm`): ERP scoring method.
- `--openai-api-key` (default: unset): OpenAI key for `--scorer llm`. If omitted, reads `OPENAI_API_KEY` from environment or `.env`.
- `--llm-model` (default: `gpt-4.1-mini`): Model to use with `--scorer llm`.
- `--openai-base-url` (default: `https://api.openai.com/v1`): Base URL for OpenAI-compatible API.

Requirements by mode:

- `--engine serpapi` requires a SerpApi key (`--serpapi-api-key` or `SERPAPI_API_KEY`).
- `--engine ddg` does not require SerpApi.
- `--scorer llm` requires an OpenAI key (`--openai-api-key` or `OPENAI_API_KEY`).

Notes:

- `.env` is auto-loaded by `erp_lookup.py`.
- CLI flags still override `.env` values (`--serpapi-api-key`, `--openai-api-key`).
- LLM scoring now uses OpenAI **Responses API** (`/v1/responses`), which supports newer models.

## Default ERP systems tracked

- AGRIS
- Agvance
- Merchant Ag
- AgTrax

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
