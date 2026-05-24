# Weekly Equity Long/Short Strategy — Implementation Plan v2

## Goal

Build a weekly cadence equity long/short signal system. Each Sunday evening, produce a ranked list of 5 longs and 5 shorts from a 100-stock universe, with theses and risk flags. Universe refreshed quarterly. Manual execution on Robinhood Gold (shorting enabled); agent produces signals, human approves trades.

Capital plan: **paper trade first**, graduate to real capital only after backtest + paper trading both validate edge.

## Tooling

- Python 3.9 (system default on this host; no newer interpreter available) with `venv` and `pip` (no poetry/uv/conda). Use `from __future__ import annotations` in every module so we can write modern type hints (`list[str]`, `dict[str, X]`, `X | None`) without runtime errors.
- SQLite for storage. Pydantic for schemas. `pandas` for tabular work.
- LLM: **Claude Code in headless mode**, billed against the user's Pro/Max subscription — *not* the Anthropic API. We do not hold an API key. All calls go through `claude --bare -p "<prompt>" --output-format json --model <id>` from a Python subprocess; `weekly_strategy/llm/client.py` is the only place that knows this. Use the latest model family:
  - `claude-haiku-4-5-20251001` for high-volume batch classification (news sentiment).
  - `claude-sonnet-4-6` for the weekly thesis writer and conviction check.
  - `claude-opus-4-7` reserved for monthly meta-review / hard cases only.
- **Subscription cost envelope:** starting 2026-06-15, Anthropic provisions a dedicated monthly credit pool for Agent SDK + headless usage, separate from interactive limits — Pro $20/mo, Max-5x $100/mo, Max-20x $200/mo. Projected spend: Stage 1 ≈ $0.60/mo (well under Pro), Stage 3 ≈ $20/mo (Pro is tight, Max-5x is comfortable). Until 2026-06-15, headless draws from the interactive plan's quota.
- **Auth invariant:** the subprocess wrapper must strip `ANTHROPIC_API_KEY` from the child env so subscription OAuth wins. If the key is set in the shell, billing silently switches to API.

## Known risk in build order

The plan defers backtesting to Stage 5. The user (cunwei.fan@tophash.io) acknowledges the risk and has explicitly chosen to prioritize signal-generation infrastructure first. **Track this as a known risk:** if Stage 2 produces signals that don't lend themselves to even a crude historical sanity check, escalate before continuing to Stage 3.

## Design principles

- Code does math, ranking, and deterministic ops. LLM does language understanding and sanity checks.
- Slow signals (quality, valuation) live in persistent dossiers. Fast signals (news, sentiment) computed weekly.
- Build vertically before horizontally: get one stock end-to-end before scaling to 100.
- Free data sources for v1. Upgrade to paid only when a specific limit is hit.
- Backtest before live. Paper-trade before real capital.
- Edge comes from patience, selection, and behavioral discipline — not speed.

## Two-stage build strategy

This plan has two major stages with clear acceptance criteria for each:

**Stage 1 — Single equity end-to-end, run concurrently on 3 names.** Build the entire pipeline working concurrently for **three** tickers from different sectors: **AAPL (Tech), JPM (Financials), XOM (Energy)**. Running three from the start catches sector/industry edge cases (financials use different XBRL tags, energy has unusual capex/FCF patterns, tech has stock-based comp dynamics) before they get baked into AAPL-specific code. No macro overlay, no cross-sectional ranking — just three independent single-stock weekly analyses. Acceptance: a clean weekly report for each of the 3 tickers that you'd actually find useful to read.

**Stage 2 — Add macro and sector layers.** Layer on the universe-level context: macro regime detection, sector rotation signals, cross-asset risk. Apply these as overlays to the per-stock scoring. Acceptance: AAPL's weekly signal now incorporates Fed posture, rate regime, and tech-sector momentum.

Subsequent stages (universe expansion, ranking, selection, backtest, paper trade, live) follow after Stage 2 is solid.

## Data sources — free stack

| Data type | Source | Library | Notes |
|---|---|---|---|
| Price/volume | yfinance | `yfinance` | Cache aggressively in SQLite |
| Fundamentals | SEC EDGAR companyfacts API | `requests` | The source of truth |
| Filings (10-K, 10-Q, 8-K) | SEC EDGAR | `requests` | Free, official |
| Earnings press releases | Company IR pages + Yahoo Finance | scraping | Manual URL map per ticker |
| News headlines | Google News RSS | `feedparser` | Per-ticker queries, dedupe required |
| Reddit sentiment | Reddit public JSON endpoints | `requests` | Unauthenticated, polite User-Agent; PRAW dropped because Reddit no longer reliably issues API keys to new individual developers |
| Macro data | FRED | `fredapi` | Solved problem, all needed series |
| Insider activity | EDGAR Form 4 + openinsider | scraping | Cluster detection |
| Institutional holdings | EDGAR 13F (quarterly) | `requests` | Lagged 45 days, treat as prior |
| Sector ETF data | yfinance | `yfinance` | XLF, XLK, XLE, etc. |
| Hacker News (tech only) | HN Algolia API | `requests` | Tech ticker bonus signal |

API keys / identification needed (all free):
- FRED API key (register at fred.stlouisfed.org)
- SEC EDGAR requires User-Agent header with contact email (no key, just identification)
- Anthropic API key for Claude LLM calls
- Reddit: **no key needed.** We use Reddit's public JSON endpoints with a polite `User-Agent` header. (Reddit's developer API key issuance has been unreliable for individual developers since 2023; the public JSON endpoints remain accessible.)

---

## STAGE 1: SINGLE EQUITY END-TO-END

Goal: get AAPL flowing through the entire pipeline. No multi-stock ranking, no macro. Just a deep weekly report for one ticker.

### Step 1.1: Project scaffolding

Create the directory structure:

```
weekly_strategy/
├── data/
│   ├── __init__.py
│   ├── fetchers.py          # All API wrappers
│   ├── storage.py           # SQLite layer
│   └── schemas.py           # Pydantic models
├── dossier/
│   ├── __init__.py
│   ├── builder.py           # Build per-stock dossier from EDGAR
│   └── data/                # JSON files per ticker
├── signals/
│   ├── __init__.py
│   ├── fundamental.py       # Quality/valuation scores from dossier
│   ├── news_sentiment.py    # LLM-based news scoring
│   └── reddit_sentiment.py  # Reddit mention + sentiment
├── llm/
│   ├── __init__.py
│   ├── client.py            # Anthropic client wrapper
│   ├── prompts.py           # Centralized prompt templates
│   └── thesis.py            # Thesis writer + conviction check
├── reporting/
│   ├── __init__.py
│   └── single_stock.py      # Stage 1 report generator
├── config/
│   ├── settings.py          # API keys, paths, constants
│   └── tickers.json         # Stage 1: ["AAPL", "JPM", "XOM"]
├── tests/
├── data_cache/              # SQLite + JSON cache (gitignored)
├── reports/                 # Generated reports (gitignored)
├── .env                     # API keys (gitignored)
├── requirements.txt
└── run_stage1.py            # Orchestrator for single-stock weekly report
```

### Step 1.2: Storage layer (hybrid: parquet + SQLite + JSON)

We use three backends, each chosen so the structure fits the access pattern. SQL is reserved for streams where `UNIQUE` constraint + time-window queries earn it; everything else stays as flat files that pandas can read directly.

| Data | Backend | Path | Rationale |
|---|---|---|---|
| Prices | Parquet, one file per ticker | `data_cache/prices/{ticker}.parquet` | Append-mostly OHLCV time series; pandas-native; columnar reads. |
| News items | SQLite | `data_cache/weekly.db` | `url UNIQUE` dedup on insert; time-window queries; grows large at Stage 3. |
| Reddit posts | SQLite | `data_cache/weekly.db` | Same shape as news: `post_id UNIQUE`, time-window queries. |
| Dossiers | JSON, one file per ticker | `dossier/data/{ticker}.json` | One blob per ticker, rewritten quarterly. |
| Weekly reports | JSON, one file per (ticker, date) | `reports/{ticker}/{YYYY-MM-DD}.json` | Append-only artefacts; convenient to read by hand. |

SQLite schema (only the two streams that benefit from it):

```sql
CREATE TABLE IF NOT EXISTS news_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT,
    source TEXT,
    url TEXT UNIQUE NOT NULL,
    published_at TIMESTAMP,
    snippet TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_pub ON news_items(ticker, published_at);

CREATE TABLE IF NOT EXISTS reddit_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    subreddit TEXT,
    post_id TEXT UNIQUE NOT NULL,
    title TEXT,
    score INTEGER,
    num_comments INTEGER,
    created_at TIMESTAMP,
    url TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reddit_ticker_created ON reddit_posts(ticker, created_at);
```

Use `sqlite3` (stdlib) — no SQLAlchemy. Inserts go through `INSERT OR IGNORE` so re-fetches are idempotent. Parquet reads/writes via `pandas` + `pyarrow`. JSON via stdlib.

**Storage API (`weekly_strategy/data/storage.py`):**

```python
# Prices (parquet)
def upsert_prices(ticker: str, df: pd.DataFrame) -> None
def load_prices(ticker: str, since: date | None = None, until: date | None = None) -> pd.DataFrame

# News / Reddit (SQLite)
def init_db() -> None  # idempotent
def insert_news(items: list[NewsItem]) -> int  # returns # newly inserted
def insert_reddit(items: list[RedditPost]) -> int
def get_news(ticker: str, since: datetime, until: datetime | None = None) -> list[NewsItem]
def get_reddit(ticker: str, since: datetime, until: datetime | None = None) -> list[RedditPost]

# Dossiers / reports (JSON)
def save_dossier(ticker: str, data: dict) -> None
def load_dossier(ticker: str) -> dict | None
def save_weekly_report(ticker: str, report_date: date, data: dict) -> Path
def list_weekly_reports(ticker: str) -> list[Path]
```

### Step 1.3: Data fetchers

Build each fetcher with a clean signature, cache results to SQLite, return typed data.

**1.3a — yfinance wrapper:**

```python
def get_price_history(ticker: str, lookback_days: int = 365) -> pd.DataFrame
def get_basic_info(ticker: str) -> dict  # sector, industry, market_cap, beta
```

Add throttling: sleep 0.5s between calls. Cache by `(ticker, date)`.

**1.3b — SEC EDGAR fetcher:**

```python
def get_cik(ticker: str) -> str  # ticker -> CIK lookup
def get_company_facts(cik: str) -> dict  # full XBRL facts JSON
def get_recent_filings(cik: str, form_types: list, days: int) -> list
```

Set `User-Agent: "YourName your@email.com"` in headers. EDGAR allows 10 req/sec.

Critical concepts in company_facts to extract:
- `Revenues` or `SalesRevenueNet` (varies by company — build mapping table)
- `NetIncomeLoss`
- `OperatingIncomeLoss`
- `GrossProfit`
- `Assets`, `Liabilities`
- `CashAndCashEquivalentsAtCarryingValue`
- `LongTermDebt`, `ShortTermBorrowings`
- `CommonStockSharesOutstanding` or `WeightedAverageNumberOfDilutedSharesOutstanding`
- `NetCashProvidedByUsedInOperatingActivities`
- `PaymentsToAcquirePropertyPlantAndEquipment` (for FCF = OCF - capex)

For each concept, filter by `fp` ('FY' for annual, 'Q1/Q2/Q3' for quarterly, no 'Q4' — Q4 is FY minus Q1+Q2+Q3) and `fy` for fiscal year. Use `filed` date for point-in-time correctness.

**1.3c — Google News RSS:**

```python
def get_news(ticker: str, query_override: str = None, days: int = 7) -> list[NewsItem]
```

URL template:
```
https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en
```

Use `feedparser` to parse. For each entry, extract title, link, published, source, summary.

Deduplication: normalize titles (lowercase, strip punctuation, first 80 chars), drop duplicates. Filter by source whitelist (start permissive, tighten later).

Source whitelist starter:
```python
TRUSTED_SOURCES = {
    'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com',
    'barrons.com', 'marketwatch.com', 'seekingalpha.com', 'finance.yahoo.com',
    'investors.com', 'forbes.com', 'businesswire.com', 'prnewswire.com',
    'globenewswire.com', 'sec.gov'
}
```

Maintain `config/ticker_queries.json` for ambiguous tickers:
```json
{
    "AAPL": "AAPL stock OR Apple Inc",
    "META": "\"Meta Platforms\" OR Facebook stock",
    "KEY": "KeyCorp stock"
}
```

**1.3d — Reddit fetcher (unauthenticated JSON):**

Decision: Reddit signal on mega-cap S&P 500 / Nasdaq 100 names is noisy because constant chatter dilutes any state-change signal. Narrow the window to **last 24 hours only** so we capture *fresh* posts that may indicate emerging narratives, not steady-state hum. Keep it but treat as experimental until backtest proves out.

Implementation: use Reddit's **public JSON endpoints** with `requests` — no API key, no PRAW dependency. Reddit's developer API key issuance has been unreliable since 2023, but the public JSON endpoints (just append `.json` to any Reddit URL) remain accessible to anyone who sends a polite `User-Agent` header.

```python
def get_reddit_mentions(ticker: str, hours: int = 24) -> RedditSnapshot
```

Subreddits to scan:
```python
SUBREDDITS = ['investing', 'stocks', 'SecurityAnalysis', 'options', 'wallstreetbets']
```

Endpoint pattern:
```
https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&restrict_sr=1&t=day&sort=new&limit=50
```

`t=day` returns posts from the last 24 hours; sort by `new` (in a 24h window, high-score posts haven't had time to accumulate votes — we want fresh mentions). Post-filter strictly to the precise 24h window using each post's `created_utc`.

Required headers:
```python
HEADERS = {"User-Agent": "weekly-strategy/0.1 (contact: cunwei.fan@tophash.io)"}
```

Rate limiting: ~60 req/min for unauth. We sleep 1.1s between requests to stay safely under, and respect `Retry-After` on 429 responses with exponential backoff.

Return:
```python
@dataclass
class RedditSnapshot:
    ticker: str
    window_hours: int                 # 24 for v1
    mention_count: int                # in window
    mention_count_prior_window: int   # the previous 24h, for delta
    mention_delta_pct: float          # (current - prior) / max(prior, 1)
    top_posts: list[RedditPost]       # top 5 by score in window
    sentiment_proxy: dict             # crude: pos/neg word count from titles
```

For the prior-window comparison, we need posts from the 24-48h-ago window. The `t=day` filter only gives us 0-24h. We can either (a) issue a second query with no time filter and post-filter, or (b) maintain our own rolling 7-day buffer in SQLite from prior weekly runs and just look back. Option (b) is cleaner and avoids extra requests — do that.

Resilience: if Reddit returns 403/429 persistently, log a warning, return an empty `RedditSnapshot`, and let the pipeline continue. Reddit data is "nice to have" not load-bearing in Stage 1.

**1.3e — Earnings press release fetcher (optional v1):**

Maintain `config/ir_urls.json`:
```json
{
    "AAPL": "https://www.apple.com/newsroom/archive/"
}
```

Simple scraper for the IR page; extract recent press release titles and dates. Skip for v1 if scraping proves brittle — we get earnings data from EDGAR 8-Ks anyway.

### Step 1.4: Dossier builder

Goal: produce a structured JSON dossier for AAPL from EDGAR company_facts.

```python
@dataclass
class Dossier:
    ticker: str
    company_name: str
    sector: str
    industry: str

    # Quality block
    revenue_3y_cagr: float
    revenue_yoy_latest: float
    revenue_qoq_latest: float
    gross_margin_current: float
    gross_margin_trend: list[float]  # last 8 quarters
    operating_margin_current: float
    fcf_margin_current: float
    roic: float
    roe: float
    share_count_change_yoy: float
    net_debt_to_ebitda: float
    fcf_conversion: float

    # Growth/momentum block
    next_earnings_date: date | None
    last_4_surprises: list[dict]  # {date, eps_actual, eps_estimate, beat_pct}

    # Valuation block (computed using current price)
    pe_trailing: float
    pe_forward: float | None  # requires consensus data; may be None in v1
    ps_ratio: float
    fcf_yield: float
    ev_to_ebitda: float

    # Catalyst block (free-form)
    upcoming_events: list[str]
    recent_8k_summary: str | None  # LLM-summarized

    # Metadata
    last_updated: datetime
    last_filing_date: date
```

Implementation:
1. Pull company_facts JSON from EDGAR
2. Extract last 5 years of quarterly and annual figures for the concepts above
3. Compute derived metrics (margins, ratios, CAGRs)
4. Pull current price from yfinance for valuation ratios
5. Store as JSON in `dossier/data/AAPL.json` and SQLite

Cadence:
- Stage 1: build once for AAPL, refresh manually as needed
- Later: quarterly refresh, plus targeted refresh when 10-Q or 10-K filed

### Step 1.5: News sentiment + fundamental scoring (two-agent cascade)

**Two-agent design**, with a third deferred:

1. **Sentiment agent** (Haiku, every item): per-headline POSITIVE/NEGATIVE/NEUTRAL × HIGH/MEDIUM/LOW × theme enum. Cheap, fast, runs on all ~50-200 items per week.
2. **Fundamental-impact agent** (Sonnet, only items classified HIGH or MEDIUM in pass 1): per-item structured judgment on *which line items move and by how much* — `areas[]` × direction × magnitude (small/medium/large) × horizon (Q/FY/multi-year) × 1-2 sentence implication. Feeds the thesis writer in Step 1.8 so it can write *"FY26 revenue tailwind"* rather than *"news was positive"*.
3. **Sectoral / cross-stock agent** — deferred to **Step 2.5** (sectoral scoring), where we'll have cross-ticker context. Reasoning about *"AAPL's AI struggles → NVDA consolidating power"* needs other tickers' dossiers and sector ETF flows that don't exist in Stage 1.

Why the cascade rather than running all three on every item: pass 2 is ~3× the per-token cost of pass 1 (Sonnet vs Haiku), but only ~30-40% of items pass the materiality filter, so total cost goes up ~2×, not 3×. LOW-materiality noise items don't deserve Sonnet attention.

For a week's worth of news on AAPL, the LLM should output structured sentiment.

Prompt structure (in `llm/prompts.py`):

```python
NEWS_SENTIMENT_PROMPT = """You are a sell-side equity analyst reviewing news for {ticker} ({company_name}) over the past week.

Below are {n} news items. For each, classify:
- Materiality: HIGH (could move stock 2%+) / MEDIUM (worth noting) / LOW (background noise)
- Sentiment: POSITIVE / NEGATIVE / NEUTRAL for the stock specifically
- Theme: one of [earnings, guidance, product, regulation, litigation, M&A, management, macro, competitive, analyst, technical, other]

Then provide:
- Overall weekly sentiment score (-10 to +10)
- Top 3 themes driving the week
- Any items that warrant follow-up

News items:
{news_items_formatted}

Respond in JSON with this schema:
{schema}
"""
```

Model choice: use **`claude-haiku-4-5-20251001`** here. Per-article classification is high-volume and structurally simple — Haiku is fast and ~5× cheaper than Sonnet, well-suited to this. Reserve Sonnet 4.6 for downstream synthesis (thesis writer in Step 1.8).

**Batching strategy** (matters because we're on subscription, not the API Batch endpoint — see Tooling note). One `claude -p` call per ticker per week, passing *all* of that week's deduped news items (typically 50-200) in a single prompt and asking for an array of classifications back. This keeps headless calls per week at O(tickers), not O(articles): Stage 1 = ~3 calls/week, Stage 3 = ~100 calls/week. If a ticker has >200 items in a week, chunk into batches of 50 with chunk-level retry on JSON parse failure.

Output stored in SQLite, used downstream in weekly report.

### Step 1.6: Reddit sentiment scoring

Lighter touch than news. Just:

```python
def score_reddit_snapshot(snapshot: RedditSnapshot) -> RedditScore:
    mention_zscore = (snapshot.mention_count - HISTORICAL_MEAN) / HISTORICAL_STD
    delta_signal = snapshot.mention_delta_vs_prior_week
    # If top posts > threshold, LLM-score the top 3 for sentiment
    if snapshot.top_posts and snapshot.mention_count > 10:
        llm_sentiment = score_posts_with_llm(snapshot.top_posts)
    return RedditScore(...)
```

Note: Reddit sentiment is most useful as a *crowding* signal. Spike in mentions → crowded trade. Persistent low mentions but stock running → institutional accumulation. These interpretations come at the report level, not the scoring level.

### Step 1.7: Fundamental scoring

Translate dossier into numerical scores. All scoring functions return 0-100.

```python
def quality_score(dossier: Dossier) -> float:
    # Composite of: ROIC, margins, balance sheet, FCF conversion
    # Z-score each component vs sector benchmarks (hardcoded medians for v1)
    # In Stage 1 (one ticker), just compute and display; no ranking yet

def valuation_score(dossier: Dossier) -> float:
    # P/E percentile vs own 5-year history
    # P/S, FCF yield, EV/EBITDA similar
    # Lower current multiples = higher score (long bias)

def momentum_score(dossier: Dossier, prices: pd.DataFrame) -> float:
    # 3m return + 6m return - 1m return (reversal)
    # Normalized
```

For Stage 1, store these as a `StockScoreBundle`:

```python
@dataclass
class StockScoreBundle:
    ticker: str
    quality_score: float
    valuation_score: float
    momentum_score: float
    news_sentiment_score: float
    reddit_sentiment_score: float
    # Macro and sector scores added in Stage 2
    week_ending: date
```

### Step 1.8: LLM thesis writer

Given the score bundle + dossier + raw news, ask LLM to write a structured weekly thesis. Use **`claude-sonnet-4-6`** here — synthesis across heterogeneous inputs benefits from the bigger model.

```python
THESIS_PROMPT = """You are writing a weekly equity research note on {ticker}.

Here is the data:
- Dossier: {dossier_json}
- Score bundle: {scores_json}
- Top news items this week: {news_top}
- Reddit activity: {reddit_summary}
- Price action: {price_summary}

Write a structured analyst note with these sections:

1. Bottom line (2-3 sentences): is this looking constructive, cautious, or neutral this week, and why?
2. What changed this week (3-5 bullets): specific developments
3. Quality & valuation snapshot (2-3 sentences): where does it stand vs history
4. Top 2 risks
5. Watch items for next week (catalysts, events)

Be specific. Cite numbers. No hedge-everything language.
"""
```

Output stored in SQLite under `weekly_reports`.

### Step 1.9: Weekly report generator

Produces a markdown report combining everything.

```
# AAPL Weekly Report — Week ending YYYY-MM-DD

## Bottom line
[LLM-generated]

## Price action
- Week return: X%
- vs SPY: X%
- Volume profile: [normal/elevated/depressed]

## Score snapshot
| Component | Score |
| Quality | XX |
| Valuation | XX |
| Momentum | XX |
| News sentiment | XX |
| Reddit sentiment | XX |

## What changed this week
[LLM-generated bullets]

## Quality & valuation
[LLM-generated commentary + dossier highlights]

## Top news items
[Top 5 news items with materiality tags]

## Reddit activity
- Mentions: X (vs Y prior week)
- Top post: [title and link]
- Sentiment proxy: [pos/neg]

## Risks
[LLM-generated]

## Watch items
[LLM-generated]

## Appendix: full dossier snapshot
[Pretty-printed dossier]
```

### Step 1.10: Stage 1 orchestrator

`run_stage1.py`:

```python
def main():
    tickers = ["AAPL", "JPM", "XOM"]

    for ticker in tickers:
        # 1. Fetch / refresh data
        prices = fetchers.get_price_history(ticker, lookback_days=365)
        news = fetchers.get_news(ticker, days=7)
        reddit = fetchers.get_reddit_mentions(ticker, hours=24)

        # 2. Build or load dossier
        dossier = dossier_builder.get_or_build(ticker)

        # 3. Compute scores
        scores = scoring.compute_all_scores(ticker, dossier, prices, news, reddit)

        # 4. LLM thesis
        thesis = thesis_writer.write(ticker, dossier, scores, news, reddit, prices)

        # 5. Generate report
        report_md = reporting.single_stock.generate(
            ticker, dossier, scores, thesis, news, reddit, prices
        )

        # 6. Save
        storage.save_weekly_report(ticker, report_md)
        output_path = f"reports/{ticker}_{date.today()}.md"
        Path(output_path).write_text(report_md)
        print(f"Report saved: {output_path}")
```

Iterate serially in Stage 1 (each ticker takes ~1-2 minutes; parallelism not worth the complexity yet). The three reports together should run in well under 10 minutes.

### Stage 1 acceptance criteria

Before moving to Stage 2:

- [ ] **All 3 tickers' dossiers** (AAPL, JPM, XOM) build cleanly from EDGAR with reasonable values (compare each to Yahoo Finance to sanity-check). Confirm financials-specific tags work for JPM and energy capex/FCF works for XOM.
- [ ] Google News RSS returns relevant articles for each ticker, deduplication works
- [ ] Reddit fetcher returns recent (24h) posts; mention counts feel right
- [ ] LLM news sentiment classification reads correctly on at least 10 articles across the 3 names
- [ ] Each ticker's weekly report is something you'd actually want to read on Monday morning
- [ ] Total pipeline (all 3 tickers) runs in < 10 minutes
- [ ] No silent failures — all errors logged

Fix any sector-specific issues before moving to Stage 2.

---

## STAGE 2: MACRO AND SECTOR LAYERS

Goal: add universe-level context that modulates the per-stock view. AAPL's signal should now incorporate what's happening with rates, Fed posture, and tech sector flows.

### Step 2.1: Macro data fetchers

**FRED time series to track:**

Rates:
- DGS2 (2-year Treasury)
- DGS10 (10-year Treasury)
- T10Y2Y (10y-2y spread)
- T10YIE (10y breakeven inflation)
- DFII10 (10y real yield) — critical for growth stocks
- DFF (Fed funds rate)
- FEDFUNDS (effective fed funds, monthly)
- MORTGAGE30US (30y mortgage)

Credit:
- BAMLH0A0HYM2 (HY OAS)
- BAMLC0A0CM (IG OAS)
- BAMLH0A0HYM2EY (HY effective yield)

Macro:
- CPIAUCSL (CPI)
- PCEPI (PCE)
- PAYEMS (nonfarm payrolls)
- UNRATE (unemployment)
- INDPRO (industrial production)
- UMCSENT (consumer sentiment)
- ICSA (initial claims, weekly)

Market:
- VIXCLS (VIX)
- DTWEXBGS (broad dollar index)

```python
def get_fred_series(series_id: str, lookback_days: int) -> pd.Series
def get_macro_snapshot() -> MacroSnapshot
```

`MacroSnapshot` structure:
```python
@dataclass
class MacroSnapshot:
    week_ending: date

    # Rates state
    yield_10y: float
    yield_10y_wow_change: float  # week-over-week change in basis points
    yield_2y: float
    yield_2y_wow_change: float
    real_yield_10y: float
    real_yield_10y_wow_change: float
    curve_2s10s: float
    curve_inverted: bool

    # Credit state
    hy_oas: float
    hy_oas_wow_change: float
    hy_regime: str  # 'tight' / 'normal' / 'wide' / 'stressed' — bucketed

    # Vol state
    vix_level: float
    vix_wow_change: float
    vix_regime: str  # 'complacent' / 'normal' / 'elevated' / 'panic'

    # Inflation expectations
    breakeven_10y: float
    breakeven_10y_wow_change: float

    # Currency
    dxy_level: float
    dxy_wow_change: float

    # Recent data prints (last 7 days)
    recent_prints: list[dict]  # {series, date, value, prior, surprise}
```

### Step 2.2: Fed communication tracker

Lightweight v1 approach: just pull headlines that mention Fed speakers from Google News.

```python
def get_fed_speak_this_week() -> list[NewsItem]
```

Query: `("Powell" OR "FOMC" OR "Federal Reserve") when:7d` on Google News RSS, filtered by trusted sources.

LLM pass to extract hawkish/dovish lean:

```python
FED_SPEAK_PROMPT = """Review these Fed-related news items from the past week.

Classify the aggregate posture as: HAWKISH / SLIGHTLY HAWKISH / NEUTRAL / SLIGHTLY DOVISH / DOVISH.

Identify any specific policy hints (rate cuts/hikes, balance sheet, forward guidance).

Items:
{items}
"""
```

Output: `FedPosture` enum and supporting commentary.

### Step 2.3: Sector data fetchers

Track 11 GICS sector ETFs:

```python
SECTOR_ETFS = {
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}
```

For each ETF compute:
- 1-week return
- 1-month return vs SPY (relative momentum)
- 3-month return vs SPY
- Volume profile vs 20-day average

```python
@dataclass
class SectorSnapshot:
    week_ending: date
    sectors: dict[str, SectorMetrics]
    leadership_ranking: list[str]  # sectors ordered by recent relative strength
    breadth: str  # 'narrow' / 'broad' — based on dispersion
```

### Step 2.4: Macro regime classifier

Translate macro snapshot into a regime label that affects scoring:

```python
@dataclass
class MacroRegime:
    risk_on_off: str  # 'risk_on' / 'neutral' / 'risk_off'
    rate_regime: str  # 'rising' / 'stable' / 'falling'
    growth_vs_value_favor: str  # 'growth' / 'value' / 'mixed'
    cyclical_vs_defensive_favor: str
    fed_posture: str  # from FedPosture
    overall_score: float  # -1 to +1, where +1 is most risk-on
```

Classification logic (deterministic, transparent — not LLM-based):

```python
def classify_regime(macro: MacroSnapshot, fed: FedPosture) -> MacroRegime:
    # Risk-on: VIX < 16, HY OAS tightening, equities up
    # Risk-off: VIX > 25, HY OAS widening, equities down
    # Growth-favored: real yields falling, Fed dovish
    # Value-favored: real yields rising, Fed hawkish
    ...
```

This is a rules-based scorecard. Document the thresholds. Tune via backtest later.

### Step 2.5: Sector scoring

For a given stock, the sector score combines sector momentum with macro regime fit.

```python
def sector_score(ticker: str, sector_snapshot: SectorSnapshot, regime: MacroRegime, dossier: Dossier) -> float:
    sector = dossier.sector
    sector_momentum = sector_snapshot.sectors[sector].momentum_3m_vs_spy
    regime_fit = is_sector_favored_by_regime(sector, regime)
    return weighted_combo(sector_momentum, regime_fit)
```

Sector-regime fit mapping (start with conventional wisdom, refine via backtest):
- Risk-on, growth-favored → XLK, XLC, XLY benefit
- Risk-off, defensive → XLP, XLU, XLV benefit
- Rising rates → XLF benefits, XLU and XLRE hurt
- Falling dollar → XLE, XLB benefit
- Inverted curve → XLF hurt (NIM compression)

### Step 2.6: Updated score bundle

Extend the Stage 1 score bundle:

```python
@dataclass
class StockScoreBundle:
    ticker: str
    week_ending: date

    # Stage 1 components
    quality_score: float
    valuation_score: float
    momentum_score: float
    news_sentiment_score: float
    reddit_sentiment_score: float

    # Stage 2 additions
    macro_regime_score: float  # universe-level, applied to all stocks with sector-specific weighting
    sector_score: float
    sector_in_favor: bool

    # Composite (Stage 2 onward)
    composite_score: float
```

Composite formula (starting weights, will tune):

```python
def composite_score(scores: StockScoreBundle, macro: MacroRegime) -> float:
    # Weights depend on regime
    if macro.risk_on_off == 'risk_off':
        # In risk-off, quality and valuation matter more
        weights = {'quality': 0.30, 'valuation': 0.25, 'momentum': 0.10,
                   'news': 0.15, 'reddit': 0.05, 'macro': 0.10, 'sector': 0.05}
    else:
        weights = {'quality': 0.20, 'valuation': 0.15, 'momentum': 0.20,
                   'news': 0.15, 'reddit': 0.10, 'macro': 0.10, 'sector': 0.10}

    return weighted_sum(scores, weights)
```

### Step 2.7: Enhanced weekly report

Add new sections to the Stage 1 report:

```
## Macro context
- Rate environment: [10y yield, weekly change, real yield]
- Credit: [HY OAS, regime]
- Vol: [VIX, regime]
- Fed posture this week: [HAWKISH/DOVISH/NEUTRAL with commentary]
- Macro regime: [risk_on / risk_off]

## Sector context
- AAPL sector: Technology (XLK)
- XLK 1w return: X%
- XLK vs SPY 3m: X% (sector leadership rank: N of 11)
- Regime fit: [favorable / unfavorable]

## Composite score: XX / 100
[Breakdown of how it was assembled]

[Rest of Stage 1 report sections continue below]
```

### Step 2.8: Stage 2 orchestrator update

```python
def main():
    ticker = "AAPL"

    # Fetch all data including Stage 2 macro/sector
    macro_snapshot = fetchers.get_macro_snapshot()
    fed_posture = fetchers.get_fed_speak_this_week_summary()
    sector_snapshot = fetchers.get_sector_snapshot()

    # Stage 1 fetches
    prices = fetchers.get_price_history(ticker, lookback_days=365)
    news = fetchers.get_news(ticker, days=7)
    reddit = fetchers.get_reddit_mentions(ticker, hours=24)
    dossier = dossier_builder.get_or_build(ticker)

    # Compute regime
    regime = macro.classify_regime(macro_snapshot, fed_posture)

    # Compute all scores including macro/sector
    scores = scoring.compute_all_scores_v2(
        ticker, dossier, prices, news, reddit,
        macro_snapshot, regime, sector_snapshot
    )

    # LLM thesis (now includes macro context)
    thesis = thesis_writer.write_v2(
        ticker, dossier, scores, news, reddit, prices,
        macro_snapshot, regime, sector_snapshot
    )

    # Generate enhanced report
    report_md = reporting.single_stock.generate_v2(
        ticker, dossier, scores, thesis,
        news, reddit, prices,
        macro_snapshot, regime, sector_snapshot
    )

    storage.save_weekly_report(ticker, report_md)
    Path(f"reports/{ticker}_{date.today()}.md").write_text(report_md)
```

### Stage 2 acceptance criteria

- [ ] FRED fetcher pulls all required series cleanly
- [ ] Macro regime classifier produces sensible labels across historical weeks (sanity-check by feeding in known regimes: Mar 2020, Jan 2022, Q4 2022)
- [ ] Sector snapshot ranks sectors meaningfully
- [ ] Fed speak summary captures actual posture (compare to known speech)
- [ ] AAPL composite score visibly shifts with macro changes
- [ ] Weekly report incorporates macro and sector commentary readably
- [ ] Pipeline still runs in < 10 minutes

Try the pipeline on different sectors at this stage too: a tech name (AAPL), a financial (JPM), an energy name (XOM), a defensive (PG). The sector-regime interactions should produce visibly different signals.

---

## STAGE 3: UNIVERSE CONSTRUCTION AND WEEKLY RANKING

Goal: expand from single-stock (Stage 1-2) to a full 100-stock universe with quarterly refresh, and produce a weekly ranked list with focus on the most actionable names.

Architectural principle: **the 100-stock universe is stable across each quarter.** Dossiers, score histories, and backtest results all depend on a stable population. Only the universe membership changes quarterly; the weekly cadence stays focused on scoring and ranking *within* the fixed universe.

### Step 3.1: Universe filter (quarterly refresh)

Build a deterministic filter that selects 100 stocks from a broader candidate pool.

**Candidate pool:** Union of S&P 500 + Nasdaq 100 = ~600 unique names. Pull constituent lists from:
- S&P 500: Wikipedia table (stable, scrapable) or maintain manually as a JSON file
- Nasdaq 100: same approach

**Hard filters (must pass all):**
- Market cap > $2B
- Average daily dollar volume (last 60 days) > $50M
- Listed > 12 months (no recent IPOs)
- No pending M&A announcements
- Has filed at least 4 quarterly reports on EDGAR
- Analyst coverage > 5 estimates (proxied via Yahoo Finance estimates count)
- Not in a delisting / bankruptcy process

**Shortability check:**
- Maintain `config/shortable_on_robinhood.json` — manually curated initially
- Mark each stock as "long-only" or "long-short" in the universe
- v1: include non-shortable names but only as long candidates

**Curation logic (filter → 100):**

```python
def construct_universe(candidates: list[str]) -> Universe:
    # 1. Apply hard filters → typically reduces to ~400 names
    passing = [t for t in candidates if passes_hard_filters(t)]

    # 2. Sector quotas: cap each GICS sector at 15 names
    by_sector = group_by_sector(passing)
    selected = []
    for sector, names in by_sector.items():
        # Rank within sector by market cap × liquidity score
        ranked = rank_by_quality_signal(names)
        selected.extend(ranked[:15])

    # 3. Thematic bonus picks (manual override list)
    # e.g., key AI infra, defense space, EV — names you want exposure to even if they don't rank top by cap
    thematic_adds = load_thematic_list()  # config/thematic_picks.json
    selected = dedupe(selected + thematic_adds)

    # 4. Trim to exactly 100 (drop lowest-liquidity if over)
    selected = trim_to_100(selected)

    return Universe(tickers=selected, constructed_at=now())
```

**Sector quotas (cap per GICS sector):** prevents over-concentration. Starting allocation:
```python
SECTOR_TARGETS = {
    'Technology': 18,
    'Financials': 12,
    'Health Care': 12,
    'Consumer Discretionary': 10,
    'Industrials': 10,
    'Communication Services': 8,
    'Consumer Staples': 8,
    'Energy': 8,
    'Materials': 5,
    'Utilities': 5,
    'Real Estate': 4
}  # sums to 100
```

These can be tilted by macro regime in v2 of the universe construction (e.g., overweight defensives in risk-off), but start static.

**Thematic picks** (config/thematic_picks.json) — names you want in the universe regardless of cap-based ranking:
```json
{
    "ai_infrastructure": ["NVDA", "AMD", "AVGO", "TSM"],
    "defense_space": ["RKLB", "LMT", "RTX", "NOC"],
    "ev_battery": ["TSLA", "ALB"]
}
```

Cap thematic picks at 10-15 total to leave room for systematic selection.

**Quarterly refresh job:** Runs once per quarter (suggest 2 weeks after quarter-end, so latest filings are in). Produces:
- Proposed adds: names newly passing filters or recently met thresholds
- Proposed drops: names that no longer meet criteria
- Sector balance vs targets
- Liquidity drift report
- Human review and approve before committing

Store universe history in SQLite:
```sql
CREATE TABLE universe_history (
    quarter TEXT,  -- e.g., '2026Q2'
    ticker TEXT,
    added_date DATE,
    removed_date DATE,
    metadata JSON,  -- liquidity, sector, etc. at time of inclusion
    PRIMARY KEY (quarter, ticker)
);
```

This history is critical for backtesting — you need to know what the universe looked like at each historical point.

### Step 3.2: Batch dossier generation

Once universe is set, generate dossiers for all 100 names.

```python
def batch_build_dossiers(universe: Universe, parallel: bool = True):
    for ticker in universe.tickers:
        try:
            dossier = dossier_builder.build(ticker)
            storage.save_dossier(ticker, dossier)
        except Exception as e:
            logger.error(f"Dossier failed for {ticker}: {e}")
            # Don't fail entire batch — log and continue
```

Considerations:
- Rate-limit EDGAR calls to 10 req/sec
- Cache aggressively — once a dossier is built, only refresh on new 10-Q/10-K
- This batch takes ~30-60 minutes the first time, much faster subsequently with caching

After batch generation, validate: spot-check 5-10 dossiers manually for sanity. Common issues to look for:
- Negative values where they shouldn't be (often a sign of XBRL concept mismatch)
- ROIC > 100% (usually a denominator issue)
- Missing share count or debt for certain industries (financials use different tags)

### Step 3.3: Batch weekly scoring

For each of the 100 stocks each week, compute the score bundle from Stage 1-2.

```python
def batch_weekly_scoring(universe: Universe, week_ending: date) -> dict[str, StockScoreBundle]:
    macro = fetchers.get_macro_snapshot()
    regime = macro_classifier.classify(macro, get_fed_posture())
    sectors = fetchers.get_sector_snapshot()

    scores = {}
    for ticker in universe.tickers:
        dossier = storage.load_dossier(ticker)
        prices = fetchers.get_price_history(ticker, lookback_days=365)
        news = fetchers.get_news(ticker, days=7)
        reddit = fetchers.get_reddit_mentions(ticker, hours=24)  # 24h fresh-mention window

        bundle = scoring.compute_full_bundle(
            ticker, dossier, prices, news, reddit,
            macro, regime, sectors
        )
        scores[ticker] = bundle
        storage.save_score_bundle(bundle)

    return scores
```

This is parallelizable but be careful with rate limits on Google News (no documented limit but reasonable) and Reddit (60 req/min).

Realistic runtime for 100 stocks: 20-40 minutes mostly LLM-bound (news sentiment scoring).

### Step 3.4: Cross-sectional normalization

After scores are computed for all 100, normalize within the universe so scores are comparable.

```python
def normalize_scores(scores: dict[str, StockScoreBundle]) -> dict[str, StockScoreBundle]:
    # For each score component, compute z-score across the 100 stocks
    for component in ['quality', 'valuation', 'momentum', 'news_sentiment', 'reddit_sentiment']:
        values = [s.get(component) for s in scores.values()]
        mean = np.mean(values)
        std = np.std(values)
        for ticker, bundle in scores.items():
            bundle.set_normalized(component, (bundle.get(component) - mean) / std)
    return scores
```

After normalization, every score component has mean 0 and std 1 across the universe. Composite scoring then combines these normalized values with the regime-dependent weights from Step 2.6.

### Step 3.5: Weekly focus list (top 20-30)

From the 100 stocks, surface the most actionable names this week.

```python
def build_focus_list(scores: dict[str, StockScoreBundle]) -> FocusList:
    ranked = sorted(scores.values(), key=lambda s: s.composite_score, reverse=True)

    # Top 15 by composite (long candidates)
    long_candidates = ranked[:15]

    # Bottom 15 by composite (short candidates)
    short_candidates = ranked[-15:]

    # Also flag any names with extreme single-component scores
    # (e.g., huge news sentiment swing, even if composite is mid)
    outliers = find_outliers(scores)

    return FocusList(
        longs=long_candidates,
        shorts=short_candidates,
        outliers=outliers
    )
```

The focus list is the input to Stage 4 (selection with constraints) — it's the wider funnel before applying sector neutrality, beta neutrality, and earnings exclusions.

### Step 3.6: Long/short selection with constraints

From the focus list, pick the actual 5 longs and 5 shorts.

```python
def select_positions(focus: FocusList, universe: Universe, prices: dict, dossiers: dict) -> Positions:
    longs = []
    shorts = []

    # Constraints:
    # - Max 2 longs per sector, max 2 shorts per sector (sector neutrality)
    # - No name reporting earnings in next 5 trading days
    # - No name with > 25% short interest (squeeze risk for shorts)
    # - Long basket beta ≈ short basket beta
    # - All shorts must be marked shortable on Robinhood

    sector_long_counts = defaultdict(int)
    sector_short_counts = defaultdict(int)

    for candidate in focus.longs:
        if len(longs) >= 5:
            break
        if not passes_long_constraints(candidate, sector_long_counts, dossiers):
            continue
        longs.append(candidate)
        sector_long_counts[dossiers[candidate.ticker].sector] += 1

    for candidate in focus.shorts:
        if len(shorts) >= 5:
            break
        if not passes_short_constraints(candidate, sector_short_counts, universe, dossiers):
            continue
        shorts.append(candidate)
        sector_short_counts[dossiers[candidate.ticker].sector] += 1

    # Beta neutralization: adjust position sizes
    longs, shorts = beta_neutralize(longs, shorts, prices, dossiers)

    return Positions(longs=longs, shorts=shorts)
```

**Turnover dampening:** If a name was in last week's portfolio and is now ranked 6-8 (just outside top 5), prefer keeping it over swapping in a new name. Stop the swap unless either:
- The held name dropped below rank 10
- The new candidate's score is materially higher (e.g., > 0.5 std dev above the held name)

This prevents whipsaw and reduces transaction costs.

### Step 3.7: LLM conviction check

For each of the 10 selected names, ask the LLM to write a thesis and verify it makes sense.

```python
CONVICTION_PROMPT = """You are reviewing a {direction} idea for {ticker} ({company_name}).

Score bundle:
{scores}

Key dossier data:
{dossier_summary}

Top news this week:
{news_top}

Macro context:
{regime_summary}

Sector context:
{sector_context}

Write:
1. A 3-sentence thesis for why this is a {direction} this week
2. The top 2 risks to this thesis
3. Confidence rating: HIGH / MEDIUM / LOW
4. Any flags or contradictions in the data

If you cannot write a coherent thesis with the given data, return confidence=LOW with explanation.
"""
```

If LLM returns LOW confidence for any selected name, drop it and substitute the next-ranked candidate from the focus list. This catches edge cases where the numerical score looks great but the underlying reality is broken (e.g., cheap on P/E because facing existential litigation; great momentum that's a pump-and-dump).

### Step 3.8: Weekly portfolio report

Final output extends the Stage 2 single-stock format to a portfolio view.

```
# Weekly Portfolio Report — Week ending YYYY-MM-DD

## Market context
[Macro regime, sector leadership, key events from last week]

## Recommended positions

### Longs (5)
1. AAPL — composite score X.XX
   - Thesis: [LLM-generated]
   - Risks: [LLM-generated]
   - Suggested weight: X%
2. ...

### Shorts (5)
1. XYZ — composite score X.XX
   - Thesis: [LLM-generated]
   - Risks: [LLM-generated]
   - Suggested weight: X%
2. ...

## Changes from last week
- Added: [names + reasons]
- Removed: [names + reasons]
- Held: [names + brief why-kept]

## Portfolio metrics
- Long basket beta: X.XX
- Short basket beta: X.XX
- Net beta: X.XX (target: -0.3 to +0.3)
- Sector exposure: [breakdown]
- Earnings risk this week: [any names reporting?]

## Watch list (next 5-10 by rank)
[For if conditions change mid-week]

## Risk flags
- [Earnings calendar]
- [Macro events]
- [Any positions approaching stop-out]

## Single-stock deep dives
[Optionally include the Stage 2-style detailed reports for the 10 selected names as appendices]
```

### Step 3.9: Stage 3 orchestrator

```python
def main_weekly():
    # 1. Load current universe (set quarterly, stable this week)
    universe = storage.load_current_universe()

    # 2. Refresh dossiers if any names reported earnings this week
    for ticker in universe.tickers:
        if had_earnings_this_week(ticker):
            dossier_builder.rebuild(ticker)

    # 3. Macro + sector fetches (universe-level)
    macro = fetchers.get_macro_snapshot()
    fed = fetchers.get_fed_speak_this_week_summary()
    regime = macro_classifier.classify(macro, fed)
    sectors = fetchers.get_sector_snapshot()

    # 4. Batch score the universe
    scores = batch_weekly_scoring(universe, regime, sectors)

    # 5. Normalize cross-sectionally
    scores = normalize_scores(scores)

    # 6. Composite + rank
    for ticker, bundle in scores.items():
        bundle.composite_score = compute_composite(bundle, regime)

    # 7. Build focus list
    focus = build_focus_list(scores)

    # 8. Apply constraints, select 5+5
    positions = select_positions(focus, universe, ...)

    # 9. LLM conviction check, substitute if needed
    positions = conviction_check_and_filter(positions)

    # 10. Generate portfolio report
    report = reporting.portfolio.generate(positions, scores, regime, sectors, macro)

    # 11. Save and output
    storage.save_weekly_report(report)
    Path(f"reports/portfolio_{date.today()}.md").write_text(report)


def main_quarterly():
    # Runs once per quarter, ~2 weeks after quarter-end
    candidates = load_candidate_pool()  # S&P 500 + Nasdaq 100
    proposed_universe = construct_universe(candidates)

    current = storage.load_current_universe()
    adds = set(proposed_universe.tickers) - set(current.tickers)
    drops = set(current.tickers) - set(proposed_universe.tickers)

    # Generate proposal report for human review
    report = generate_universe_change_proposal(current, proposed_universe, adds, drops)
    Path(f"reports/universe_proposal_{quarter()}.md").write_text(report)

    # After human approval, commit
    if input("Approve universe change? [y/N]: ").lower() == 'y':
        storage.commit_universe(proposed_universe)
        batch_build_dossiers(proposed_universe)
```

### Stage 3 acceptance criteria

- [ ] Quarterly universe construction produces sensible 100-stock list with proper sector balance
- [ ] Batch dossier build completes for all 100 names without silent failures
- [ ] Weekly batch scoring runs in < 1 hour
- [ ] Cross-sectional z-scoring produces normal-ish distributions per component
- [ ] Long/short selection respects all constraints (no sector concentration, no earnings names, etc.)
- [ ] Beta-neutral target achievable (long beta and short beta within 0.3 of each other)
- [ ] LLM conviction check catches at least one bad pick per month (false-positive elimination working)
- [ ] Weekly portfolio report is actionable — you'd execute from it without further research
- [ ] Universe history table populated for future backtest use

Run the full pipeline live for at least 4 consecutive weeks before moving to Stage 4. Watch for week-to-week stability — major reshuffling of positions every week suggests the signal is too noisy.

---

## STAGE 4 onward — outline only, details TBD

These come after Stage 3 is solid. Tracking here so we don't lose sight.

### Stage 4: Risk and execution rules

- Position sizing logic (volatility-adjusted)
- Sector exposure caps (enforced at portfolio level)
- Beta-neutral construction with size adjustments
- Turnover budget enforcement
- Stop-out rules per position and per book
- Pre-trade risk check before generating final report

### Stage 5: Backtesting framework

- Historical data assembly with point-in-time correctness
- Reconstruct historical universes (use universe_history table)
- Walk-forward simulation engine
- Performance metrics (Sharpe, hit rate, drawdown, factor attribution)
- Realistic transaction cost modeling (spreads, borrow fees, slippage)

### Stage 6: Paper trading

- Generate weekly signals live
- Simulate trades without capital
- Track live-vs-backtest divergence
- 8 weeks minimum before considering real money

### Stage 7: Live deployment

- Start at 25% intended capital
- Scale up if live performance tracks paper
- Monthly post-mortem
- Quarterly system audit and parameter retune

### Stretch goals

- Insider activity layer (Form 4 cluster detection)
- 13F-based smart money positioning overlay
- Options overlay for hedging
- Alternative data (Google Trends, HN for tech, etc.)
- Automated execution (only after 6+ months successful manual)

---

## What Claude is good for in this system

**Use the LLM for:**
- Reading news articles → structured sentiment + materiality + theme classification
- Parsing earnings call transcripts and 8-Ks for material events
- Generating natural-language theses from numerical scores
- Summarizing Fed communications for posture extraction
- Reviewing top/bottom ranked stocks for sanity (the conviction check)
- Writing the weekly report prose

**Don't use the LLM for:**
- Computing numerical scores from data (use code — deterministic, auditable, fast)
- Ranking decisions (rules-based composite)
- Macro regime classification (rules-based scorecard, transparent and tunable)
- Trade execution

---

## Realistic expectations

- First 6 months will likely underperform passive QQQ+SPY split. Normal.
- Target: by month 18, demonstrate persistent alpha after costs vs passive benchmark.
- Realistic Sharpe range: 0.3-0.8 if executed well. Above 1.0 is suspicious (overfit).
- Realistic directional hit rate: 52-56% per name.
- If by month 18 alpha is not demonstrable → revert to passive and use the agent for monitoring only.
- API costs: $0/month for v1. Anthropic LLM costs maybe $10-30/month at weekly cadence for full universe.

---

## First milestone

End of Stage 1: AAPL, JPM, and XOM each flow through the entire pipeline producing a clean, readable weekly report. The three reports run from `run_stage1.py` in one shot. Don't move past this milestone until all three are clean.

When opening a Claude Code session, attach this file and start with Step 1.1.

