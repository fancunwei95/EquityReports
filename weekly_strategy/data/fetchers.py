from __future__ import annotations

"""Network fetchers: yfinance, SEC EDGAR, Google News RSS, Reddit public JSON.

Each fetcher:
* Throttles to stay under each provider's rate ceiling.
* Returns typed records (pydantic models from `data.schemas` or a pandas
  DataFrame for prices); caching to storage is a separate concern.
* For prices specifically, ``get_price_history`` also writes through to
  ``storage.upsert_prices`` because that's the only sane way to keep the
  per-ticker parquet warm across runs.
"""

import json
import re
import time
from datetime import date, datetime, timedelta
from typing import Iterable
from urllib.parse import quote_plus, urlparse

import feedparser
import pandas as pd
import requests
import yfinance as yf

from weekly_strategy.config import settings
from weekly_strategy.data import storage
from weekly_strategy.data.schemas import NewsItem, RedditPost, RedditSnapshot

# ---------------------------------------------------------------------------
# Shared throttling helpers
# ---------------------------------------------------------------------------


class _Throttle:
    """Minimum-interval throttle. Not thread-safe; we run sequentially."""

    def __init__(self, min_interval_s: float) -> None:
        self._min = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min:
            time.sleep(self._min - elapsed)
        self._last = time.monotonic()


_yf_throttle = _Throttle(0.5)
_sec_throttle = _Throttle(0.11)        # EDGAR caps at 10 req/s; stay under.
_reddit_throttle = _Throttle(1.1)      # Reddit unauth ~60 req/min.


# ---------------------------------------------------------------------------
# 1.3a -- yfinance
# ---------------------------------------------------------------------------


def get_price_history(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV from yfinance, write through to parquet, return the merged frame."""
    _yf_throttle.wait()
    raw = yf.Ticker(ticker).history(period=f"{lookback_days}d", auto_adjust=False)
    if raw.empty:
        return storage.load_prices(ticker)
    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    # yfinance returns a tz-aware DatetimeIndex; storage normalizes to date.
    storage.upsert_prices(ticker, raw)
    return storage.load_prices(ticker)


def get_basic_info(ticker: str) -> dict:
    """Sector, industry, market cap, beta. yfinance's `.info` is best-effort."""
    _yf_throttle.wait()
    info = yf.Ticker(ticker).info or {}
    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "beta": info.get("beta"),
        "currency": info.get("currency"),
    }


# ---------------------------------------------------------------------------
# 1.3b -- SEC EDGAR
# ---------------------------------------------------------------------------

_EDGAR_DIR = settings.DATA_CACHE_DIR / "edgar"
_EDGAR_DIR.mkdir(parents=True, exist_ok=True)

_TICKER_MAP_CACHE: dict[str, str] | None = None


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": settings.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def _load_ticker_map() -> dict[str, str]:
    """ticker (upper) -> 10-digit CIK string."""
    global _TICKER_MAP_CACHE
    if _TICKER_MAP_CACHE is not None:
        return _TICKER_MAP_CACHE
    cache_path = _EDGAR_DIR / "company_tickers.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
    else:
        _sec_throttle.wait()
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_sec_headers(),
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()
        cache_path.write_text(json.dumps(raw))
    mapping: dict[str, str] = {}
    for entry in raw.values():
        mapping[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    _TICKER_MAP_CACHE = mapping
    return mapping


def get_cik(ticker: str) -> str:
    mapping = _load_ticker_map()
    cik = mapping.get(ticker.upper())
    if cik is None:
        raise ValueError(f"No CIK found for ticker {ticker!r}")
    return cik


def get_company_facts(cik: str, force_refresh: bool = False) -> dict:
    """Cached companyfacts XBRL JSON. Dossier builder controls refresh cadence."""
    cache_path = _EDGAR_DIR / f"CIK{cik}.json"
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())
    _sec_throttle.wait()
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=_sec_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_path.write_text(json.dumps(data))
    return data


def get_recent_filings(
    cik: str,
    form_types: Iterable[str],
    days: int = 90,
) -> list[dict]:
    """Subset of EDGAR submissions: forms filed within the last ``days`` days."""
    _sec_throttle.wait()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=_sec_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    primary = recent.get("primaryDocument", [])
    cutoff = date.today() - timedelta(days=days)
    wanted = set(form_types)
    out: list[dict] = []
    for i, form in enumerate(forms):
        if form not in wanted:
            continue
        fd = date.fromisoformat(dates[i])
        if fd < cutoff:
            continue
        out.append(
            {
                "form": form,
                "filing_date": fd,
                "accession": accs[i],
                "primary_document": primary[i] if i < len(primary) else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 1.3c -- Google News RSS
# ---------------------------------------------------------------------------

# Keep the whitelist loose at Stage 1: any source name / URL that *contains*
# one of these tokens passes. Tighten when we get real noise.
TRUSTED_SOURCE_KEYWORDS: frozenset[str] = frozenset(
    {
        "reuters", "bloomberg", "wsj", "wall street journal",
        "financial times", "cnbc", "barron", "marketwatch",
        "seeking alpha", "yahoo", "forbes", "businesswire",
        "business wire", "prnewswire", "pr newswire",
        "globenewswire", "globe newswire", "sec.gov",
        "ap news", "apnews", "the motley fool", "fool.com",
        "investors.com", "investor's business daily",
    }
)

_QUERIES_PATH = settings.PACKAGE_ROOT / "config" / "ticker_queries.json"


def _query_for(ticker: str, override: str | None) -> str:
    if override is not None:
        return override
    if _QUERIES_PATH.exists():
        overrides = json.loads(_QUERIES_PATH.read_text())
        q = overrides.get(ticker.upper())
        if q:
            return q
    return f"{ticker.upper()} stock"


def _normalize_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()[:80]


def _source_name_and_host(entry) -> tuple[str, str]:
    name = ""
    src = entry.get("source") if isinstance(entry, dict) else getattr(entry, "source", None)
    if src is not None:
        name = (src.get("title") if isinstance(src, dict) else getattr(src, "title", "")) or ""
        href = (src.get("href") if isinstance(src, dict) else getattr(src, "href", "")) or ""
    else:
        href = entry.get("link", "") if isinstance(entry, dict) else getattr(entry, "link", "")
    try:
        host = urlparse(href).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        host = ""
    return name, host


def _is_trusted(source_name: str, host: str) -> bool:
    hay = f"{source_name} {host}".lower()
    return any(kw in hay for kw in TRUSTED_SOURCE_KEYWORDS)


def fetch_news(
    ticker: str,
    query_override: str | None = None,
    days: int = 7,
) -> list[NewsItem]:
    """Pull Google News RSS, dedup by normalized title, filter to trusted sources."""
    q = _query_for(ticker, query_override)
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(q)}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(url)
    seen: set[str] = set()
    items: list[NewsItem] = []
    for entry in feed.entries:
        title = entry.get("title") or ""
        norm = _normalize_title(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        source_name, host = _source_name_and_host(entry)
        if not _is_trusted(source_name, host):
            continue
        published = entry.get("published_parsed")
        published_dt = datetime(*published[:6]) if published else None
        items.append(
            NewsItem(
                ticker=ticker.upper(),
                title=title,
                source=source_name or host or None,
                url=entry.get("link") or "",
                published_at=published_dt,
                snippet=(entry.get("summary") or "")[:500] or None,
            )
        )
    return items


# ---------------------------------------------------------------------------
# 1.3d -- Reddit public JSON
# ---------------------------------------------------------------------------

SUBREDDITS: tuple[str, ...] = (
    "investing",
    "stocks",
    "SecurityAnalysis",
    "options",
    "wallstreetbets",
)


def _reddit_headers() -> dict[str, str]:
    return {"User-Agent": settings.REDDIT_USER_AGENT}


def _reddit_get(url: str, max_retries: int = 3) -> dict | None:
    """GET a Reddit JSON endpoint with exponential backoff on 429.

    Returns ``None`` on 403/404 or persistent failure -- callers treat Reddit
    data as nice-to-have, not load-bearing.
    """
    for attempt in range(max_retries):
        _reddit_throttle.wait()
        r = requests.get(url, headers=_reddit_headers(), timeout=20)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        if r.status_code == 429:
            wait_s = float(r.headers.get("Retry-After", 2 ** attempt))
            time.sleep(min(wait_s, 30.0))
            continue
        if r.status_code in (403, 404):
            return None
        # Other 5xx etc -- back off and retry.
        time.sleep(2 ** attempt)
    return None


def fetch_reddit_recent(ticker: str, hours: int = 24) -> list[RedditPost]:
    """Posts mentioning ``ticker`` across ``SUBREDDITS`` within the last ``hours``."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    seen: set[str] = set()
    out: list[RedditPost] = []
    for sub in SUBREDDITS:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json?"
            f"q={quote_plus(ticker)}&restrict_sr=1&t=day&sort=new&limit=50"
        )
        payload = _reddit_get(url)
        if not payload:
            continue
        for child in payload.get("data", {}).get("children", []):
            d = child.get("data") or {}
            post_id = d.get("id")
            if not post_id or post_id in seen:
                continue
            created_utc = d.get("created_utc")
            if not created_utc:
                continue
            created_at = datetime.utcfromtimestamp(float(created_utc))
            if created_at < cutoff:
                continue
            seen.add(post_id)
            out.append(
                RedditPost(
                    ticker=ticker.upper(),
                    subreddit=sub,
                    post_id=post_id,
                    title=d.get("title"),
                    score=int(d.get("score") or 0),
                    num_comments=int(d.get("num_comments") or 0),
                    created_at=created_at,
                    url="https://www.reddit.com" + (d.get("permalink") or ""),
                )
            )
    return out


def build_reddit_snapshot(
    ticker: str,
    current_posts: list[RedditPost],
    window_hours: int = 24,
) -> RedditSnapshot:
    """Roll up current vs. prior window. Prior data comes from our SQLite buffer."""
    now = datetime.utcnow()
    prior_start = now - timedelta(hours=window_hours * 2)
    prior_end = now - timedelta(hours=window_hours)
    prior_posts = storage.get_reddit(ticker, since=prior_start, until=prior_end)
    prior_n = len(prior_posts)
    delta = (len(current_posts) - prior_n) / max(prior_n, 1)
    top = sorted(current_posts, key=lambda p: p.score, reverse=True)[:5]
    return RedditSnapshot(
        ticker=ticker.upper(),
        window_hours=window_hours,
        mention_count=len(current_posts),
        mention_count_prior_window=prior_n,
        mention_delta_pct=delta,
        top_posts=top,
    )
