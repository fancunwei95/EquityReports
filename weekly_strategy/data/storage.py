from __future__ import annotations

"""Hybrid storage layer.

Three backends, picked per access pattern (see plan.md Step 1.2):

* Parquet  -- prices (one file per ticker, append-mostly OHLCV).
* SQLite   -- news_items + reddit_posts (UNIQUE-on-insert dedup, time-window queries).
* JSON     -- dossiers + weekly reports (one blob per ticker / per (ticker, date)).
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from weekly_strategy.config import settings
from weekly_strategy.data.schemas import NewsItem, RedditPost

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PRICES_DIR = settings.DATA_CACHE_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)


def _price_path(ticker: str) -> Path:
    return PRICES_DIR / f"{ticker.upper()}.parquet"


def _dossier_path(ticker: str) -> Path:
    return settings.DOSSIER_DIR / f"{ticker.upper()}.json"


def _report_dir(ticker: str) -> Path:
    d = settings.REPORTS_DIR / ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# SQLite -- news + reddit only
# ---------------------------------------------------------------------------

_SCHEMA = """
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
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    # No PARSE_DECLTYPES: we serialize timestamps as ISO strings ourselves
    # via _iso / _parse_dt. The stdlib TIMESTAMP converter expects a space
    # separator ("YYYY-MM-DD HH:MM:SS") and chokes on isoformat's "T".
    conn = sqlite3.connect(settings.SQLITE_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and indices if they don't exist. Idempotent."""
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def insert_news(items: Iterable[NewsItem]) -> int:
    """INSERT OR IGNORE -- duplicate urls are silently skipped. Returns # newly inserted."""
    rows = [
        (n.ticker, n.title, n.source, n.url, _iso(n.published_at), n.snippet)
        for n in items
    ]
    if not rows:
        return 0
    with _conn() as conn:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO news_items "
            "(ticker, title, source, url, published_at, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        return cur.rowcount


def insert_reddit(items: Iterable[RedditPost]) -> int:
    rows = [
        (
            p.ticker,
            p.subreddit,
            p.post_id,
            p.title,
            p.score,
            p.num_comments,
            _iso(p.created_at),
            p.url,
        )
        for p in items
    ]
    if not rows:
        return 0
    with _conn() as conn:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO reddit_posts "
            "(ticker, subreddit, post_id, title, score, num_comments, created_at, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return cur.rowcount


def _parse_dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def get_news(
    ticker: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[NewsItem]:
    sql = "SELECT ticker, title, source, url, published_at, snippet FROM news_items WHERE ticker = ?"
    params: list[object] = [ticker.upper()]
    if since is not None:
        sql += " AND published_at >= ?"
        params.append(since.isoformat())
    if until is not None:
        sql += " AND published_at < ?"
        params.append(until.isoformat())
    sql += " ORDER BY published_at DESC"
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        NewsItem(
            ticker=r["ticker"],
            title=r["title"],
            source=r["source"],
            url=r["url"],
            published_at=_parse_dt(r["published_at"]),
            snippet=r["snippet"],
        )
        for r in rows
    ]


def get_reddit(
    ticker: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[RedditPost]:
    sql = (
        "SELECT ticker, subreddit, post_id, title, score, num_comments, created_at, url "
        "FROM reddit_posts WHERE ticker = ?"
    )
    params: list[object] = [ticker.upper()]
    if since is not None:
        sql += " AND created_at >= ?"
        params.append(since.isoformat())
    if until is not None:
        sql += " AND created_at < ?"
        params.append(until.isoformat())
    sql += " ORDER BY created_at DESC"
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        RedditPost(
            ticker=r["ticker"],
            subreddit=r["subreddit"],
            post_id=r["post_id"],
            title=r["title"],
            score=r["score"] or 0,
            num_comments=r["num_comments"] or 0,
            created_at=_parse_dt(r["created_at"]),
            url=r["url"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Parquet -- prices
# ---------------------------------------------------------------------------

_PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    """Merge ``df`` into the per-ticker parquet, keyed by date.

    ``df`` must have a DatetimeIndex (or a ``date`` column) and at least one of
    the OHLCV columns. New dates are appended; overlapping dates are overwritten
    by the new row. Returns the number of rows in the merged file.
    """
    if df.empty:
        return _row_count(ticker)

    incoming = _normalize_price_df(df)
    path = _price_path(ticker)
    if path.exists():
        existing = _read_prices_file(path)
        # Drop rows in existing whose date appears in incoming; then concat.
        existing = existing.loc[~existing.index.isin(incoming.index)]
        merged = pd.concat([existing, incoming]).sort_index()
    else:
        merged = incoming.sort_index()

    # Reset to a column so the parquet stays explicit about the date.
    out = merged.reset_index()
    out.to_parquet(path, index=False)
    return len(merged)


def _read_prices_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index).date
    df.index.name = "date"
    return df


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index).date
    out.index.name = "date"
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    # Keep only known price columns that are present.
    keep = [c for c in _PRICE_COLS if c in out.columns]
    return out[keep]


def load_prices(
    ticker: str,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    path = _price_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=_PRICE_COLS)
    df = _read_prices_file(path).sort_index()
    if since is not None:
        df = df.loc[df.index >= since]
    if until is not None:
        df = df.loc[df.index < until]
    return df


def _row_count(ticker: str) -> int:
    path = _price_path(ticker)
    if not path.exists():
        return 0
    return len(pd.read_parquet(path))


# ---------------------------------------------------------------------------
# JSON -- dossiers + weekly reports
# ---------------------------------------------------------------------------


def save_dossier(ticker: str, data: dict) -> Path:
    path = _dossier_path(ticker)
    payload = {
        "ticker": ticker.upper(),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "data": data,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_dossier(ticker: str) -> dict | None:
    path = _dossier_path(ticker)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_weekly_report(ticker: str, report_date: date, data: dict) -> Path:
    path = _report_dir(ticker) / f"{report_date.isoformat()}.json"
    payload = {
        "ticker": ticker.upper(),
        "report_date": report_date.isoformat(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "data": data,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def list_weekly_reports(ticker: str) -> list[Path]:
    d = _report_dir(ticker)
    return sorted(d.glob("*.json"))
