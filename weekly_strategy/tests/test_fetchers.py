from __future__ import annotations

"""Unit tests for the four fetchers (yfinance, EDGAR, Google News, Reddit).

Every network call is monkeypatched so the suite runs offline. ``time.sleep``
is also stubbed so throttle delays don't slow the run.
"""

import importlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated storage + fresh fetchers module
# ---------------------------------------------------------------------------


@pytest.fixture
def fetchers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings_mod, "DOSSIER_DIR", tmp_path / "dossiers")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    for d in (
        settings_mod.DATA_CACHE_DIR,
        settings_mod.REPORTS_DIR,
        settings_mod.DOSSIER_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.data.storage as storage_mod
    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()

    import weekly_strategy.data.fetchers as fetchers_mod
    fetchers_mod = importlib.reload(fetchers_mod)

    # Make throttles instant.
    import time as time_mod
    monkeypatch.setattr(time_mod, "sleep", lambda *_a, **_k: None)

    return fetchers_mod


# ---------------------------------------------------------------------------
# 1.3a -- yfinance
# ---------------------------------------------------------------------------


class _FakeYfTicker:
    def __init__(self, df: pd.DataFrame, info: dict) -> None:
        self._df = df
        self.info = info

    def history(self, period: str, auto_adjust: bool):  # noqa: ARG002
        return self._df


def test_yfinance_history_writes_through_to_parquet(fetchers, monkeypatch):
    idx = pd.to_datetime(["2026-05-19", "2026-05-20", "2026-05-21"])
    raw = pd.DataFrame(
        {
            "Open": [180.0, 181.0, 182.0],
            "High": [182.0, 183.0, 184.0],
            "Low": [179.0, 180.0, 181.0],
            "Close": [181.5, 182.5, 183.5],
            "Adj Close": [181.0, 182.0, 183.0],
            "Volume": [50_000_000, 51_000_000, 52_000_000],
        },
        index=idx,
    )
    monkeypatch.setattr(
        fetchers.yf, "Ticker", lambda t: _FakeYfTicker(raw, {"sector": "Tech"})
    )
    df = fetchers.get_price_history("AAPL", lookback_days=30)
    assert list(df.index) == [date(2026, 5, 19), date(2026, 5, 20), date(2026, 5, 21)]
    assert df.loc[date(2026, 5, 21), "adj_close"] == 183.0

    # A second call (e.g., the next week) should merge, not duplicate.
    raw2 = pd.DataFrame(
        {
            "Open": [183.0],
            "High": [185.0],
            "Low": [182.5],
            "Close": [184.5],
            "Adj Close": [184.0],
            "Volume": [53_000_000],
        },
        index=pd.to_datetime(["2026-05-22"]),
    )
    monkeypatch.setattr(fetchers.yf, "Ticker", lambda t: _FakeYfTicker(raw2, {}))
    df2 = fetchers.get_price_history("AAPL")
    assert len(df2) == 4
    assert df2.loc[date(2026, 5, 22), "close"] == 184.5


def test_yfinance_basic_info_pulls_fields(fetchers, monkeypatch):
    info = {
        "shortName": "JPMorgan Chase & Co.",
        "sector": "Financial Services",
        "industry": "Banks - Diversified",
        "marketCap": 600_000_000_000,
        "beta": 1.12,
        "currency": "USD",
    }
    monkeypatch.setattr(
        fetchers.yf,
        "Ticker",
        lambda t: _FakeYfTicker(pd.DataFrame(), info),
    )
    out = fetchers.get_basic_info("JPM")
    assert out["ticker"] == "JPM"
    assert out["sector"] == "Financial Services"
    assert out["market_cap"] == 600_000_000_000
    assert out["beta"] == pytest.approx(1.12)


# ---------------------------------------------------------------------------
# 1.3b -- SEC EDGAR
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, data, status_code: int = 200, headers: dict | None = None):
        self._data = data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_edgar_cik_lookup_and_companyfacts_cache(fetchers, monkeypatch):
    ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMorgan Chase & Co."},
    }
    facts = {"cik": 320193, "facts": {"us-gaap": {"Revenues": {}}}}
    calls: list[str] = []

    def fake_get(url, headers=None, timeout=None):  # noqa: ARG001
        calls.append(url)
        if "company_tickers.json" in url:
            return _FakeResp(ticker_map)
        if "companyfacts/CIK0000320193.json" in url:
            return _FakeResp(facts)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(fetchers.requests, "get", fake_get)

    cik = fetchers.get_cik("aapl")
    assert cik == "0000320193"

    # Second lookup is cached in-memory -- no extra HTTP call.
    fetchers.get_cik("JPM")
    assert sum("company_tickers" in c for c in calls) == 1

    out = fetchers.get_company_facts(cik)
    assert out["cik"] == 320193

    # Companyfacts is disk-cached -- second call hits disk, no new HTTP.
    fetchers.get_company_facts(cik)
    assert sum("companyfacts" in c for c in calls) == 1


def test_edgar_recent_filings_filter_by_form_and_date(fetchers, monkeypatch):
    today = date.today()
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K", "8-K"],
                "filingDate": [
                    (today - timedelta(days=5)).isoformat(),    # 10-K, recent  -> keep
                    (today - timedelta(days=20)).isoformat(),   # 10-Q, recent  -> keep
                    (today - timedelta(days=200)).isoformat(),  # 8-K, too old -> drop
                    (today - timedelta(days=2)).isoformat(),    # 8-K, recent  -> keep
                ],
                "accessionNumber": ["a", "b", "c", "d"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
            }
        }
    }

    def fake_get(url, headers=None, timeout=None):  # noqa: ARG001
        return _FakeResp(submissions)

    monkeypatch.setattr(fetchers.requests, "get", fake_get)
    rows = fetchers.get_recent_filings("0000320193", ["10-K", "10-Q", "8-K"], days=90)
    forms = sorted(r["form"] for r in rows)
    assert forms == ["10-K", "10-Q", "8-K"]


def test_edgar_get_cik_raises_on_unknown(fetchers, monkeypatch):
    ticker_map = {"0": {"cik_str": 1, "ticker": "AAPL", "title": "Apple"}}

    def fake_get(url, headers=None, timeout=None):  # noqa: ARG001
        return _FakeResp(ticker_map)

    monkeypatch.setattr(fetchers.requests, "get", fake_get)
    with pytest.raises(ValueError):
        fetchers.get_cik("ZZZZ")


# ---------------------------------------------------------------------------
# 1.3c -- Google News RSS
# ---------------------------------------------------------------------------


def _entry(title: str, source_title: str, link: str, published: tuple | None = None):
    src = SimpleNamespace(title=source_title, href=link)
    e = SimpleNamespace(
        title=title,
        link=link,
        source=src,
        summary="snippet body",
        published_parsed=published,
    )
    # feedparser entries support .get() via dict-like access in real code;
    # our normalize uses both getattr and .get, so wrap minimally.
    e.get = lambda key, default=None: getattr(e, key, default)
    return e


def test_news_dedup_and_whitelist(fetchers, monkeypatch):
    entries = [
        _entry("Apple Q2 earnings beat",      "Reuters",            "https://reuters.com/a"),
        _entry("Apple Q2 earnings beat!!",    "Bloomberg",          "https://bloomberg.com/b"),  # dup title
        _entry("Random pump-and-dump blog",   "Joe's Stock Tips",   "https://joeblog.example/c"),  # untrusted
        _entry("Apple unveils new chip",      "Yahoo Finance",      "https://finance.yahoo.com/d"),
        _entry("",                             "Reuters",            "https://reuters.com/e"),    # empty title
    ]
    feed = SimpleNamespace(entries=entries)
    monkeypatch.setattr(fetchers.feedparser, "parse", lambda _url: feed)

    items = fetchers.fetch_news("AAPL", days=7)
    titles = [i.title for i in items]
    assert titles == ["Apple Q2 earnings beat", "Apple unveils new chip"]
    assert all(i.ticker == "AAPL" for i in items)


def test_news_query_override_used(fetchers, monkeypatch):
    captured: dict = {}

    def fake_parse(url):
        captured["url"] = url
        return SimpleNamespace(entries=[])

    monkeypatch.setattr(fetchers.feedparser, "parse", fake_parse)
    fetchers.fetch_news("META", query_override='"Meta Platforms" stock')
    assert "Meta+Platforms" in captured["url"] or "Meta%20Platforms" in captured["url"]


# ---------------------------------------------------------------------------
# 1.3d -- Reddit
# ---------------------------------------------------------------------------


def _reddit_listing(posts: list[dict]) -> dict:
    return {"data": {"children": [{"data": p} for p in posts]}}


def test_reddit_window_filter_and_dedup(fetchers, monkeypatch):
    now = datetime.utcnow()
    in_window_ts = (now - timedelta(hours=2)).timestamp()
    out_of_window_ts = (now - timedelta(hours=30)).timestamp()
    base_post = {
        "id": "p1",
        "title": "AAPL DD",
        "score": 100,
        "num_comments": 25,
        "created_utc": in_window_ts,
        "permalink": "/r/stocks/comments/p1/",
    }
    dup_post = dict(base_post, id="p1", title="AAPL DD edited")  # duplicate id
    old_post = dict(base_post, id="p2", created_utc=out_of_window_ts)
    fresh_post = dict(base_post, id="p3", score=200)

    by_url: dict[str, dict] = {}
    for sub, posts in (
        ("investing", [base_post]),
        ("stocks", [dup_post, old_post]),
        ("SecurityAnalysis", []),
        ("options", []),
        ("wallstreetbets", [fresh_post]),
    ):
        url_part = f"/r/{sub}/search.json"
        by_url[url_part] = _reddit_listing(posts)

    def fake_get(url, headers=None, timeout=None):  # noqa: ARG001
        for key, payload in by_url.items():
            if key in url:
                return _FakeResp(payload)
        return _FakeResp({}, status_code=404)

    monkeypatch.setattr(fetchers.requests, "get", fake_get)

    posts = fetchers.fetch_reddit_recent("AAPL", hours=24)
    ids = sorted(p.post_id for p in posts)
    assert ids == ["p1", "p3"]


def test_reddit_429_then_success(fetchers, monkeypatch):
    state = {"calls": 0}

    def fake_get(url, headers=None, timeout=None):  # noqa: ARG001
        state["calls"] += 1
        if state["calls"] == 1:
            return _FakeResp({}, status_code=429, headers={"Retry-After": "0"})
        return _FakeResp(_reddit_listing([]))

    monkeypatch.setattr(fetchers.requests, "get", fake_get)
    payload = fetchers._reddit_get("https://www.reddit.com/r/x/search.json?q=A")
    assert payload == {"data": {"children": []}}
    assert state["calls"] == 2


def test_reddit_snapshot_uses_prior_window_from_storage(fetchers, monkeypatch):
    from weekly_strategy.data.schemas import RedditPost

    # Prime storage with prior-window posts (24-48h ago).
    now = datetime.utcnow()
    prior = [
        RedditPost(
            ticker="AAPL",
            subreddit="stocks",
            post_id=f"old{i}",
            title=f"old {i}",
            score=10,
            num_comments=2,
            created_at=now - timedelta(hours=36),
            url="https://x",
        )
        for i in range(3)
    ]
    from weekly_strategy.data import storage
    storage.insert_reddit(prior)

    current = [
        RedditPost(
            ticker="AAPL",
            subreddit="stocks",
            post_id=f"new{i}",
            title=f"new {i}",
            score=50 + i,
            num_comments=5,
            created_at=now - timedelta(hours=2),
            url="https://x",
        )
        for i in range(6)
    ]
    snap = fetchers.build_reddit_snapshot("AAPL", current, window_hours=24)
    assert snap.mention_count == 6
    assert snap.mention_count_prior_window == 3
    assert snap.mention_delta_pct == pytest.approx(1.0)  # (6 - 3) / 3
    assert len(snap.top_posts) == 5
    assert snap.top_posts[0].score >= snap.top_posts[-1].score
