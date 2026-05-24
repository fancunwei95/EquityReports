from __future__ import annotations

"""Roundtrip tests for the hybrid storage layer.

These tests redirect every storage path to a temp directory so the real
data_cache/, reports/, and dossier/data/ trees aren't touched.
"""

import importlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reload settings + storage with paths pointing at tmp_path."""
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

    # Re-import so module-level PRICES_DIR is recomputed against the patched settings.
    import weekly_strategy.data.storage as storage_mod

    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()
    return storage_mod


# ---------------------------------------------------------------------------
# Parquet prices
# ---------------------------------------------------------------------------


def _price_df(dates: list[date], close_start: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [close_start + i for i in range(len(dates))],
            "high": [close_start + i + 1 for i in range(len(dates))],
            "low": [close_start + i - 1 for i in range(len(dates))],
            "close": [close_start + i for i in range(len(dates))],
            "adj_close": [close_start + i for i in range(len(dates))],
            "volume": [1_000_000 + i for i in range(len(dates))],
        },
        index=pd.to_datetime(dates),
    )


def test_prices_upsert_appends_new_dates(storage):
    d0, d1, d2 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    storage.upsert_prices("AAPL", _price_df([d0, d1]))
    storage.upsert_prices("AAPL", _price_df([d2], close_start=200.0))

    out = storage.load_prices("AAPL")
    assert list(out.index) == [d0, d1, d2]
    assert out.loc[d2, "close"] == 200.0


def test_prices_upsert_overwrites_overlapping_dates(storage):
    d0, d1 = date(2026, 1, 5), date(2026, 1, 6)
    storage.upsert_prices("AAPL", _price_df([d0, d1], close_start=100.0))
    # Second write overwrites d1 with a different close.
    storage.upsert_prices("AAPL", _price_df([d1], close_start=999.0))

    out = storage.load_prices("AAPL")
    assert list(out.index) == [d0, d1]
    assert out.loc[d1, "close"] == 999.0


def test_prices_load_window(storage):
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]
    storage.upsert_prices("JPM", _price_df(days))
    out = storage.load_prices("JPM", since=date(2026, 1, 4), until=date(2026, 1, 7))
    assert list(out.index) == [date(2026, 1, 4), date(2026, 1, 5), date(2026, 1, 6)]


def test_prices_load_missing_returns_empty(storage):
    out = storage.load_prices("NOPE")
    assert out.empty


# ---------------------------------------------------------------------------
# SQLite -- news + reddit
# ---------------------------------------------------------------------------


def test_news_dedup_on_url(storage):
    from weekly_strategy.data.schemas import NewsItem

    items = [
        NewsItem(
            ticker="AAPL",
            title="A",
            source="src",
            url="https://x/1",
            published_at=datetime(2026, 5, 1, 12, 0),
        ),
        NewsItem(
            ticker="AAPL",
            title="A (dup)",
            source="src",
            url="https://x/1",
            published_at=datetime(2026, 5, 1, 12, 0),
        ),
        NewsItem(
            ticker="AAPL",
            title="B",
            source="src",
            url="https://x/2",
            published_at=datetime(2026, 5, 2, 12, 0),
        ),
    ]
    inserted = storage.insert_news(items)
    assert inserted == 2

    # Re-inserting the same set should be a no-op.
    assert storage.insert_news(items) == 0

    rows = storage.get_news("AAPL")
    assert len(rows) == 2
    assert {r.url for r in rows} == {"https://x/1", "https://x/2"}


def test_news_time_window(storage):
    from weekly_strategy.data.schemas import NewsItem

    items = [
        NewsItem(
            ticker="JPM",
            url=f"https://x/{i}",
            published_at=datetime(2026, 5, i + 1, 9, 0),
        )
        for i in range(5)
    ]
    storage.insert_news(items)
    rows = storage.get_news(
        "JPM",
        since=datetime(2026, 5, 2),
        until=datetime(2026, 5, 5),
    )
    # since inclusive, until exclusive -> days 2,3,4
    assert {r.url for r in rows} == {"https://x/1", "https://x/2", "https://x/3"}


def test_reddit_dedup_on_post_id(storage):
    from weekly_strategy.data.schemas import RedditPost

    items = [
        RedditPost(
            ticker="XOM",
            subreddit="stocks",
            post_id="abc",
            title="oil",
            score=10,
            num_comments=3,
            created_at=datetime(2026, 5, 1),
        ),
        RedditPost(
            ticker="XOM",
            subreddit="stocks",
            post_id="abc",  # duplicate
            title="oil (edited)",
            score=12,
            num_comments=4,
            created_at=datetime(2026, 5, 1),
        ),
    ]
    assert storage.insert_reddit(items) == 1
    rows = storage.get_reddit("XOM")
    assert len(rows) == 1
    # First write wins because INSERT OR IGNORE on conflict.
    assert rows[0].title == "oil"


# ---------------------------------------------------------------------------
# JSON -- dossiers + weekly reports
# ---------------------------------------------------------------------------


def test_dossier_roundtrip(storage):
    storage.save_dossier("AAPL", {"sector": "Tech", "fcf_margin": 0.27})
    blob = storage.load_dossier("AAPL")
    assert blob is not None
    assert blob["ticker"] == "AAPL"
    assert blob["data"]["sector"] == "Tech"


def test_dossier_missing_returns_none(storage):
    assert storage.load_dossier("MISSING") is None


def test_weekly_report_append_and_list(storage):
    storage.save_weekly_report("AAPL", date(2026, 5, 17), {"score": 1.2})
    storage.save_weekly_report("AAPL", date(2026, 5, 24), {"score": 1.4})
    paths = storage.list_weekly_reports("AAPL")
    assert [p.name for p in paths] == ["2026-05-17.json", "2026-05-24.json"]
