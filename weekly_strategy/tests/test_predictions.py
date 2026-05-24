from __future__ import annotations

"""Tests for data/predictions.py daily snapshot persistence."""

import importlib
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import weekly_strategy.data.storage as storage_mod
    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()
    import weekly_strategy.data.predictions as mod
    mod = importlib.reload(mod)
    return mod


def _bundle(ticker, **overrides):
    from weekly_strategy.data.schemas import StockScoreBundle
    base = dict(
        ticker=ticker, week_ending=date(2026, 5, 24),
        quality_score=70.0, valuation_score=40.0, momentum_score=80.0,
        composite_z=60.0,
    )
    base.update(overrides)
    return StockScoreBundle(**base)


def _portfolio(longs=("AAPL",), shorts=("XOM",)):
    from weekly_strategy.data.schemas import Portfolio, PortfolioPosition
    return Portfolio(
        week_ending=date(2026, 5, 24),
        longs=[
            PortfolioPosition(ticker=t, direction="long",
                              composite_z=60.0, sector="Technology", beta=1.0)
            for t in longs
        ],
        shorts=[
            PortfolioPosition(ticker=t, direction="short",
                              composite_z=30.0, sector="Energy", beta=0.5)
            for t in shorts
        ],
    )


def test_save_writes_bundles_and_portfolio(predictions):
    bundles = {"AAPL": _bundle("AAPL"), "XOM": _bundle("XOM", composite_z=30.0)}
    out = predictions.save_daily_snapshot(
        as_of=date(2026, 5, 24),
        bundles=bundles,
        portfolio=_portfolio(),
    )
    assert (out / "bundles.parquet").exists()
    assert (out / "portfolio.json").exists()
    df = pd.read_parquet(out / "bundles.parquet")
    assert set(df["ticker"]) == {"AAPL", "XOM"}
    assert df.loc[df["ticker"] == "AAPL", "composite_z"].iloc[0] == 60.0


def test_save_writes_price_snapshot_when_tickers_passed(predictions):
    from weekly_strategy.data import storage
    px = pd.DataFrame(
        {"close": [180.0, 182.0], "adj_close": [180.0, 182.0]},
        index=pd.to_datetime(["2026-05-20", "2026-05-21"]),
    )
    storage.upsert_prices("AAPL", px)
    out = predictions.save_daily_snapshot(
        as_of=date(2026, 5, 24),
        bundles={"AAPL": _bundle("AAPL")},
        portfolio=_portfolio(),
        universe_tickers=["AAPL"],
    )
    pf = pd.read_parquet(out / "prices.parquet")
    assert pf.iloc[0]["close"] == 182.0  # latest close


def test_load_history_concats_all_snapshots(predictions):
    bundles_a = {"AAPL": _bundle("AAPL", composite_z=60.0)}
    bundles_b = {"AAPL": _bundle("AAPL", composite_z=70.0)}
    predictions.save_daily_snapshot(
        as_of=date(2026, 5, 24), bundles=bundles_a, portfolio=_portfolio(),
    )
    predictions.save_daily_snapshot(
        as_of=date(2026, 5, 25), bundles=bundles_b, portfolio=_portfolio(),
    )
    hist = predictions.load_snapshot_history()
    assert len(hist) == 2
    assert sorted(hist["as_of"].tolist()) == ["2026-05-24", "2026-05-25"]


def test_load_previous_book_returns_most_recent_prior(predictions):
    predictions.save_daily_snapshot(
        as_of=date(2026, 5, 23),
        bundles={"OLD": _bundle("OLD")},
        portfolio=_portfolio(longs=("OLDLONG",), shorts=("OLDSHORT",)),
    )
    predictions.save_daily_snapshot(
        as_of=date(2026, 5, 24),
        bundles={"NEW": _bundle("NEW")},
        portfolio=_portfolio(longs=("NEWLONG",), shorts=("NEWSHORT",)),
    )
    prev_long, prev_short = predictions.load_previous_book(date(2026, 5, 25))
    # Most-recent prior to 5/25 is 5/24.
    assert prev_long == {"NEWLONG"}
    assert prev_short == {"NEWSHORT"}


def test_load_previous_book_empty_when_no_history(predictions):
    prev_long, prev_short = predictions.load_previous_book(date(2026, 5, 24))
    assert prev_long == set()
    assert prev_short == set()


def test_list_snapshot_dates_orders_chronologically(predictions):
    for d in [date(2026, 5, 24), date(2026, 5, 22), date(2026, 5, 23)]:
        predictions.save_daily_snapshot(
            as_of=d, bundles={"X": _bundle("X")}, portfolio=_portfolio(),
        )
    assert predictions.list_snapshot_dates() == [
        date(2026, 5, 22), date(2026, 5, 23), date(2026, 5, 24),
    ]
