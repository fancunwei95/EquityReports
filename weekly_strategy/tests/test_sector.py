from __future__ import annotations

"""Tests for data/sector.py. Uses synthetic parquet caches; no network."""

import importlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sector_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.data.storage as storage_mod
    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()

    import weekly_strategy.data.sector as mod
    mod = importlib.reload(mod)
    return mod


def _prices_series(values: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    end = date(2026, 5, 22)
    idx = [end - timedelta(days=len(values) - 1 - i) for i in range(len(values))]
    df = pd.DataFrame({"close": values}, index=pd.to_datetime(idx))
    if volumes is not None:
        df["volume"] = volumes
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["adj_close"] = df["close"]
    return df


def _seed_prices(sector_mod, ticker: str, *, growth_per_day: float = 0.001,
                 n_days: int = 80, vol_ratio_late: float = 1.0):
    """Plant a parquet cache for ``ticker`` so load_prices(ticker) returns data."""
    from weekly_strategy.data import storage
    closes = [100.0 * (1.0 + growth_per_day) ** i for i in range(n_days)]
    vols = [1_000_000.0] * (n_days - 5) + [1_000_000.0 * vol_ratio_late] * 5
    storage.upsert_prices(ticker, _prices_series(closes, vols))


# ---------------------------------------------------------------------------
# Compute helpers
# ---------------------------------------------------------------------------


def test_trailing_returns_basic(sector_mod):
    closes = [100.0 + i * 0.5 for i in range(80)]
    df = _prices_series(closes)
    rets = sector_mod._trailing_returns(df)
    # 1m: closes[-1]/closes[-22] - 1
    assert rets["1m"] == pytest.approx(closes[-1] / closes[-22] - 1, abs=1e-6)
    assert rets["3m"] == pytest.approx(closes[-1] / closes[-64] - 1, abs=1e-6)
    assert rets["1w"] == pytest.approx(closes[-1] / closes[-6] - 1, abs=1e-6)


def test_volume_profile_picks_up_late_spike(sector_mod):
    closes = [100.0] * 30
    # Volumes flat 1M, then last 5 days at 2M (relative ratio 2.0)
    vols = [1_000_000.0] * 25 + [2_000_000.0] * 5
    profile = sector_mod._volume_profile(_prices_series(closes, vols))
    assert profile == pytest.approx(2.0)


def test_volume_profile_none_when_short(sector_mod):
    profile = sector_mod._volume_profile(_prices_series([100.0] * 10, [1.0] * 10))
    assert profile is None


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------


def test_get_sector_snapshot_full(sector_mod):
    # SPY grows 0.05%/day; XLF beats it; XLE underperforms.
    _seed_prices(sector_mod, sector_mod.BENCHMARK, growth_per_day=0.0005)
    fast_growers = {"XLF", "XLK", "XLC"}
    slow = {"XLE", "XLB", "XLU"}
    for etf in sector_mod.SECTOR_ETFS:
        if etf in fast_growers:
            _seed_prices(sector_mod, etf, growth_per_day=0.0015)
        elif etf in slow:
            _seed_prices(sector_mod, etf, growth_per_day=0.0000)
        else:
            _seed_prices(sector_mod, etf, growth_per_day=0.0005)

    snap = sector_mod.get_sector_snapshot(week_ending=date(2026, 5, 22))
    assert len(snap.sectors) == 11
    # All sectors have at least 1m/3m returns since we seeded 80 days.
    assert all(m.return_1m is not None for m in snap.sectors.values())
    # rel_1m should be positive for fast growers, negative for slow.
    for etf in fast_growers:
        assert snap.sectors[etf].rel_1m > 0
    for etf in slow:
        assert snap.sectors[etf].rel_1m < 0
    # Leadership ordering puts the fast growers first.
    top3 = set(snap.leadership_ranking[:3])
    assert top3 == fast_growers
    # Breadth = narrow when dispersion is high.
    assert snap.breadth in ("narrow", "broad")


def test_get_sector_snapshot_degrades_when_no_spy(sector_mod):
    """SPY missing -> rel_1m falls back to None on every sector."""
    _seed_prices(sector_mod, "XLF", growth_per_day=0.001)
    snap = sector_mod.get_sector_snapshot()
    assert "XLF" in snap.sectors
    assert snap.sectors["XLF"].return_1m is not None
    assert snap.sectors["XLF"].rel_1m is None
    # Leadership ranking is empty because no rel_1m values are present.
    assert snap.leadership_ranking == []


def test_metrics_for_sector_label_lookup(sector_mod):
    _seed_prices(sector_mod, sector_mod.BENCHMARK, growth_per_day=0.0005)
    _seed_prices(sector_mod, "XLF", growth_per_day=0.001)
    snap = sector_mod.get_sector_snapshot()
    m = snap.metrics_for_sector("Financials")
    assert m is not None
    assert m.etf == "XLF"
    # yfinance commonly reports "Financial Services" -- not in our snapshot keys,
    # but the orchestrator should map via SECTOR_TO_ETF first.
    assert sector_mod.SECTOR_TO_ETF["Financial Services"] == "XLF"
