from __future__ import annotations

"""Tests for data/macro.py. FRED is mocked; tests run offline."""

import importlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def macro_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "FRED_API_KEY", "test-key")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.data.macro as mod
    mod = importlib.reload(mod)
    return mod


class _FakeFred:
    """Stand-in for fredapi.Fred. Returns canned series per id."""

    SERIES_DATA: dict[str, list[tuple[str, float]]] = {}

    def __init__(self, api_key: str):
        assert api_key == "test-key"

    def get_series(self, series_id, observation_start=None):
        data = self.SERIES_DATA.get(series_id, [])
        if not data:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([d for d, _ in data])
        vals = [v for _, v in data]
        return pd.Series(vals, index=idx, name=series_id)


@pytest.fixture
def fake_fred_module(monkeypatch):
    """Make `from fredapi import Fred` resolve to our fake."""
    import types
    fake_module = types.ModuleType("fredapi")
    fake_module.Fred = _FakeFred  # type: ignore[attr-defined]
    import sys
    monkeypatch.setitem(sys.modules, "fredapi", fake_module)
    return fake_module


def _set_series(sid: str, points: list[tuple[str, float]]):
    _FakeFred.SERIES_DATA[sid] = points


def _clear_fake_data():
    _FakeFred.SERIES_DATA.clear()


# ---------------------------------------------------------------------------
# get_fred_series
# ---------------------------------------------------------------------------


def test_get_fred_series_caches_to_parquet(macro_mod, fake_fred_module):
    _clear_fake_data()
    _set_series("DGS10", [("2026-05-15", 4.40), ("2026-05-22", 4.50)])
    s = macro_mod.get_fred_series("DGS10")
    assert list(s.values) == [4.40, 4.50]

    # Cache file exists; second call reads it (without hitting our fake).
    cache_file = macro_mod._cache_path("DGS10")
    assert cache_file.exists()

    # Mutate the fake to confirm the cached version is used.
    _set_series("DGS10", [("2026-05-22", 999.0)])
    cached = macro_mod.get_fred_series("DGS10")
    assert list(cached.values) == [4.40, 4.50]

    # force_refresh hits the network again.
    refreshed = macro_mod.get_fred_series("DGS10", force_refresh=True)
    assert list(refreshed.values) == [999.0]


def test_get_fred_series_raises_when_no_api_key(macro_mod, fake_fred_module, monkeypatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "FRED_API_KEY", None)
    import importlib
    mod = importlib.reload(macro_mod)
    with pytest.raises(mod.FredAuthError):
        mod.get_fred_series("DGS10", force_refresh=True)


# ---------------------------------------------------------------------------
# Regime classifiers
# ---------------------------------------------------------------------------


def test_classify_hy_buckets(macro_mod):
    assert macro_mod._classify_hy(None) == "unknown"
    assert macro_mod._classify_hy(300) == "tight"
    assert macro_mod._classify_hy(450) == "normal"
    assert macro_mod._classify_hy(600) == "wide"
    assert macro_mod._classify_hy(800) == "stressed"


def test_classify_vix_buckets(macro_mod):
    assert macro_mod._classify_vix(None) == "unknown"
    assert macro_mod._classify_vix(12) == "complacent"
    assert macro_mod._classify_vix(17) == "normal"
    assert macro_mod._classify_vix(25) == "elevated"
    assert macro_mod._classify_vix(40) == "panic"


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------


def test_get_macro_snapshot_builds_full_record(macro_mod, fake_fred_module):
    _clear_fake_data()
    today = date.today()
    last_week = (today - timedelta(days=7)).isoformat()
    today_s = today.isoformat()

    _set_series("DGS10",       [(last_week, 4.40), (today_s, 4.55)])     # +15 bps
    _set_series("DGS2",        [(last_week, 4.80), (today_s, 4.70)])     # -10 bps
    _set_series("T10Y2Y",      [(last_week, -0.30), (today_s, -0.15)])   # inverted
    _set_series("T10YIE",      [(last_week, 2.30), (today_s, 2.35)])
    _set_series("DFII10",      [(last_week, 2.05), (today_s, 2.10)])
    _set_series("DFF",         [(today_s, 5.25)])
    _set_series("MORTGAGE30US",[(today_s, 7.10)])
    _set_series("BAMLH0A0HYM2",[(last_week, 3.50), (today_s, 3.80)])     # 350bps -> 380bps
    _set_series("BAMLC0A0CM",  [(today_s, 1.10)])
    _set_series("VIXCLS",      [(last_week, 18.0), (today_s, 22.5)])     # +4.5
    _set_series("DTWEXBGS",    [(last_week, 102.0), (today_s, 103.02)])  # +1%

    # Drop a fresh CPI print in last 3 days for recent_prints coverage.
    _set_series(
        "CPIAUCSL",
        [
            ((today - timedelta(days=40)).isoformat(), 308.0),
            ((today - timedelta(days=2)).isoformat(), 309.0),
        ],
    )

    snap = macro_mod.get_macro_snapshot(week_ending=today)

    assert snap.yield_10y == pytest.approx(4.55)
    assert snap.yield_10y_wow_change_bps == pytest.approx(15.0, abs=0.001)
    assert snap.yield_2y == pytest.approx(4.70)
    assert snap.yield_2y_wow_change_bps == pytest.approx(-10.0, abs=0.001)
    assert snap.curve_2s10s_bps == pytest.approx(-15.0, abs=0.001)
    assert snap.curve_inverted is True
    assert snap.hy_oas_bps == pytest.approx(380.0)
    assert snap.hy_regime == "normal"  # 380 -> in [350, 500)
    assert snap.vix_level == pytest.approx(22.5)
    assert snap.vix_wow_change == pytest.approx(4.5)
    assert snap.vix_regime == "elevated"
    assert snap.breakeven_10y == pytest.approx(2.35)
    assert snap.dxy_wow_change_pct == pytest.approx(0.01, abs=1e-4)

    # Recent prints picked up the CPI release.
    cpi_print = [p for p in snap.recent_prints if p["series"] == "CPIAUCSL"]
    assert cpi_print
    assert cpi_print[0]["change"] == pytest.approx(1.0)


def test_get_macro_snapshot_degrades_when_series_missing(macro_mod, fake_fred_module):
    """If FRED returns empty for some series, snapshot still builds with None fields."""
    _clear_fake_data()
    _set_series("DGS10", [(date.today().isoformat(), 4.5)])  # only one series
    snap = macro_mod.get_macro_snapshot(week_ending=date.today())
    assert snap.yield_10y == pytest.approx(4.5)
    # Everything else falls back to None / unknown defaults.
    assert snap.vix_level is None
    assert snap.vix_regime == "unknown"
    assert snap.curve_2s10s_bps is None
    assert snap.curve_inverted is False  # default
    assert snap.hy_regime == "unknown"


def test_get_macro_snapshot_clears_cache_between_tests(macro_mod):
    """Sanity: cache file pattern is per-series, so a re-run with same tmp_path
    sees the prior series. We rely on the tmp_path fixture to isolate."""
    # tmp_path is fresh per test in pytest, so this is just an explicit check.
    cache_dir = macro_mod._MACRO_CACHE
    assert cache_dir.exists()
