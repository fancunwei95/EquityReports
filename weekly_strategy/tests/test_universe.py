from __future__ import annotations

"""Tests for dossier/universe.py construction logic."""

import importlib
import json
from datetime import date, datetime
from pathlib import Path

import pytest


@pytest.fixture
def universe_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import weekly_strategy.data.storage as storage_mod
    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()
    import weekly_strategy.dossier.universe as mod
    mod = importlib.reload(mod)
    return mod


def _meta(market_cap: float, sector: str, industry: str = "X",
          beta: float = 1.0) -> dict:
    return {
        "ticker": "?", "name": "?",
        "sector": sector, "industry": industry,
        "market_cap": market_cap, "beta": beta, "currency": "USD",
    }


# ---------------------------------------------------------------------------
# Pure construction
# ---------------------------------------------------------------------------


def test_construct_universe_respects_sector_quotas(universe_mod):
    # 4 names per sector, quota = 2 -> only top-2 by mcap make it through.
    meta = {}
    for sector_idx, sector in enumerate(["Technology", "Financial Services"]):
        for i in range(4):
            t = f"{'TECH' if sector_idx == 0 else 'FIN'}{i}"
            meta[t] = _meta(market_cap=(4 - i) * 10e9, sector=sector)

    u = universe_mod.construct_universe(
        quarter="2026Q2",
        candidates=list(meta),
        thematic_picks=[],
        sector_targets={"Technology": 2, "Financial Services": 2},
        min_market_cap=1_000_000_000,
        target_size=4,
        metadata=meta,
    )
    assert len(u.entries) == 4
    assert u.sector_counts["Technology"] == 2
    assert u.sector_counts["Financial Services"] == 2
    tech = [e for e in u.entries if e.sector == "Technology"]
    assert {e.ticker for e in tech} == {"TECH0", "TECH1"}  # top-2 by mcap


def test_construct_universe_applies_liquidity_floor(universe_mod):
    meta = {
        "BIG": _meta(market_cap=10e9, sector="Technology"),
        "TINY": _meta(market_cap=1e9, sector="Technology"),  # below $5B floor
    }
    u = universe_mod.construct_universe(
        quarter="2026Q2",
        candidates=["BIG", "TINY"],
        thematic_picks=[],
        sector_targets={"Technology": 5},
        min_market_cap=5_000_000_000,
        target_size=5,
        metadata=meta,
    )
    assert [e.ticker for e in u.entries] == ["BIG"]


def test_construct_universe_layers_thematic_picks(universe_mod):
    meta = {
        "AAA": _meta(market_cap=100e9, sector="Technology"),
        "BBB": _meta(market_cap=80e9, sector="Technology"),
        "NICHE": _meta(market_cap=2e9, sector="Healthcare"),  # below floor
    }
    u = universe_mod.construct_universe(
        quarter="2026Q2",
        candidates=["AAA", "BBB"],   # NICHE not in candidates
        thematic_picks=["NICHE"],
        sector_targets={"Technology": 5},
        min_market_cap=5_000_000_000,
        target_size=5,
        metadata=meta,
    )
    # NICHE made it via thematic despite failing liquidity + sector quota.
    tickers = {e.ticker for e in u.entries}
    assert tickers == {"AAA", "BBB", "NICHE"}
    niche = u.get("NICHE")
    assert niche.included_reason == "thematic"
    assert u.n_thematic_added == 1


def test_construct_universe_trims_to_target_keeping_thematic(universe_mod):
    """When the universe overshoots target_size, drop lowest-cap cap_rank picks
    first; protect thematic picks."""
    meta = {}
    for i in range(8):
        meta[f"BIG{i}"] = _meta(market_cap=(100 - i) * 1e9, sector="Technology")
    meta["NICHE"] = _meta(market_cap=2e9, sector="Healthcare")

    u = universe_mod.construct_universe(
        quarter="2026Q2",
        candidates=list(meta),
        thematic_picks=["NICHE"],
        sector_targets={"Technology": 8, "Healthcare": 8},
        min_market_cap=1_000_000_000,
        target_size=5,
        metadata=meta,
    )
    # 5 total; NICHE is thematic and is preserved despite being smallest cap.
    assert len(u.entries) == 5
    assert u.get("NICHE") is not None
    # Cap-rank picks should be the 4 largest BIGs.
    cap_picks = [e.ticker for e in u.entries if e.included_reason == "cap_rank"]
    assert set(cap_picks) == {"BIG0", "BIG1", "BIG2", "BIG3"}


def test_universe_get_and_tickers_helpers(universe_mod):
    meta = {"X": _meta(market_cap=10e9, sector="Technology")}
    u = universe_mod.construct_universe(
        quarter="2026Q2",
        candidates=["X"], thematic_picks=[],
        sector_targets={"Technology": 5}, min_market_cap=1e9,
        target_size=5, metadata=meta,
    )
    assert u.tickers == ["X"]
    assert u.get("x").ticker == "X"  # case-insensitive lookup
    assert u.get("MISSING") is None


def test_quarter_label():
    from weekly_strategy.dossier import universe
    assert universe._current_quarter_label(today=date(2026, 2, 14)) == "2026Q1"
    assert universe._current_quarter_label(today=date(2026, 5, 24)) == "2026Q2"
    assert universe._current_quarter_label(today=date(2026, 11, 30)) == "2026Q4"


# ---------------------------------------------------------------------------
# Storage roundtrip
# ---------------------------------------------------------------------------


def test_save_and_load_universe(universe_mod, tmp_path):
    from weekly_strategy.data import storage
    meta = {"X": _meta(market_cap=10e9, sector="Technology")}
    u = universe_mod.construct_universe(
        quarter="2026Q2", candidates=["X"], thematic_picks=[],
        sector_targets={"Technology": 5}, min_market_cap=1e9,
        target_size=5, metadata=meta,
    )
    path = storage.save_universe(u)
    assert path.exists()
    loaded = storage.load_universe("2026Q2")
    assert loaded is not None
    assert loaded.tickers == ["X"]
    # load_current_universe also works.
    cur = storage.load_current_universe()
    assert cur.quarter == "2026Q2"


def test_load_thematic_picks_from_default_path():
    from weekly_strategy.dossier import universe
    picks = universe.load_thematic_picks()
    # config/thematic_picks.json contains at least NVDA + LMT + TSLA.
    assert "NVDA" in picks
    assert "LMT" in picks
    assert "TSLA" in picks


def test_load_candidate_pool_dedupes():
    from weekly_strategy.dossier import universe
    pool = universe.load_candidate_pool()
    assert len(pool) == len(set(pool))
    # Sanity: a few well-known mega-caps are in the snapshot.
    for must in ("AAPL", "MSFT", "JPM", "XOM"):
        assert must in pool


# ---------------------------------------------------------------------------
# Step 3.2: batch dossier generation
# ---------------------------------------------------------------------------


def _u(tickers: list[str]):
    from datetime import datetime as _dt
    from weekly_strategy.data.schemas import Universe, UniverseEntry
    return Universe(
        quarter="2026Q2",
        constructed_at=_dt(2026, 5, 24),
        entries=[
            UniverseEntry(ticker=t, sector="Technology",
                          market_cap=10e9, included_reason="cap_rank")
            for t in tickers
        ],
    )


def test_batch_build_isolates_per_ticker_failures(universe_mod, monkeypatch):
    from weekly_strategy.data.schemas import Dossier
    from weekly_strategy.dossier import builder as bb

    def fake_build(ticker):
        if ticker == "BAD":
            raise RuntimeError("EDGAR ate it")
        return Dossier(ticker=ticker, last_updated=datetime(2026, 5, 24))

    monkeypatch.setattr(bb, "build_dossier", fake_build)
    monkeypatch.setattr(universe_mod.dossier_builder, "build_dossier", fake_build)
    res = universe_mod.batch_build_dossiers(
        _u(["AAPL", "BAD", "JPM"]), log=lambda *_a, **_kw: None,
    )
    assert res.succeeded == ["AAPL", "JPM"]
    assert res.failed == [("BAD", "RuntimeError: EDGAR ate it")]
    assert res.n_total == 3


def test_batch_build_records_warnings(universe_mod, monkeypatch):
    from weekly_strategy.data.schemas import Dossier

    def fake_build(ticker):
        # ROIC > 100% is a known XBRL denominator issue and should warn.
        return Dossier(ticker=ticker, last_updated=datetime(2026, 5, 24), roic=1.5)

    monkeypatch.setattr(universe_mod.dossier_builder, "build_dossier", fake_build)
    res = universe_mod.batch_build_dossiers(
        _u(["AAPL"]), log=lambda *_a, **_kw: None,
    )
    assert res.succeeded == ["AAPL"]
    assert any("ROIC > 100%" in w[1] for w in res.warnings)


def test_validate_dossier_flags_negative_revenue(universe_mod):
    from weekly_strategy.data.schemas import Dossier
    d = Dossier(ticker="X", last_updated=datetime(2026, 5, 24),
                revenue_latest_fy=-1_000_000_000)
    flags = universe_mod._validate_dossier(d)
    assert any("Negative revenue" in f for f in flags)


def test_validate_dossier_clean_when_sensible(universe_mod):
    from weekly_strategy.data.schemas import Dossier
    d = Dossier(
        ticker="AAPL", last_updated=datetime(2026, 5, 24),
        roic=0.55, gross_margin_current=0.46,
        operating_margin_current=0.31, revenue_latest_fy=400e9,
        market_cap=4.5e12, current_price=308.0,
    )
    assert universe_mod._validate_dossier(d) == []
