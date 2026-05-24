from __future__ import annotations

"""Tests for signals/batch_scoring.py.

batch_weekly_scoring is integration-shaped (network + storage + LLM cascade);
those paths are tested in the per-ticker modules. Here we focus on:
* per-ticker error isolation in the batch loop
* normalize_scores math (z-scores + composite_z)
"""

import importlib
import statistics
from datetime import date, datetime
from pathlib import Path

import pytest


@pytest.fixture
def mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import weekly_strategy.data.storage as storage_mod
    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()
    import weekly_strategy.signals.batch_scoring as m
    m = importlib.reload(m)
    return m


def _bundle(ticker: str, **overrides):
    from weekly_strategy.data.schemas import StockScoreBundle
    base = dict(
        ticker=ticker, week_ending=date(2026, 5, 24),
        quality_score=50.0, valuation_score=50.0, momentum_score=50.0,
        news_sentiment_score=0.0, sector_score_value=50.0,
    )
    base.update(overrides)
    return StockScoreBundle(**base)


# ---------------------------------------------------------------------------
# batch_weekly_scoring: error isolation
# ---------------------------------------------------------------------------


def test_batch_weekly_scoring_isolates_failures(mod, monkeypatch):
    """Mock _score_one so we can inject controlled failures."""
    def fake_score_one(*, ticker, **kw):
        if ticker == "BAD":
            raise RuntimeError("storage gone")
        return _bundle(ticker, quality_score=80.0)

    monkeypatch.setattr(mod, "_score_one", fake_score_one)

    from weekly_strategy.data.schemas import Universe, UniverseEntry
    universe = Universe(
        quarter="2026Q2", constructed_at=datetime(2026, 5, 24),
        entries=[UniverseEntry(ticker=t) for t in ("AAPL", "BAD", "JPM")],
    )
    res = mod.batch_weekly_scoring(
        universe, week_ending=date(2026, 5, 24),
        log=lambda *_a, **_kw: None,
    )
    assert set(res.bundles) == {"AAPL", "JPM"}
    assert res.failed == [("BAD", "RuntimeError: storage gone")]


# ---------------------------------------------------------------------------
# normalize_scores: z-score math + composite_z
# ---------------------------------------------------------------------------


def test_normalize_scores_zero_mean_unit_std(mod):
    """After normalization, each component should have z-scores that average
    to ~0 across the universe with pstdev ~1."""
    bundles = {
        "A": _bundle("A", quality_score=30.0),
        "B": _bundle("B", quality_score=50.0),
        "C": _bundle("C", quality_score=70.0),
    }
    out = mod.normalize_scores(bundles)
    zs = [out[t].z_quality for t in ("A", "B", "C")]
    # Mean of zs should be ~0 (small floating tolerance).
    assert sum(zs) == pytest.approx(0.0, abs=1e-9)
    # Population stdev of z-scores is 1 by construction.
    assert statistics.pstdev(zs) == pytest.approx(1.0, abs=1e-9)
    # A is the worst quality (z < 0); C is the best (z > 0).
    assert out["A"].z_quality < 0 < out["C"].z_quality


def test_normalize_scores_handles_zero_stdev(mod):
    """If every stock has the same value for a component, z = 0 across."""
    bundles = {
        "A": _bundle("A", quality_score=50.0),
        "B": _bundle("B", quality_score=50.0),
    }
    out = mod.normalize_scores(bundles)
    assert out["A"].z_quality == 0.0
    assert out["B"].z_quality == 0.0


def test_normalize_scores_missing_value_falls_to_zero(mod):
    """A ticker missing a raw component shouldn't drag the normalization stats,
    and gets z=0 for that component (neutral)."""
    bundles = {
        "A": _bundle("A", news_sentiment_score=None),
        "B": _bundle("B", news_sentiment_score=0.6),
        "C": _bundle("C", news_sentiment_score=-0.4),
    }
    out = mod.normalize_scores(bundles)
    # B is positive on news vs the mean of {0.6, -0.4} = 0.1, so z > 0.
    assert out["B"].z_news_sentiment > 0
    assert out["C"].z_news_sentiment < 0
    # A had no news_sentiment, so it gets z=0 (neutral).
    assert out["A"].z_news_sentiment == 0.0


def test_composite_z_maps_to_0_100(mod):
    """Composite z=0 -> 50; z=+2 -> 100; z=-2 -> 0 (clamped beyond)."""
    # Build a single-bundle universe where everything is mean.
    bundles = {"A": _bundle("A")}
    out = mod.normalize_scores(bundles)
    assert out["A"].composite_z == pytest.approx(50.0)

    # Construct a universe where A is +2 std on every component.
    bundles2 = {
        "A": _bundle("A", quality_score=70, valuation_score=70, momentum_score=70,
                     news_sentiment_score=1.0, sector_score_value=70),
        "B": _bundle("B", quality_score=50, valuation_score=50, momentum_score=50,
                     news_sentiment_score=0.0, sector_score_value=50),
        "C": _bundle("C", quality_score=30, valuation_score=30, momentum_score=30,
                     news_sentiment_score=-1.0, sector_score_value=30),
    }
    out2 = mod.normalize_scores(bundles2)
    # For three evenly-spaced values around the mean, pstdev = sqrt(2/3)*step,
    # so the high-end z-score is sqrt(3/2) ~= 1.2247.
    # All weights sum to 1, so weighted_z = 1.2247 and
    # composite_z = 50 + 25*1.2247 = ~80.62.
    expected_high = 50.0 + 25.0 * (1.5 ** 0.5)
    assert out2["A"].composite_z == pytest.approx(expected_high, abs=1e-6)
    assert out2["B"].composite_z == pytest.approx(50.0)
    assert out2["C"].composite_z == pytest.approx(100.0 - expected_high, abs=1e-6)


def test_composite_z_uses_risk_off_weights_in_recession(mod):
    """In recession, valuation and quality carry more weight."""
    from weekly_strategy.data.schemas import MacroRegime
    regime = MacroRegime(
        week_ending=date(2026, 5, 24),
        rate_regime="cutting", financial_conditions="tightening",
        cycle_phase="recession",
    )
    # Build a universe where exactly one bundle has high quality and high val.
    bundles = {
        "A": _bundle("A", quality_score=90, valuation_score=90,
                     momentum_score=10, sector_score_value=10,
                     news_sentiment_score=-0.5),
        "B": _bundle("B", quality_score=50, valuation_score=50,
                     momentum_score=50, sector_score_value=50,
                     news_sentiment_score=0.0),
        "C": _bundle("C", quality_score=30, valuation_score=30,
                     momentum_score=80, sector_score_value=80,
                     news_sentiment_score=0.5),
    }
    risk_on_out = mod.normalize_scores(bundles, regime=None)
    risk_off_out = mod.normalize_scores(bundles, regime=regime)
    # In risk-off, A's quality+valuation weight more, so its composite_z
    # is higher than in risk-on.
    assert risk_off_out["A"].composite_z > risk_on_out["A"].composite_z