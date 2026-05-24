from __future__ import annotations

"""Tests for signals/fundamental.py. Pure math; no LLM, no I/O."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from weekly_strategy.data.schemas import Dossier
from weekly_strategy.signals import fundamental as fnd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dossier(**overrides) -> Dossier:
    defaults = dict(
        ticker="X",
        last_updated=datetime.utcnow(),
    )
    defaults.update(overrides)
    return Dossier(**defaults)


def _prices(values: list[float], end: date | None = None) -> pd.DataFrame:
    end = end or date(2026, 5, 22)
    idx = [end - timedelta(days=i) for i in range(len(values) - 1, -1, -1)]
    return pd.DataFrame({"close": values}, index=pd.to_datetime(idx))


# ---------------------------------------------------------------------------
# linear helper
# ---------------------------------------------------------------------------


def test_linear_positive_slope():
    assert fnd._linear(0.0, 0.0, 0.20) == 0.0
    assert fnd._linear(0.10, 0.0, 0.20) == pytest.approx(50.0)
    assert fnd._linear(0.20, 0.0, 0.20) == 100.0
    assert fnd._linear(0.50, 0.0, 0.20) == 100.0  # clamped
    assert fnd._linear(-0.10, 0.0, 0.20) == 0.0   # clamped


def test_linear_negative_slope_for_cheaper_is_better():
    # P/E: 35 -> 0, 10 -> 100
    assert fnd._linear(35.0, 35.0, 10.0) == 0.0
    assert fnd._linear(22.5, 35.0, 10.0) == pytest.approx(50.0)
    assert fnd._linear(10.0, 35.0, 10.0) == 100.0


# ---------------------------------------------------------------------------
# quality_score
# ---------------------------------------------------------------------------


def test_quality_high_quality_tech_co():
    d = _dossier(
        roic=0.20,                       # 100
        operating_margin_current=0.30,   # 100
        fcf_margin_current=0.20,         # 100
        net_debt_to_ebitda=0.0,          # 100
        share_count_change_yoy=-0.05,    # 100 (buyback)
    )
    assert fnd.quality_score(d) == pytest.approx(100.0)


def test_quality_low_quality_high_leverage_co():
    d = _dossier(
        roic=0.05,
        operating_margin_current=0.05,
        fcf_margin_current=0.02,
        net_debt_to_ebitda=3.0,          # 0
        share_count_change_yoy=0.05,     # 0 (dilution)
    )
    # roic=25, opm=16.7, fcf=10, ndebt=0, share=0 -> mean ~10.3
    assert 5 < fnd.quality_score(d) < 20


def test_quality_skips_missing_components():
    d_full = _dossier(roic=0.10, operating_margin_current=0.15)  # mean of 50, 50 = 50
    assert fnd.quality_score(d_full) == pytest.approx(50.0)
    d_empty = _dossier()
    assert fnd.quality_score(d_empty) == 50.0  # neutral when nothing's present


# ---------------------------------------------------------------------------
# valuation_score
# ---------------------------------------------------------------------------


def test_valuation_cheap_stock():
    d = _dossier(
        fcf_yield=0.08,        # 100
        pe_trailing=10.0,      # 100
        ps_ratio=1.0,          # 100
        ev_to_ebitda=8.0,      # 100
    )
    assert fnd.valuation_score(d) == pytest.approx(100.0)


def test_valuation_expensive_stock():
    d = _dossier(
        fcf_yield=0.02,        # 0
        pe_trailing=40.0,      # 0 (clamped, past 35)
        ps_ratio=15.0,         # 0 (clamped)
        ev_to_ebitda=30.0,     # 0 (clamped)
    )
    assert fnd.valuation_score(d) == pytest.approx(0.0)


def test_valuation_skips_negative_pe():
    # Negative P/E (loss-making) -> not a sane "cheaper" comp; we skip.
    d = _dossier(fcf_yield=0.05, pe_trailing=-15.0)
    # fcf_yield=0.05 -> _linear(0.05, 0.02, 0.08) = 50
    # pe skipped -> mean of [50] = 50
    assert fnd.valuation_score(d) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# returns + momentum
# ---------------------------------------------------------------------------


def test_compute_returns_trailing_windows():
    # 130 days of closes climbing 100 -> 130
    closes = [100 + i * 0.25 for i in range(130)]
    px = _prices(closes)
    out = fnd.compute_returns(px)
    # 1m return: latest / -22nd vs end
    assert out["return_1m"] == pytest.approx((closes[-1] / closes[-22] - 1), abs=1e-6)
    assert out["return_3m"] == pytest.approx((closes[-1] / closes[-64] - 1), abs=1e-6)
    assert out["return_6m"] == pytest.approx((closes[-1] / closes[-127] - 1), abs=1e-6)


def test_compute_returns_short_history_returns_none():
    px = _prices([100, 101, 102])  # < any window
    out = fnd.compute_returns(px)
    assert out["return_1m"] is None
    assert out["return_3m"] is None
    assert out["return_6m"] is None


def test_momentum_score_uptrend():
    # 130 days, steadily up 30%
    closes = [100 + i * 0.25 for i in range(130)]
    score = fnd.momentum_score(_prices(closes))
    assert 80 < score <= 100  # strong uptrend should be high


def test_momentum_score_downtrend():
    closes = [130 - i * 0.25 for i in range(130)]
    score = fnd.momentum_score(_prices(closes))
    assert 0 <= score < 20


def test_momentum_score_short_history_neutral():
    score = fnd.momentum_score(_prices([100, 101]))
    assert score == 50.0


# ---------------------------------------------------------------------------
# build_score_bundle integration
# ---------------------------------------------------------------------------


def _regime(financial_conditions="easing", cycle_phase="mid",
            rate_regime="cutting"):
    from weekly_strategy.data.schemas import MacroRegime
    return MacroRegime(
        week_ending=date(2026, 5, 24),
        rate_regime=rate_regime,
        financial_conditions=financial_conditions,
        cycle_phase=cycle_phase,
        narrative="x",
    )


# ---------------------------------------------------------------------------
# Stage 2: macro_regime_score
# ---------------------------------------------------------------------------


def test_macro_regime_score_easing_mid_cycle_positive():
    r = _regime(financial_conditions="easing", cycle_phase="mid",
                rate_regime="cutting")
    assert fnd.macro_regime_score(r) > 50  # easing + cutting bumps it up


def test_macro_regime_score_recession_low():
    r = _regime(financial_conditions="tightening", cycle_phase="recession")
    assert fnd.macro_regime_score(r) < 30


def test_macro_regime_score_none_is_neutral():
    assert fnd.macro_regime_score(None) == 50.0


# ---------------------------------------------------------------------------
# Stage 2: composite_score
# ---------------------------------------------------------------------------


def _bundle(**overrides):
    from weekly_strategy.data.schemas import StockScoreBundle
    base = dict(
        ticker="X", week_ending=date(2026, 5, 24),
        quality_score=80, valuation_score=40, momentum_score=70,
        news_sentiment_score=0.4, reddit_sentiment=0.0,
        macro_regime_score=60, sector_score_value=70,
    )
    base.update(overrides)
    return StockScoreBundle(**base)


def test_composite_uses_risk_on_weights_in_easing():
    b = _bundle()
    r = _regime(financial_conditions="easing", cycle_phase="mid")
    c = fnd.composite_score(b, r)
    # Hand-check: 0.20*80 + 0.15*40 + 0.20*70 + 0.15*70 + 0.10*50 + 0.10*60 + 0.10*70
    # = 16 + 6 + 14 + 10.5 + 5 + 6 + 7 = 64.5
    assert c == pytest.approx(64.5, abs=0.5)


def test_composite_uses_risk_off_weights_in_tightening():
    b = _bundle()
    r = _regime(financial_conditions="tightening", cycle_phase="late")
    c = fnd.composite_score(b, r)
    # 0.30*80 + 0.25*40 + 0.10*70 + 0.15*70 + 0.05*50 + 0.10*60 + 0.05*70
    # = 24 + 10 + 7 + 10.5 + 2.5 + 6 + 3.5 = 63.5
    assert c == pytest.approx(63.5, abs=0.5)


def test_composite_includes_sector_in_score():
    high_sector = _bundle(sector_score_value=100)
    low_sector = _bundle(sector_score_value=0)
    r = _regime()
    assert fnd.composite_score(high_sector, r) > fnd.composite_score(low_sector, r)


def test_build_score_bundle_populates_stage2_fields():
    closes = [100 + i * 0.2 for i in range(130)]
    px = _prices(closes)
    d = _dossier(roic=0.15, operating_margin_current=0.20, fcf_yield=0.05)
    r = _regime(financial_conditions="easing", cycle_phase="mid")
    sector_info = {
        "score": 75.0, "etf": "XLK", "sector_name": "Technology",
        "rank": 2, "leaders_out_of": 11, "regime_fit": 70.0,
        "momentum_component": 80.0, "rel_1m": 0.08, "in_favor": True,
    }
    bundle = fnd.build_score_bundle(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=d, prices=px,
        news_sentiment_score=0.4,
        macro_regime=r, sector_score_info=sector_info,
    )
    assert bundle.macro_regime_score is not None
    assert bundle.sector_score_value == 75.0
    assert bundle.sector_etf == "XLK"
    assert bundle.sector_rank == 2
    assert bundle.sector_in_favor is True
    assert bundle.composite_score is not None
    assert 0 <= bundle.composite_score <= 100


def test_build_score_bundle_carries_through_external_signals():
    d = _dossier(roic=0.15, operating_margin_current=0.20, fcf_yield=0.05)
    closes = [100 + i * 0.2 for i in range(130)]
    bundle = fnd.build_score_bundle(
        ticker="AAPL",
        week_ending=date(2026, 5, 24),
        dossier=d,
        prices=_prices(closes),
        news_sentiment_score=0.42,
        news_noise_ratio=0.20,
        reddit_sentiment=0.10,
        reddit_is_crowded=True,
    )
    assert bundle.ticker == "AAPL"
    assert bundle.week_ending == date(2026, 5, 24)
    assert 0 <= bundle.quality_score <= 100
    assert 0 <= bundle.valuation_score <= 100
    assert 0 <= bundle.momentum_score <= 100
    assert bundle.news_sentiment_score == pytest.approx(0.42)
    assert bundle.reddit_is_crowded is True
    assert bundle.return_6m is not None
