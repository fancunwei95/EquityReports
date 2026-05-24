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
