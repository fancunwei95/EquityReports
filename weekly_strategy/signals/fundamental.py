from __future__ import annotations

"""Fundamental + momentum scoring from a Dossier and price history.

All scoring functions return 0-100 (higher = more attractive for a long).
Pure code, no LLM -- per the project's "code does math, LLM does language"
rule.

Stage 1 design choices:
* No cross-sectional ranking. We're only running 3 tickers; relative
  scoring needs the wider universe (Stage 3).
* No 5-year history percentile for valuation. We don't have it yet.
  Use absolute thresholds calibrated to "what looks rich vs cheap"
  in current US equity context.
* Missing fields (e.g., financials have no gross margin) contribute
  nothing -- the component is just skipped. The composite is the mean
  of present components, not zero-filled.
"""

from datetime import date
from statistics import mean
from typing import Optional

import pandas as pd

from weekly_strategy.data.schemas import Dossier, MacroRegime, StockScoreBundle


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _linear(value: float, zero_at: float, hundred_at: float) -> float:
    """Linear map: returns 0 when ``value == zero_at``, 100 when ``hundred_at``.

    Works whether ``hundred_at > zero_at`` (positive slope) or vice versa
    (negative slope -- useful for "cheaper is better" valuation fields).
    """
    if hundred_at == zero_at:
        return 50.0
    pct = (value - zero_at) / (hundred_at - zero_at)
    return _clamp(pct * 100.0)


# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------


def quality_score(dossier: Dossier) -> float:
    """0-100 quality composite. Mean of the present components.

    Anchors (calibrated to large-cap US):
      ROIC                : 0%  -> 0,    20% -> 100
      Operating margin    : 0%  -> 0,    30% -> 100
      FCF margin          : 0%  -> 0,    20% -> 100
      Net debt / EBITDA   : 3x  -> 0,    0x  -> 100  (inverted)
      Share count YoY     : +5% -> 0,   -5%  -> 100  (buybacks reward)
    """
    components: list[float] = []
    if dossier.roic is not None:
        components.append(_linear(dossier.roic, 0.0, 0.20))
    if dossier.operating_margin_current is not None:
        components.append(_linear(dossier.operating_margin_current, 0.0, 0.30))
    if dossier.fcf_margin_current is not None:
        components.append(_linear(dossier.fcf_margin_current, 0.0, 0.20))
    if dossier.net_debt_to_ebitda is not None:
        components.append(_linear(dossier.net_debt_to_ebitda, 3.0, 0.0))
    if dossier.share_count_change_yoy is not None:
        components.append(_linear(dossier.share_count_change_yoy, 0.05, -0.05))
    if not components:
        return 50.0
    return float(mean(components))


def valuation_score(dossier: Dossier) -> float:
    """0-100 valuation composite. Higher = cheaper.

    Anchors:
      FCF yield   : 2% -> 0,    8% -> 100
      P/E trail   : 35x-> 0,    10x-> 100
      P/S         : 10x-> 0,    1x -> 100
      EV/EBITDA   : 25x-> 0,    8x -> 100
    """
    components: list[float] = []
    if dossier.fcf_yield is not None:
        components.append(_linear(dossier.fcf_yield, 0.02, 0.08))
    if dossier.pe_trailing is not None and dossier.pe_trailing > 0:
        components.append(_linear(dossier.pe_trailing, 35.0, 10.0))
    if dossier.ps_ratio is not None and dossier.ps_ratio > 0:
        components.append(_linear(dossier.ps_ratio, 10.0, 1.0))
    if dossier.ev_to_ebitda is not None and dossier.ev_to_ebitda > 0:
        components.append(_linear(dossier.ev_to_ebitda, 25.0, 8.0))
    if not components:
        return 50.0
    return float(mean(components))


# ---------------------------------------------------------------------------
# Returns + momentum
# ---------------------------------------------------------------------------

_TRADING_DAYS = {"1m": 21, "3m": 63, "6m": 126}


def compute_returns(prices: pd.DataFrame) -> dict[str, Optional[float]]:
    """Trailing return over 1m / 3m / 6m. Returns None when not enough history."""
    out: dict[str, Optional[float]] = {f"return_{k}": None for k in _TRADING_DAYS}
    if prices is None or prices.empty or "close" not in prices.columns:
        return out
    closes = prices["close"].dropna().astype(float)
    if closes.empty:
        return out
    latest = closes.iloc[-1]
    for label, days in _TRADING_DAYS.items():
        if len(closes) > days:
            base = closes.iloc[-days - 1]
            if base > 0:
                out[f"return_{label}"] = float(latest / base - 1.0)
    return out


def momentum_score(prices: pd.DataFrame) -> float:
    """0-100 momentum composite from the three trailing returns.

    Composite = r_6m + r_3m - r_1m  (subtracting 1m introduces a mild
    short-term reversal -- stocks up sharply *this* month often give some
    back). Score is mapped from a +/-30% composite range to 0-100.
    """
    rets = compute_returns(prices)
    parts = [rets["return_3m"], rets["return_6m"]]
    if rets["return_1m"] is not None:
        parts.append(-rets["return_1m"])  # mean-reversion contribution
    present = [r for r in parts if r is not None]
    if not present:
        return 50.0
    composite = sum(present) / len(present)
    return _linear(composite, -0.10, 0.10)


# ---------------------------------------------------------------------------
# Bundle assembler
# ---------------------------------------------------------------------------


def build_score_bundle(
    *,
    ticker: str,
    week_ending: date,
    dossier: Dossier,
    prices: pd.DataFrame,
    news_sentiment_score: float | None = None,
    news_noise_ratio: float | None = None,
    reddit_sentiment: float | None = None,
    reddit_is_crowded: bool = False,
    # Stage 2 overlays. None preserves Stage 1 behavior.
    macro_regime: MacroRegime | None = None,
    sector_score_info: dict | None = None,
) -> StockScoreBundle:
    rets = compute_returns(prices)
    macro_score = macro_regime_score(macro_regime) if macro_regime is not None else None
    sector_score_value = sector_score_info["score"] if sector_score_info else None
    sector_etf = sector_score_info["etf"] if sector_score_info else None
    sector_rank = sector_score_info.get("rank") if sector_score_info else None
    sector_in_favor = bool(sector_score_info.get("in_favor")) if sector_score_info else False

    bundle = StockScoreBundle(
        ticker=ticker.upper(),
        week_ending=week_ending,
        quality_score=quality_score(dossier),
        valuation_score=valuation_score(dossier),
        momentum_score=momentum_score(prices),
        return_1m=rets["return_1m"],
        return_3m=rets["return_3m"],
        return_6m=rets["return_6m"],
        news_sentiment_score=news_sentiment_score,
        news_noise_ratio=news_noise_ratio,
        reddit_sentiment=reddit_sentiment,
        reddit_is_crowded=reddit_is_crowded,
        macro_regime_score=macro_score,
        sector_score_value=sector_score_value,
        sector_etf=sector_etf,
        sector_rank=sector_rank,
        sector_in_favor=sector_in_favor,
        composite_score=None,  # set below if we have enough inputs
    )
    composite = composite_score(bundle, macro_regime) if macro_regime is not None else None
    return bundle.model_copy(update={"composite_score": composite})


# ---------------------------------------------------------------------------
# Stage 2: macro regime score + composite
# ---------------------------------------------------------------------------


def macro_regime_score(regime: MacroRegime | None) -> float:
    """Universe-level 0-100 view: higher = more supportive of equities.

    Heuristic, not optimised. Backtest in Stage 5 should refine the deltas.
    """
    if regime is None:
        return 50.0
    score = 50.0
    fc = regime.financial_conditions
    if fc == "easing":
        score += 10
    elif fc == "tightening":
        score -= 12

    cp = regime.cycle_phase
    if cp == "early":
        score += 10
    elif cp == "late":
        score -= 5
    elif cp == "recession":
        score -= 20

    # rate_regime: cutting is supportive; tightening_market is a tax
    rr = regime.rate_regime
    if rr in ("cutting", "easing_market"):
        score += 5
    elif rr == "tightening_market":
        score -= 5

    return _clamp(score)


_RISK_OFF_WEIGHTS = {
    "quality":   0.30, "valuation": 0.25, "momentum":  0.10,
    "news":      0.15, "reddit":    0.05, "macro":     0.10, "sector": 0.05,
}
_RISK_ON_WEIGHTS = {
    "quality":   0.20, "valuation": 0.15, "momentum":  0.20,
    "news":      0.15, "reddit":    0.10, "macro":     0.10, "sector": 0.10,
}


def composite_score(
    bundle: StockScoreBundle, regime: MacroRegime | None,
) -> float:
    """Weighted 0-100. Risk-off shifts weight to quality + valuation."""
    risk_off = regime is not None and (
        regime.financial_conditions == "tightening"
        or regime.cycle_phase == "recession"
    )
    weights = _RISK_OFF_WEIGHTS if risk_off else _RISK_ON_WEIGHTS

    # Map [-1, +1] sentiment to [0, 100] for sentiment-style components.
    def sent_to_100(x: float | None) -> float:
        if x is None:
            return 50.0
        return _clamp(50.0 + x * 50.0)

    components = {
        "quality":   bundle.quality_score,
        "valuation": bundle.valuation_score,
        "momentum":  bundle.momentum_score,
        "news":      sent_to_100(bundle.news_sentiment_score),
        "reddit":    sent_to_100(bundle.reddit_sentiment),
        "macro":     bundle.macro_regime_score if bundle.macro_regime_score is not None else 50.0,
        "sector":    bundle.sector_score_value if bundle.sector_score_value is not None else 50.0,
    }
    return sum(weights[k] * components[k] for k in weights)
