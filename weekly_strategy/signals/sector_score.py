from __future__ import annotations

"""Sector scoring + cross-stock LLM (Step 2.5).

Two responsibilities, both consumed by the Stage 2 score bundle and
thesis writer:

1. ``sector_score`` -- deterministic 0-100 from a ticker's sector ETF
   leadership rank + macro-regime fit. Plan.md's "rules-based scorecard
   for now, tune via backtest later."

2. ``cross_stock_implications`` -- one Sonnet call (the deferred "agent
   3" from the news cascade discussion in Step 1.5). Reads news, sector,
   regime; returns implications both directions plus related tickers.
"""

from datetime import date
from typing import Iterable

from weekly_strategy.data.schemas import (
    CrossStockImplications,
    Dossier,
    MacroRegime,
    NewsItem,
    SectorSnapshot,
)
from weekly_strategy.data.sector import SECTOR_ETFS, SECTOR_TO_ETF
from weekly_strategy.llm import client, prompts


# Regime fit table: sectors that benefit in each regime state.
_CYCLICAL_ETFS = frozenset({"XLK", "XLY", "XLC", "XLI", "XLB"})
_DEFENSIVE_ETFS = frozenset({"XLP", "XLU", "XLV"})
# XLF (financials), XLE (energy), XLRE (real estate) are regime-specific
# rather than pure cyclical/defensive -- handled below.

NEWS_ITEMS_FOR_CROSS_STOCK = 6


# ---------------------------------------------------------------------------
# Deterministic sector score
# ---------------------------------------------------------------------------


def is_risk_on(regime: MacroRegime | None) -> bool:
    """Risk-on when financial conditions aren't tightening AND we're not in recession."""
    if regime is None:
        return True
    if regime.financial_conditions == "tightening":
        return False
    if regime.cycle_phase == "recession":
        return False
    return True


def sector_etf_for(dossier: Dossier) -> str | None:
    if dossier.sector is None:
        return None
    return SECTOR_TO_ETF.get(dossier.sector)


def regime_fit_for_etf(etf: str, regime: MacroRegime | None) -> float:
    """Return a 0-100 regime-fit score for the given sector ETF."""
    risk_on = is_risk_on(regime)
    if etf in _CYCLICAL_ETFS:
        return 70.0 if risk_on else 30.0
    if etf in _DEFENSIVE_ETFS:
        return 30.0 if risk_on else 70.0
    if etf == "XLF":
        rr = regime.rate_regime if regime else "unknown"
        # Banks: NIM expansion in tightening, but punished by inverted curve.
        if rr in ("hiking_or_holding", "tightening_market"):
            return 70.0
        if rr in ("cutting", "easing_market"):
            return 35.0
        return 50.0
    if etf == "XLE":
        # Energy: benefits when financial conditions easing (commodity rally),
        # also a regime-hedge when HY OAS is tight/normal (i.e., no credit stress).
        hy = regime.hy_regime if regime else "unknown"
        if hy in ("tight", "normal") and risk_on:
            return 60.0
        if hy in ("wide", "stressed"):
            return 65.0  # mild defensive bid via commodity hedge
        return 50.0
    if etf == "XLRE":
        # Real estate: hurt by rising rates, helped by cuts.
        rr = regime.rate_regime if regime else "unknown"
        if rr in ("cutting", "easing_market"):
            return 65.0
        if rr in ("hiking_or_holding", "tightening_market"):
            return 35.0
        return 50.0
    return 50.0


def sector_momentum_component(
    etf: str | None, snap: SectorSnapshot | None,
) -> tuple[float, int | None]:
    """Score by leadership rank: 1st = 100, last (11th) = 0. Linear in between."""
    if etf is None or snap is None or not snap.leadership_ranking:
        return 50.0, None
    ranking = snap.leadership_ranking
    if etf not in ranking:
        return 50.0, None
    rank = ranking.index(etf)  # 0-indexed: 0 = best
    n = len(ranking)
    # rank 0 -> 100, rank n-1 -> 0
    score = 100.0 * (n - 1 - rank) / max(n - 1, 1)
    return score, rank + 1  # human-readable: 1-indexed


def sector_score(
    *,
    dossier: Dossier,
    sector_snap: SectorSnapshot | None,
    regime: MacroRegime | None,
    momentum_weight: float = 0.6,
) -> dict:
    """0-100 sector score with the inputs surfaced for the report."""
    etf = sector_etf_for(dossier)
    mom_score, rank_one_indexed = sector_momentum_component(etf, sector_snap)
    fit = regime_fit_for_etf(etf, regime) if etf else 50.0
    score = momentum_weight * mom_score + (1 - momentum_weight) * fit
    sector_metrics = (
        sector_snap.sectors.get(etf) if (etf and sector_snap and etf in sector_snap.sectors) else None
    )
    return {
        "score": score,
        "etf": etf,
        "sector_name": SECTOR_ETFS.get(etf, "") if etf else "",
        "rank": rank_one_indexed,
        "leaders_out_of": len(sector_snap.leadership_ranking) if sector_snap else None,
        "regime_fit": fit,
        "momentum_component": mom_score,
        "rel_1m": sector_metrics.rel_1m if sector_metrics else None,
        "in_favor": fit >= 60.0,
    }


# ---------------------------------------------------------------------------
# Cross-stock LLM agent (deferred from Step 1.5 pass 3)
# ---------------------------------------------------------------------------


def cross_stock_implications(
    ticker: str,
    *,
    dossier: Dossier,
    news_items: Iterable[NewsItem],
    sector_snap: SectorSnapshot | None,
    regime: MacroRegime | None,
    week_ending: date,
    company_name: str | None = None,
    model: str = client.MODEL_SONNET,
) -> CrossStockImplications:
    """One Sonnet call. Sectoral / cross-stock read."""
    etf = sector_etf_for(dossier)
    sector_name = SECTOR_ETFS.get(etf, "") if etf else ""

    # Filter to material items only -- LOW noise doesn't help the reasoning.
    items = [
        it for it in news_items
        if it.classification is not None
        and it.classification.materiality in ("HIGH", "MEDIUM")
    ]
    items.sort(
        key=lambda it: abs(
            it.classification.sentiment_value * it.classification.materiality_weight
        ),
        reverse=True,
    )
    items = items[:NEWS_ITEMS_FOR_CROSS_STOCK]
    if not items:
        # Nothing material this week -- skip the LLM call.
        return CrossStockImplications(
            ticker=ticker.upper(),
            week_ending=week_ending,
            sector_etf=etf,
            summary="No material news this week; cross-stock read deferred.",
        )

    prompt = prompts.CROSS_STOCK_USER_TEMPLATE.format(
        ticker=ticker.upper(),
        company_suffix=f" ({company_name})" if company_name else "",
        sector_etf=etf or "?",
        sector_name=sector_name or "?",
        sector_block=_render_sector(etf, sector_snap),
        regime_block=_render_regime(regime),
        news_block=_render_news(items),
    )
    try:
        parsed, _resp = client.ask_json(
            prompt, model=model, system_prompt=prompts.CROSS_STOCK_SYSTEM,
        )
    except client.ClaudeCliError:
        return CrossStockImplications(
            ticker=ticker.upper(),
            week_ending=week_ending,
            sector_etf=etf,
            summary="LLM unavailable for cross-stock pass.",
        )

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        parsed = {}
    return CrossStockImplications.coerce(
        ticker=ticker, week_ending=week_ending, sector_etf=etf, raw=parsed,
    )


def _render_sector(etf: str | None, snap: SectorSnapshot | None) -> str:
    if etf is None or snap is None:
        return "  (no sector context)"
    m = snap.sectors.get(etf)
    lines = [f"  ETF: {etf}", f"  breadth: {snap.breadth}"]
    if m is not None:
        lines.append(
            f"  returns -- 1w {_pct(m.return_1w)} · 1m {_pct(m.return_1m)} · "
            f"3m {_pct(m.return_3m)}"
        )
        if m.rel_1m is not None:
            lines.append(f"  rel vs SPY -- 1m {_pct(m.rel_1m)} · 3m {_pct(m.rel_3m)}")
        if m.volume_vs_20d is not None:
            lines.append(f"  volume vs 20d avg: {m.volume_vs_20d:.2f}x")
    if snap.leadership_ranking:
        rank = (
            snap.leadership_ranking.index(etf) + 1
            if etf in snap.leadership_ranking else "?"
        )
        lines.append(f"  rank: {rank} of {len(snap.leadership_ranking)}")
        lines.append(
            "  leaders: " + ", ".join(snap.leadership_ranking[:3])
            + " | laggards: " + ", ".join(snap.leadership_ranking[-3:])
        )
    return "\n".join(lines)


def _render_regime(regime: MacroRegime | None) -> str:
    if regime is None:
        return "  (no macro regime computed)"
    lines = [
        f"  rate_regime          : {regime.rate_regime}",
        f"  financial_conditions : {regime.financial_conditions}",
        f"  cycle_phase          : {regime.cycle_phase}",
    ]
    if regime.narrative:
        lines.append(f"  narrative: {regime.narrative}")
    return "\n".join(lines)


def _render_news(items: list[NewsItem]) -> str:
    lines: list[str] = []
    for it in items:
        c = it.classification
        contrib = (c.sentiment_value * c.materiality_weight) if c else 0
        title = (it.title or "").strip()
        lines.append(
            f"  - [{contrib:+.2f} {c.materiality:6s} {c.theme}] {title[:120]}"
        )
        if it.fundamental_impact:
            fi = it.fundamental_impact
            lines.append(
                f"      impact: {fi.direction} {fi.magnitude} {fi.horizon} on {', '.join(fi.areas)}"
            )
            if fi.implication:
                lines.append(f"      -> {fi.implication[:200]}")
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return f"{x*100:+.2f}%" if x is not None else "n/a"
