from __future__ import annotations

"""Weekly thesis writer (Step 1.8).

Synthesizes the deterministic inputs (dossier, scores, top news with
fundamental impacts, reddit, price action) into a structured JSON
analyst note via one Sonnet call. The thesis is the final
language-rendered artifact for the week -- everything upstream funnels
through here for the human to read.
"""

import json
from datetime import date
from typing import Iterable

import pandas as pd

from weekly_strategy.data.schemas import (
    CrossStockImplications,
    Dossier,
    MacroRegime,
    MacroSnapshot,
    NewsItem,
    NewsScore,
    RedditScore,
    SectorSnapshot,
    StockScoreBundle,
    WeeklyThesis,
)
from weekly_strategy.llm import client, prompts


# Cap how many news items we cite in the prompt -- thesis writer doesn't
# need 200 headlines, just the ones most likely to drive the week's narrative.
NEWS_ITEMS_FOR_THESIS = 8


def write_thesis(
    *,
    ticker: str,
    week_ending: date,
    dossier: Dossier,
    scores: StockScoreBundle,
    news_score: NewsScore | None = None,
    news_items: Iterable[NewsItem] | None = None,
    reddit_score: RedditScore | None = None,
    prices: pd.DataFrame | None = None,
    # Stage 2 additions -- all optional so Stage 1 callers still work.
    macro_snapshot: MacroSnapshot | None = None,
    macro_regime: MacroRegime | None = None,
    sector_snap: SectorSnapshot | None = None,
    cross_stock: CrossStockImplications | None = None,
    model: str = client.MODEL_SONNET,
    company_name: str | None = None,
) -> WeeklyThesis:
    """One Sonnet call. Returns a structured WeeklyThesis."""
    prompt = _build_prompt(
        ticker=ticker,
        week_ending=week_ending,
        dossier=dossier,
        scores=scores,
        news_score=news_score,
        news_items=news_items,
        reddit_score=reddit_score,
        prices=prices,
        macro_snapshot=macro_snapshot,
        macro_regime=macro_regime,
        sector_snap=sector_snap,
        cross_stock=cross_stock,
        company_name=company_name,
    )
    parsed, resp = client.ask_json(
        prompt, model=model, system_prompt=prompts.THESIS_SYSTEM
    )
    return _materialize_thesis(
        ticker=ticker, week_ending=week_ending, parsed=parsed,
        model=model, cost_usd=resp.cost_usd,
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_prompt(
    *,
    ticker: str,
    week_ending: date,
    dossier: Dossier,
    scores: StockScoreBundle,
    news_score: NewsScore | None,
    news_items: Iterable[NewsItem] | None,
    reddit_score: RedditScore | None,
    prices: pd.DataFrame | None,
    macro_snapshot: MacroSnapshot | None,
    macro_regime: MacroRegime | None,
    sector_snap: SectorSnapshot | None,
    cross_stock: CrossStockImplications | None,
    company_name: str | None,
) -> str:
    suffix = f" ({company_name})" if company_name else ""
    return prompts.THESIS_USER_TEMPLATE.format(
        ticker=ticker.upper(),
        company_suffix=suffix,
        week_ending=week_ending.isoformat(),
        dossier_block=_render_dossier(dossier),
        scores_block=_render_scores(scores),
        macro_block=_render_macro_block(macro_snapshot, macro_regime),
        sector_block=_render_sector_block(scores, sector_snap),
        cross_stock_block=_render_cross_stock(cross_stock),
        news_block=_render_news(news_score, news_items),
        reddit_block=_render_reddit(reddit_score),
        price_block=_render_prices(prices, scores),
    )


def _render_macro_block(
    snap: MacroSnapshot | None, regime: MacroRegime | None,
) -> str:
    if snap is None and regime is None:
        return "  (no macro data this run)"
    lines: list[str] = []
    if snap is not None:
        if snap.yield_10y is not None:
            wow = snap.yield_10y_wow_change_bps
            lines.append(
                f"  10y yield  : {snap.yield_10y:.2f}%"
                + (f" ({wow:+.1f} bps wow)" if wow is not None else "")
            )
        if snap.real_yield_10y is not None:
            lines.append(f"  10y real   : {snap.real_yield_10y:.2f}%")
        if snap.curve_2s10s_bps is not None:
            inv = " (inverted)" if snap.curve_inverted else ""
            lines.append(f"  2s10s curve: {snap.curve_2s10s_bps:.0f} bps{inv}")
        if snap.hy_oas_bps is not None:
            lines.append(f"  HY OAS     : {snap.hy_oas_bps:.0f} bps ({snap.hy_regime})")
        if snap.vix_level is not None:
            wow = snap.vix_wow_change
            lines.append(
                f"  VIX        : {snap.vix_level:.1f} ({snap.vix_regime})"
                + (f"  wow {wow:+.1f}" if wow is not None else "")
            )
        if snap.dxy_level is not None:
            lines.append(f"  DXY        : {snap.dxy_level:.2f}")
    if regime is not None:
        lines.append(f"  rate_regime          : {regime.rate_regime}")
        lines.append(f"  financial_conditions : {regime.financial_conditions}")
        lines.append(f"  cycle_phase          : {regime.cycle_phase}")
        if regime.narrative:
            lines.append(f"  narrative: {regime.narrative}")
    return "\n".join(lines) if lines else "  (no macro data this run)"


def _render_sector_block(
    scores: StockScoreBundle, snap: SectorSnapshot | None,
) -> str:
    if scores.sector_etf is None and snap is None:
        return "  (no sector context this run)"
    lines: list[str] = []
    if scores.sector_etf:
        in_favor = "favorable" if scores.sector_in_favor else "unfavorable"
        lines.append(
            f"  stock sector ETF: {scores.sector_etf}"
            + (f"  rank {scores.sector_rank}" if scores.sector_rank else "")
            + f"  ({in_favor} regime fit)"
        )
        if scores.sector_score_value is not None:
            lines.append(f"  sector_score: {scores.sector_score_value:.1f}/100")
    if snap is not None and snap.leadership_ranking:
        m = snap.sectors.get(scores.sector_etf) if scores.sector_etf else None
        if m is not None:
            if m.rel_1m is not None or m.return_1m is not None:
                lines.append(
                    "  sector returns -- "
                    f"1m {_pct(m.return_1m)} (vs SPY {_pct(m.rel_1m)})  "
                    f"3m {_pct(m.return_3m)} (vs SPY {_pct(m.rel_3m)})"
                )
            if m.volume_vs_20d is not None:
                lines.append(f"  sector volume vs 20d avg: {m.volume_vs_20d:.2f}x")
        lines.append(
            "  market breadth: " + snap.breadth
            + "  |  leaders: " + ", ".join(snap.leadership_ranking[:3])
            + "  |  laggards: " + ", ".join(snap.leadership_ranking[-3:])
        )
    return "\n".join(lines) if lines else "  (no sector context this run)"


def _render_cross_stock(cs: CrossStockImplications | None) -> str:
    if cs is None:
        return "  (cross-stock pass not run)"
    lines: list[str] = []
    if cs.summary:
        lines.append(f"  summary: {cs.summary}")
    if cs.implications_for_sector:
        lines.append("  this stock -> sector / peers:")
        for s in cs.implications_for_sector:
            lines.append(f"    - {s}")
    if cs.implications_from_sector:
        lines.append("  sector / macro -> this stock:")
        for s in cs.implications_from_sector:
            lines.append(f"    - {s}")
    if cs.related_tickers:
        lines.append("  related tickers: " + ", ".join(cs.related_tickers))
    return "\n".join(lines) if lines else "  (no cross-stock implications this week)"


def _render_dossier(d: Dossier) -> str:
    rows: list[tuple[str, str]] = [
        ("sector", d.sector or "?"),
        ("industry", d.industry or "?"),
        ("revenue (latest FY)", _money(d.revenue_latest_fy)),
        ("revenue CAGR 3y", _pct(d.revenue_3y_cagr)),
        ("revenue YoY (latest Q)", _pct(d.revenue_yoy_latest)),
        ("revenue QoQ", _pct(d.revenue_qoq_latest)),
        ("gross margin", _pct(d.gross_margin_current)),
        ("operating margin", _pct(d.operating_margin_current)),
        ("FCF margin", _pct(d.fcf_margin_current)),
        ("ROE", _pct(d.roe)),
        ("ROIC", _pct(d.roic)),
        ("net debt / EBITDA", _ratio(d.net_debt_to_ebitda)),
        ("share count YoY", _pct(d.share_count_change_yoy)),
        ("market cap", _money(d.market_cap)),
        ("P/E (trail)", _ratio(d.pe_trailing)),
        ("P/S", _ratio(d.ps_ratio)),
        ("FCF yield", _pct(d.fcf_yield)),
        ("EV / EBITDA", _ratio(d.ev_to_ebitda)),
        ("last 10-K/10-Q filing", str(d.last_filing_date) if d.last_filing_date else "?"),
    ]
    return "\n".join(f"  {label:<24}: {val}" for label, val in rows)


def _render_scores(s: StockScoreBundle) -> str:
    return (
        f"  quality   : {s.quality_score:5.1f}\n"
        f"  valuation : {s.valuation_score:5.1f}\n"
        f"  momentum  : {s.momentum_score:5.1f}\n"
        f"  return 1m / 3m / 6m: {_pct(s.return_1m)} / {_pct(s.return_3m)} / {_pct(s.return_6m)}"
    )


def _render_news(
    score: NewsScore | None, items: Iterable[NewsItem] | None,
) -> str:
    if score is None or score.n_total == 0:
        return "  (no news in window)"
    lines: list[str] = [
        f"  n_items={score.n_total}, classified={score.n_classified}, "
        f"sentiment_score={score.sentiment_score:+.2f}, noise_ratio={score.noise_ratio:.2f}",
    ]
    if score.top_themes:
        lines.append("  top_themes: " + ", ".join(
            f"{t.theme}({t.count})" for t in score.top_themes
        ))
    # Pick the highest-|contribution| items that also have fundamental impact data.
    if items is not None:
        item_list = [it for it in items if it.classification is not None]
        item_list.sort(
            key=lambda it: abs(
                it.classification.sentiment_value
                * it.classification.materiality_weight
            ),
            reverse=True,
        )
        if item_list:
            lines.append("  top items:")
            for it in item_list[:NEWS_ITEMS_FOR_THESIS]:
                contrib = (
                    it.classification.sentiment_value
                    * it.classification.materiality_weight
                )
                head = (
                    f"    [{contrib:+.2f} {it.classification.materiality:6s} "
                    f"{it.classification.theme}] {(it.title or '')[:100]}"
                )
                lines.append(head)
                if it.fundamental_impact is not None:
                    fi = it.fundamental_impact
                    lines.append(
                        f"        impact: {fi.direction} {fi.magnitude} {fi.horizon} "
                        f"on {', '.join(fi.areas)}"
                    )
                    if fi.implication:
                        lines.append(f"        -> {fi.implication[:240]}")
    return "\n".join(lines)


def _render_reddit(r: RedditScore | None) -> str:
    if r is None or r.mention_count == 0:
        return "  (no reddit activity in window)"
    parts = [
        f"  mentions: {r.mention_count} (prior {r.mention_count_prior_window}, "
        f"delta {r.mention_delta_pct:+.0%})",
    ]
    if r.is_crowded:
        parts.append("  *** flag: crowded ***")
    if r.llm_sentiment is not None:
        parts.append(f"  llm_sentiment: {r.llm_sentiment:+.2f}")
    if r.llm_themes:
        parts.append(f"  themes: {', '.join(r.llm_themes)}")
    return "\n".join(parts)


def _render_prices(prices: pd.DataFrame | None, s: StockScoreBundle) -> str:
    if prices is None or prices.empty or "close" not in prices.columns:
        return "  (no price history)"
    last = float(prices["close"].iloc[-1])
    lines = [f"  last close: ${last:,.2f}"]
    if s.return_1m is not None:
        lines.append(f"  trailing returns: 1m {s.return_1m:+.1%}, "
                     f"3m {s.return_3m:+.1%}, 6m {s.return_6m:+.1%}")
    # 52w hi/lo if we have enough.
    closes = prices["close"].astype(float)
    if len(closes) > 20:
        hi, lo = float(closes.max()), float(closes.min())
        lines.append(f"  range over loaded window: ${lo:,.2f} - ${hi:,.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output materialization
# ---------------------------------------------------------------------------


def _materialize_thesis(
    *,
    ticker: str,
    week_ending: date,
    parsed,
    model: str,
    cost_usd: float,
) -> WeeklyThesis:
    """Robust parse: tolerate dict wrappers, list-of-strings vs string-only sections."""
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        parsed = {}

    return WeeklyThesis(
        ticker=ticker.upper(),
        week_ending=week_ending,
        bottom_line=str(parsed.get("bottom_line") or "(model returned no bottom_line)"),
        what_changed=_as_string_list(parsed.get("what_changed")),
        quality_and_valuation=(parsed.get("quality_and_valuation") or None) or None,
        risks=_as_string_list(parsed.get("risks")),
        watch_items=_as_string_list(parsed.get("watch_items")),
        model=model,
        cost_usd=cost_usd,
    )


def _as_string_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    return [str(raw)]


# ---------------------------------------------------------------------------
# Small formatters
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return f"{x*100:+.2f}%" if x is not None else "n/a"


def _ratio(x: float | None) -> str:
    return f"{x:.2f}x" if x is not None else "n/a"


def _money(x: float | None) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1e12:
        return f"${x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"
