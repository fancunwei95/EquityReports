from __future__ import annotations

"""Step 3.7: LLM conviction check.

For each selected position, one Sonnet call asks: does this {direction}
{ticker} actually make sense given the scores, dossier, news, and macro
context? Returns HIGH / MEDIUM / LOW.

``filter_by_conviction`` walks the Portfolio, runs the check on each
position, drops LOWs, and pulls the next-ranked focus-list candidate
in as a substitute (subject to the same sector / exclusion constraints
the original selection used). This catches the case where the numerical
score looks great but the underlying reality is broken -- e.g., cheap
P/E because of pending litigation, momentum that's actually a pump.
"""

from collections import defaultdict
from datetime import date
from typing import Iterable

from weekly_strategy.data.schemas import (
    ConvictionCheck,
    Dossier,
    FocusList,
    FocusListItem,
    MacroRegime,
    NewsItem,
    NewsScore,
    Portfolio,
    PortfolioPosition,
    SectorSnapshot,
    StockScoreBundle,
)
from weekly_strategy.llm import client, prompts


# ---------------------------------------------------------------------------
# Single-position check
# ---------------------------------------------------------------------------


def conviction_check(
    position: PortfolioPosition,
    *,
    bundle: StockScoreBundle,
    dossier: Dossier | None = None,
    news_score: NewsScore | None = None,
    news_items: Iterable[NewsItem] | None = None,
    macro_regime: MacroRegime | None = None,
    sector_snap: SectorSnapshot | None = None,
    company_name: str | None = None,
    model: str = client.MODEL_SONNET,
) -> ConvictionCheck:
    """One Sonnet call. Adversarial verdict on a single position."""
    user_prompt = prompts.CONVICTION_USER_TEMPLATE.format(
        direction=position.direction,
        ticker=position.ticker,
        company_suffix=f" ({company_name})" if company_name else "",
        scores_block=_render_scores(bundle),
        dossier_block=_render_dossier_snippet(dossier),
        news_block=_render_news(news_score, news_items),
        macro_sector_block=_render_macro_sector(
            macro_regime, sector_snap, bundle.sector_etf,
        ),
    )
    try:
        parsed, resp = client.ask_json(
            user_prompt, model=model, system_prompt=prompts.CONVICTION_SYSTEM,
        )
        cost = resp.cost_usd
    except client.ClaudeCliError:
        # If the LLM is unreachable, treat as MEDIUM (don't auto-drop the position).
        return ConvictionCheck(
            ticker=position.ticker, direction=position.direction,
            confidence="MEDIUM",
            thesis=None,
            risks=[],
            flags=["LLM call failed -- treated as MEDIUM confidence"],
        )

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        parsed = {}
    return ConvictionCheck.coerce(
        ticker=position.ticker, direction=position.direction,
        cost_usd=cost, raw=parsed,
    )


# ---------------------------------------------------------------------------
# Whole-portfolio filter with substitution
# ---------------------------------------------------------------------------


def filter_by_conviction(
    portfolio: Portfolio,
    *,
    focus: FocusList,
    bundles: dict[str, StockScoreBundle],
    dossiers: dict[str, Dossier] | None = None,
    news_scores: dict[str, NewsScore] | None = None,
    news_items_by_ticker: dict[str, list[NewsItem]] | None = None,
    company_names: dict[str, str] | None = None,
    macro_regime: MacroRegime | None = None,
    sector_snap: SectorSnapshot | None = None,
    max_per_sector: int = 2,
    drop_on_low: bool = True,
) -> tuple[Portfolio, dict[str, ConvictionCheck]]:
    """Run conviction_check on each position. Drop LOW. Substitute from focus.

    Returns the filtered ``Portfolio`` and a dict of every check that ran
    (including substitutes), keyed by ticker.
    """
    dossiers = dossiers or {}
    news_scores = news_scores or {}
    news_items_by_ticker = news_items_by_ticker or {}
    company_names = company_names or {}

    checks: dict[str, ConvictionCheck] = {}
    new_longs, sector_long = _walk_side(
        portfolio.longs, focus.long_candidates, "long",
        bundles, dossiers, news_scores, news_items_by_ticker,
        company_names, macro_regime, sector_snap,
        max_per_sector, drop_on_low, checks,
    )
    new_shorts, sector_short = _walk_side(
        portfolio.shorts, focus.short_candidates, "short",
        bundles, dossiers, news_scores, news_items_by_ticker,
        company_names, macro_regime, sector_snap,
        max_per_sector, drop_on_low, checks,
    )
    updated = portfolio.model_copy(update={"longs": new_longs, "shorts": new_shorts})
    return updated, checks


def _walk_side(
    positions: list[PortfolioPosition],
    candidates: list[FocusListItem],
    side: str,
    bundles: dict[str, StockScoreBundle],
    dossiers: dict[str, Dossier],
    news_scores: dict[str, NewsScore],
    news_items_by_ticker: dict[str, list[NewsItem]],
    company_names: dict[str, str],
    macro_regime: MacroRegime | None,
    sector_snap: SectorSnapshot | None,
    max_per_sector: int,
    drop_on_low: bool,
    checks: dict[str, ConvictionCheck],
) -> tuple[list[PortfolioPosition], dict[str, int]]:
    """Run conviction on every position; substitute LOWs from the focus list."""
    sector_counts: dict[str, int] = defaultdict(int)
    kept: list[PortfolioPosition] = []
    selected_tickers: set[str] = set()

    for pos in positions:
        check = _check_position(
            pos, bundles, dossiers, news_scores, news_items_by_ticker,
            company_names, macro_regime, sector_snap,
        )
        checks[pos.ticker] = check
        if drop_on_low and check.confidence == "LOW":
            continue
        kept.append(pos)
        selected_tickers.add(pos.ticker)
        if pos.sector:
            sector_counts[pos.sector] += 1

    # Substitute up to the original count from the focus list, skipping any
    # name that's already in the book (kept) or that violates sector cap.
    target = len(positions)
    if len(kept) < target:
        for candidate in candidates:
            if len(kept) >= target:
                break
            if candidate.ticker in selected_tickers:
                continue
            if candidate.sector and sector_counts[candidate.sector] >= max_per_sector:
                continue
            sub_pos = PortfolioPosition(
                ticker=candidate.ticker,
                direction=side,
                composite_score=candidate.composite_score,
                composite_z=candidate.composite_z,
                sector=candidate.sector,
                beta=candidate.beta,
                weight=1.0 / max(1, target),
                selection_reason="substitute",
            )
            check = _check_position(
                sub_pos, bundles, dossiers, news_scores, news_items_by_ticker,
                company_names, macro_regime, sector_snap,
            )
            checks[sub_pos.ticker] = check
            if drop_on_low and check.confidence == "LOW":
                continue
            kept.append(sub_pos)
            selected_tickers.add(sub_pos.ticker)
            if sub_pos.sector:
                sector_counts[sub_pos.sector] += 1
    return kept, sector_counts


def _check_position(
    pos: PortfolioPosition,
    bundles: dict[str, StockScoreBundle],
    dossiers: dict[str, Dossier],
    news_scores: dict[str, NewsScore],
    news_items_by_ticker: dict[str, list[NewsItem]],
    company_names: dict[str, str],
    macro_regime: MacroRegime | None,
    sector_snap: SectorSnapshot | None,
) -> ConvictionCheck:
    return conviction_check(
        pos,
        bundle=bundles.get(pos.ticker) or _placeholder_bundle(pos),
        dossier=dossiers.get(pos.ticker),
        news_score=news_scores.get(pos.ticker),
        news_items=news_items_by_ticker.get(pos.ticker),
        macro_regime=macro_regime,
        sector_snap=sector_snap,
        company_name=company_names.get(pos.ticker),
    )


def _placeholder_bundle(pos: PortfolioPosition) -> StockScoreBundle:
    """Build a minimal bundle when the position references a ticker we no longer
    have full score data for. Keeps conviction_check from crashing on edge cases."""
    return StockScoreBundle(
        ticker=pos.ticker, week_ending=date.today(),
        quality_score=50.0, valuation_score=50.0, momentum_score=50.0,
        composite_score=pos.composite_score, composite_z=pos.composite_z,
        sector_etf=None,
    )


# ---------------------------------------------------------------------------
# Render helpers (kept compact -- the LLM gets the gist, not every field)
# ---------------------------------------------------------------------------


def _render_scores(b: StockScoreBundle) -> str:
    lines = [
        f"  composite : {(b.composite_z or b.composite_score or 0):.1f}",
        f"  quality   : {b.quality_score:.1f}",
        f"  valuation : {b.valuation_score:.1f}",
        f"  momentum  : {b.momentum_score:.1f}",
    ]
    if b.sector_score_value is not None:
        lines.append(f"  sector    : {b.sector_score_value:.1f} ({b.sector_etf or '?'})")
    if b.news_sentiment_score is not None:
        lines.append(f"  news_sent : {b.news_sentiment_score:+.2f}")
    if b.return_1m is not None:
        lines.append(
            f"  returns   : 1m {b.return_1m*100:+.1f}% / "
            f"3m {(b.return_3m or 0)*100:+.1f}% / "
            f"6m {(b.return_6m or 0)*100:+.1f}%"
        )
    return "\n".join(lines)


def _render_dossier_snippet(d: Dossier | None) -> str:
    if d is None:
        return "  (no dossier on file)"
    lines = []
    if d.sector:
        lines.append(f"  sector: {d.sector} | industry: {d.industry or '?'}")
    if d.revenue_3y_cagr is not None:
        lines.append(f"  revenue 3y CAGR: {d.revenue_3y_cagr*100:+.2f}%")
    if d.operating_margin_current is not None:
        lines.append(f"  operating margin: {d.operating_margin_current*100:+.2f}%")
    if d.fcf_yield is not None:
        lines.append(f"  FCF yield: {d.fcf_yield*100:+.2f}%")
    if d.pe_trailing is not None:
        lines.append(f"  P/E (trail): {d.pe_trailing:.2f}x")
    if d.net_debt_to_ebitda is not None:
        lines.append(f"  net debt / EBITDA: {d.net_debt_to_ebitda:.2f}x")
    return "\n".join(lines) if lines else "  (dossier present but mostly empty)"


def _render_news(score: NewsScore | None, items: Iterable[NewsItem] | None) -> str:
    if score is None or score.n_total == 0:
        return "  (no news in window)"
    lines = [
        f"  n_items={score.n_total}, classified={score.n_classified}, "
        f"sentiment={score.sentiment_score:+.2f}, noise={score.noise_ratio:.0%}",
    ]
    if items is None:
        return "\n".join(lines)
    material = [
        it for it in items
        if it.classification is not None
        and it.classification.materiality in ("HIGH", "MEDIUM")
    ]
    material.sort(
        key=lambda it: abs(it.classification.sentiment_value * it.classification.materiality_weight),
        reverse=True,
    )
    for it in material[:4]:
        c = it.classification
        contrib = c.sentiment_value * c.materiality_weight
        lines.append(
            f"    [{contrib:+.2f} {c.materiality:6s} {c.theme}] "
            f"{(it.title or '')[:80]}"
        )
    return "\n".join(lines)


def _render_macro_sector(
    regime: MacroRegime | None,
    snap: SectorSnapshot | None,
    sector_etf: str | None,
) -> str:
    lines: list[str] = []
    if regime is not None:
        lines.append(
            f"  macro: rate={regime.rate_regime} "
            f"conditions={regime.financial_conditions} cycle={regime.cycle_phase}"
        )
        if regime.narrative:
            lines.append(f"  > {regime.narrative}")
    if snap is not None and sector_etf:
        m = snap.sectors.get(sector_etf)
        if m is not None and m.rel_1m is not None:
            rank_pos = (
                snap.leadership_ranking.index(sector_etf) + 1
                if sector_etf in snap.leadership_ranking else None
            )
            rank_str = f" (rank {rank_pos} of 11)" if rank_pos else ""
            lines.append(
                f"  sector {sector_etf}{rank_str}: 1m rel SPY {m.rel_1m*100:+.2f}%"
            )
    return "\n".join(lines) if lines else "  (no macro/sector context)"
