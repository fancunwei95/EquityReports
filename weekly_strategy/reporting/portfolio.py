from __future__ import annotations

"""Step 3.8: weekly portfolio Markdown report.

Renders the L/S book + macro/sector context + watch list + changes vs
last week + risk flags. The single-stock format from Stage 1-2 stays
useful as an appendix (this report links to per-ticker notes by path);
callers can drop those in if they want.
"""

from datetime import date
from pathlib import Path
from typing import Iterable

from weekly_strategy.config import settings
from weekly_strategy.data.schemas import (
    ConvictionCheck,
    FedPosture,
    FocusList,
    MacroRegime,
    MacroSnapshot,
    Portfolio,
    PortfolioPosition,
    SectorSnapshot,
    StockScoreBundle,
)


def render_portfolio_markdown(
    portfolio: Portfolio,
    *,
    bundles: dict[str, StockScoreBundle] | None = None,
    convictions: dict[str, ConvictionCheck] | None = None,
    focus: FocusList | None = None,
    macro_snapshot: MacroSnapshot | None = None,
    macro_regime: MacroRegime | None = None,
    fed_posture: FedPosture | None = None,
    sector_snap: SectorSnapshot | None = None,
    previous_longs: set[str] | None = None,
    previous_shorts: set[str] | None = None,
    earnings_calendar: dict[str, str] | None = None,
) -> str:
    """Compose the full portfolio Markdown.

    Optional inputs are rendered as separate sections; missing inputs drop
    their sections rather than emitting placeholders.
    """
    bundles = bundles or {}
    convictions = convictions or {}
    previous_longs = previous_longs or set()
    previous_shorts = previous_shorts or set()
    earnings_calendar = earnings_calendar or {}

    parts: list[str] = []
    parts.append(_render_header(portfolio))
    if macro_snapshot or macro_regime or fed_posture:
        parts.append(_render_market_context(macro_snapshot, macro_regime, fed_posture, sector_snap))
    parts.append(_render_positions("Longs", portfolio.longs, convictions, bundles))
    parts.append(_render_positions("Shorts", portfolio.shorts, convictions, bundles))
    parts.append(_render_changes(portfolio, previous_longs, previous_shorts))
    parts.append(_render_portfolio_metrics(portfolio))
    if focus is not None:
        parts.append(_render_watch_list(focus, portfolio))
    parts.append(_render_risk_flags(portfolio, earnings_calendar))
    if portfolio.rejected:
        parts.append(_render_rejections(portfolio.rejected))
    return "\n\n".join(p for p in parts if p)


def save_portfolio_markdown(
    portfolio: Portfolio, markdown: str,
) -> Path:
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.REPORTS_DIR / f"portfolio_{portfolio.week_ending.isoformat()}.md"
    path.write_text(markdown)
    return path


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(p: Portfolio) -> str:
    return (
        "# Weekly Portfolio Report\n\n"
        f"Week ending **{p.week_ending.isoformat()}**  \n"
        f"Longs: **{len(p.longs)}** · Shorts: **{len(p.shorts)}**  \n"
        f"Net beta: **{_fmt_beta(p.net_beta)}**"
    )


def _render_market_context(
    snap: MacroSnapshot | None,
    regime: MacroRegime | None,
    fed: FedPosture | None,
    sector_snap: SectorSnapshot | None,
) -> str:
    lines = ["## Market context", ""]
    if regime is not None:
        lines.append(
            f"**Regime**: rate `{regime.rate_regime}` · "
            f"conditions `{regime.financial_conditions}` · "
            f"cycle `{regime.cycle_phase}`"
        )
        if regime.narrative:
            lines.append("")
            lines.append(f"> {regime.narrative}")
    if fed is not None and fed.n_items > 0:
        lines.append("")
        lines.append(f"**Fed posture** ({fed.n_items} items): `{fed.posture}`")
        if fed.summary:
            lines.append(fed.summary)
    if snap is not None:
        rates = []
        if snap.yield_10y is not None:
            rates.append(f"10y {snap.yield_10y:.2f}%")
        if snap.curve_2s10s_bps is not None:
            rates.append(f"2s10s {snap.curve_2s10s_bps:.0f}bps"
                         + (" (inverted)" if snap.curve_inverted else ""))
        if snap.hy_oas_bps is not None:
            rates.append(f"HY OAS {snap.hy_oas_bps:.0f}bps")
        if snap.vix_level is not None:
            rates.append(f"VIX {snap.vix_level:.1f}")
        if rates:
            lines.append("")
            lines.append("**Rates / credit / vol**: " + " · ".join(rates))
    if sector_snap is not None and sector_snap.leadership_ranking:
        lines.append("")
        lines.append(
            f"**Sectors**: breadth `{sector_snap.breadth}` · "
            f"leaders {', '.join(sector_snap.leadership_ranking[:3])} · "
            f"laggards {', '.join(sector_snap.leadership_ranking[-3:])}"
        )
    return "\n".join(lines)


def _render_positions(
    label: str,
    positions: list[PortfolioPosition],
    convictions: dict[str, ConvictionCheck],
    bundles: dict[str, StockScoreBundle],
) -> str:
    if not positions:
        return f"## {label}\n\n_(none selected)_"
    lines = [f"## {label} ({len(positions)})", ""]
    for i, p in enumerate(positions, 1):
        score = (p.composite_z if p.composite_z is not None else p.composite_score) or 0.0
        sector = p.sector or "?"
        beta_str = _fmt_beta(p.beta)
        weight_pct = (p.weight or 0) * 100
        lines.append(
            f"### {i}. `{p.ticker}` — {sector} (β={beta_str}, weight={weight_pct:.1f}%)"
        )
        lines.append(
            f"Composite: **{score:.2f}** · Selection: `{p.selection_reason}`"
        )
        check = convictions.get(p.ticker)
        if check is not None:
            lines.append(f"Conviction: **{check.confidence}**")
            if check.thesis:
                lines.append("")
                lines.append(f"> {check.thesis}")
            if check.risks:
                lines.append("")
                lines.append("Top risks:")
                for r in check.risks:
                    lines.append(f"- {r}")
            if check.flags:
                lines.append("")
                lines.append("**Data flags**:")
                for f in check.flags:
                    lines.append(f"- {f}")
        lines.append("")
    return "\n".join(lines)


def _render_changes(
    p: Portfolio, prev_longs: set[str], prev_shorts: set[str],
) -> str:
    cur_longs = {pos.ticker for pos in p.longs}
    cur_shorts = {pos.ticker for pos in p.shorts}
    added_long = sorted(cur_longs - prev_longs)
    dropped_long = sorted(prev_longs - cur_longs)
    held_long = sorted(cur_longs & prev_longs)
    added_short = sorted(cur_shorts - prev_shorts)
    dropped_short = sorted(prev_shorts - cur_shorts)
    held_short = sorted(cur_shorts & prev_shorts)
    if not (prev_longs or prev_shorts):
        return "## Changes vs last week\n\n_(no prior week on file -- this is the first run)_"
    lines = ["## Changes vs last week", ""]
    lines.append(f"**Longs added**: {_fmt_tickers(added_long)}")
    lines.append(f"**Longs dropped**: {_fmt_tickers(dropped_long)}")
    lines.append(f"**Longs held**: {_fmt_tickers(held_long)}")
    lines.append("")
    lines.append(f"**Shorts added**: {_fmt_tickers(added_short)}")
    lines.append(f"**Shorts dropped**: {_fmt_tickers(dropped_short)}")
    lines.append(f"**Shorts held**: {_fmt_tickers(held_short)}")
    return "\n".join(lines)


def _render_portfolio_metrics(p: Portfolio) -> str:
    sectors_long: dict[str, int] = {}
    sectors_short: dict[str, int] = {}
    for pos in p.longs:
        if pos.sector:
            sectors_long[pos.sector] = sectors_long.get(pos.sector, 0) + 1
    for pos in p.shorts:
        if pos.sector:
            sectors_short[pos.sector] = sectors_short.get(pos.sector, 0) + 1
    lines = ["## Portfolio metrics", ""]
    lines.append(f"- Long basket beta: {_fmt_beta(p.long_basket_beta)}")
    lines.append(f"- Short basket beta: {_fmt_beta(p.short_basket_beta)}")
    lines.append(f"- Net beta: {_fmt_beta(p.net_beta)}")
    if sectors_long:
        lines.append("- Long sector exposure: " + ", ".join(
            f"{s} ({n})" for s, n in sorted(sectors_long.items())
        ))
    if sectors_short:
        lines.append("- Short sector exposure: " + ", ".join(
            f"{s} ({n})" for s, n in sorted(sectors_short.items())
        ))
    total_weight_long = sum(p_.weight for p_ in p.longs)
    total_weight_short = sum(p_.weight for p_ in p.shorts)
    lines.append(
        f"- Gross exposure (sum of |weights|): "
        f"long {total_weight_long*100:.1f}% / short {total_weight_short*100:.1f}%"
    )
    return "\n".join(lines)


def _render_watch_list(focus: FocusList, p: Portfolio) -> str:
    picked = {pos.ticker for pos in p.longs} | {pos.ticker for pos in p.shorts}
    next_longs = [it for it in focus.long_candidates if it.ticker not in picked][:5]
    next_shorts = [it for it in focus.short_candidates if it.ticker not in picked][:5]
    lines = ["## Watch list (next-ranked by composite)", ""]
    if next_longs:
        lines.append("**Long bench**: " + ", ".join(
            f"{it.ticker} (z={it.composite_z:+.2f})"
            for it in next_longs if it.composite_z is not None
        ))
    if next_shorts:
        lines.append("**Short bench**: " + ", ".join(
            f"{it.ticker} (z={it.composite_z:+.2f})"
            for it in next_shorts if it.composite_z is not None
        ))
    if focus.outliers:
        lines.append("")
        lines.append("**Outliers** (extreme single-component scores):")
        for o in focus.outliers:
            lines.append(f"- {o.ticker}: {o.outlier_reason}")
    return "\n".join(lines)


def _render_risk_flags(p: Portfolio, earnings: dict[str, str]) -> str:
    lines = ["## Risk flags", ""]
    if earnings:
        positions = list(p.longs) + list(p.shorts)
        flagged = [
            (pos.ticker, earnings.get(pos.ticker))
            for pos in positions if pos.ticker in earnings
        ]
        if flagged:
            lines.append("**Earnings this week:**")
            for t, d in flagged:
                lines.append(f"- {t}: {d}")
    if p.net_beta is not None and abs(p.net_beta) > 0.3:
        lines.append(
            f"- Net beta {p.net_beta:+.2f} outside target band (±0.30) -- "
            "consider rebalancing weights."
        )
    if not p.long_basket_beta or not p.short_basket_beta:
        lines.append("- Beta data incomplete on one or both baskets -- "
                     "net-beta neutralization could not be enforced this week.")
    if len(lines) == 2:
        lines.append("_(none flagged)_")
    return "\n".join(lines)


def _render_rejections(rejected: list[dict]) -> str:
    lines = ["## Rejections from initial focus list", ""]
    for r in rejected:
        lines.append(f"- `{r.get('ticker')}` ({r.get('side')}): {r.get('reason')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small formatters
# ---------------------------------------------------------------------------


def _fmt_beta(x: float | None) -> str:
    return f"{x:+.2f}" if x is not None else "n/a"


def _fmt_tickers(tickers: Iterable[str]) -> str:
    lst = list(tickers)
    if not lst:
        return "_(none)_"
    return ", ".join(f"`{t}`" for t in lst)
