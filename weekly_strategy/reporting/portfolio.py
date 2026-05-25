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
    Dossier,
    FedPosture,
    FocusList,
    MacroRegime,
    MacroSnapshot,
    NewsItem,
    Portfolio,
    PortfolioPosition,
    RedditScore,
    SectorSnapshot,
    StockScoreBundle,
)


# ---------------------------------------------------------------------------
# Per-position deterministic reasoning
# ---------------------------------------------------------------------------


_Z_COMPONENT_LABEL = {
    "z_quality":   "quality",
    "z_valuation": "valuation",
    "z_momentum":  "momentum",
    "z_news_sentiment": "news sentiment",
    "z_sector":    "sector",
}


def _explain_position(
    position: PortfolioPosition,
    bundle: StockScoreBundle | None,
    dossier: Dossier | None,
    sector_snap: SectorSnapshot | None,
    news_items: list[NewsItem] | None = None,
    reddit_score: RedditScore | None = None,
) -> dict:
    """Build a structured reasoning packet for one position.

    Returns dict with keys: headline (str), drivers (list[str]),
    context (list[str]), fundamentals (list[str] chips), news_summary,
    sentiment_summary (HTML strs). Pure code -- no LLM call. Inputs are
    whatever's available from the run.
    """
    out: dict = {
        "headline": "",
        "drivers": [],
        "context": [],
        "fundamentals": [],
        "news_summary": "",
        "sentiment_summary": "",
    }
    if bundle is None:
        out["headline"] = "No score data available for this position."
        return out

    is_long = position.direction == "long"
    composite_z_score = bundle.composite_z if bundle.composite_z is not None else 50.0
    weighted_z = (composite_z_score - 50.0) / 25.0  # invert the [-2,+2]z -> [0,100] map
    side_phrase = "ranks at the TOP of the universe" if is_long else "ranks at the BOTTOM of the universe"

    out["headline"] = (
        f"Composite {composite_z_score:.1f}/100 (weighted z {weighted_z:+.2f}σ) — "
        f"{position.ticker} {side_phrase} on the regime-weighted ranking this run."
    )

    # --- score drivers ---
    z_components: list[tuple[str, float]] = []
    for attr in ("z_quality", "z_valuation", "z_momentum",
                 "z_news_sentiment", "z_sector"):
        v = getattr(bundle, attr, None)
        if v is None:
            continue
        z_components.append((_Z_COMPONENT_LABEL[attr], float(v)))

    if z_components:
        if is_long:
            ranked = sorted(z_components, key=lambda kv: kv[1], reverse=True)
            sign_word = "tailwind"; drag_word = "drag"
        else:
            ranked = sorted(z_components, key=lambda kv: kv[1])
            sign_word = "headwind"; drag_word = "lift"

        favored = [c for c in ranked[:3] if abs(c[1]) >= 0.3]
        if favored:
            parts = [f"<b>{name}</b> ({z:+.2f}σ)" for name, z in favored]
            out["drivers"].append(
                f"Main {sign_word}{'s' if len(favored) > 1 else ''}: "
                + ", ".join(parts) + "."
            )

        # Counter-signal: must actually be against the direction of the call.
        if is_long:
            counter_pool = sorted(
                [c for c in z_components if c[1] <= -0.5],
                key=lambda kv: kv[1],
            )
        else:
            counter_pool = sorted(
                [c for c in z_components if c[1] >= 0.5],
                key=lambda kv: kv[1], reverse=True,
            )
        if counter_pool:
            name, z = counter_pool[0]
            out["drivers"].append(
                f"Counter-signal: <b>{name}</b> {z:+.2f}σ acts as a {drag_word}."
            )

    # --- price action ---
    rets: list[str] = []
    if bundle.return_1m is not None:
        rets.append(f"1m {bundle.return_1m*100:+.1f}%")
    if bundle.return_3m is not None:
        rets.append(f"3m {bundle.return_3m*100:+.1f}%")
    if bundle.return_6m is not None:
        rets.append(f"6m {bundle.return_6m*100:+.1f}%")
    if rets:
        out["context"].append("Trailing returns: " + " · ".join(rets) + ".")

    # --- sector context ---
    if bundle.sector_etf and sector_snap is not None:
        m = sector_snap.sectors.get(bundle.sector_etf)
        rank = bundle.sector_rank
        favor = "favorable" if bundle.sector_in_favor else "unfavorable"
        rank_phrase = f"rank {rank}/11" if rank else "rank n/a"
        rel_phrase = ""
        if m is not None and m.rel_1m is not None:
            rel_phrase = f", 1m vs SPY {m.rel_1m*100:+.2f}%"
        out["context"].append(
            f"Sector <b>{bundle.sector_etf}</b> ({position.sector or '?'}): "
            f"{rank_phrase}{rel_phrase}; regime fit <b>{favor}</b>."
        )

    # --- fundamentals chips ---
    if dossier is not None:
        bits: list[str] = []
        if dossier.revenue_3y_cagr is not None:
            bits.append(f"rev 3y CAGR {dossier.revenue_3y_cagr*100:+.1f}%")
        if dossier.operating_margin_current is not None:
            bits.append(f"op margin {dossier.operating_margin_current*100:.1f}%")
        if dossier.fcf_margin_current is not None:
            bits.append(f"FCF margin {dossier.fcf_margin_current*100:.1f}%")
        if dossier.roe is not None and abs(dossier.roe) > 0.005:
            bits.append(f"ROE {dossier.roe*100:.1f}%")
        if dossier.pe_trailing is not None and 0 < dossier.pe_trailing < 200:
            bits.append(f"P/E {dossier.pe_trailing:.1f}x")
        if dossier.fcf_yield is not None:
            bits.append(f"FCF yield {dossier.fcf_yield*100:.2f}%")
        if dossier.net_debt_to_ebitda is not None:
            bits.append(f"netDebt/EBITDA {dossier.net_debt_to_ebitda:.2f}x")
        if dossier.share_count_change_yoy is not None and abs(dossier.share_count_change_yoy) > 0.005:
            bits.append(f"shareCt YoY {dossier.share_count_change_yoy*100:+.1f}%")
        if bits:
            out["fundamentals"] = bits[:6]

    # --- news summary (top material items + fundamental impacts) ---
    if news_items:
        material = [
            it for it in news_items
            if it.classification is not None
            and it.classification.materiality in ("HIGH", "MEDIUM")
        ]
        material.sort(
            key=lambda it: abs(
                it.classification.sentiment_value * it.classification.materiality_weight
            ),
            reverse=True,
        )
        if material:
            # Aggregate sentiment summary header
            wsum = sum(it.classification.materiality_weight for it in material)
            agg = sum(
                it.classification.sentiment_value * it.classification.materiality_weight
                for it in material
            ) / max(wsum, 1e-9)
            sent_label = (
                "positive" if agg > 0.15 else "negative" if agg < -0.15 else "neutral"
            )
            header = (
                f"<div class='news-aggregate'>"
                f"<strong>{len(material)}</strong> material items this week · "
                f"aggregate sentiment "
                f"<span class='news-tag {sent_label}'>{agg:+.2f}</span>"
                f"</div>"
            )
            top = material[:4]
            lines: list[str] = []
            for it in top:
                c = it.classification
                contrib = c.sentiment_value * c.materiality_weight
                title = (it.title or "")[:110].replace("<", "&lt;").replace(">", "&gt;")
                fi_block = ""
                if it.fundamental_impact is not None:
                    fi = it.fundamental_impact
                    fi_block = (
                        f"<div class='fi'>↳ <b>{fi.direction}</b> · {fi.magnitude} · "
                        f"{fi.horizon} on {', '.join(fi.areas)}"
                    )
                    if fi.implication:
                        impl = fi.implication[:220].replace("<", "&lt;").replace(">", "&gt;")
                        fi_block += f"<br><span class='fi-impl'>{impl}</span>"
                    fi_block += "</div>"
                lines.append(
                    f"<div class='news-item'>"
                    f"<span class='news-tag {c.sentiment.lower()}'>{c.sentiment}</span> "
                    f"<span class='news-mat'>{c.materiality}</span> "
                    f"<span class='news-theme'>{c.theme}</span> "
                    f"<a href='{it.url}'>{title}</a>"
                    f" <em>(contrib {contrib:+.2f})</em>"
                    f"{fi_block}"
                    f"</div>"
                )
            out["news_summary"] = header + "".join(lines)

    # --- reddit sentiment summary ---
    if reddit_score is not None and reddit_score.mention_count > 0:
        bits: list[str] = []
        bits.append(
            f"<strong>{reddit_score.mention_count}</strong> mentions (last 24h"
            f"; prior {reddit_score.mention_count_prior_window}, "
            f"delta {reddit_score.mention_delta_pct:+.0%})"
        )
        if reddit_score.is_crowded:
            bits.append("<span class='reddit-flag'>⚠ crowded</span>")
        if reddit_score.llm_sentiment is not None:
            lab = (
                "positive" if reddit_score.llm_sentiment > 0.15
                else "negative" if reddit_score.llm_sentiment < -0.15
                else "neutral"
            )
            bits.append(
                f"LLM sentiment "
                f"<span class='news-tag {lab}'>{reddit_score.llm_sentiment:+.2f}</span>"
            )
        if reddit_score.llm_themes:
            chips = "".join(f"<span class='chip'>{t}</span>" for t in reddit_score.llm_themes)
            bits.append(f"Themes: <span class='chips inline'>{chips}</span>")
        out["sentiment_summary"] = " · ".join(bits)

    return out


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
    dossiers: dict[str, "Dossier"] | None = None,  # for per-position reasoning
    news_items_by_ticker: dict[str, list["NewsItem"]] | None = None,  # for news themes
    reddit_scores: dict[str, "RedditScore"] | None = None,             # for per-position reddit
    top_shown: int = 5,    # show top-N per side prominently; fold the rest
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
    parts.append(_render_positions(
        "Longs", portfolio.longs, convictions, bundles,
        dossiers=dossiers, sector_snap=sector_snap,
        news_items_by_ticker=news_items_by_ticker,
        reddit_scores=reddit_scores, top_shown=top_shown,
    ))
    parts.append(_render_positions(
        "Shorts", portfolio.shorts, convictions, bundles,
        dossiers=dossiers, sector_snap=sector_snap,
        news_items_by_ticker=news_items_by_ticker,
        reddit_scores=reddit_scores, top_shown=top_shown,
    ))
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
    """Write to ``daily_reports/{YYYY-MM-DD}.md`` at the repo root.

    One file per as-of date; re-running on the same date overwrites in
    place (so daily reruns don't pile up duplicates).
    """
    settings.DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.DAILY_REPORTS_DIR / f"{portfolio.week_ending.isoformat()}.md"
    path.write_text(markdown)
    return path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="icon" type="image/svg+xml" href="../assets/favicon.svg">
<style>
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 980px;
  margin: 2em auto;
  padding: 0 1em 4em;
  color: #1c1c1c;
  line-height: 1.55;
  background: #f6f7f9;
}}
h1 {{
  color: #0f1e3a;
  border-bottom: 3px solid #2a5fa0;
  padding-bottom: 0.4em;
  margin-bottom: 0.6em;
  font-size: 1.9em;
}}
h2 {{
  color: #1a2540;
  border-bottom: 1px solid #c8d2e0;
  padding-bottom: 0.25em;
  margin-top: 2em;
  font-size: 1.4em;
}}
h3 {{ margin-top: 1.2em; color: #1a2540; }}
table {{
  border-collapse: collapse;
  margin: 0.6em 0;
  font-size: 0.95em;
  background: #fff;
  border-radius: 4px;
  overflow: hidden;
}}
th, td {{ border: 1px solid #e0e4ea; padding: 0.4em 0.8em; text-align: left; }}
th {{ background: #eef1f6; color: #2a3b5c; }}
code {{
  background: #eef1f6;
  padding: 0.05em 0.35em;
  border-radius: 3px;
  font-size: 0.92em;
  color: #2a3b5c;
}}
blockquote {{
  border-left: 4px solid #2a5fa0;
  margin: 0.6em 0;
  padding: 0.4em 1em;
  color: #2a3b5c;
  background: #eef3fa;
  border-radius: 0 4px 4px 0;
  font-style: italic;
}}
hr {{ border: 0; border-top: 1px solid #c8d2e0; margin: 2em 0; }}
ul, ol {{ padding-left: 1.4em; }}
a {{ color: #2a5fa0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* === Position cards === */
.position {{
  background: #fff;
  border-radius: 8px;
  padding: 1em 1.2em 0.8em;
  margin: 0.9em 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  border-left: 5px solid #888;
}}
.position.long  {{ border-left-color: #1f8a5c; }}
.position.short {{ border-left-color: #c14242; }}

.position-head {{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.6em;
  margin-bottom: 0.3em;
}}
.side-badge {{
  font-size: 0.75em;
  font-weight: 700;
  padding: 0.1em 0.55em;
  border-radius: 3px;
  color: #fff;
  letter-spacing: 0.04em;
}}
.side-badge.long  {{ background: #1f8a5c; }}
.side-badge.short {{ background: #c14242; }}
.position-head .rank {{
  color: #6c7a93;
  font-size: 0.95em;
  font-weight: 600;
}}
.position-head .ticker {{
  font-size: 1.4em;
  font-weight: 700;
  color: #0f1e3a;
  letter-spacing: 0.02em;
}}
.position-head .meta {{
  color: #6c7a93;
  font-size: 0.9em;
}}
.position-headline {{
  color: #2a3b5c;
  font-size: 0.98em;
  margin-bottom: 0.6em;
  padding-bottom: 0.55em;
  border-bottom: 1px dashed #d6dce5;
}}

.position-stats {{
  display: flex;
  flex-wrap: wrap;
  gap: 1em;
  margin: 0.5em 0 0.8em;
  font-size: 0.86em;
}}
.position-stats .stat {{
  display: flex;
  flex-direction: column;
  line-height: 1.3;
}}
.position-stats .lbl {{
  color: #6c7a93;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.position-stats .val {{
  color: #1a2540;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}
.conviction-high   {{ color: #1f8a5c; }}
.conviction-medium {{ color: #b87800; }}
.conviction-low    {{ color: #c14242; }}

.reasoning-block {{
  margin: 0.55em 0;
  padding: 0.45em 0;
}}
.reasoning-block .label {{
  font-size: 0.76em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #6c7a93;
  margin-bottom: 0.25em;
  font-weight: 600;
}}
.reasoning-block ul {{ margin: 0.2em 0; padding-left: 1.2em; }}
.reasoning-block li {{ margin: 0.18em 0; color: #2a3b5c; }}
.reasoning-block.llm {{ background: #f4f7fb; padding: 0.6em 0.8em; border-radius: 4px; }}
.reasoning-block .sub-label {{
  font-size: 0.82em;
  font-weight: 600;
  color: #1a2540;
  margin-top: 0.4em;
}}
.reasoning-block .sub-label.flags {{ color: #c14242; }}

.chips {{ display: flex; flex-wrap: wrap; gap: 0.4em; }}
.chip {{
  background: #eef1f6;
  border: 1px solid #d6dce5;
  border-radius: 12px;
  padding: 0.15em 0.7em;
  font-size: 0.82em;
  color: #2a3b5c;
  font-variant-numeric: tabular-nums;
}}

.news-aggregate {{
  font-size: 0.88em;
  color: #2a3b5c;
  margin-bottom: 0.45em;
  padding-bottom: 0.3em;
  border-bottom: 1px dashed #d6dce5;
}}
.news-item {{
  font-size: 0.88em;
  line-height: 1.5;
  margin: 0.4em 0;
  padding: 0.35em 0;
  border-top: 1px dotted #e0e4ea;
}}
.news-item:first-child {{ border-top: 0; }}
.news-item .fi {{
  margin: 0.25em 0 0 1.4em;
  font-size: 0.92em;
  color: #4a5775;
}}
.news-item .fi-impl {{
  display: block;
  margin-top: 0.2em;
  font-style: italic;
  color: #6c7a93;
}}
.news-list {{ font-size: 0.88em; line-height: 1.6; }}
.reddit-summary {{ font-size: 0.88em; color: #2a3b5c; }}
.reddit-flag {{
  color: #c14242;
  font-weight: 600;
  font-size: 0.85em;
}}
.chips.inline {{ display: inline-flex; }}

/* === Folded "Others" section === */
details.position-fold {{
  background: #fff;
  border: 1px solid #d6dce5;
  border-radius: 6px;
  margin: 1em 0;
  padding: 0.55em 0.9em;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
details.position-fold summary {{
  cursor: pointer;
  font-weight: 600;
  color: #2a5fa0;
  user-select: none;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 0.5em;
}}
details.position-fold summary::before {{
  content: "▶";
  font-size: 0.7em;
  color: #6c7a93;
  transition: transform 0.15s ease;
  display: inline-block;
}}
details.position-fold[open] summary::before {{
  transform: rotate(90deg);
}}
details.position-fold[open] summary {{
  border-bottom: 1px solid #e0e4ea;
  padding-bottom: 0.5em;
  margin-bottom: 0.6em;
}}
details.position-fold.long summary {{ color: #1f8a5c; }}
details.position-fold.short summary {{ color: #c14242; }}
.fold-body {{ padding-top: 0.25em; }}
.fold-body .position {{
  margin: 0.6em 0;
  padding: 0.7em 1em 0.55em;
  border-left-width: 4px;
}}
.fold-body .position-headline {{
  font-size: 0.92em;
  padding-bottom: 0.4em;
  margin-bottom: 0.4em;
}}
.fold-body .position-head .ticker {{ font-size: 1.15em; }}
.fold-body .position-stats {{ margin: 0.3em 0 0.55em; }}
.news-tag {{
  display: inline-block;
  padding: 0 0.4em;
  border-radius: 3px;
  font-size: 0.72em;
  font-weight: 700;
  letter-spacing: 0.03em;
  color: #fff;
  vertical-align: middle;
}}
.news-tag.positive {{ background: #1f8a5c; }}
.news-tag.negative {{ background: #c14242; }}
.news-tag.neutral  {{ background: #888; }}
.news-mat {{
  display: inline-block;
  padding: 0 0.4em;
  border-radius: 3px;
  font-size: 0.72em;
  font-weight: 700;
  background: #eef1f6;
  color: #2a3b5c;
  vertical-align: middle;
}}
.news-theme {{
  display: inline-block;
  padding: 0 0.4em;
  border-radius: 3px;
  font-size: 0.72em;
  background: #fff8e6;
  border: 1px solid #ecd99b;
  color: #6c5410;
  vertical-align: middle;
}}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def render_html(markdown_text: str, *, title: str) -> str:
    """Convert the Markdown report to a styled standalone HTML page.

    The position cards contain raw HTML (div + class), which the
    ``md_in_html`` extension passes through verbatim. Other Markdown
    elements (tables, lists, headings) still render normally.
    """
    import markdown as _md
    body = _md.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
    )
    return _HTML_TEMPLATE.format(title=title, content=body)


def save_portfolio_html(
    portfolio: Portfolio, markdown_text: str,
) -> Path:
    """Write a styled HTML sibling to the Markdown report."""
    settings.DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html(
        markdown_text,
        title=f"Portfolio — {portfolio.week_ending.isoformat()}",
    )
    path = settings.DAILY_REPORTS_DIR / f"{portfolio.week_ending.isoformat()}.html"
    path.write_text(html)
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
    *,
    dossiers: dict[str, Dossier] | None = None,
    sector_snap: SectorSnapshot | None = None,
    news_items_by_ticker: dict[str, list[NewsItem]] | None = None,
    reddit_scores: dict[str, RedditScore] | None = None,
    top_shown: int = 5,
) -> str:
    """Render a side (longs or shorts) as HTML cards with reasoning.

    Top ``top_shown`` positions render as expanded cards; the remainder go
    inside a folded ``<details>`` block so the report stays scannable while
    keeping the full data available for inspection.
    """
    if not positions:
        return f"## {label}\n\n_(none selected)_"
    dossiers = dossiers or {}
    news_items_by_ticker = news_items_by_ticker or {}
    reddit_scores = reddit_scores or {}
    side_class = "long" if label.lower().startswith("long") else "short"
    badge = "LONG" if side_class == "long" else "SHORT"

    top = positions[:top_shown]
    rest = positions[top_shown:]
    label_singular = label.rstrip("s").lower()  # "long" or "short"

    if rest:
        header = f"## {label} (top {len(top)} of {len(positions)})"
    else:
        header = f"## {label} ({len(positions)})"
    out: list[str] = [header, ""]

    # Top-K full cards.
    for i, p in enumerate(top, 1):
        out.append(_format_position_card(
            p, i, side_class, badge,
            bundles, convictions, dossiers, sector_snap,
            news_items_by_ticker, reddit_scores, compact=False,
        ))
        out.append("")

    # Rest folded.
    if rest:
        out.append(
            f"<details class='position-fold {side_class}'>"
            f"<summary>Show {len(rest)} more {label_singular} "
            f"candidate{'s' if len(rest) != 1 else ''} (ranks "
            f"{top_shown + 1}-{len(positions)})</summary>"
        )
        out.append("<div class='fold-body'>")
        for i, p in enumerate(rest, top_shown + 1):
            out.append(_format_position_card(
                p, i, side_class, badge,
                bundles, convictions, dossiers, sector_snap,
                news_items_by_ticker, reddit_scores, compact=True,
            ))
        out.append("</div></details>")

    return "\n".join(out)


def _format_position_card(
    p: PortfolioPosition,
    i: int,
    side_class: str,
    badge: str,
    bundles: dict[str, StockScoreBundle],
    convictions: dict[str, ConvictionCheck],
    dossiers: dict[str, Dossier],
    sector_snap: SectorSnapshot | None,
    news_items_by_ticker: dict[str, list[NewsItem]],
    reddit_scores: dict[str, RedditScore],
    *,
    compact: bool,
) -> str:
    """One position card. ``compact`` shows the headline + scores + chips
    only; the full version adds the reasoning + news + reddit blocks."""
    bundle = bundles.get(p.ticker) if bundles else None
    dossier = dossiers.get(p.ticker)
    news_items = news_items_by_ticker.get(p.ticker)
    reddit_score = reddit_scores.get(p.ticker)
    reasoning = _explain_position(
        p, bundle, dossier, sector_snap, news_items, reddit_score,
    )
    composite_disp = (
        f"{p.composite_z:.2f}" if p.composite_z is not None
        else (f"{p.composite_score:.2f}" if p.composite_score is not None else "n/a")
    )
    beta_str = _fmt_beta(p.beta)
    weight_pct = (p.weight or 0) * 100
    check = convictions.get(p.ticker)

    klass = f"position {side_class}" + (" compact" if compact else "")
    out: list[str] = [f"<div class='{klass}'>"]
    out.append(
        f"<div class='position-head'>"
        f"<span class='side-badge {side_class}'>{badge}</span>"
        f"<span class='rank'>#{i}</span>"
        f"<span class='ticker'>{p.ticker}</span>"
        f"<span class='meta'>{p.sector or '?'} · β {beta_str} · weight {weight_pct:.1f}%</span>"
        f"</div>"
    )
    out.append(f"<div class='position-headline'>{reasoning['headline']}</div>")

    stat_bits = [
        f"<span class='stat'><span class='lbl'>composite</span>"
        f"<span class='val'>{composite_disp}</span></span>",
        f"<span class='stat'><span class='lbl'>selection</span>"
        f"<span class='val'>{p.selection_reason}</span></span>",
    ]
    if check is not None:
        stat_bits.append(
            f"<span class='stat'><span class='lbl'>conviction</span>"
            f"<span class='val conviction-{check.confidence.lower()}'>{check.confidence}</span></span>"
        )
    out.append("<div class='position-stats'>" + "".join(stat_bits) + "</div>")

    # Drivers always shown (even in compact -- one line, high info density).
    if reasoning["drivers"]:
        out.append("<div class='reasoning-block'><div class='label'>Why this rank</div><ul>")
        for line in reasoning["drivers"]:
            out.append(f"<li>{line}</li>")
        out.append("</ul></div>")

    # Fundamentals chips: shown in compact too, they're tiny.
    if reasoning["fundamentals"]:
        chips = "".join(
            f"<span class='chip'>{b}</span>" for b in reasoning["fundamentals"]
        )
        out.append(
            f"<div class='reasoning-block'><div class='label'>Fundamentals</div>"
            f"<div class='chips'>{chips}</div></div>"
        )

    # Full-only sections.
    if not compact:
        if reasoning["context"]:
            out.append("<div class='reasoning-block'><div class='label'>Context</div><ul>")
            for line in reasoning["context"]:
                out.append(f"<li>{line}</li>")
            out.append("</ul></div>")
        if reasoning["news_summary"]:
            out.append(
                f"<div class='reasoning-block'><div class='label'>News &amp; fundamental impact</div>"
                f"<div class='news-list'>{reasoning['news_summary']}</div></div>"
            )
        if reasoning["sentiment_summary"]:
            out.append(
                f"<div class='reasoning-block'><div class='label'>Reddit chatter (24h)</div>"
                f"<div class='reddit-summary'>{reasoning['sentiment_summary']}</div></div>"
            )
        if check is not None and (check.thesis or check.risks or check.flags):
            out.append("<div class='reasoning-block llm'><div class='label'>LLM conviction</div>")
            if check.thesis:
                out.append(f"<blockquote>{check.thesis}</blockquote>")
            if check.risks:
                out.append("<div class='sub-label'>Top risks</div><ul>")
                for r in check.risks:
                    out.append(f"<li>{r}</li>")
                out.append("</ul>")
            if check.flags:
                out.append("<div class='sub-label flags'>Data flags</div><ul>")
                for f in check.flags:
                    out.append(f"<li>{f}</li>")
                out.append("</ul>")
            out.append("</div>")

    out.append("</div>")
    return "\n".join(out)


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
