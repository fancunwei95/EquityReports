from __future__ import annotations

"""Markdown rendering of a per-ticker weekly thesis.

The structured ``WeeklyThesis`` is what downstream tools consume; this
module produces the human-readable artifact you'd actually open and read
on Sunday evening. Save target: ``reports/{TICKER}/{week_ending}.md``.
"""

from datetime import date
from pathlib import Path
from typing import Iterable

from weekly_strategy.config import settings
from weekly_strategy.data.schemas import (
    Dossier,
    NewsItem,
    NewsScore,
    RedditScore,
    StockScoreBundle,
    WeeklyThesis,
)


def render_markdown(
    thesis: WeeklyThesis,
    *,
    dossier: Dossier,
    scores: StockScoreBundle,
    news_score: NewsScore | None = None,
    news_items: Iterable[NewsItem] | None = None,
    reddit_score: RedditScore | None = None,
) -> str:
    """Render a single-stock weekly report as Markdown."""
    parts: list[str] = []
    parts.append(_render_header(thesis, dossier, scores))
    parts.append(_render_bottom_line(thesis))
    parts.append(_render_scores_table(scores, dossier))
    parts.append(_render_what_changed(thesis))
    parts.append(_render_quality_valuation(thesis))
    parts.append(_render_risks(thesis))
    parts.append(_render_watch_items(thesis))
    if news_score is not None and news_score.n_total > 0:
        parts.append(_render_news_section(news_score, news_items))
    if reddit_score is not None and reddit_score.mention_count > 0:
        parts.append(_render_reddit_section(reddit_score))
    parts.append(_render_dossier_appendix(dossier))
    parts.append(_render_footer(thesis))
    return "\n\n".join(p for p in parts if p)


def save_markdown(
    thesis: WeeklyThesis,
    markdown: str,
) -> Path:
    """Write to reports/{TICKER}/{week_ending}.md and return the path."""
    out_dir = settings.REPORTS_DIR / thesis.ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{thesis.week_ending.isoformat()}.md"
    path.write_text(markdown)
    return path


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(thesis: WeeklyThesis, dossier: Dossier, scores: StockScoreBundle) -> str:
    name = dossier.company_name or thesis.ticker
    sector = dossier.sector or "?"
    industry = dossier.industry or "?"
    return (
        f"# {thesis.ticker} — Weekly Note\n\n"
        f"**{name}** | {sector} / {industry}  \n"
        f"Week ending {thesis.week_ending.isoformat()}"
    )


def _render_bottom_line(thesis: WeeklyThesis) -> str:
    return f"## Bottom line\n\n{thesis.bottom_line}"


def _render_scores_table(scores: StockScoreBundle, dossier: Dossier) -> str:
    lines = [
        "## Scorecard",
        "",
        "| Score      | Value |",
        "|------------|------:|",
        f"| Quality    | {scores.quality_score:5.1f} |",
        f"| Valuation  | {scores.valuation_score:5.1f} |",
        f"| Momentum   | {scores.momentum_score:5.1f} |",
    ]
    if scores.news_sentiment_score is not None:
        lines.append(f"| News sent. | {scores.news_sentiment_score:+5.2f} |")
    if scores.reddit_sentiment is not None:
        lines.append(f"| Reddit     | {scores.reddit_sentiment:+5.2f} |")
    lines.append("")
    lines.append(
        f"Trailing returns: 1m **{_pct(scores.return_1m)}** · "
        f"3m **{_pct(scores.return_3m)}** · 6m **{_pct(scores.return_6m)}**  \n"
        f"Last close: **{_money(dossier.current_price, sign=False)}** · "
        f"Market cap: **{_money(dossier.market_cap)}**"
    )
    return "\n".join(lines)


def _render_what_changed(thesis: WeeklyThesis) -> str:
    if not thesis.what_changed:
        return ""
    return "## What changed this week\n\n" + "\n".join(
        f"- {b}" for b in thesis.what_changed
    )


def _render_quality_valuation(thesis: WeeklyThesis) -> str:
    if not thesis.quality_and_valuation:
        return ""
    return f"## Quality & valuation snapshot\n\n{thesis.quality_and_valuation}"


def _render_risks(thesis: WeeklyThesis) -> str:
    if not thesis.risks:
        return ""
    return "## Top risks\n\n" + "\n".join(f"- {r}" for r in thesis.risks)


def _render_watch_items(thesis: WeeklyThesis) -> str:
    if not thesis.watch_items:
        return ""
    return "## Watch items for next week\n\n" + "\n".join(
        f"- {w}" for w in thesis.watch_items
    )


def _render_news_section(
    score: NewsScore, items: Iterable[NewsItem] | None,
) -> str:
    lines = [
        "## News & fundamental implications",
        "",
        f"_{score.n_classified}/{score.n_total} items classified · "
        f"weighted sentiment {score.sentiment_score:+.2f} · "
        f"noise ratio {score.noise_ratio:.0%}_",
    ]
    if score.top_themes:
        lines.append(
            "Themes (HIGH+MED count): "
            + ", ".join(f"**{t.theme}** ({t.count})" for t in score.top_themes)
        )
    if items is not None:
        material = [
            it for it in items
            if it.classification is not None
            and it.classification.materiality in ("HIGH", "MEDIUM")
        ]
        if material:
            material.sort(
                key=lambda it: abs(
                    it.classification.sentiment_value
                    * it.classification.materiality_weight
                ),
                reverse=True,
            )
            lines.append("")
            for it in material[:6]:
                title = (it.title or "").replace("|", "/")
                contrib = (
                    it.classification.sentiment_value
                    * it.classification.materiality_weight
                )
                tag = (
                    f"`{it.classification.materiality}` "
                    f"`{it.classification.theme}` "
                    f"`contrib {contrib:+.2f}`"
                )
                lines.append(f"- [{title}]({it.url}) — {tag}")
                if it.fundamental_impact is not None:
                    fi = it.fundamental_impact
                    lines.append(
                        f"   - **{fi.direction} / {fi.magnitude} / {fi.horizon}** "
                        f"on {', '.join(fi.areas)}"
                    )
                    if fi.implication:
                        lines.append(f"   - _{fi.implication}_")
    return "\n".join(lines)


def _render_reddit_section(r: RedditScore) -> str:
    lines = [
        "## Reddit chatter (24h)",
        "",
        f"Mentions: **{r.mention_count}** (prior {r.mention_count_prior_window}, "
        f"delta {r.mention_delta_pct:+.0%})",
    ]
    if r.is_crowded:
        lines.append("> **Flag: crowded** — mention spike or absolute volume above threshold")
    if r.llm_sentiment is not None:
        lines.append(f"LLM sentiment: **{r.llm_sentiment:+.2f}**")
    if r.llm_themes:
        lines.append("Themes: " + ", ".join(f"_{t}_" for t in r.llm_themes))
    return "\n".join(lines)


def _render_dossier_appendix(d: Dossier) -> str:
    rows = [
        ("Revenue (FY)", _money(d.revenue_latest_fy)),
        ("Revenue 3y CAGR", _pct(d.revenue_3y_cagr)),
        ("Gross margin", _pct(d.gross_margin_current)),
        ("Operating margin", _pct(d.operating_margin_current)),
        ("FCF margin", _pct(d.fcf_margin_current)),
        ("ROE", _pct(d.roe)),
        ("ROIC", _pct(d.roic)),
        ("Net debt / EBITDA", _ratio(d.net_debt_to_ebitda)),
        ("Share count YoY", _pct(d.share_count_change_yoy)),
        ("P/E (trail)", _ratio(d.pe_trailing)),
        ("P/S", _ratio(d.ps_ratio)),
        ("FCF yield", _pct(d.fcf_yield)),
        ("EV / EBITDA", _ratio(d.ev_to_ebitda)),
        ("Last filing", str(d.last_filing_date) if d.last_filing_date else "n/a"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        "## Dossier (latest fundamentals)\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"{body}"
    )


def _render_footer(thesis: WeeklyThesis) -> str:
    cost = f"${thesis.cost_usd:.4f}" if thesis.cost_usd is not None else "n/a"
    model = thesis.model or "?"
    return (
        "---\n\n"
        f"_Generated by `weekly_strategy` · model: `{model}` · "
        f"thesis cost: {cost}_"
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return f"{x*100:+.2f}%" if x is not None else "n/a"


def _ratio(x: float | None) -> str:
    return f"{x:.2f}x" if x is not None else "n/a"


def _money(x: float | None, *, sign: bool = True) -> str:
    if x is None:
        return "n/a"
    prefix = "$" if sign else "$"
    if abs(x) >= 1e12:
        return f"{prefix}{x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"{prefix}{x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"{prefix}{x/1e6:.2f}M"
    return f"{prefix}{x:,.2f}"
