from __future__ import annotations

"""Tests for reporting/single_stock.py markdown rendering."""

from datetime import date, datetime
from pathlib import Path

import pytest

from weekly_strategy.data.schemas import (
    Dossier,
    FundamentalImpact,
    NewsClassification,
    NewsItem,
    NewsScore,
    RedditScore,
    StockScoreBundle,
    ThemeCount,
    WeeklyThesis,
)
from weekly_strategy.reporting import single_stock


def _dossier() -> Dossier:
    return Dossier(
        ticker="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        revenue_latest_fy=400e9,
        gross_margin_current=0.46,
        operating_margin_current=0.31,
        fcf_margin_current=0.24,
        roe=1.05,
        pe_trailing=40.5,
        fcf_yield=0.022,
        current_price=308.0,
        market_cap=4.5e12,
        last_updated=datetime(2026, 5, 24),
        last_filing_date=date(2026, 5, 1),
    )


def _scores() -> StockScoreBundle:
    return StockScoreBundle(
        ticker="AAPL",
        week_ending=date(2026, 5, 24),
        quality_score=93.1, valuation_score=0.7, momentum_score=80.1,
        return_1m=0.13, return_3m=0.16, return_6m=0.15,
        news_sentiment_score=0.48,
    )


def _thesis() -> WeeklyThesis:
    return WeeklyThesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        bottom_line="Constructive on momentum, hold-not-add on valuation.",
        what_changed=["Q2 beat", "CEO transition story"],
        quality_and_valuation="Quality 93, valuation 0.7.",
        risks=["CEO transition", "China margin"],
        watch_items=["WWDC", "China 618"],
        model="claude-sonnet-4-6", cost_usd=0.0361,
    )


def test_render_markdown_contains_all_sections():
    md = single_stock.render_markdown(
        _thesis(), dossier=_dossier(), scores=_scores(),
    )
    assert "# AAPL — Weekly Note" in md
    assert "Apple Inc." in md
    assert "Technology / Consumer Electronics" in md
    assert "## Bottom line" in md
    assert "Constructive on momentum" in md
    assert "## Scorecard" in md
    assert "| Quality    |  93.1 |" in md
    assert "## What changed this week" in md
    assert "- Q2 beat" in md
    assert "## Top risks" in md
    assert "- CEO transition" in md
    assert "## Watch items for next week" in md
    assert "- WWDC" in md
    assert "## Dossier (latest fundamentals)" in md
    assert "model: `claude-sonnet-4-6`" in md


def test_render_includes_news_section_with_fundamental_impact():
    items = [
        NewsItem(
            ticker="AAPL", title="Apple Q2 beats", source="reuters.com",
            url="https://x/1", published_at=datetime(2026, 5, 20),
            classification=NewsClassification(
                sentiment="POSITIVE", materiality="HIGH", theme="earnings",
            ),
            fundamental_impact=FundamentalImpact(
                areas=["revenue", "gross_margin"], direction="POSITIVE",
                magnitude="medium", horizon="FY26",
                implication="Beat reinforces FY26 path.",
            ),
        ),
        NewsItem(
            ticker="AAPL", title="Noise", source="x.com", url="https://x/2",
            published_at=datetime(2026, 5, 21),
            classification=NewsClassification(
                sentiment="NEUTRAL", materiality="LOW", theme="other",
            ),
        ),
    ]
    score = NewsScore(
        ticker="AAPL",
        window_start=datetime(2026, 5, 17), window_end=datetime(2026, 5, 24),
        n_total=2, n_classified=2, sentiment_score=0.50,
        top_themes=[ThemeCount(theme="earnings", count=1)],
        noise_ratio=0.5,
    )
    md = single_stock.render_markdown(
        _thesis(), dossier=_dossier(), scores=_scores(),
        news_score=score, news_items=items,
    )
    assert "## News & fundamental implications" in md
    assert "[Apple Q2 beats](https://x/1)" in md
    # HIGH+MED items rendered; LOW item should not appear under top items.
    assert "Noise" not in md.split("## News")[1].split("## ")[0]
    # Fundamental impact callout
    assert "POSITIVE / medium / FY26" in md
    assert "Beat reinforces FY26 path." in md


def test_render_includes_reddit_when_present():
    md = single_stock.render_markdown(
        _thesis(), dossier=_dossier(), scores=_scores(),
        reddit_score=RedditScore(
            ticker="AAPL", window_hours=24, mention_count=14,
            mention_count_prior_window=10, mention_delta_pct=0.4,
            llm_sentiment=0.20, llm_themes=["AI hype", "earnings"],
            is_crowded=False,
        ),
    )
    assert "## Reddit chatter (24h)" in md
    assert "Mentions: **14**" in md
    assert "_AI hype_" in md


def test_render_omits_optional_sections_when_empty():
    thin_thesis = WeeklyThesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        bottom_line="just a bottom line",
    )
    md = single_stock.render_markdown(
        thin_thesis, dossier=_dossier(), scores=_scores(),
    )
    assert "## What changed" not in md
    assert "## Top risks" not in md
    assert "## Watch items" not in md
    assert "## Reddit" not in md
    assert "## News" not in md


def test_save_markdown_writes_to_per_ticker_dir(tmp_path, monkeypatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path / "reports")
    # Reload renderer so it picks up patched settings reference.
    import importlib
    import weekly_strategy.reporting.single_stock as mod
    mod = importlib.reload(mod)

    md = mod.render_markdown(_thesis(), dossier=_dossier(), scores=_scores())
    path = mod.save_markdown(_thesis(), md)
    assert path.exists()
    assert path.parent.name == "AAPL"
    assert path.name == "2026-05-24.md"
    assert "# AAPL" in path.read_text()
