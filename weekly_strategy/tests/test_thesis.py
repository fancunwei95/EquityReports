from __future__ import annotations

"""Tests for llm/thesis.py. LLM mocked."""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import pytest

from weekly_strategy.data.schemas import (
    Dossier,
    NewsClassification,
    NewsItem,
    NewsScore,
    RedditScore,
    StockScoreBundle,
    ThemeCount,
    FundamentalImpact,
)
from weekly_strategy.llm import thesis


@dataclass
class _FakeResp:
    text: str = ""
    cost_usd: float = 0.012
    session_id: str = "sess"
    raw: dict = None  # type: ignore[assignment]


def _stub(monkeypatch, payload):
    captured = {"calls": 0}

    def fake_ask_json(prompt, model, system_prompt=None, timeout_s=120.0):
        captured["calls"] += 1
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        captured["model"] = model
        return payload, _FakeResp()

    monkeypatch.setattr(thesis.client, "ask_json", fake_ask_json)
    return captured


def _dossier() -> Dossier:
    return Dossier(
        ticker="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        is_financials=False,
        revenue_latest_fy=400_000_000_000,
        revenue_3y_cagr=0.018,
        revenue_yoy_latest=0.05,
        gross_margin_current=0.46,
        operating_margin_current=0.31,
        fcf_margin_current=0.24,
        roe=1.05,
        roic=0.55,
        net_debt_to_ebitda=0.3,
        share_count_change_yoy=-0.026,
        current_price=308.0,
        market_cap=4_500_000_000_000,
        pe_trailing=40.5,
        ps_ratio=11.2,
        fcf_yield=0.022,
        ev_to_ebitda=31.6,
        last_updated=datetime(2026, 5, 24),
        last_filing_date=date(2026, 5, 1),
    )


def _scores() -> StockScoreBundle:
    return StockScoreBundle(
        ticker="AAPL",
        week_ending=date(2026, 5, 24),
        quality_score=93.1,
        valuation_score=0.7,
        momentum_score=72.0,
        return_1m=0.04,
        return_3m=0.09,
        return_6m=0.18,
        news_sentiment_score=0.48,
        news_noise_ratio=0.20,
        reddit_sentiment=0.10,
        reddit_is_crowded=False,
    )


def _news_score() -> NewsScore:
    return NewsScore(
        ticker="AAPL",
        window_start=datetime(2026, 5, 17),
        window_end=datetime(2026, 5, 24),
        n_total=18,
        n_classified=18,
        sentiment_score=0.48,
        top_themes=[
            ThemeCount(theme="analyst", count=8),
            ThemeCount(theme="earnings", count=2),
        ],
        noise_ratio=0.22,
    )


def _news_items() -> list[NewsItem]:
    cls_high = NewsClassification(
        sentiment="POSITIVE", materiality="HIGH", theme="earnings",
        rationale="Q beat",
    )
    fi = FundamentalImpact(
        areas=["revenue", "gross_margin"], direction="POSITIVE",
        magnitude="medium", horizon="FY26",
        implication="Beat reinforces FY26 revenue path.",
    )
    return [
        NewsItem(
            ticker="AAPL", title="Apple Q2 beats expectations",
            source="reuters.com", url="https://x/1",
            published_at=datetime(2026, 5, 20),
            classification=cls_high, fundamental_impact=fi,
        ),
        NewsItem(
            ticker="AAPL", title="Routine analyst note", source="yahoo.com",
            url="https://x/2", published_at=datetime(2026, 5, 21),
            classification=NewsClassification(
                sentiment="NEUTRAL", materiality="LOW", theme="analyst",
                rationale="no new info",
            ),
        ),
    ]


def _reddit_score() -> RedditScore:
    return RedditScore(
        ticker="AAPL", window_hours=24, mention_count=14,
        mention_count_prior_window=10, mention_delta_pct=0.4,
        llm_sentiment=0.15, llm_themes=["AI hype", "earnings"],
        is_crowded=False,
    )


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    return pd.DataFrame({"close": [200.0 + i * 0.5 for i in range(120)]}, index=idx)


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


def test_prompt_includes_all_input_blocks(monkeypatch):
    _stub(monkeypatch, {"bottom_line": "ok"})
    thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
        news_score=_news_score(), news_items=_news_items(),
        reddit_score=_reddit_score(), prices=_prices(),
        company_name="Apple Inc.",
    )


def test_prompt_renders_dossier_and_scores(monkeypatch):
    captured = _stub(monkeypatch, {"bottom_line": "ok"})
    thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
        news_score=_news_score(), news_items=_news_items(),
        reddit_score=_reddit_score(), prices=_prices(),
        company_name="Apple Inc.",
    )
    p = captured["prompt"]
    # Dossier fields
    assert "Technology" in p
    assert "+46.00%" in p  # gross margin
    assert "40.50x" in p   # PE
    # Scores
    assert "quality   :  93.1" in p
    assert "valuation :   0.7" in p
    # News
    assert "Apple Q2 beats expectations" in p
    assert "POSITIVE medium FY26 on revenue, gross_margin" in p
    # Reddit
    assert "mentions: 14" in p
    assert "AI hype" in p
    # System prompt is set
    assert "sell-side equity analyst" in captured["system_prompt"]


def test_prompt_truncates_to_8_news_items(monkeypatch):
    captured = _stub(monkeypatch, {"bottom_line": "ok"})
    # Build 12 classified items
    many = [
        NewsItem(
            ticker="AAPL", title=f"item {i}",
            source="reuters.com", url=f"https://x/{i}",
            published_at=datetime(2026, 5, 20),
            classification=NewsClassification(
                sentiment="POSITIVE", materiality="HIGH", theme="earnings",
            ),
        )
        for i in range(12)
    ]
    thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
        news_score=_news_score(), news_items=many,
        reddit_score=None, prices=None,
        company_name="Apple Inc.",
    )
    p = captured["prompt"]
    # First 8 should appear; later ones should not (assuming default ordering by contrib).
    assert "item 0" in p
    assert "item 7" in p
    assert "item 11" not in p


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def test_parses_full_structured_response(monkeypatch):
    _stub(monkeypatch, {
        "bottom_line": "Constructive on Apple this week.",
        "what_changed": ["Q2 beat", "AI overhaul announced"],
        "quality_and_valuation": "High-quality but expensive.",
        "risks": ["China demand softening", "AI execution risk"],
        "watch_items": ["FY26 guidance", "WWDC announcements"],
    })
    t = thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
    )
    assert t.bottom_line.startswith("Constructive")
    assert t.what_changed == ["Q2 beat", "AI overhaul announced"]
    assert t.quality_and_valuation == "High-quality but expensive."
    assert len(t.risks) == 2
    assert len(t.watch_items) == 2
    assert t.cost_usd == pytest.approx(0.012)


def test_tolerates_dict_wrapped_response(monkeypatch):
    _stub(monkeypatch, [{"bottom_line": "wrapped in list"}])
    t = thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
    )
    assert t.bottom_line == "wrapped in list"


def test_tolerates_string_section_when_list_expected(monkeypatch):
    _stub(monkeypatch, {
        "bottom_line": "ok",
        "what_changed": "single bullet as string",
        "risks": "another single string",
    })
    t = thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
    )
    assert t.what_changed == ["single bullet as string"]
    assert t.risks == ["another single string"]


def test_defaults_when_model_returns_nothing_useful(monkeypatch):
    _stub(monkeypatch, "definitely not a dict")
    t = thesis.write_thesis(
        ticker="AAPL", week_ending=date(2026, 5, 24),
        dossier=_dossier(), scores=_scores(),
    )
    assert "bottom_line" in t.bottom_line  # default placeholder string
    assert t.what_changed == []
    assert t.risks == []
