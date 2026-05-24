from __future__ import annotations

"""Tests for signals/news_sentiment.py.

LLM calls are mocked; the classifier never hits the network in this suite.
"""

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def news_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings_mod, "DOSSIER_DIR", tmp_path / "dossiers")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    for d in (settings_mod.DATA_CACHE_DIR, settings_mod.REPORTS_DIR, settings_mod.DOSSIER_DIR):
        d.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.data.storage as storage_mod
    storage_mod = importlib.reload(storage_mod)
    storage_mod.init_db()

    import weekly_strategy.signals.news_sentiment as mod
    mod = importlib.reload(mod)
    return mod


@dataclass
class _FakeResp:
    text: str = ""
    cost_usd: float = 0.0
    session_id: str = "sess"
    raw: dict = None  # type: ignore[assignment]


def _stub_classifier(monkeypatch, mod, response_payload: Any) -> dict:
    """Replace client.ask_json so classify_batch returns ``response_payload``."""
    captured: dict = {"calls": 0}

    def fake_ask_json(prompt, model, system_prompt=None, timeout_s=120.0):
        captured["calls"] += 1
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return response_payload, _FakeResp(text=json.dumps(response_payload))

    monkeypatch.setattr(mod.client, "ask_json", fake_ask_json)
    return captured


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_prompt_includes_index_source_and_truncated_snippet():
    from weekly_strategy.llm.prompts import format_news_batch_user_prompt
    long_snippet = "Apple " * 100
    items = [
        {"title": "Apple beats Q2", "source": "reuters.com",
         "snippet": long_snippet, "published_at": "2026-05-19T10:00:00"},
        {"title": "China weakness", "source": "bloomberg.com",
         "snippet": "China sales down", "published_at": "2026-05-20T08:00:00"},
    ]
    out = format_news_batch_user_prompt("AAPL", items, company_name="Apple Inc.")
    assert "AAPL" in out
    assert "(Apple Inc.)" in out
    assert "0. [reuters.com | 2026-05-19] Apple beats Q2" in out
    assert "1. [bloomberg.com | 2026-05-20] China weakness" in out
    # Snippet truncated, with ellipsis somewhere.
    assert "..." in out
    # Exact count baked into the prompt.
    assert "of the 2 items" in out


# ---------------------------------------------------------------------------
# Classifier alignment / robustness
# ---------------------------------------------------------------------------


def test_classify_batch_aligns_by_index(news_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    items = [
        NewsItem(ticker="AAPL", title=t, url=f"u{i}",
                 published_at=datetime(2026, 5, 20))
        for i, t in enumerate(["A", "B", "C"])
    ]
    # Model returns out-of-order entries; aligner should sort by 'i'.
    payload = [
        {"i": 2, "sentiment": "NEGATIVE", "materiality": "HIGH", "theme": "litigation"},
        {"i": 0, "sentiment": "POSITIVE", "materiality": "MEDIUM", "theme": "earnings"},
        {"i": 1, "sentiment": "NEUTRAL", "materiality": "LOW", "theme": "macro"},
    ]
    captured = _stub_classifier(monkeypatch, news_mod, payload)
    classifications = news_mod.classify_batch("AAPL", items)
    assert captured["calls"] == 1
    assert [c.sentiment for c in classifications] == ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    assert [c.theme for c in classifications] == ["earnings", "macro", "litigation"]


def test_classify_batch_handles_missing_entries(news_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    items = [
        NewsItem(ticker="AAPL", title=t, url=f"u{i}", published_at=datetime(2026, 5, 20))
        for i, t in enumerate(["A", "B", "C"])
    ]
    # Model returns only 2 of 3 -- missing slot must fill with default.
    payload = [
        {"i": 0, "sentiment": "POSITIVE", "materiality": "HIGH", "theme": "earnings"},
        {"i": 2, "sentiment": "NEGATIVE", "materiality": "LOW", "theme": "other"},
    ]
    _stub_classifier(monkeypatch, news_mod, payload)
    cls = news_mod.classify_batch("AAPL", items)
    assert len(cls) == 3
    assert cls[1].sentiment == "NEUTRAL"
    assert cls[1].materiality == "LOW"


def test_classify_batch_coerces_unknown_enum_values(news_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    items = [NewsItem(ticker="AAPL", title="x", url="u0", published_at=datetime(2026, 5, 20))]
    # Garbage values -> coerce to safe defaults.
    _stub_classifier(monkeypatch, news_mod, [
        {"i": 0, "sentiment": "bullish", "materiality": "MAJOR", "theme": "ipo"},
    ])
    cls = news_mod.classify_batch("AAPL", items)
    assert cls[0].sentiment == "NEUTRAL"
    assert cls[0].materiality == "LOW"
    assert cls[0].theme == "other"


def test_classify_batch_unwraps_dict_wrapper(news_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    items = [NewsItem(ticker="AAPL", title="x", url="u0", published_at=datetime(2026, 5, 20))]
    # Model wraps array under "items" key.
    _stub_classifier(monkeypatch, news_mod, {
        "items": [{"i": 0, "sentiment": "POSITIVE", "materiality": "HIGH", "theme": "earnings"}]
    })
    cls = news_mod.classify_batch("AAPL", items)
    assert cls[0].sentiment == "POSITIVE"


# ---------------------------------------------------------------------------
# Aggregator (pure math)
# ---------------------------------------------------------------------------


def _classified(title: str, url: str, sent: str, mat: str, theme: str):
    from weekly_strategy.data.schemas import NewsClassification, NewsItem
    return NewsItem(
        ticker="AAPL",
        title=title,
        url=url,
        source="reuters.com",
        published_at=datetime(2026, 5, 20),
        classification=NewsClassification(
            sentiment=sent, materiality=mat, theme=theme, rationale=None,
        ),
    )


def test_aggregate_weighted_sentiment(news_mod):
    items = [
        _classified("a", "u1", "POSITIVE", "HIGH", "earnings"),
        _classified("b", "u2", "POSITIVE", "MEDIUM", "earnings"),
        _classified("c", "u3", "NEGATIVE", "LOW", "litigation"),
        _classified("d", "u4", "NEUTRAL", "LOW", "macro"),
    ]
    score = news_mod.aggregate("AAPL", items,
                               datetime(2026, 5, 14), datetime(2026, 5, 21))
    # Weights: HIGH=1.0, MED=0.5, LOW=0.2, LOW=0.2 -> sum 1.9
    # Numerator: +1*1.0 + 1*0.5 + (-1)*0.2 + 0*0.2 = 1.3
    # -> 1.3 / 1.9 = 0.6842
    assert score.sentiment_score == pytest.approx(0.6842, abs=1e-3)
    assert score.n_total == 4
    assert score.n_classified == 4
    assert score.noise_ratio == pytest.approx(0.5)  # 2 of 4 are LOW
    # Top themes: earnings has 2 HIGH+MED; litigation/macro both LOW so excluded.
    assert [t.theme for t in score.top_themes] == ["earnings"]
    assert score.top_themes[0].count == 2


def test_aggregate_top_items_by_abs_contribution(news_mod):
    items = [
        _classified("strong neg",  "u1", "NEGATIVE", "HIGH",   "litigation"),
        _classified("mild pos",    "u2", "POSITIVE", "MEDIUM", "earnings"),
        _classified("noise",       "u3", "NEUTRAL",  "LOW",    "other"),
        _classified("strong pos",  "u4", "POSITIVE", "HIGH",   "product"),
    ]
    score = news_mod.aggregate("AAPL", items,
                               datetime(2026, 5, 14), datetime(2026, 5, 21))
    top_urls = [t.url for t in score.top_items]
    # |contribution|: u1=1.0, u4=1.0, u2=0.5, u3=0.0
    # u1/u4 tied at top; either could be first. Both should appear before u2.
    assert top_urls[0] in {"u1", "u4"}
    assert top_urls[1] in {"u1", "u4"}
    assert top_urls[2] == "u2"


def test_aggregate_empty_window(news_mod):
    score = news_mod.aggregate("AAPL", [],
                               datetime(2026, 5, 14), datetime(2026, 5, 21))
    assert score.n_total == 0
    assert score.sentiment_score == 0.0


# ---------------------------------------------------------------------------
# Orchestrator -- idempotent re-runs
# ---------------------------------------------------------------------------


def test_score_news_idempotent_skips_already_classified(news_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    from weekly_strategy.data import storage

    items = [
        NewsItem(ticker="AAPL", title="t1", url="https://x/1",
                 source="reuters.com", published_at=datetime(2026, 5, 20)),
        NewsItem(ticker="AAPL", title="t2", url="https://x/2",
                 source="bloomberg.com", published_at=datetime(2026, 5, 21)),
    ]
    storage.insert_news(items)

    captured = _stub_classifier(monkeypatch, news_mod, [
        {"i": 0, "sentiment": "POSITIVE", "materiality": "HIGH", "theme": "earnings"},
        {"i": 1, "sentiment": "NEGATIVE", "materiality": "MEDIUM", "theme": "competitive"},
    ])
    score1 = news_mod.score_news(
        "AAPL", since=datetime(2026, 5, 14), until=datetime(2026, 5, 22),
    )
    assert score1.n_classified == 2
    assert captured["calls"] == 1

    # Second run: no unclassified items remain, so no LLM call.
    score2 = news_mod.score_news(
        "AAPL", since=datetime(2026, 5, 14), until=datetime(2026, 5, 22),
    )
    assert captured["calls"] == 1  # unchanged
    assert score2.sentiment_score == pytest.approx(score1.sentiment_score)
