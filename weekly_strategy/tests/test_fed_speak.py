from __future__ import annotations

"""Tests for the Fed speak tracker (Step 2.2)."""

import importlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def macro_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "FRED_API_KEY", "test-key")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.data.macro as mod
    mod = importlib.reload(mod)
    return mod


@dataclass
class _FakeResp:
    text: str = ""
    cost_usd: float = 0.0
    session_id: str = "sess"
    raw: dict = None  # type: ignore[assignment]


def _entry(title, source_title, link):
    src = SimpleNamespace(title=source_title, href=link)
    e = SimpleNamespace(title=title, link=link, source=src,
                        summary="snippet", published_parsed=(2026, 5, 20, 9, 0, 0))
    e.get = lambda key, default=None: getattr(e, key, default)
    return e


def _stub_llm(monkeypatch, mod, payload):
    captured = {"calls": 0}

    def fake_ask_json(prompt, model, system_prompt=None, timeout_s=120.0):
        captured["calls"] += 1
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return payload, _FakeResp()

    monkeypatch.setattr(mod.client, "ask_json", fake_ask_json)
    return captured


# ---------------------------------------------------------------------------
# fetch_fed_speak_news
# ---------------------------------------------------------------------------


def test_fetch_fed_news_dedupes_and_whitelists(macro_mod, monkeypatch):
    feed = SimpleNamespace(entries=[
        _entry("Powell signals openness to cuts", "Reuters", "https://reuters.com/a"),
        _entry("Powell signals openness to cuts!!", "Bloomberg", "https://bloomberg.com/b"),  # dup
        _entry("Random untrusted Fed rumor", "Joe's Tips", "https://joes.example/c"),         # filtered
        _entry("Williams pushes back on near-term cuts", "WSJ", "https://wsj.com/d"),
    ])
    monkeypatch.setattr(macro_mod.feedparser, "parse", lambda _url: feed)
    items = macro_mod.fetch_fed_speak_news(days=7)
    titles = [it.title for it in items]
    assert "Powell signals openness to cuts" in titles
    assert "Williams pushes back on near-term cuts" in titles
    assert len(items) == 2
    assert all(it.ticker == "FED" for it in items)


# ---------------------------------------------------------------------------
# classify_fed_posture
# ---------------------------------------------------------------------------


def test_classify_returns_neutral_when_no_items(macro_mod):
    posture = macro_mod.classify_fed_posture([], week_ending=date(2026, 5, 24))
    assert posture.posture == "NEUTRAL"
    assert posture.n_items == 0
    assert posture.summary is None


def test_classify_parses_full_response(macro_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    items = [
        NewsItem(ticker="FED", title="Powell hints at pause", url="u1",
                 source="reuters.com", published_at=datetime(2026, 5, 20)),
        NewsItem(ticker="FED", title="Williams: more data needed", url="u2",
                 source="wsj.com", published_at=datetime(2026, 5, 21)),
    ]
    captured = _stub_llm(monkeypatch, macro_mod, {
        "posture": "SLIGHTLY_HAWKISH",
        "summary": "Speakers signaled patience.",
        "policy_hints": ["pause messaging", "watching CPI"],
        "key_speakers": ["Powell", "Williams"],
    })
    posture = macro_mod.classify_fed_posture(items, week_ending=date(2026, 5, 24))
    assert captured["calls"] == 1
    assert "Powell hints at pause" in captured["prompt"]
    assert posture.posture == "SLIGHTLY_HAWKISH"
    assert posture.summary == "Speakers signaled patience."
    assert posture.policy_hints == ["pause messaging", "watching CPI"]
    assert posture.key_speakers == ["Powell", "Williams"]
    assert posture.n_items == 2


def test_classify_coerces_unknown_posture(macro_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    _stub_llm(monkeypatch, macro_mod, {"posture": "very hawkish maybe?"})
    posture = macro_mod.classify_fed_posture(
        [NewsItem(ticker="FED", title="x", url="u1",
                  published_at=datetime(2026, 5, 20))],
        week_ending=date(2026, 5, 24),
    )
    assert posture.posture == "NEUTRAL"  # safe default on unknown


def test_classify_handles_dashes_and_spaces_in_posture(macro_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem
    _stub_llm(monkeypatch, macro_mod, {"posture": "slightly-dovish"})
    posture = macro_mod.classify_fed_posture(
        [NewsItem(ticker="FED", title="x", url="u1",
                  published_at=datetime(2026, 5, 20))],
        week_ending=date(2026, 5, 24),
    )
    assert posture.posture == "SLIGHTLY_DOVISH"


def test_classify_degrades_on_llm_error(macro_mod, monkeypatch):
    from weekly_strategy.data.schemas import NewsItem

    def boom(*_a, **_kw):
        raise macro_mod.client.ClaudeCliError("outage")
    monkeypatch.setattr(macro_mod.client, "ask_json", boom)
    posture = macro_mod.classify_fed_posture(
        [NewsItem(ticker="FED", title="x", url="u1",
                  published_at=datetime(2026, 5, 20))],
        week_ending=date(2026, 5, 24),
    )
    assert posture.posture == "NEUTRAL"
    assert posture.n_items == 1
