from __future__ import annotations

"""Tests for llm/conviction.py."""

from dataclasses import dataclass
from datetime import date

import pytest

from weekly_strategy.data.schemas import (
    FocusList, FocusListItem, Portfolio, PortfolioPosition, StockScoreBundle,
)
from weekly_strategy.llm import conviction


@dataclass
class _FakeResp:
    text: str = ""
    cost_usd: float = 0.012
    session_id: str = "sess"
    raw: dict = None  # type: ignore[assignment]


def _stub(monkeypatch, payload_or_factory):
    """payload_or_factory can be a static dict or a callable(prompt) -> dict."""
    captured = {"calls": 0}

    def fake_ask_json(prompt, model, system_prompt=None, timeout_s=120.0):
        captured["calls"] += 1
        captured["last_prompt"] = prompt
        payload = (
            payload_or_factory(prompt) if callable(payload_or_factory)
            else payload_or_factory
        )
        return payload, _FakeResp()

    monkeypatch.setattr(conviction.client, "ask_json", fake_ask_json)
    return captured


def _bundle(ticker):
    return StockScoreBundle(
        ticker=ticker, week_ending=date(2026, 5, 24),
        quality_score=70, valuation_score=60, momentum_score=80,
        composite_z=1.5, sector_etf="XLK",
    )


def _position(ticker, side="long", sector="Technology"):
    return PortfolioPosition(
        ticker=ticker, direction=side, sector=sector,
        composite_z=1.5, beta=1.0, weight=0.20,
    )


# ---------------------------------------------------------------------------
# Single check
# ---------------------------------------------------------------------------


def test_conviction_check_parses_full_response(monkeypatch):
    _stub(monkeypatch, {
        "thesis": "Quality compounder with tech tailwinds and FY26 visibility.",
        "risks": ["valuation reset", "AI capex digestion"],
        "confidence": "HIGH",
        "flags": [],
    })
    c = conviction.conviction_check(
        _position("AAPL"), bundle=_bundle("AAPL"),
    )
    assert c.ticker == "AAPL"
    assert c.confidence == "HIGH"
    assert len(c.risks) == 2
    assert c.thesis.startswith("Quality compounder")


def test_conviction_check_coerces_unknown_confidence(monkeypatch):
    _stub(monkeypatch, {"confidence": "very high maybe?"})
    c = conviction.conviction_check(_position("AAPL"), bundle=_bundle("AAPL"))
    assert c.confidence == "MEDIUM"


def test_conviction_check_handles_llm_failure(monkeypatch):
    def boom(*_a, **_kw):
        raise conviction.client.ClaudeCliError("outage")
    monkeypatch.setattr(conviction.client, "ask_json", boom)
    c = conviction.conviction_check(_position("AAPL"), bundle=_bundle("AAPL"))
    assert c.confidence == "MEDIUM"  # neutral fallback (do not auto-drop)
    assert any("LLM call failed" in f for f in c.flags)


def test_conviction_check_includes_score_and_direction_in_prompt(monkeypatch):
    captured = _stub(monkeypatch, {"confidence": "MEDIUM"})
    conviction.conviction_check(_position("XOM", side="short"), bundle=_bundle("XOM"))
    p = captured["last_prompt"]
    assert "short XOM" in p
    assert "quality" in p.lower()


# ---------------------------------------------------------------------------
# filter_by_conviction
# ---------------------------------------------------------------------------


def _portfolio(longs, shorts):
    return Portfolio(
        week_ending=date(2026, 5, 24),
        longs=[_position(t) for t in longs],
        shorts=[_position(t, side="short") for t in shorts],
    )


def _focus(longs, shorts):
    return FocusList(
        week_ending=date(2026, 5, 24),
        long_candidates=[
            FocusListItem(
                ticker=t, composite_z=2.0 - i*0.1, sector="Technology", beta=1.0,
                rank=i + 1,
            )
            for i, t in enumerate(longs)
        ],
        short_candidates=[
            FocusListItem(
                ticker=t, composite_z=-2.0 + i*0.1, sector="Healthcare", beta=1.0,
                rank=len(shorts) - i,
            )
            for i, t in enumerate(shorts)
        ],
    )


def test_filter_drops_low_and_substitutes_from_focus(monkeypatch):
    # AAPL gets LOW; MSFT is the next focus candidate and gets HIGH.
    def factory(prompt):
        if "long AAPL" in prompt:
            return {"confidence": "LOW", "flags": ["scores inconsistent"]}
        return {"confidence": "HIGH", "thesis": "good"}

    captured = _stub(monkeypatch, factory)
    portfolio = _portfolio(longs=["AAPL"], shorts=[])
    focus = _focus(longs=["AAPL", "MSFT", "NVDA"], shorts=[])
    bundles = {t: _bundle(t) for t in ("AAPL", "MSFT", "NVDA")}

    new, checks = conviction.filter_by_conviction(
        portfolio, focus=focus, bundles=bundles,
    )
    assert [p.ticker for p in new.longs] == ["MSFT"]
    assert new.longs[0].selection_reason == "substitute"
    assert checks["AAPL"].confidence == "LOW"
    assert checks["MSFT"].confidence == "HIGH"


def test_filter_keeps_high_and_medium(monkeypatch):
    _stub(monkeypatch, {"confidence": "MEDIUM"})
    portfolio = _portfolio(longs=["A", "B"], shorts=["X"])
    focus = _focus(longs=["A", "B"], shorts=["X"])
    new, checks = conviction.filter_by_conviction(
        portfolio, focus=focus, bundles={t: _bundle(t) for t in ("A", "B", "X")},
    )
    assert {p.ticker for p in new.longs} == {"A", "B"}
    assert {p.ticker for p in new.shorts} == {"X"}


def test_filter_substitution_respects_sector_cap(monkeypatch):
    """If two longs go LOW in the same sector, the substitute must not violate
    max_per_sector when a third tech name is the next candidate."""
    def factory(prompt):
        if "long T1" in prompt or "long T2" in prompt:
            return {"confidence": "LOW"}
        return {"confidence": "HIGH"}

    _stub(monkeypatch, factory)
    portfolio = _portfolio(longs=["T1", "T2"], shorts=[])
    focus = FocusList(
        week_ending=date(2026, 5, 24),
        long_candidates=[
            FocusListItem(ticker="T1", composite_z=2.0, sector="Technology", rank=1),
            FocusListItem(ticker="T2", composite_z=1.8, sector="Technology", rank=2),
            FocusListItem(ticker="T3", composite_z=1.6, sector="Technology", rank=3),
            FocusListItem(ticker="F1", composite_z=1.4, sector="Financials", rank=4),
            FocusListItem(ticker="F2", composite_z=1.2, sector="Financials", rank=5),
        ],
    )
    new, _checks = conviction.filter_by_conviction(
        portfolio, focus=focus,
        bundles={t: _bundle(t) for t in ("T1", "T2", "T3", "F1", "F2")},
        max_per_sector=2,
    )
    # T1+T2 dropped LOW. Substitutes from focus must respect Tech cap=2.
    # Since both originals (Tech) are gone, T3 (Tech) brings count back to 1,
    # then F1 (Fin) is next.
    new_tickers = [p.ticker for p in new.longs]
    assert "T3" in new_tickers
    assert "F1" in new_tickers
    # Never 3 Tech total in book (we have 2 slots).
    tech_count = sum(1 for p in new.longs if p.sector == "Technology")
    assert tech_count <= 2
