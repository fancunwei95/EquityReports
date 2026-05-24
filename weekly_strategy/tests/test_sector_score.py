from __future__ import annotations

"""Tests for signals/sector_score.py."""

import importlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pytest


@pytest.fixture
def mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import weekly_strategy.signals.sector_score as m
    m = importlib.reload(m)
    return m


@dataclass
class _FakeResp:
    text: str = ""
    cost_usd: float = 0.0
    session_id: str = "sess"
    raw: dict = None  # type: ignore[assignment]


def _stub_llm(monkeypatch, mod, payload):
    captured = {"calls": 0}

    def fake_ask_json(prompt, model, system_prompt=None, timeout_s=120.0):
        captured["calls"] += 1
        captured["prompt"] = prompt
        return payload, _FakeResp()
    monkeypatch.setattr(mod.client, "ask_json", fake_ask_json)
    return captured


def _dossier(sector="Technology"):
    from weekly_strategy.data.schemas import Dossier
    return Dossier(
        ticker="AAPL", company_name="Apple Inc.", sector=sector,
        last_updated=datetime(2026, 5, 24),
    )


def _regime(rate="cutting", finc="easing", phase="mid",
            hy="normal", vix="normal", narrative="x"):
    from weekly_strategy.data.schemas import MacroRegime
    return MacroRegime(
        week_ending=date(2026, 5, 24),
        rate_regime=rate, financial_conditions=finc, cycle_phase=phase,
        narrative=narrative,
    )


def _snap(*, leaders=None, sectors=None, breadth="narrow"):
    """Build a minimal SectorSnapshot."""
    from weekly_strategy.data.schemas import SectorMetrics, SectorSnapshot
    sectors = sectors or {}
    return SectorSnapshot(
        week_ending=date(2026, 5, 24),
        sectors={
            etf: (m if isinstance(m, SectorMetrics)
                  else SectorMetrics(etf=etf, sector="X", rel_1m=0.0))
            for etf, m in sectors.items()
        },
        leadership_ranking=leaders or [],
        breadth=breadth,
    )


# ---------------------------------------------------------------------------
# Regime fit
# ---------------------------------------------------------------------------


def test_cyclical_etf_favored_in_risk_on(mod):
    r = _regime(finc="easing", phase="mid")
    assert mod.regime_fit_for_etf("XLK", r) == 70.0
    assert mod.regime_fit_for_etf("XLY", r) == 70.0


def test_defensive_etf_favored_in_risk_off(mod):
    r = _regime(finc="tightening")
    assert mod.regime_fit_for_etf("XLP", r) == 70.0
    assert mod.regime_fit_for_etf("XLK", r) == 30.0


def test_xlf_favored_when_rates_hiking(mod):
    assert mod.regime_fit_for_etf("XLF", _regime(rate="hiking_or_holding")) == 70.0
    assert mod.regime_fit_for_etf("XLF", _regime(rate="cutting")) == 35.0


def test_xlre_hurt_when_rates_rising(mod):
    assert mod.regime_fit_for_etf("XLRE", _regime(rate="hiking_or_holding")) == 35.0
    assert mod.regime_fit_for_etf("XLRE", _regime(rate="cutting")) == 65.0


def test_regime_fit_none_regime_falls_to_50(mod):
    # No regime -> assume risk_on by default; cyclicals still get 70.
    assert mod.regime_fit_for_etf("XLK", None) == 70.0
    assert mod.regime_fit_for_etf("XLF", None) == 50.0  # falls through to default


# ---------------------------------------------------------------------------
# Momentum component
# ---------------------------------------------------------------------------


def test_momentum_component_linear_in_rank(mod):
    snap = _snap(leaders=["XLK", "XLF", "XLE", "XLY", "XLB"])
    s_top, rank_top = mod.sector_momentum_component("XLK", snap)
    s_mid, rank_mid = mod.sector_momentum_component("XLE", snap)
    s_low, rank_low = mod.sector_momentum_component("XLB", snap)
    assert s_top == 100.0 and rank_top == 1
    assert s_mid == pytest.approx(50.0) and rank_mid == 3
    assert s_low == 0.0 and rank_low == 5


def test_momentum_component_missing_etf(mod):
    snap = _snap(leaders=["XLK", "XLF"])
    s, rank = mod.sector_momentum_component("XLE", snap)
    assert s == 50.0  # neutral
    assert rank is None


# ---------------------------------------------------------------------------
# sector_score
# ---------------------------------------------------------------------------


def test_sector_score_full(mod):
    from weekly_strategy.data.schemas import SectorMetrics
    snap = _snap(
        leaders=["XLK", "XLF", "XLE", "XLY", "XLB"],
        sectors={"XLK": SectorMetrics(etf="XLK", sector="Technology", rel_1m=0.10)},
    )
    regime = _regime(finc="easing", phase="mid")  # risk_on, XLK favored
    out = mod.sector_score(
        dossier=_dossier(sector="Technology"),
        sector_snap=snap, regime=regime,
    )
    # XLK is rank 1 -> momentum 100, fit 70. With 60/40 weighting: 60+28=88
    assert out["score"] == pytest.approx(88.0)
    assert out["etf"] == "XLK"
    assert out["rank"] == 1
    assert out["in_favor"] is True
    assert out["rel_1m"] == pytest.approx(0.10)


def test_sector_score_unknown_sector_returns_neutral(mod):
    snap = _snap(leaders=["XLK"])
    regime = _regime()
    out = mod.sector_score(
        dossier=_dossier(sector="Unknown Sector"),
        sector_snap=snap, regime=regime,
    )
    assert out["etf"] is None
    # mom=50, fit=50 -> 50
    assert out["score"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Cross-stock implications
# ---------------------------------------------------------------------------


def _news(ticker="AAPL", n_material=3):
    from weekly_strategy.data.schemas import NewsClassification, NewsItem
    items = []
    for i in range(n_material):
        items.append(NewsItem(
            ticker=ticker, title=f"material item {i}",
            url=f"u{i}", source="reuters.com",
            published_at=datetime(2026, 5, 20),
            classification=NewsClassification(
                sentiment="POSITIVE", materiality="HIGH" if i == 0 else "MEDIUM",
                theme="earnings", rationale="...",
            ),
        ))
    # Add a LOW item that should be filtered out.
    items.append(NewsItem(
        ticker=ticker, title="noise item", url="ulow", source="x.com",
        published_at=datetime(2026, 5, 20),
        classification=NewsClassification(
            sentiment="NEUTRAL", materiality="LOW", theme="other",
        ),
    ))
    return items


def test_cross_stock_calls_llm_with_material_items_only(mod, monkeypatch):
    captured = _stub_llm(monkeypatch, mod, {
        "summary": "Tech leadership, AAPL captures dollar share",
        "implications_for_sector": ["Lifts XLK sentiment via Services beats"],
        "implications_from_sector": ["Tech +10% rel SPY is supportive"],
        "related_tickers": ["googl", "msft", "smsi"],  # last one is junk; should drop chars >6
    })
    out = mod.cross_stock_implications(
        "AAPL",
        dossier=_dossier(),
        news_items=_news(),
        sector_snap=_snap(leaders=["XLK", "XLF"]),
        regime=_regime(),
        week_ending=date(2026, 5, 24),
        company_name="Apple Inc.",
    )
    assert captured["calls"] == 1
    # The prompt enumerates only the material items, not the LOW one.
    assert "material item 0" in captured["prompt"]
    assert "noise item" not in captured["prompt"]
    assert out.sector_etf == "XLK"
    assert out.related_tickers == ["GOOGL", "MSFT", "SMSI"]
    assert "Tech leadership" in out.summary


def test_cross_stock_skips_llm_when_no_material(mod, monkeypatch):
    # No material items -> short-circuit, no LLM call.
    captured = _stub_llm(monkeypatch, mod, {"summary": "should not be used"})
    out = mod.cross_stock_implications(
        "AAPL", dossier=_dossier(),
        news_items=[],
        sector_snap=_snap(leaders=["XLK"]),
        regime=_regime(),
        week_ending=date(2026, 5, 24),
    )
    assert captured["calls"] == 0
    assert "No material news" in out.summary


def test_cross_stock_degrades_on_llm_failure(mod, monkeypatch):
    def boom(*_a, **_kw):
        raise mod.client.ClaudeCliError("outage")
    monkeypatch.setattr(mod.client, "ask_json", boom)
    out = mod.cross_stock_implications(
        "AAPL", dossier=_dossier(),
        news_items=_news(),
        sector_snap=_snap(leaders=["XLK"]),
        regime=_regime(),
        week_ending=date(2026, 5, 24),
    )
    assert out.sector_etf == "XLK"
    assert "LLM unavailable" in out.summary
