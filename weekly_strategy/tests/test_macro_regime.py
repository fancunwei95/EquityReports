from __future__ import annotations

"""Tests for signals/macro_regime.py."""

import importlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest


@pytest.fixture
def regime_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.signals.macro_regime as mod
    mod = importlib.reload(mod)
    return mod


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


def _snap(**overrides):
    from weekly_strategy.data.schemas import MacroSnapshot
    base = dict(week_ending=date(2026, 5, 24))
    base.update(overrides)
    return MacroSnapshot(**base)


def _fed(posture="NEUTRAL", n_items=2, summary="..."):
    from weekly_strategy.data.schemas import FedPosture
    return FedPosture(
        week_ending=date(2026, 5, 24),
        posture=posture, n_items=n_items, summary=summary,
    )


# ---------------------------------------------------------------------------
# rate_regime
# ---------------------------------------------------------------------------


def test_rate_regime_uses_fed_posture_when_available(regime_mod):
    snap = _snap()
    assert regime_mod.classify_rate_regime(snap, _fed("HAWKISH")) == "hiking_or_holding"
    assert regime_mod.classify_rate_regime(snap, _fed("SLIGHTLY_DOVISH")) == "cutting"
    assert regime_mod.classify_rate_regime(snap, _fed("NEUTRAL")) == "on_hold"


def test_rate_regime_falls_back_to_long_end_when_no_fed_read(regime_mod):
    # No Fed news this week.
    fed_empty = _fed(posture="NEUTRAL", n_items=0)
    assert regime_mod.classify_rate_regime(
        _snap(yield_10y_wow_change_bps=20.0), fed_empty,
    ) == "tightening_market"
    assert regime_mod.classify_rate_regime(
        _snap(yield_10y_wow_change_bps=-15.0), fed_empty,
    ) == "easing_market"
    assert regime_mod.classify_rate_regime(
        _snap(yield_10y_wow_change_bps=2.0), fed_empty,
    ) == "on_hold"
    assert regime_mod.classify_rate_regime(_snap(), fed_empty) == "unknown"


# ---------------------------------------------------------------------------
# financial_conditions
# ---------------------------------------------------------------------------


def test_financial_conditions_tightening_via_majority_vote(regime_mod):
    snap = _snap(
        hy_oas_wow_change_bps=15,             # tight
        real_yield_10y_wow_change_bps=8,      # tight
        vix_wow_change=3.0,                   # tight
        dxy_wow_change_pct=-0.001,            # flat
    )
    assert regime_mod.classify_financial_conditions(snap) == "tightening"


def test_financial_conditions_easing_via_majority_vote(regime_mod):
    snap = _snap(
        hy_oas_wow_change_bps=-15,            # ease
        real_yield_10y_wow_change_bps=-8,     # ease
        vix_wow_change=-3.0,                  # ease
        dxy_wow_change_pct=0.001,             # flat
    )
    assert regime_mod.classify_financial_conditions(snap) == "easing"


def test_financial_conditions_stable_on_mixed_signals(regime_mod):
    snap = _snap(
        hy_oas_wow_change_bps=15,             # tight
        real_yield_10y_wow_change_bps=-8,     # ease
        vix_wow_change=3.0,                   # tight
        dxy_wow_change_pct=-0.01,             # ease
    )
    assert regime_mod.classify_financial_conditions(snap) == "stable"


def test_financial_conditions_unknown_when_no_data(regime_mod):
    assert regime_mod.classify_financial_conditions(_snap()) == "unknown"


# ---------------------------------------------------------------------------
# Full classify_regime + LLM synthesis
# ---------------------------------------------------------------------------


def test_classify_regime_full(regime_mod, monkeypatch):
    snap = _snap(
        yield_10y=4.5, yield_10y_wow_change_bps=12.0,
        yield_2y=4.7, hy_oas_bps=380, hy_oas_wow_change_bps=15,
        vix_level=22.0, vix_wow_change=3.0, hy_regime="normal", vix_regime="elevated",
    )
    fed = _fed("SLIGHTLY_HAWKISH", n_items=3, summary="Speakers signaled patience.")
    captured = _stub_llm(monkeypatch, regime_mod, {
        "cycle_phase": "late",
        "narrative": "Late-cycle: HY OAS at 380bps but widening 15bps, vol elevated.",
        "risks": ["credit dislocation", "growth shock"],
    })
    out = regime_mod.classify_regime(snap, fed=fed)
    assert captured["calls"] == 1
    assert out.rate_regime == "hiking_or_holding"
    # finc: HY tight, real None, vix tight, dxy None -> 2 tight signals < 3 needed
    assert out.financial_conditions == "stable"
    assert out.cycle_phase == "late"
    assert "HY OAS" in out.narrative
    assert out.risks == ["credit dislocation", "growth shock"]


def test_classify_regime_coerces_unknown_cycle_phase(regime_mod, monkeypatch):
    _stub_llm(monkeypatch, regime_mod, {
        "cycle_phase": "weird_phase", "narrative": "x", "risks": [],
    })
    out = regime_mod.classify_regime(_snap(), fed=_fed())
    assert out.cycle_phase == "unknown"


def test_classify_regime_degrades_on_llm_failure(regime_mod, monkeypatch):
    def boom(*_a, **_kw):
        raise regime_mod.client.ClaudeCliError("outage")
    monkeypatch.setattr(regime_mod.client, "ask_json", boom)
    snap = _snap(yield_10y_wow_change_bps=20.0)
    out = regime_mod.classify_regime(snap, fed=_fed(n_items=0))
    # Deterministic buckets still populated.
    assert out.rate_regime == "tightening_market"
    assert out.cycle_phase == "unknown"
    assert out.narrative is None
