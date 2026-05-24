from __future__ import annotations

"""Macro regime classifier (Step 2.4).

Two deterministic labels + one LLM synthesis:

* ``rate_regime``          -- direction of monetary policy: cutting,
                              on_hold, hiking_or_holding (from FedPosture)
                              or tightening/easing_market (from the bond
                              market alone if no Fed read is available).
* ``financial_conditions`` -- easing / stable / tightening, derived from
                              week-on-week direction of HY OAS, real yields,
                              VIX, and DXY (majority vote).
* ``cycle_phase``          -- Sonnet read across the snapshot + posture.

The narrative + risks come from the same Sonnet call. Degrades to
unknown / empty when the LLM call fails -- a macro layer should never
crash a weekly run.
"""

import json
from datetime import date

from weekly_strategy.data.schemas import FedPosture, MacroRegime, MacroSnapshot
from weekly_strategy.llm import client, prompts


# Threshold parameters (basis points / fractional). Calibrated to "noticeable
# weekly move, not noise".
_RATE_BPS_THRESHOLD = 10.0      # 10y move that we'd call "market easing/tightening"
_HY_BPS_THRESHOLD = 10.0
_REAL_BPS_THRESHOLD = 5.0
_VIX_THRESHOLD = 2.0
_DXY_THRESHOLD = 0.005


def classify_rate_regime(snap: MacroSnapshot, fed: FedPosture | None) -> str:
    """Prefer the Fed's stated posture; fall back to the bond market."""
    if fed is not None and fed.n_items > 0:
        if fed.posture in ("HAWKISH", "SLIGHTLY_HAWKISH"):
            return "hiking_or_holding"
        if fed.posture in ("DOVISH", "SLIGHTLY_DOVISH"):
            return "cutting"
        if fed.posture == "NEUTRAL":
            return "on_hold"
    # No Fed read -- defer to long-end direction.
    delta = snap.yield_10y_wow_change_bps
    if delta is None:
        return "unknown"
    if delta > _RATE_BPS_THRESHOLD:
        return "tightening_market"
    if delta < -_RATE_BPS_THRESHOLD:
        return "easing_market"
    return "on_hold"


def classify_financial_conditions(snap: MacroSnapshot) -> str:
    """Majority vote of HY OAS / real yields / VIX / DXY weekly changes.

    Higher OAS, real yields, VIX, or DXY all tighten financial conditions.
    """
    signals: list[str] = []
    if snap.hy_oas_wow_change_bps is not None:
        signals.append(_sign(snap.hy_oas_wow_change_bps, _HY_BPS_THRESHOLD))
    if snap.real_yield_10y_wow_change_bps is not None:
        signals.append(_sign(snap.real_yield_10y_wow_change_bps, _REAL_BPS_THRESHOLD))
    if snap.vix_wow_change is not None:
        signals.append(_sign(snap.vix_wow_change, _VIX_THRESHOLD))
    if snap.dxy_wow_change_pct is not None:
        signals.append(_sign(snap.dxy_wow_change_pct, _DXY_THRESHOLD))

    if not signals:
        return "unknown"
    tight = signals.count("tight")
    ease = signals.count("ease")
    if tight >= 3 and tight > ease:
        return "tightening"
    if ease >= 3 and ease > tight:
        return "easing"
    return "stable"


def _sign(value: float, threshold: float) -> str:
    if value > threshold:
        return "tight"
    if value < -threshold:
        return "ease"
    return "flat"


# ---------------------------------------------------------------------------
# LLM synthesis (cycle phase + narrative)
# ---------------------------------------------------------------------------


def classify_regime(
    snap: MacroSnapshot,
    *,
    fed: FedPosture | None = None,
    model: str = client.MODEL_SONNET,
) -> MacroRegime:
    """Build a full MacroRegime: deterministic buckets + Sonnet synthesis."""
    rate = classify_rate_regime(snap, fed)
    finc = classify_financial_conditions(snap)

    snapshot_block = _render_snapshot_for_llm(snap)
    fed_block = _render_fed_for_llm(fed)
    user_prompt = prompts.MACRO_REGIME_USER_TEMPLATE.format(
        week_ending=snap.week_ending.isoformat(),
        snapshot_block=snapshot_block,
        fed_block=fed_block,
        rate_regime=rate,
        financial_conditions=finc,
        hy_regime=snap.hy_regime,
        vix_regime=snap.vix_regime,
    )

    try:
        parsed, _resp = client.ask_json(
            user_prompt, model=model, system_prompt=prompts.MACRO_REGIME_SYSTEM,
        )
    except client.ClaudeCliError:
        return MacroRegime(
            week_ending=snap.week_ending,
            rate_regime=rate,
            financial_conditions=finc,
            cycle_phase="unknown",
            narrative=None,
        )

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        parsed = {}

    return MacroRegime.coerce_llm(
        week_ending=snap.week_ending,
        rate_regime=rate,
        financial_conditions=finc,
        raw=parsed,
    )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_snapshot_for_llm(snap: MacroSnapshot) -> str:
    lines: list[str] = []

    def line(label: str, value: str) -> None:
        lines.append(f"  {label:<24}: {value}")

    line("10y yield", _pct_pct(snap.yield_10y, snap.yield_10y_wow_change_bps))
    line("2y yield",  _pct_pct(snap.yield_2y, snap.yield_2y_wow_change_bps))
    line("Real 10y", _pct_pct(snap.real_yield_10y, snap.real_yield_10y_wow_change_bps))
    line("Curve 2s10s", _bps_only(snap.curve_2s10s_bps))
    if snap.curve_inverted:
        line("Curve state", "inverted")
    line("HY OAS",  _bps_with_wow(snap.hy_oas_bps, snap.hy_oas_wow_change_bps))
    line("IG OAS",  _bps_only(snap.ig_oas_bps))
    line("VIX",     _level_change(snap.vix_level, snap.vix_wow_change))
    line("Breakeven 10y", _pct_pct(snap.breakeven_10y, snap.breakeven_10y_wow_change_bps))
    line("DXY",     _level_pct(snap.dxy_level, snap.dxy_wow_change_pct))
    line("Fed funds", _pct(snap.fed_funds))
    line("30y mortgage", _pct(snap.mortgage_30y))
    if snap.recent_prints:
        lines.append("  recent prints:")
        for p in snap.recent_prints:
            lines.append(
                f"    - {p.get('label', p.get('series'))}: {p.get('value')} "
                f"(prior {p.get('prior')}, change {p.get('change')})"
            )
    return "\n".join(lines)


def _render_fed_for_llm(fed: FedPosture | None) -> str:
    if fed is None or fed.n_items == 0:
        return "  (no Fed speak this week)"
    lines = [f"  posture: {fed.posture} ({fed.n_items} items)"]
    if fed.summary:
        lines.append(f"  summary: {fed.summary}")
    if fed.policy_hints:
        lines.append("  hints: " + "; ".join(fed.policy_hints))
    if fed.key_speakers:
        lines.append("  speakers: " + ", ".join(fed.key_speakers))
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return f"{x:.2f}%" if x is not None else "n/a"


def _pct_pct(level: float | None, wow_bps: float | None) -> str:
    if level is None:
        return "n/a"
    if wow_bps is None:
        return f"{level:.2f}%"
    return f"{level:.2f}% ({wow_bps:+.1f} bps wow)"


def _bps_only(bps: float | None) -> str:
    return f"{bps:.0f} bps" if bps is not None else "n/a"


def _bps_with_wow(bps: float | None, wow: float | None) -> str:
    if bps is None:
        return "n/a"
    if wow is None:
        return f"{bps:.0f} bps"
    return f"{bps:.0f} bps ({wow:+.1f} wow)"


def _level_change(level: float | None, delta: float | None) -> str:
    if level is None:
        return "n/a"
    if delta is None:
        return f"{level:.1f}"
    return f"{level:.1f} ({delta:+.1f} wow)"


def _level_pct(level: float | None, pct: float | None) -> str:
    if level is None:
        return "n/a"
    if pct is None:
        return f"{level:.2f}"
    return f"{level:.2f} ({pct*100:+.2f}% wow)"
