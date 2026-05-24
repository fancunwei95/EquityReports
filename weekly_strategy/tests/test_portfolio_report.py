from __future__ import annotations

"""Tests for reporting/portfolio.py."""

from datetime import date, datetime
from pathlib import Path

import pytest

from weekly_strategy.data.schemas import (
    ConvictionCheck, FedPosture, FocusList, FocusListItem,
    MacroRegime, MacroSnapshot, Portfolio, PortfolioPosition,
    SectorMetrics, SectorSnapshot, StockScoreBundle,
)
from weekly_strategy.reporting import portfolio as port


def _pos(ticker, side="long", sector="Technology", beta=1.0, comp_z=1.5):
    return PortfolioPosition(
        ticker=ticker, direction=side, sector=sector,
        composite_z=comp_z, beta=beta, weight=0.20,
    )


def _portfolio(longs=("AAPL", "MSFT"), shorts=("XYZ", "ABC"),
               net_beta=0.05, rejected=None):
    return Portfolio(
        week_ending=date(2026, 5, 24),
        longs=[_pos(t) for t in longs],
        shorts=[_pos(t, side="short", sector="Healthcare", beta=0.8) for t in shorts],
        long_basket_beta=1.0, short_basket_beta=-0.95, net_beta=net_beta,
        rejected=rejected or [],
    )


# ---------------------------------------------------------------------------
# Section presence
# ---------------------------------------------------------------------------


def test_renders_header_and_position_sections():
    p = _portfolio()
    md = port.render_portfolio_markdown(p)
    assert "# Weekly Portfolio Report" in md
    assert "Week ending **2026-05-24**" in md
    assert "## Longs (2)" in md
    assert "## Shorts (2)" in md
    assert "`AAPL`" in md and "`XYZ`" in md


def test_renders_market_context_when_inputs_present():
    regime = MacroRegime(
        week_ending=date(2026, 5, 24),
        rate_regime="cutting", financial_conditions="easing",
        cycle_phase="mid", narrative="Soft-landing read holding.",
    )
    snap = MacroSnapshot(
        week_ending=date(2026, 5, 24),
        yield_10y=4.50, curve_2s10s_bps=15.0, hy_oas_bps=380.0,
        vix_level=18.0,
    )
    fed = FedPosture(week_ending=date(2026, 5, 24), posture="SLIGHTLY_DOVISH",
                     n_items=4, summary="Patience messaging.")
    sector_snap = SectorSnapshot(
        week_ending=date(2026, 5, 24),
        sectors={},
        leadership_ranking=["XLK", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLB"],
        breadth="broad",
    )
    md = port.render_portfolio_markdown(
        _portfolio(),
        macro_snapshot=snap, macro_regime=regime,
        fed_posture=fed, sector_snap=sector_snap,
    )
    assert "## Market context" in md
    assert "rate `cutting`" in md
    assert "SLIGHTLY_DOVISH" in md
    assert "10y 4.50%" in md
    assert "leaders XLK, XLF, XLE" in md
    assert "Soft-landing read" in md


def test_renders_conviction_thesis_and_risks():
    p = _portfolio(longs=("AAPL",))
    checks = {
        "AAPL": ConvictionCheck(
            ticker="AAPL", direction="long", confidence="HIGH",
            thesis="Quality compounder with FY26 catalysts.",
            risks=["valuation reset", "AI execution"],
            flags=["news sentiment +0.48 vs valuation 0.7 -- tension noted"],
        ),
    }
    md = port.render_portfolio_markdown(p, convictions=checks)
    assert "Conviction: **HIGH**" in md
    assert "Quality compounder" in md
    assert "valuation reset" in md
    assert "tension noted" in md  # data flags rendered


def test_changes_vs_last_week():
    p = _portfolio(longs=("AAPL", "MSFT"), shorts=("XYZ", "ABC"))
    md = port.render_portfolio_markdown(
        p,
        previous_longs={"AAPL", "GOOGL"},   # AAPL held, GOOGL dropped, MSFT added
        previous_shorts={"XYZ"},             # ABC added
    )
    # Looking for specific list contents in the changes section.
    changes_block = md.split("## Changes vs last week")[1].split("## ")[0]
    assert "`MSFT`" in changes_block          # added
    assert "`GOOGL`" in changes_block          # dropped
    assert "`AAPL`" in changes_block           # held
    assert "`ABC`" in changes_block            # short added


def test_portfolio_metrics_section():
    p = _portfolio(net_beta=0.05)
    md = port.render_portfolio_markdown(p)
    assert "## Portfolio metrics" in md
    assert "Long basket beta: +1.00" in md
    assert "Net beta: +0.05" in md
    assert "Technology (2)" in md   # long sector exposure
    assert "Healthcare (2)" in md   # short sector exposure


def test_watch_list_excludes_already_selected():
    focus = FocusList(
        week_ending=date(2026, 5, 24),
        long_candidates=[
            FocusListItem(ticker=t, composite_z=2.0 - i*0.1, rank=i+1)
            for i, t in enumerate(["AAPL", "MSFT", "GOOGL", "NVDA"])
        ],
        outliers=[
            FocusListItem(ticker="META", composite_z=0.0,
                          outlier_reason="extreme news sentiment (z=+2.40)"),
        ],
    )
    p = _portfolio(longs=("AAPL", "MSFT"), shorts=())
    md = port.render_portfolio_markdown(p, focus=focus)
    assert "## Watch list" in md
    assert "GOOGL" in md   # next ranked, not in book
    assert "NVDA" in md
    # AAPL/MSFT already in book; shouldn't be repeated in the long bench list.
    assert md.count("AAPL") == 1  # only in the position section
    assert "META" in md   # outlier surfaced


def test_risk_flags_for_earnings_and_net_beta():
    p = _portfolio(net_beta=0.45)  # outside ±0.30 band
    md = port.render_portfolio_markdown(
        p, earnings_calendar={"AAPL": "2026-05-27", "MSFT": "2026-05-28"},
    )
    assert "## Risk flags" in md
    assert "AAPL: 2026-05-27" in md
    assert "outside target band" in md


def test_renders_rejections_when_present():
    rejected = [
        {"ticker": "X1", "side": "long", "reason": "sector cap (Technology)"},
        {"ticker": "Y1", "side": "short", "reason": "earnings within exclusion window"},
    ]
    p = _portfolio(rejected=rejected)
    md = port.render_portfolio_markdown(p)
    assert "## Rejections from initial focus list" in md
    assert "`X1` (long): sector cap" in md


def test_save_writes_to_daily_reports_dir(tmp_path, monkeypatch):
    import importlib
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DAILY_REPORTS_DIR", tmp_path / "daily_reports")
    import weekly_strategy.reporting.portfolio as mod
    mod = importlib.reload(mod)
    p = _portfolio()
    md = mod.render_portfolio_markdown(p)
    path = mod.save_portfolio_markdown(p, md)
    assert path.exists()
    assert path.parent.name == "daily_reports"
    assert path.name == "2026-05-24.md"
