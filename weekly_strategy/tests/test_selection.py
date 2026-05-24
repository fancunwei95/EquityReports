from __future__ import annotations

"""Tests for signals/selection.py focus list + L/S selection."""

from datetime import date

import pytest

from weekly_strategy.data.schemas import StockScoreBundle
from weekly_strategy.signals import selection as sel


def _b(ticker, composite_z=None, composite=50.0, sector=None, beta=None,
       z_quality=None, z_news=None):
    return StockScoreBundle(
        ticker=ticker, week_ending=date(2026, 5, 24),
        quality_score=50, valuation_score=50, momentum_score=50,
        composite_score=composite, composite_z=composite_z,
        z_quality=z_quality, z_news_sentiment=z_news,
    )


def _build_bundles(spec):
    """spec is list of (ticker, composite_z)."""
    return {t: _b(t, composite_z=z) for t, z in spec}


# ---------------------------------------------------------------------------
# Focus list
# ---------------------------------------------------------------------------


def test_focus_list_top_and_bottom_by_z():
    bundles = _build_bundles([
        ("A", 2.0), ("B", 1.0), ("C", 0.5), ("D", 0.0),
        ("E", -0.5), ("F", -1.0), ("G", -2.0),
    ])
    fl = sel.build_focus_list(bundles, week_ending=date(2026, 5, 24),
                              top_k_longs=3, top_k_shorts=3)
    assert [i.ticker for i in fl.long_candidates] == ["A", "B", "C"]
    # Shorts: worst-first (G is rank=1 from the worst-end).
    assert [i.ticker for i in fl.short_candidates] == ["G", "F", "E"]
    # Long ranks count from the best (1, 2, 3).
    assert [i.rank for i in fl.long_candidates] == [1, 2, 3]
    # Short ranks count from the worst end (so G is 7, F is 6, E is 5 in a 7-name universe).
    assert [i.rank for i in fl.short_candidates] == [7, 6, 5]


def test_focus_list_falls_back_to_composite_score():
    bundles = {
        "A": _b("A", composite_z=None, composite=80.0),
        "B": _b("B", composite_z=None, composite=40.0),
    }
    fl = sel.build_focus_list(bundles, week_ending=date(2026, 5, 24),
                              top_k_longs=2, top_k_shorts=2)
    assert fl.long_candidates[0].ticker == "A"
    assert fl.short_candidates[0].ticker == "B"


def test_focus_list_outliers_picks_extreme_components():
    bundles = {
        "A": _b("A", composite_z=0.0, z_news=2.5),   # extreme news, mid composite
        "B": _b("B", composite_z=0.5, z_news=0.1),
        "C": _b("C", composite_z=-0.5, z_quality=-2.0),
    }
    fl = sel.build_focus_list(bundles, week_ending=date(2026, 5, 24),
                              top_k_longs=2, top_k_shorts=2,
                              outlier_threshold_z=1.5)
    outlier_tickers = {o.ticker for o in fl.outliers}
    assert "A" in outlier_tickers  # |z_news|=2.5
    assert "C" in outlier_tickers  # |z_quality|=2.0
    assert "B" not in outlier_tickers


def test_focus_list_attaches_sector_and_beta():
    bundles = {"A": _b("A", composite_z=2.0)}
    fl = sel.build_focus_list(
        bundles, week_ending=date(2026, 5, 24),
        top_k_longs=1, top_k_shorts=1,
        sector_lookup={"A": "Technology"},
        beta_lookup={"A": 1.25},
    )
    assert fl.long_candidates[0].sector == "Technology"
    assert fl.long_candidates[0].beta == 1.25


# ---------------------------------------------------------------------------
# Selection -- sector neutrality
# ---------------------------------------------------------------------------


def _fl_with_sectors(sectors_long, sectors_short, *, betas_long=None, betas_short=None):
    """Build a FocusList directly. sectors_long is a list of (ticker, sector)."""
    from weekly_strategy.data.schemas import FocusList, FocusListItem
    betas_long = betas_long or {}
    betas_short = betas_short or {}
    longs = [
        FocusListItem(
            ticker=t, composite_z=2.0 - i * 0.1, sector=s,
            beta=betas_long.get(t), rank=i + 1,
        )
        for i, (t, s) in enumerate(sectors_long)
    ]
    shorts = [
        FocusListItem(
            ticker=t, composite_z=-2.0 + i * 0.1, sector=s,
            beta=betas_short.get(t), rank=len(sectors_short) - i,
        )
        for i, (t, s) in enumerate(sectors_short)
    ]
    return FocusList(
        week_ending=date(2026, 5, 24),
        long_candidates=longs, short_candidates=shorts,
    )


def test_select_respects_sector_neutrality():
    fl = _fl_with_sectors(
        [
            ("T1", "Tech"), ("T2", "Tech"), ("T3", "Tech"), ("T4", "Tech"),
            ("F1", "Fin"),  ("F2", "Fin"),  ("H1", "HC"),   ("H2", "HC"),
        ],
        [
            ("S1", "Tech"), ("S2", "Tech"), ("S3", "Tech"),
            ("S4", "Fin"),  ("S5", "Fin"),  ("S6", "HC"),
        ],
    )
    port = sel.select_positions(fl, week_ending=date(2026, 5, 24),
                                position_count=5, max_per_sector=2)
    long_sectors = [p.sector for p in port.longs]
    short_sectors = [p.sector for p in port.shorts]
    assert long_sectors.count("Tech") <= 2
    assert short_sectors.count("Tech") <= 2
    # Names rejected for sector cap appear in rejected list.
    assert any("sector cap" in r["reason"] for r in port.rejected if r["side"] == "long")


# ---------------------------------------------------------------------------
# Selection -- exclusions
# ---------------------------------------------------------------------------


def test_select_excludes_names_reporting_earnings():
    fl = _fl_with_sectors(
        [("A", "Tech"), ("B", "Tech"), ("C", "Fin"), ("D", "HC"), ("E", "Industrials"), ("F", "Energy")],
        [("X", "Tech"), ("Y", "Fin"), ("Z", "HC")],
    )
    port = sel.select_positions(
        fl, week_ending=date(2026, 5, 24),
        position_count=5, earnings_exclude={"A", "C"},
    )
    long_tickers = {p.ticker for p in port.longs}
    assert "A" not in long_tickers
    assert "C" not in long_tickers
    assert any("earnings within" in r["reason"] for r in port.rejected)


def test_select_filters_short_by_short_interest():
    fl = _fl_with_sectors(
        [("L1", "Tech")],
        [("S1", "Tech"), ("S2", "Fin")],
    )
    port = sel.select_positions(
        fl, week_ending=date(2026, 5, 24),
        position_count=2, short_interest={"S1": 0.40, "S2": 0.10},
        short_interest_max=0.25,
    )
    short_tickers = {p.ticker for p in port.shorts}
    # S1 has 40% short interest, gets dropped; S2 (10%) is kept.
    assert short_tickers == {"S2"}


# ---------------------------------------------------------------------------
# Turnover dampening
# ---------------------------------------------------------------------------


def test_select_prefers_held_names_in_focus():
    """If a held name is still in the focus list (rank <= 10), it should be
    selected before a fresh top-ranked candidate."""
    fl = _fl_with_sectors(
        [("FRESH", "Tech"), ("HELD", "Tech")],  # FRESH ranks first by composite_z
        [("S1", "HC")],
    )
    port = sel.select_positions(
        fl, week_ending=date(2026, 5, 24),
        position_count=1, max_per_sector=1,
        held_long_tickers={"HELD"},
    )
    assert port.longs[0].ticker == "HELD"
    assert port.longs[0].selection_reason == "turnover_dampening"


def test_select_drops_held_name_if_past_rank_threshold():
    """If the held name has fallen past TURNOVER_RANK_THRESHOLD, no longer protected."""
    fl = _fl_with_sectors(
        [(f"T{i}", "Tech") for i in range(11)],  # 11 names; HELD is at rank 11
        [],
    )
    # Force the last name's rank past the threshold.
    from weekly_strategy.data.schemas import FocusListItem
    new_long = [
        it.model_copy(update={"rank": 11 if it.ticker == "T10" else it.rank})
        for it in fl.long_candidates
    ]
    fl = fl.model_copy(update={"long_candidates": new_long})
    port = sel.select_positions(
        fl, week_ending=date(2026, 5, 24),
        position_count=1, max_per_sector=11,
        held_long_tickers={"T10"},
    )
    # T10 not protected; T0 (top-ranked) wins.
    assert port.longs[0].ticker == "T0"


# ---------------------------------------------------------------------------
# Beta neutralization
# ---------------------------------------------------------------------------


def test_select_beta_neutralizes_short_weights():
    fl = _fl_with_sectors(
        [(f"L{i}", "Tech") for i in range(5)] + [(f"L{i}", "Fin") for i in range(5, 10)],
        [(f"S{i}", "Tech") for i in range(5)] + [(f"S{i}", "Fin") for i in range(5, 10)],
        betas_long={f"L{i}": 1.0 for i in range(10)},   # long basket beta = 1.0
        betas_short={f"S{i}": 0.5 for i in range(10)},  # short basket beta = 0.5
    )
    port = sel.select_positions(
        fl, week_ending=date(2026, 5, 24),
        position_count=5, max_per_sector=3,
    )
    assert port.long_basket_beta == pytest.approx(1.0, abs=1e-6)
    # Shorts get scaled to match long beta magnitude (0.5 -> 1.0 effective).
    assert port.short_basket_beta == pytest.approx(1.0, abs=1e-6)
    assert port.net_beta == pytest.approx(0.0, abs=1e-6)


def test_select_keeps_unknown_beta_safe():
    """If any short has no beta, basket beta is None and we don't rescale."""
    fl = _fl_with_sectors(
        [("L1", "Tech")],
        [("S1", "Tech"), ("S2", "Fin")],
        betas_long={"L1": 1.0}, betas_short={"S1": 0.7},  # S2 missing beta
    )
    port = sel.select_positions(
        fl, week_ending=date(2026, 5, 24), position_count=2,
    )
    assert port.short_basket_beta is None
