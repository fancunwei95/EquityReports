from __future__ import annotations

"""Tests for the dossier builder.

Uses small synthetic companyfacts JSON rather than real EDGAR fixtures so
the cases under test are explicit. The orchestrator test mocks the fetchers
and storage to keep the suite offline + isolated.
"""

import importlib
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated storage and fresh builder
# ---------------------------------------------------------------------------


@pytest.fixture
def builder_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    import weekly_strategy.data.fetchers as fetchers_mod
    fetchers_mod = importlib.reload(fetchers_mod)

    import weekly_strategy.dossier.builder as builder_module
    builder_module = importlib.reload(builder_module)
    return builder_module


# ---------------------------------------------------------------------------
# Helpers to build synthetic companyfacts JSON
# ---------------------------------------------------------------------------


def _fy(end: str, filed: str, val: float, *, start: str | None = None) -> dict:
    """Annual entry. start is None for balance-sheet items (point-in-time)."""
    return {
        "val": val, "end": end, "start": start, "filed": filed,
        "fp": "FY", "fy": int(end[:4]), "form": "10-K",
    }


def _q(fp: str, end: str, start: str, val: float, filed: str | None = None) -> dict:
    """Discrete quarterly entry. ~90 day duration."""
    return {
        "val": val, "end": end, "start": start, "filed": filed or end,
        "fp": fp, "fy": int(end[:4]), "form": "10-Q",
    }


def _pit(end: str, filed: str, val: float) -> dict:
    """Point-in-time entry (balance sheet)."""
    return {
        "val": val, "end": end, "start": None, "filed": filed,
        "fp": "FY", "fy": int(end[:4]), "form": "10-K",
    }


def _facts(concepts: dict[str, list[dict]]) -> dict:
    """Wrap entries into the nested EDGAR companyfacts shape.

    Each concept is keyed by canonical name; this helper maps it to the
    correct (namespace, tag) per builder.CONCEPT_TAGS so the extractor finds
    it on the first try.
    """
    from weekly_strategy.dossier.builder import CONCEPT_TAGS
    facts: dict = {"facts": {}}
    for concept, entries in concepts.items():
        ns, tag = CONCEPT_TAGS[concept][0]
        unit = "shares" if concept == "shares_outstanding" else "USD"
        facts["facts"].setdefault(ns, {})[tag] = {"units": {unit: entries}}
    return facts


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_extract_concept_picks_first_matching_tag(builder_mod):
    # Put revenue under the second-choice tag; extractor should still find it.
    facts = {
        "facts": {
            "us-gaap": {
                "SalesRevenueNet": {
                    "units": {"USD": [_fy("2025-12-31", "2026-02-01", 100.0)]}
                }
            }
        }
    }
    entries = builder_mod.extract_concept(facts, "revenue")
    assert len(entries) == 1
    assert entries[0]["val"] == 100.0


def test_latest_fy_and_quarter(builder_mod):
    entries = [
        _fy("2023-12-31", "2024-02-01", 100),
        _fy("2024-12-31", "2025-02-01", 110),
        _q("Q1", "2025-03-31", "2025-01-01", 27),
        _q("Q2", "2025-06-30", "2025-04-01", 29),
        # Cumulative YTD entry: 180-day span. Should NOT be picked over discrete Q2.
        {"val": 56, "start": "2025-01-01", "end": "2025-06-30", "fp": "Q2",
         "fy": 2025, "form": "10-Q", "filed": "2025-08-01"},
    ]
    fy = builder_mod.latest_fy(entries)
    assert fy["val"] == 110
    q = builder_mod.latest_quarter(entries)
    assert q["val"] == 29  # discrete Q2, not the 56 cumulative.


def test_cagr_basic(builder_mod):
    # 100 -> 133.1 over 3 years = 10% CAGR
    assert builder_mod.cagr(100, 133.1, 3) == pytest.approx(0.10, abs=1e-4)
    assert builder_mod.cagr(0, 100, 3) is None
    assert builder_mod.cagr(100, 100, 0) is None
    assert builder_mod.cagr(None, 100, 3) is None


def test_safe_div(builder_mod):
    assert builder_mod.safe_div(10, 2) == 5
    assert builder_mod.safe_div(10, 0) is None
    assert builder_mod.safe_div(None, 2) is None


def test_pct_change_handles_negative_base(builder_mod):
    pair = (
        {"val": -100, "end": "2024-03-31", "start": "2024-01-01", "fp": "Q1"},
        {"val": -80, "end": "2025-03-31", "start": "2025-01-01", "fp": "Q1"},
    )
    # (-80 - -100) / abs(-100) = 0.2
    assert builder_mod._pct_change(pair) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# End-to-end: _assemble with synthetic facts
# ---------------------------------------------------------------------------


def _full_synthetic_facts():
    """5 years of annual revenue + 8 quarters + balance sheet for a tech-style co."""
    revenue_annual = [
        _fy("2020-12-31", "2021-02-01", 200_000),
        _fy("2021-12-31", "2022-02-01", 230_000),
        _fy("2022-12-31", "2023-02-01", 250_000),
        _fy("2023-12-31", "2024-02-01", 280_000),
        _fy("2024-12-31", "2025-02-01", 300_000),
    ]
    revenue_quarters = [
        _q("Q1", "2023-03-31", "2023-01-01", 70_000),
        _q("Q2", "2023-06-30", "2023-04-01", 72_000),
        _q("Q3", "2023-09-30", "2023-07-01", 68_000),
        _q("Q1", "2024-03-31", "2024-01-01", 75_000),
        _q("Q2", "2024-06-30", "2024-04-01", 78_000),
        _q("Q3", "2024-09-30", "2024-07-01", 72_000),
        _q("Q1", "2025-03-31", "2025-01-01", 80_000),
        _q("Q2", "2025-06-30", "2025-04-01", 84_000),
    ]
    gross_quarters = [
        _q(e["fp"], e["end"], e["start"], e["val"] * 0.4) for e in revenue_quarters
    ]
    return {
        "revenue": revenue_annual + revenue_quarters,
        "gross_profit": [_fy("2024-12-31", "2025-02-01", 120_000)] + gross_quarters,
        "net_income": [_fy("2024-12-31", "2025-02-01", 60_000)],
        "operating_income": [_fy("2024-12-31", "2025-02-01", 90_000)],
        "ocf": [_fy("2024-12-31", "2025-02-01", 80_000)],
        "capex": [_fy("2024-12-31", "2025-02-01", 20_000)],
        "depreciation": [_fy("2024-12-31", "2025-02-01", 10_000)],
        "stockholders_equity": [_pit("2024-12-31", "2025-02-01", 150_000)],
        "cash": [_pit("2024-12-31", "2025-02-01", 30_000)],
        "long_term_debt": [_pit("2024-12-31", "2025-02-01", 50_000)],
        "short_term_debt": [_pit("2024-12-31", "2025-02-01", 5_000)],
        "shares_outstanding": [
            _pit("2023-12-31", "2024-02-01", 1_050_000),
            _pit("2024-12-31", "2025-02-01", 1_000_000),
        ],
        "assets": [_pit("2024-12-31", "2025-02-01", 400_000)],
        "liabilities": [_pit("2024-12-31", "2025-02-01", 250_000)],
    }


def test_assemble_tech_dossier(builder_mod):
    facts = _facts(_full_synthetic_facts())
    info = {"name": "Synth Inc.", "sector": "Technology", "industry": "Software"}
    d = builder_mod._assemble(
        ticker="SYNTH", facts=facts, info=info, current_price=150.0
    )
    assert d.ticker == "SYNTH"
    assert d.is_financials is False
    assert d.revenue_latest_fy == 300_000
    # 200_000 (FY2021, 3 years before FY2024) -> 300_000 in 3 years
    # cagr uses entries[-4] which is FY2021 = 230_000 actually. Let me recompute.
    # annual_history(.., 5) returns the last 5 annual: 2020..2024. rev_3y_ago is index -4 = 2021 (230_000).
    # cagr(230_000, 300_000, 3) = (300/230)^(1/3) - 1 ~ 0.0928
    assert d.revenue_3y_cagr == pytest.approx(0.0928, abs=1e-3)
    # Latest discrete Q is Q2-2025 (84k); year-ago Q2 = Q2-2024 (78k). YoY = 6/78 ~ 0.0769.
    assert d.revenue_yoy_latest == pytest.approx(0.0769, abs=1e-3)
    # QoQ: Q2-2025 vs Q1-2025: (84-80)/80 = 0.05
    assert d.revenue_qoq_latest == pytest.approx(0.05, abs=1e-4)
    # FY margins: gross = 120/300 = 0.4; op = 90/300 = 0.3; fcf = (80-20)/300 = 0.2.
    assert d.gross_margin_current == pytest.approx(0.4)
    assert d.operating_margin_current == pytest.approx(0.3)
    assert d.fcf_margin_current == pytest.approx(0.2)
    assert d.fcf_conversion == pytest.approx(60_000 / 60_000)  # FCF=60k, NI=60k
    # net debt = (50k + 5k) - 30k = 25k. EBITDA = op income + D&A = 100k. -> 0.25
    assert d.net_debt_to_ebitda == pytest.approx(0.25)
    # ROE = NI / equity = 60k / 150k = 0.4
    assert d.roe == pytest.approx(0.4)
    # Share count change: 1_000_000 / 1_050_000 - 1 = -0.0476 (buyback)
    assert d.share_count_change_yoy == pytest.approx(-0.0476, abs=1e-3)
    # Market cap = 150 * 1_000_000 = 150_000_000
    assert d.market_cap == pytest.approx(150_000_000)
    # PE = 150 / (60_000 / 1_000_000) = 150 / 0.06 = 2500
    assert d.pe_trailing == pytest.approx(2500)
    # P/S = 150_000_000 / 300_000 = 500
    assert d.ps_ratio == pytest.approx(500)
    # Gross margin trend uses 8 quarters; should be 8 entries of 0.4
    assert len(d.gross_margin_trend) == 8
    assert all(gm == pytest.approx(0.4) for gm in d.gross_margin_trend)


def test_assemble_financials_skips_gross_margin(builder_mod):
    facts = _facts({
        "revenue": [_fy("2024-12-31", "2025-02-01", 100_000)],  # not used for ratios
        "net_income": [_fy("2024-12-31", "2025-02-01", 20_000)],
        "operating_income": [_fy("2024-12-31", "2025-02-01", 25_000)],
        "stockholders_equity": [_pit("2024-12-31", "2025-02-01", 200_000)],
        "cash": [_pit("2024-12-31", "2025-02-01", 50_000)],
        # No gross_profit, no FCF concepts -- typical for a bank.
    })
    info = {"name": "Big Bank", "sector": "Financial Services", "industry": "Banks"}
    d = builder_mod._assemble(
        ticker="BANK", facts=facts, info=info, current_price=100.0
    )
    assert d.is_financials is True
    assert d.gross_margin_current is None
    assert d.gross_margin_trend == []
    assert d.fcf_margin_current is None
    assert d.roe == pytest.approx(0.10)  # 20k / 200k


def test_build_dossier_orchestrator_writes_json(builder_mod, monkeypatch):
    """Full path: mock fetchers + price history, verify storage.save_dossier was called."""
    from weekly_strategy.data import storage
    # Seed price history so build_dossier can compute current price.
    px = pd.DataFrame(
        {"close": [99.0, 100.0, 101.0]},
        index=pd.to_datetime(["2026-05-19", "2026-05-20", "2026-05-21"]),
    )
    storage.upsert_prices("AAPL", px)

    facts = _facts(_full_synthetic_facts())
    monkeypatch.setattr(builder_mod.fetchers, "get_cik", lambda t: "0000320193")
    monkeypatch.setattr(builder_mod.fetchers, "get_company_facts", lambda c: facts)
    monkeypatch.setattr(
        builder_mod.fetchers, "get_basic_info",
        lambda t: {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    )

    d = builder_mod.build_dossier("AAPL")
    assert d.ticker == "AAPL"
    assert d.current_price == 101.0  # from the seeded prices

    # The on-disk JSON should be there and round-trippable.
    blob = storage.load_dossier("AAPL")
    assert blob is not None
    assert blob["ticker"] == "AAPL"
    assert blob["data"]["ticker"] == "AAPL"
    assert blob["data"]["sector"] == "Technology"
