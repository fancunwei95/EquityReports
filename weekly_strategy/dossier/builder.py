from __future__ import annotations

"""Build a per-ticker Dossier from EDGAR companyfacts + yfinance basics.

EDGAR's XBRL is messy in two specific ways that this module spends most of
its complexity on:

1. **Tag drift across companies.** "Revenues" is the canonical tag, but some
   filers use ``SalesRevenueNet``, ``RevenueFromContractWithCustomerExcludingAssessedTax``,
   etc. We try a candidate list per concept.

2. **Quarterly = YTD-cumulative vs discrete-Q.** A 10-Q filing reports values
   with a ``start`` / ``end`` date pair. The duration tells us whether it's
   a discrete quarter (~90 days) or YTD-cumulative (180/270 days). For
   "latest quarter" semantics we want discrete-Q only.

Financials sector branch: banks/insurers don't report Revenues / GrossProfit
in a way that makes margin math meaningful. ``is_financials`` is set on the
Dossier so downstream scoring can skip those fields.
"""

from datetime import date, datetime
from typing import Iterable, Sequence

import pandas as pd

from weekly_strategy.data import fetchers, storage
from weekly_strategy.data.schemas import Dossier


# ---------------------------------------------------------------------------
# Concept-tag candidate table
# ---------------------------------------------------------------------------

# Each concept maps to a tag-search-order list. We try tags in order and
# return the first one that has values. All concepts live in the us-gaap
# namespace unless noted; "dei" (Document and Entity Information) is used
# for share counts in some cases.

CONCEPT_TAGS: dict[str, list[tuple[str, str]]] = {
    "revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "assets": [("us-gaap", "Assets")],
    "liabilities": [("us-gaap", "Liabilities")],
    "stockholders_equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "Cash"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ],
    "long_term_debt": [
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtNoncurrent"),
    ],
    "short_term_debt": [
        ("us-gaap", "ShortTermBorrowings"),
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "DebtCurrent"),
    ],
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ],
    "ocf": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ],
    "depreciation": [
        ("us-gaap", "DepreciationAndAmortization"),
        ("us-gaap", "DepreciationDepletionAndAmortization"),
        ("us-gaap", "Depreciation"),
    ],
}


def _entries_for(facts: dict, namespace: str, tag: str) -> list[dict]:
    """Flatten ``facts.{namespace}.{tag}.units.*`` into a single list, tagging unit."""
    concept = facts.get("facts", {}).get(namespace, {}).get(tag)
    if not concept:
        return []
    units = concept.get("units") or {}
    out: list[dict] = []
    for unit_key, items in units.items():
        for it in items:
            out.append(
                {
                    "val": it.get("val"),
                    "end": it.get("end"),
                    "start": it.get("start"),
                    "filed": it.get("filed"),
                    "fp": it.get("fp"),
                    "fy": it.get("fy"),
                    "form": it.get("form"),
                    "unit": unit_key,
                }
            )
    return out


def extract_concept(facts: dict, concept: str) -> list[dict]:
    """Try each candidate tag in order; return the first non-empty entries list."""
    for ns, tag in CONCEPT_TAGS.get(concept, []):
        entries = _entries_for(facts, ns, tag)
        if entries:
            return entries
    return []


def latest_fy(entries: Sequence[dict]) -> dict | None:
    """Most recent annual entry. Annual entries have fp='FY' and span ~365 days."""
    annual = [e for e in entries if e.get("fp") == "FY"]
    if not annual:
        return None
    return max(annual, key=lambda e: e.get("end") or "")


def latest_quarter(entries: Sequence[dict]) -> dict | None:
    """Most recent discrete quarter (fp in Q1/Q2/Q3, span ~90 days).

    Falls back to any quarterly entry if no discrete-Q is found (some filers
    report only YTD-cumulative).
    """
    quarterly = [e for e in entries if e.get("fp") in ("Q1", "Q2", "Q3")]
    if not quarterly:
        return None
    discrete = [e for e in quarterly if _is_discrete_quarter(e)]
    pool = discrete or quarterly
    return max(pool, key=lambda e: e.get("end") or "")


def _is_discrete_quarter(entry: dict) -> bool:
    s, e = entry.get("start"), entry.get("end")
    if not s or not e:
        # Balance-sheet items have no start (point-in-time); treat as discrete.
        return s is None
    try:
        delta = (date.fromisoformat(e) - date.fromisoformat(s)).days
    except ValueError:
        return False
    return 80 <= delta <= 100


def annual_history(entries: Sequence[dict], n: int) -> list[dict]:
    """Last ``n`` annual entries, oldest first."""
    annual = sorted(
        (e for e in entries if e.get("fp") == "FY"),
        key=lambda e: e.get("end") or "",
    )
    return annual[-n:]


def quarterly_history(entries: Sequence[dict], n: int) -> list[dict]:
    """Last ``n`` discrete-quarter entries, oldest first."""
    quarterly = [
        e for e in entries
        if e.get("fp") in ("Q1", "Q2", "Q3") and _is_discrete_quarter(e)
    ]
    quarterly.sort(key=lambda e: e.get("end") or "")
    return quarterly[-n:]


# ---------------------------------------------------------------------------
# Derived metric helpers (pure)
# ---------------------------------------------------------------------------


def cagr(start_val: float | None, end_val: float | None, years: int) -> float | None:
    if start_val is None or end_val is None or start_val <= 0 or years <= 0:
        return None
    return (end_val / start_val) ** (1.0 / years) - 1.0


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def latest_filing_date(*concepts_entries: Iterable[dict]) -> date | None:
    """Most recent ``filed`` date across all provided entry lists."""
    dates: list[date] = []
    for entries in concepts_entries:
        for e in entries:
            f = e.get("filed")
            if f:
                try:
                    dates.append(date.fromisoformat(f))
                except ValueError:
                    continue
    return max(dates) if dates else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_FINANCIALS_SECTORS = {"Financial Services", "Financials"}


def build_dossier(ticker: str) -> Dossier:
    """End-to-end: fetch + parse + compute + persist. Returns the Dossier."""
    cik = fetchers.get_cik(ticker)
    facts = fetchers.get_company_facts(cik)
    info = fetchers.get_basic_info(ticker)
    prices = storage.load_prices(ticker)
    current_price = float(prices["close"].iloc[-1]) if not prices.empty else None

    dossier = _assemble(ticker=ticker, facts=facts, info=info, current_price=current_price)
    storage.save_dossier(ticker, dossier.model_dump(mode="json"))
    return dossier


def _assemble(
    *,
    ticker: str,
    facts: dict,
    info: dict,
    current_price: float | None,
) -> Dossier:
    """The pure half: given fetched inputs, compute everything. Easy to test."""
    sector = info.get("sector")
    is_financials = sector in _FINANCIALS_SECTORS

    # Pull every concept once.
    concepts = {k: extract_concept(facts, k) for k in CONCEPT_TAGS}

    # Revenue history (annual + quarterly).
    rev_annual = annual_history(concepts["revenue"], 5)
    rev_latest_fy = rev_annual[-1] if rev_annual else None
    rev_3y_ago = rev_annual[-4] if len(rev_annual) >= 4 else None
    rev_latest_q = latest_quarter(concepts["revenue"])
    rev_yoy_pair = _yoy_quarter_pair(concepts["revenue"])
    rev_qoq_pair = _consecutive_quarters(concepts["revenue"])

    revenue_3y_cagr = cagr(
        rev_3y_ago["val"] if rev_3y_ago else None,
        rev_latest_fy["val"] if rev_latest_fy else None,
        3,
    )
    revenue_yoy_latest = _pct_change(rev_yoy_pair)
    revenue_qoq_latest = _pct_change(rev_qoq_pair)

    # Margins: latest FY (annual gives the cleanest cross-company comparison).
    net_income_fy = latest_fy(concepts["net_income"])
    op_income_fy = latest_fy(concepts["operating_income"])
    gross_profit_fy = latest_fy(concepts["gross_profit"])
    ocf_fy = latest_fy(concepts["ocf"])
    capex_fy = latest_fy(concepts["capex"])
    dep_fy = latest_fy(concepts["depreciation"])

    rev_fy_val = rev_latest_fy["val"] if rev_latest_fy else None
    gross_margin = safe_div(_val(gross_profit_fy), rev_fy_val) if not is_financials else None
    operating_margin = safe_div(_val(op_income_fy), rev_fy_val)
    fcf = (_val(ocf_fy) or 0) - (_val(capex_fy) or 0) if ocf_fy and capex_fy else None
    fcf_margin = safe_div(fcf, rev_fy_val)
    fcf_conversion = safe_div(fcf, _val(net_income_fy))

    # Gross margin trend across last 8 discrete quarters (skipped for financials).
    if is_financials:
        gross_margin_trend: list[float] = []
    else:
        gross_margin_trend = _quarterly_margin_trend(
            concepts["gross_profit"], concepts["revenue"], n=8
        )

    # Balance sheet.
    equity = latest_fy(concepts["stockholders_equity"]) or _latest_pit(concepts["stockholders_equity"])
    cash_pit = _latest_pit(concepts["cash"])
    ltd_pit = _latest_pit(concepts["long_term_debt"])
    std_pit = _latest_pit(concepts["short_term_debt"])
    total_debt = (_val(ltd_pit) or 0) + (_val(std_pit) or 0)
    net_debt = total_debt - (_val(cash_pit) or 0)
    ebitda = ((_val(op_income_fy) or 0) + (_val(dep_fy) or 0)) if op_income_fy else None
    net_debt_to_ebitda = safe_div(net_debt, ebitda) if ebitda and ebitda > 0 else None

    # Returns.
    roe = safe_div(_val(net_income_fy), _val(equity))
    invested_capital = (_val(equity) or 0) + total_debt
    roic = safe_div(_val(op_income_fy), invested_capital) if invested_capital > 0 else None

    # Share count YoY.
    sh_annual = annual_history(concepts["shares_outstanding"], 2) or _pit_annual_pair(
        concepts["shares_outstanding"]
    )
    if len(sh_annual) >= 2:
        share_count_change_yoy = _pct_change((sh_annual[-2], sh_annual[-1]))
    else:
        share_count_change_yoy = None

    # Valuation: requires current price + shares.
    shares_now = _latest_pit(concepts["shares_outstanding"])
    shares_val = _val(shares_now)
    market_cap = (current_price * shares_val) if (current_price and shares_val) else None
    pe_trailing = safe_div(current_price, safe_div(_val(net_income_fy), shares_val))
    ps_ratio = safe_div(market_cap, rev_fy_val)
    fcf_yield = safe_div(fcf, market_cap)
    enterprise_value = (market_cap + net_debt) if market_cap is not None else None
    ev_to_ebitda = safe_div(enterprise_value, ebitda) if ebitda and ebitda > 0 else None

    last_filing = latest_filing_date(*concepts.values())

    return Dossier(
        ticker=ticker.upper(),
        company_name=info.get("name"),
        sector=sector,
        industry=info.get("industry"),
        is_financials=is_financials,
        revenue_latest_fy=rev_fy_val,
        revenue_3y_cagr=revenue_3y_cagr,
        revenue_yoy_latest=revenue_yoy_latest,
        revenue_qoq_latest=revenue_qoq_latest,
        gross_margin_current=gross_margin,
        gross_margin_trend=gross_margin_trend,
        operating_margin_current=operating_margin,
        fcf_margin_current=fcf_margin,
        roic=roic,
        roe=roe,
        share_count_change_yoy=share_count_change_yoy,
        net_debt_to_ebitda=net_debt_to_ebitda,
        fcf_conversion=fcf_conversion,
        current_price=current_price,
        market_cap=market_cap,
        pe_trailing=pe_trailing,
        ps_ratio=ps_ratio,
        fcf_yield=fcf_yield,
        ev_to_ebitda=ev_to_ebitda,
        last_updated=datetime.utcnow(),
        last_filing_date=last_filing,
    )


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------


def _val(entry: dict | None) -> float | None:
    if entry is None:
        return None
    v = entry.get("val")
    return float(v) if v is not None else None


def _latest_pit(entries: Sequence[dict]) -> dict | None:
    """Latest point-in-time entry (balance-sheet items: no start, just end)."""
    pit = [e for e in entries if e.get("start") is None]
    if pit:
        return max(pit, key=lambda e: e.get("end") or "")
    # Fall back to any entry; some filers report PIT items with a start too.
    if not entries:
        return None
    return max(entries, key=lambda e: e.get("end") or "")


def _yoy_quarter_pair(entries: Sequence[dict]) -> tuple[dict, dict] | None:
    """Returns (year-ago-Q, latest-Q) where both are discrete quarters of the same fp."""
    latest = latest_quarter(entries)
    if latest is None:
        return None
    target_fp = latest.get("fp")
    target_end = latest.get("end")
    if not target_end:
        return None
    same_fp = [
        e for e in entries
        if e.get("fp") == target_fp and _is_discrete_quarter(e) and e.get("end") != target_end
    ]
    if not same_fp:
        return None
    year_ago = max(same_fp, key=lambda e: e.get("end") or "")
    return year_ago, latest


def _consecutive_quarters(entries: Sequence[dict]) -> tuple[dict, dict] | None:
    """Returns (prior-Q, latest-Q) consecutive discrete quarters."""
    qs = quarterly_history(entries, 2)
    if len(qs) < 2:
        return None
    return qs[0], qs[1]


def _pct_change(pair: tuple[dict, dict] | None) -> float | None:
    if pair is None:
        return None
    prior, latest = pair
    p, l = _val(prior), _val(latest)
    if p is None or l is None or p == 0:
        return None
    return (l - p) / abs(p)


def _quarterly_margin_trend(
    gp_entries: Sequence[dict],
    rev_entries: Sequence[dict],
    n: int,
) -> list[float]:
    """Discrete-quarter gross margins, oldest first. Pairs by ``end`` date."""
    gp_by_end = {e["end"]: e for e in quarterly_history(gp_entries, n)}
    rev_by_end = {e["end"]: e for e in quarterly_history(rev_entries, n)}
    common = sorted(set(gp_by_end) & set(rev_by_end))
    out: list[float] = []
    for end in common:
        gm = safe_div(_val(gp_by_end[end]), _val(rev_by_end[end]))
        if gm is not None:
            out.append(gm)
    return out


def _pit_annual_pair(entries: Sequence[dict]) -> list[dict]:
    """For PIT concepts like share count: pull two annual snapshots ~1y apart."""
    pit = [e for e in entries if e.get("start") is None and e.get("end")]
    if not pit:
        return []
    pit.sort(key=lambda e: e["end"])
    if len(pit) < 2:
        return pit
    # Return the latest and the entry closest to ~365 days earlier.
    latest = pit[-1]
    target = (date.fromisoformat(latest["end"]) - pd.Timedelta(days=365)).isoformat()
    prior = min(pit[:-1], key=lambda e: abs((date.fromisoformat(e["end"]) - date.fromisoformat(target)).days))
    return [prior, latest]
