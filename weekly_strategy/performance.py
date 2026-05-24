from __future__ import annotations

"""Portfolio performance tracker for the public dashboard.

On each daily run:

* ``record_portfolio_entry(portfolio)`` captures the entry close for the
  top-N longs + shorts and persists the open portfolio to
  ``docs/data/portfolios.json``.

* ``update_performance()`` walks every persisted portfolio, marks open
  ones with current close prices, closes any portfolio that has been held
  for ``HOLDING_TRADING_DAYS`` market sessions, and computes returns.

* ``write_performance_view()`` derives the small JSON that ``index.html``
  reads: cumulative return time series, last-5 open portfolios, closed-
  portfolio table, top-level stats.

Long return = (exit / entry - 1). Short return = -(exit / entry - 1).
L/S return per portfolio = mean(long_returns) - mean(short_returns)
                          (equal-weight in each basket).

Cumulative L/S return = sum of per-portfolio L/S returns. (Simple sum
rather than compounding: matches how people intuit weekly P&L over
1-week non-overlapping books. With many overlapping books in flight,
compounding would over- or under-state depending on rebalancing.)
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from weekly_strategy.config import settings
from weekly_strategy.data import storage
from weekly_strategy.data.schemas import Portfolio


HOLDING_TRADING_DAYS = 5     # close 5 sessions after entry
TOP_N_PER_SIDE = 5           # only track the top 5 visible per side
MAX_OPEN_DISPLAY = 5         # last N open portfolios surfaced in the UI
PORTFOLIOS_FILE = settings.DOCS_DATA_DIR / "portfolios.json"
PERFORMANCE_FILE = settings.DOCS_DATA_DIR / "performance.json"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    settings.DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_portfolios() -> list[dict]:
    if not PORTFOLIOS_FILE.exists():
        return []
    return json.loads(PORTFOLIOS_FILE.read_text())


def _save_portfolios(portfolios: list[dict]) -> None:
    _ensure_dirs()
    PORTFOLIOS_FILE.write_text(json.dumps(portfolios, indent=2, default=str))


def _save_performance(perf: dict) -> None:
    _ensure_dirs()
    PERFORMANCE_FILE.write_text(json.dumps(perf, indent=2, default=str))


# ---------------------------------------------------------------------------
# Price helpers (trading-day aware)
# ---------------------------------------------------------------------------


def _close_at_or_before(ticker: str, target: date) -> tuple[date, float] | None:
    """Latest close at or before ``target``. Returns (date, close).

    For an entry-price snapshot on a Sunday "week_ending", this returns
    the prior Friday's close (or whatever the most recent trading day was).
    """
    prices = storage.load_prices(ticker)
    if prices is None or prices.empty:
        return None
    closes = prices["close"].astype(float).dropna()
    if closes.empty:
        return None
    series = closes[[d <= target for d in closes.index]]
    if series.empty:
        return None
    d = series.index[-1]
    return d, float(series.iloc[-1])


def _latest_close(ticker: str) -> tuple[date, float] | None:
    prices = storage.load_prices(ticker)
    if prices is None or prices.empty:
        return None
    closes = prices["close"].astype(float).dropna()
    if closes.empty:
        return None
    return closes.index[-1], float(closes.iloc[-1])


def _trading_days_between(entry: date, end: date, ticker: str) -> int:
    """Trading days elapsed between entry (exclusive) and end (inclusive),
    derived from actual price observations for ``ticker`` rather than a
    calendar approximation (handles holidays + early-close days)."""
    prices = storage.load_prices(ticker)
    if prices is None or prices.empty:
        return 0
    in_window = [d for d in prices.index if entry < d <= end]
    return len(in_window)


# ---------------------------------------------------------------------------
# Recording entry
# ---------------------------------------------------------------------------


def record_portfolio_entry(
    portfolio: Portfolio,
    *,
    top_n: int = TOP_N_PER_SIDE,
) -> dict | None:
    """Snapshot the top-N longs + shorts at portfolio creation.

    Returns the entry record if a new one was recorded; ``None`` if this
    entry_date is already on file (so re-running a same-day report doesn't
    create duplicates).
    """
    entry_date = portfolio.week_ending
    portfolios = _load_portfolios()
    if any(p.get("entry_date") == entry_date.isoformat() for p in portfolios):
        return None

    longs = portfolio.longs[:top_n]
    shorts = portfolio.shorts[:top_n]
    record: dict = {
        "entry_date": entry_date.isoformat(),
        "status": "open",
        "longs": [],
        "shorts": [],
    }
    for p in longs:
        snap = _close_at_or_before(p.ticker, entry_date)
        if snap is None:
            continue
        d, close = snap
        record["longs"].append({
            "ticker": p.ticker,
            "sector": p.sector,
            "composite_z": p.composite_z,
            "entry_date_used": d.isoformat(),
            "entry_close": close,
        })
    for p in shorts:
        snap = _close_at_or_before(p.ticker, entry_date)
        if snap is None:
            continue
        d, close = snap
        record["shorts"].append({
            "ticker": p.ticker,
            "sector": p.sector,
            "composite_z": p.composite_z,
            "entry_date_used": d.isoformat(),
            "entry_close": close,
        })

    portfolios.append(record)
    _save_portfolios(portfolios)
    return record


# ---------------------------------------------------------------------------
# Updating / closing
# ---------------------------------------------------------------------------


def update_performance() -> dict:
    """Mark to market all open portfolios; close any that hit holding window.

    Returns the regenerated performance.json payload.
    """
    portfolios = _load_portfolios()
    today = date.today()
    changed = False

    for p in portfolios:
        if p.get("status") != "open":
            continue
        entry = date.fromisoformat(p["entry_date"])
        # Use the first long as a "trading calendar reference"; SPY would be
        # better, but any liquid name's price index is close enough.
        ref_ticker = (p["longs"] + p["shorts"])[0]["ticker"] if (p["longs"] or p["shorts"]) else None
        if ref_ticker is None:
            continue
        held = _trading_days_between(entry, today, ref_ticker)
        p["days_held"] = held

        # Mark current
        for side_name, sign in (("longs", 1.0), ("shorts", -1.0)):
            for pos in p[side_name]:
                snap = _latest_close(pos["ticker"])
                if snap is None:
                    pos["current_close"] = None
                    pos["current_return"] = None
                    continue
                _d, close = snap
                pos["current_close"] = close
                if pos["entry_close"]:
                    raw = close / pos["entry_close"] - 1.0
                    pos["current_return"] = sign * raw

        # Compute basket current returns
        l_rets = [pos["current_return"] for pos in p["longs"]
                  if pos.get("current_return") is not None]
        s_rets = [pos["current_return"] for pos in p["shorts"]
                  if pos.get("current_return") is not None]
        p["current_long_return"] = sum(l_rets) / len(l_rets) if l_rets else None
        p["current_short_return"] = sum(s_rets) / len(s_rets) if s_rets else None
        if p["current_long_return"] is not None and p["current_short_return"] is not None:
            # current_short_return is already signed (+) when shorts dropped
            p["current_ls_return"] = p["current_long_return"] + p["current_short_return"]
        else:
            p["current_ls_return"] = None

        # Close if held long enough
        if held >= HOLDING_TRADING_DAYS:
            for side_name, sign in (("longs", 1.0), ("shorts", -1.0)):
                for pos in p[side_name]:
                    pos["exit_close"] = pos.get("current_close")
                    pos["exit_date"] = today.isoformat()
                    if pos.get("entry_close") and pos.get("exit_close"):
                        raw = pos["exit_close"] / pos["entry_close"] - 1.0
                        pos["return"] = sign * raw
                    else:
                        pos["return"] = None
            l_final = [pos["return"] for pos in p["longs"] if pos.get("return") is not None]
            s_final = [pos["return"] for pos in p["shorts"] if pos.get("return") is not None]
            p["long_basket_return"] = sum(l_final) / len(l_final) if l_final else None
            p["short_basket_return"] = sum(s_final) / len(s_final) if s_final else None
            if p["long_basket_return"] is not None and p["short_basket_return"] is not None:
                p["ls_return"] = p["long_basket_return"] + p["short_basket_return"]
            else:
                p["ls_return"] = None
            p["exit_date"] = today.isoformat()
            p["status"] = "closed"
            changed = True

    _save_portfolios(portfolios)
    return write_performance_view(portfolios)


def write_performance_view(portfolios: list[dict] | None = None) -> dict:
    """Build the small JSON that index.html reads."""
    if portfolios is None:
        portfolios = _load_portfolios()
    portfolios = sorted(portfolios, key=lambda p: p["entry_date"])

    closed = [p for p in portfolios if p.get("status") == "closed"]
    open_ = [p for p in portfolios if p.get("status") == "open"]

    cum = 0.0
    cum_series: list[dict] = []
    for p in closed:
        ret = p.get("ls_return") or 0.0
        cum += ret
        cum_series.append({
            "exit_date": p.get("exit_date"),
            "entry_date": p["entry_date"],
            "ls_return": ret,
            "cumulative_return": cum,
        })

    current_open_pnl = sum(
        (p.get("current_ls_return") or 0.0) for p in open_
    )

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "stats": {
            "n_closed": len(closed),
            "n_open": len(open_),
            "cumulative_return": cum,
            "current_open_pnl": current_open_pnl,
        },
        "cumulative_series": cum_series,
        "open_portfolios": list(reversed(open_))[:MAX_OPEN_DISPLAY],
        "closed_portfolios": list(reversed(closed)),
    }
    _save_performance(payload)
    return payload
