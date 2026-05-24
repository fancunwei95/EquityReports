from __future__ import annotations

"""Sector ETF fetcher + cross-sector snapshot.

Tracks the 11 GICS sector SPDR ETFs and benchmarks them against SPY. Returns
are computed from cached parquet prices (same storage layer as per-ticker
prices). Reuses the existing yfinance fetcher so we don't have a second
network path to maintain.
"""

import statistics
from datetime import date

import pandas as pd

from weekly_strategy.data import fetchers, storage
from weekly_strategy.data.schemas import SectorMetrics, SectorSnapshot


# ETF ticker -> friendly sector name. Keys are the truth; values mirror the
# GICS sector labels yfinance reports for individual stocks (see SECTOR_TO_ETF
# below for mapping yfinance's sector strings back to these ETFs).
SECTOR_ETFS: dict[str, str] = {
    "XLF":  "Financials",
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

# yfinance reports sector as one of these strings on Ticker.info -- map back
# to the canonical ETF. Includes a few alternate spellings yfinance uses.
SECTOR_TO_ETF: dict[str, str] = {
    "Financial Services": "XLF",
    "Financials":         "XLF",
    "Technology":         "XLK",
    "Energy":             "XLE",
    "Healthcare":         "XLV",
    "Health Care":        "XLV",
    "Consumer Cyclical":  "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples":   "XLP",
    "Industrials":        "XLI",
    "Basic Materials":    "XLB",
    "Materials":          "XLB",
    "Utilities":          "XLU",
    "Real Estate":        "XLRE",
    "Communication Services": "XLC",
}

BENCHMARK = "SPY"

# Trading-day windows
_WINDOWS = {"1w": 5, "1m": 21, "3m": 63}
_VOL_RECENT = 5
_VOL_TRAILING = 20

# Breadth threshold on stdev of relative 1m returns across sectors.
_BREADTH_NARROW_STD = 0.03


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_sector_prices(*, lookback_days: int = 180) -> None:
    """Fetch prices for SPY + all sector ETFs (parquet-cached via storage)."""
    fetchers.get_price_history(BENCHMARK, lookback_days=lookback_days)
    for etf in SECTOR_ETFS:
        fetchers.get_price_history(etf, lookback_days=lookback_days)


# ---------------------------------------------------------------------------
# Pure compute helpers
# ---------------------------------------------------------------------------


def _trailing_returns(prices: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in _WINDOWS}
    if prices is None or prices.empty or "close" not in prices.columns:
        return out
    closes = prices["close"].astype(float).dropna()
    if closes.empty:
        return out
    latest = float(closes.iloc[-1])
    for label, n in _WINDOWS.items():
        if len(closes) > n:
            base = float(closes.iloc[-n - 1])
            if base > 0:
                out[label] = latest / base - 1.0
    return out


def _volume_profile(prices: pd.DataFrame) -> float | None:
    """Last 5-day avg volume / trailing 20-day avg (the 20 days before that 5)."""
    if prices is None or prices.empty or "volume" not in prices.columns:
        return None
    vol = prices["volume"].astype(float).dropna()
    if len(vol) < _VOL_RECENT + _VOL_TRAILING:
        return None
    recent = vol.iloc[-_VOL_RECENT:].mean()
    trailing = vol.iloc[-(_VOL_RECENT + _VOL_TRAILING):-_VOL_RECENT].mean()
    if trailing == 0:
        return None
    return float(recent / trailing)


def _rel(sector_ret: float | None, spy_ret: float | None) -> float | None:
    if sector_ret is None or spy_ret is None:
        return None
    return sector_ret - spy_ret


# ---------------------------------------------------------------------------
# Snapshot assembler
# ---------------------------------------------------------------------------


def get_sector_snapshot(*, week_ending: date | None = None) -> SectorSnapshot:
    """Assemble the cross-sector snapshot from cached parquet prices.

    Caller should typically have called ``fetch_sector_prices()`` once at
    the start of the run so the parquet files exist.
    """
    we = week_ending or date.today()

    spy = storage.load_prices(BENCHMARK)
    spy_rets = _trailing_returns(spy)

    metrics: dict[str, SectorMetrics] = {}
    for etf, name in SECTOR_ETFS.items():
        prices = storage.load_prices(etf)
        rets = _trailing_returns(prices)
        metrics[etf] = SectorMetrics(
            etf=etf,
            sector=name,
            return_1w=rets["1w"],
            return_1m=rets["1m"],
            return_3m=rets["3m"],
            rel_1m=_rel(rets["1m"], spy_rets["1m"]),
            rel_3m=_rel(rets["3m"], spy_rets["3m"]),
            volume_vs_20d=_volume_profile(prices),
        )

    leadership = sorted(
        (m for m in metrics.values() if m.rel_1m is not None),
        key=lambda m: m.rel_1m,
        reverse=True,
    )
    leadership_ranking = [m.etf for m in leadership]

    breadth = "unknown"
    rel_1m_vals = [m.rel_1m for m in metrics.values() if m.rel_1m is not None]
    if len(rel_1m_vals) >= 4:
        std = statistics.pstdev(rel_1m_vals)
        breadth = "narrow" if std >= _BREADTH_NARROW_STD else "broad"

    return SectorSnapshot(
        week_ending=we,
        sectors=metrics,
        leadership_ranking=leadership_ranking,
        breadth=breadth,
    )
