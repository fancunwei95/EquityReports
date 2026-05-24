from __future__ import annotations

"""Daily prediction snapshot persistence.

Each Stage 3 run emits one snapshot to ``data_cache/predictions/{as_of}/``:
* ``bundles.parquet`` -- one row per ticker, every score component including
                         the cross-sectional z-scores. This is the raw
                         feature matrix calibration will regress against.
* ``portfolio.json``  -- the L/S selections + the macro/sector regime that
                         drove them. Joins back to the bundles via ticker.
* ``prices.parquet``  -- close + adj_close for every universe ticker at
                         ``as_of`` (so we don't depend on the prices/
                         parquet cache still having the right snapshot
                         when we run calibration weeks later).

After N days, ``load_snapshot_history()`` concatenates all bundle rows so
calibration can compute forward returns + information coefficients.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from weekly_strategy.config import settings
from weekly_strategy.data import storage
from weekly_strategy.data.schemas import (
    MacroRegime,
    Portfolio,
    SectorSnapshot,
    StockScoreBundle,
)


PREDICTIONS_DIR = settings.DATA_CACHE_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _snapshot_dir(as_of: date) -> Path:
    d = PREDICTIONS_DIR / as_of.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_daily_snapshot(
    *,
    as_of: date,
    bundles: dict[str, StockScoreBundle],
    portfolio: Portfolio,
    macro_regime: MacroRegime | None = None,
    sector_snap: SectorSnapshot | None = None,
    universe_tickers: Iterable[str] | None = None,
) -> Path:
    """Persist one daily snapshot. Returns the snapshot directory."""
    out = _snapshot_dir(as_of)

    # 1. Bundles -> parquet
    rows: list[dict] = []
    for ticker, b in bundles.items():
        rows.append({
            "as_of": as_of.isoformat(),
            "ticker": ticker,
            # Raw components
            "quality": b.quality_score,
            "valuation": b.valuation_score,
            "momentum": b.momentum_score,
            "news_sentiment": b.news_sentiment_score,
            "news_noise_ratio": b.news_noise_ratio,
            "reddit_sentiment": b.reddit_sentiment,
            "sector_score": b.sector_score_value,
            "macro_regime_score": b.macro_regime_score,
            # Composites
            "composite_score": b.composite_score,
            "composite_z": b.composite_z,
            # Cross-sectional z-scores
            "z_quality": b.z_quality,
            "z_valuation": b.z_valuation,
            "z_momentum": b.z_momentum,
            "z_news": b.z_news_sentiment,
            "z_sector": b.z_sector,
            # Returns at snapshot time
            "return_1m": b.return_1m,
            "return_3m": b.return_3m,
            "return_6m": b.return_6m,
            "sector_etf": b.sector_etf,
            "sector_rank": b.sector_rank,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out / "bundles.parquet", index=False)

    # 2. Portfolio + context -> JSON
    payload = {
        "as_of": as_of.isoformat(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "portfolio": portfolio.model_dump(mode="json"),
        "macro_regime": macro_regime.model_dump(mode="json") if macro_regime else None,
        "sector_breadth": sector_snap.breadth if sector_snap else None,
        "sector_leadership": (
            sector_snap.leadership_ranking if sector_snap else []
        ),
    }
    (out / "portfolio.json").write_text(json.dumps(payload, indent=2, default=str))

    # 3. Closing prices at as_of (so calibration doesn't depend on the
    # live price cache having the right snapshot weeks later).
    if universe_tickers is not None:
        price_rows: list[dict] = []
        for t in universe_tickers:
            prices = storage.load_prices(t)
            if prices.empty:
                continue
            last = prices.iloc[-1]
            price_rows.append({
                "as_of": as_of.isoformat(),
                "ticker": t,
                "close": float(last.get("close", float("nan"))),
                "adj_close": float(last.get("adj_close", float("nan"))),
            })
        if price_rows:
            pd.DataFrame(price_rows).to_parquet(out / "prices.parquet", index=False)

    return out


# ---------------------------------------------------------------------------
# Loading + history
# ---------------------------------------------------------------------------


def list_snapshot_dates() -> list[date]:
    """All persisted snapshot dates, oldest first."""
    out: list[date] = []
    for d in sorted(PREDICTIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            out.append(date.fromisoformat(d.name))
        except ValueError:
            continue
    return out


def load_bundles(as_of: date) -> pd.DataFrame:
    p = _snapshot_dir(as_of) / "bundles.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def load_portfolio(as_of: date) -> dict | None:
    p = _snapshot_dir(as_of) / "portfolio.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_prices_snapshot(as_of: date) -> pd.DataFrame:
    p = _snapshot_dir(as_of) / "prices.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def load_snapshot_history() -> pd.DataFrame:
    """Concat every saved bundle snapshot. Used by Stage 5 calibration."""
    frames: list[pd.DataFrame] = []
    for d in list_snapshot_dates():
        bundle = load_bundles(d)
        if not bundle.empty:
            frames.append(bundle)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_previous_book(as_of: date) -> tuple[set[str], set[str]]:
    """Read the most recent prior snapshot's portfolio for turnover dampening.

    Returns (previous_longs, previous_shorts) as ticker sets. Empty sets
    when no prior snapshot exists.
    """
    history = [d for d in list_snapshot_dates() if d < as_of]
    if not history:
        return set(), set()
    prev = load_portfolio(history[-1])
    if prev is None:
        return set(), set()
    longs = {p["ticker"] for p in prev.get("portfolio", {}).get("longs", [])}
    shorts = {p["ticker"] for p in prev.get("portfolio", {}).get("shorts", [])}
    return longs, shorts
