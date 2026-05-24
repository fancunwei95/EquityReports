from __future__ import annotations

"""Step 3.1: universe construction.

Builds the ~100-name working universe from a candidate snapshot (S&P 500
+ Nasdaq 100, baked into config/candidate_universe.json) by:

1. Pulling basic_info for each candidate via yfinance (sector, market
   cap, beta).
2. Applying a min-market-cap liquidity filter.
3. Bucketing by GICS sector and taking the top N per sector against the
   sector quotas in SECTOR_TARGETS.
4. Layering thematic_picks on top with dedup.
5. Trimming to exactly TARGET_SIZE (drop lowest-cap if overshoot).

For Stage 3 the candidate list is static. The quarterly refresh job in
Step 3.9 invokes ``construct_universe`` against an updated snapshot and
hands back a proposal for human review before commit.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

from weekly_strategy.config import settings
from weekly_strategy.data import fetchers, storage
from weekly_strategy.data.schemas import Universe, UniverseEntry


# Sector cap allocation (sums to 100). From plan.md Step 3.1.
SECTOR_TARGETS: dict[str, int] = {
    "Technology":             18,
    "Financial Services":     12,
    "Healthcare":             12,
    "Consumer Cyclical":      10,
    "Industrials":            10,
    "Communication Services":  8,
    "Consumer Defensive":      8,
    "Energy":                  8,
    "Basic Materials":         5,
    "Utilities":               5,
    "Real Estate":             4,
}

TARGET_SIZE = 100
MIN_MARKET_CAP = 5_000_000_000      # $5B liquidity floor
THEMATIC_MAX = 15                   # ceiling on thematic adds

_CANDIDATE_PATH = settings.PACKAGE_ROOT / "config" / "candidate_universe.json"
_THEMATIC_PATH = settings.PACKAGE_ROOT / "config" / "thematic_picks.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_candidate_pool(path: Path | None = None) -> list[str]:
    p = path or _CANDIDATE_PATH
    raw = json.loads(p.read_text())
    tickers = raw.get("tickers") or []
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        u = t.upper()
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def load_thematic_picks(path: Path | None = None) -> list[str]:
    """Flatten the themed buckets into a single deduped list."""
    p = path or _THEMATIC_PATH
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    out: list[str] = []
    seen: set[str] = set()
    for key, group in raw.items():
        if key.startswith("_") or not isinstance(group, list):
            continue
        for t in group:
            u = str(t).upper()
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
    return out[:THEMATIC_MAX]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _fetch_metadata(tickers: Iterable[str]) -> dict[str, dict]:
    """basic_info per ticker. Skips on failure -- yfinance can return junk for some names."""
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            info = fetchers.get_basic_info(t)
            if not info.get("sector") or not info.get("market_cap"):
                continue
            out[t] = info
        except Exception:
            continue
    return out


def construct_universe(
    *,
    quarter: str | None = None,
    candidates: list[str] | None = None,
    thematic_picks: list[str] | None = None,
    sector_targets: dict[str, int] | None = None,
    min_market_cap: float = MIN_MARKET_CAP,
    target_size: int = TARGET_SIZE,
    metadata: dict[str, dict] | None = None,
) -> Universe:
    """Run the full universe construction pipeline.

    ``metadata`` lets tests inject pre-fetched basic_info dicts so the
    network call to yfinance is skipped.
    """
    quarter = quarter or _current_quarter_label()
    candidates = candidates or load_candidate_pool()
    thematic = thematic_picks if thematic_picks is not None else load_thematic_picks()
    sector_targets = sector_targets or SECTOR_TARGETS

    meta = metadata if metadata is not None else _fetch_metadata(candidates)
    # Liquidity filter.
    eligible = {
        t: info for t, info in meta.items()
        if (info.get("market_cap") or 0) >= min_market_cap
    }

    # 1. Thematic picks FIRST so they get the "thematic" label even if they
    #    also appear in the candidate pool (the alternative -- cap_rank first
    #    -- means an overshoot could trim them before the thematic protection
    #    kicks in).
    selected: list[UniverseEntry] = []
    picked: set[str] = set()
    for t in thematic:
        info = meta.get(t, {})
        selected.append(
            UniverseEntry(
                ticker=t,
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=info.get("market_cap"),
                beta=info.get("beta"),
                included_reason="thematic",
            )
        )
        picked.add(t)

    # 2. Bucket eligible (post-liquidity) candidates by sector, take top-N
    #    per sector by market cap. Skip anything already in thematic.
    by_sector: dict[str, list[tuple[str, dict]]] = {}
    for t, info in eligible.items():
        if t in picked:
            continue
        sector = info.get("sector") or "Unknown"
        by_sector.setdefault(sector, []).append((t, info))

    for sector, names in by_sector.items():
        names.sort(key=lambda kv: kv[1].get("market_cap") or 0, reverse=True)
        quota = sector_targets.get(sector, 0)
        for ticker, info in names[:quota]:
            if ticker in picked:
                continue
            picked.add(ticker)
            selected.append(
                UniverseEntry(
                    ticker=ticker,
                    sector=sector,
                    industry=info.get("industry"),
                    market_cap=info.get("market_cap"),
                    beta=info.get("beta"),
                    included_reason="cap_rank",
                )
            )

    # Trim to target_size, dropping the lowest-cap cap-rank picks first
    # (preserve thematic picks since they exist for non-cap reasons).
    if len(selected) > target_size:
        selected.sort(
            key=lambda e: (
                e.included_reason != "thematic",
                -(e.market_cap or 0),
            )
        )
        selected = selected[:target_size]

    # Final ordering by sector then market cap (descending).
    selected.sort(
        key=lambda e: (e.sector or "ZZZ", -(e.market_cap or 0)),
    )

    sector_counts: dict[str, int] = {}
    for e in selected:
        if e.sector:
            sector_counts[e.sector] = sector_counts.get(e.sector, 0) + 1

    n_thematic = sum(1 for e in selected if e.included_reason == "thematic")

    return Universe(
        quarter=quarter,
        constructed_at=datetime.utcnow(),
        entries=selected,
        sector_counts=sector_counts,
        n_thematic_added=n_thematic,
    )


def _current_quarter_label(*, today: date | None = None) -> str:
    d = today or date.today()
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"
