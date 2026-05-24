from __future__ import annotations

"""One-shot bootstrap: build the curated thematic 100-stock universe from
``config/thematic_universe.json``, pull yfinance metadata for each ticker,
and persist it as the current working Universe so run_stage3 picks it up.

Run once at project start (or whenever the curated list changes):

    python -m weekly_strategy.scripts.bootstrap_thematic_universe

Idempotent: re-running overwrites the same per-quarter snapshot.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from weekly_strategy.config import settings
from weekly_strategy.data import fetchers, storage
from weekly_strategy.data.schemas import Universe, UniverseEntry
from weekly_strategy.dossier import universe as universe_mod


_THEMATIC_PATH = settings.PACKAGE_ROOT / "config" / "thematic_universe.json"


def _load_curated() -> tuple[str, list[str]]:
    raw = json.loads(_THEMATIC_PATH.read_text())
    quarter = raw.get("quarter") or universe_mod._current_quarter_label()
    buckets = raw.get("buckets") or {}
    # Flatten + dedupe, preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for _bucket, names in buckets.items():
        for t in names:
            u = t.upper()
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
    return quarter, out


def main() -> int:
    storage.init_db()
    quarter, tickers = _load_curated()
    print(f"[bootstrap] quarter={quarter}  tickers={len(tickers)}")

    entries: list[UniverseEntry] = []
    failed: list[tuple[str, str]] = []
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        try:
            info = fetchers.get_basic_info(ticker)
            entries.append(UniverseEntry(
                ticker=ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=info.get("market_cap"),
                beta=info.get("beta"),
                included_reason="thematic",
            ))
            sector = info.get("sector") or "?"
            mc = (info.get("market_cap") or 0) / 1e9
            print(f"  [{i:3d}/{len(tickers)}] {ticker:<6} {sector:<28} ${mc:>7.1f}B")
        except Exception as e:
            failed.append((ticker, f"{type(e).__name__}: {e}"))
            print(f"  [{i:3d}/{len(tickers)}] {ticker:<6} FAILED: {e}")

    sector_counts: dict[str, int] = {}
    for e in entries:
        if e.sector:
            sector_counts[e.sector] = sector_counts.get(e.sector, 0) + 1

    universe = Universe(
        quarter=quarter,
        constructed_at=datetime.utcnow(),
        entries=entries,
        sector_counts=sector_counts,
        n_thematic_added=len(entries),
    )
    path = storage.save_universe(universe)
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"saved -> {path}  ({elapsed:.1f}s)")
    print(f"{len(entries)} accepted / {len(failed)} failed")
    print()
    print("GICS sector breakdown:")
    for s, n in sorted(sector_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<28} {n}")
    if failed:
        print()
        print("Failed tickers:")
        for t, err in failed:
            print(f"  {t}: {err[:80]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
