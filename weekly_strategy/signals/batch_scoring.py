from __future__ import annotations

"""Step 3.3-3.4: batch weekly scoring + cross-sectional normalization.

``batch_weekly_scoring`` reuses every Stage 1-2 per-ticker primitive but
ratchets them into a universe-wide loop with per-ticker isolation. The
macro / sector / regime context is pulled once and shared across all
tickers (cheap -- one yfinance batch + one FRED batch + one or two
Sonnet synthesis calls).

``normalize_scores`` computes z-scores for each scoring component across
the universe, attaches them to each bundle as ``z_*`` fields, and recomputes
``composite_z`` from regime-weighted z-scores. This is what plan.md Step
3.4 means by "normalize within the universe so scores are comparable" --
once you have 100 stocks, you want to know who's *relatively* good on
quality, not just absolutely.
"""

import statistics
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from weekly_strategy.data import fetchers, storage
from weekly_strategy.data.schemas import (
    CrossStockImplications,
    Dossier,
    MacroRegime,
    MacroSnapshot,
    SectorSnapshot,
    StockScoreBundle,
    Universe,
)
from weekly_strategy.signals import fundamental as fnd
from weekly_strategy.signals import news_sentiment, reddit_sentiment, sector_score


@dataclass
class BatchScoreResult:
    bundles: dict[str, StockScoreBundle] = field(default_factory=dict)
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def batch_weekly_scoring(
    universe: Universe,
    *,
    week_ending: date,
    macro_snapshot: MacroSnapshot | None = None,
    macro_regime: MacroRegime | None = None,
    sector_snap: SectorSnapshot | None = None,
    news_days: int = 7,
    price_lookback_days: int = 365,
    reddit_hours: int = 24,
    skip_reddit: bool = False,
    skip_news: bool = False,
    log=print,
) -> BatchScoreResult:
    """Compute a StockScoreBundle for every name in ``universe``.

    Per-ticker pipeline: prices -> news cascade -> reddit -> sector score
    -> bundle. Errors on one ticker do not stop the batch.
    """
    res = BatchScoreResult()
    t_start = time.time()
    since = datetime.combine(week_ending, datetime.min.time()) - timedelta(days=news_days)

    for i, ticker in enumerate(universe.tickers, 1):
        try:
            log(f"  [{i:3d}/{len(universe.tickers)}] {ticker}...")
            bundle = _score_one(
                ticker=ticker,
                week_ending=week_ending,
                since=since,
                macro_regime=macro_regime,
                sector_snap=sector_snap,
                news_days=news_days,
                price_lookback_days=price_lookback_days,
                reddit_hours=reddit_hours,
                skip_reddit=skip_reddit,
                skip_news=skip_news,
            )
            res.bundles[ticker] = bundle
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            res.failed.append((ticker, err))
            log(f"  FAIL {ticker}: {err}")

    res.elapsed_s = time.time() - t_start
    log(
        f"batch_weekly_scoring done in {res.elapsed_s:.1f}s -- "
        f"{len(res.bundles)} ok / {len(res.failed)} failed"
    )
    return res


def _score_one(
    *,
    ticker: str,
    week_ending: date,
    since: datetime,
    macro_regime: MacroRegime | None,
    sector_snap: SectorSnapshot | None,
    news_days: int,
    price_lookback_days: int,
    reddit_hours: int,
    skip_reddit: bool,
    skip_news: bool,
) -> StockScoreBundle:
    # 1. Prices (refreshed if stale; the underlying parquet cache handles re-pulls)
    fetchers.get_price_history(ticker, lookback_days=price_lookback_days)
    prices = storage.load_prices(ticker)

    # 2. News (fetch + cascade)
    news_sent_score = None
    news_noise = None
    if not skip_news:
        info = fetchers.get_basic_info(ticker)
        items = fetchers.fetch_news(ticker, days=news_days)
        if items:
            storage.insert_news(items)
        ns = news_sentiment.score_news(
            ticker, since=since, company_name=info.get("name"),
        )
        news_sent_score = ns.sentiment_score
        news_noise = ns.noise_ratio

    # 3. Reddit (optional)
    reddit_sent = None
    reddit_crowded = False
    if not skip_reddit:
        posts = fetchers.fetch_reddit_recent(ticker, hours=reddit_hours)
        if posts:
            storage.insert_reddit(posts)
        snap = fetchers.build_reddit_snapshot(
            ticker, current_posts=posts, window_hours=reddit_hours,
        )
        rscore = reddit_sentiment.score_reddit_snapshot(snap)
        reddit_sent = rscore.llm_sentiment
        reddit_crowded = rscore.is_crowded

    # 4. Sector score (from cached sector_snap; pure compute)
    blob = storage.load_dossier(ticker)
    if blob is None:
        raise RuntimeError("no cached dossier; run batch_build_dossiers first")
    dossier = Dossier(**blob["data"])
    sector_info = sector_score.sector_score(
        dossier=dossier, sector_snap=sector_snap, regime=macro_regime,
    )

    # 5. Bundle
    return fnd.build_score_bundle(
        ticker=ticker,
        week_ending=week_ending,
        dossier=dossier,
        prices=prices,
        news_sentiment_score=news_sent_score,
        news_noise_ratio=news_noise,
        reddit_sentiment=reddit_sent,
        reddit_is_crowded=reddit_crowded,
        macro_regime=macro_regime,
        sector_score_info=sector_info,
    )


# ---------------------------------------------------------------------------
# Cross-sectional normalization
# ---------------------------------------------------------------------------

_NORMALIZED_COMPONENTS: list[tuple[str, str]] = [
    # (bundle attribute name, output attribute name)
    ("quality_score", "z_quality"),
    ("valuation_score", "z_valuation"),
    ("momentum_score", "z_momentum"),
    ("news_sentiment_score", "z_news_sentiment"),
    ("sector_score_value", "z_sector"),
]

# Same weights as the raw composite, just applied to z-scores. The mapping
# from z back to a 0-100 axis happens at the end (z=+2 -> 100, z=-2 -> 0).
_RISK_OFF_WEIGHTS_Z = {
    "z_quality":   0.30, "z_valuation": 0.25, "z_momentum":  0.10,
    "z_news_sentiment": 0.20, "z_sector": 0.15,
}
_RISK_ON_WEIGHTS_Z = {
    "z_quality":   0.20, "z_valuation": 0.15, "z_momentum":  0.20,
    "z_news_sentiment": 0.20, "z_sector": 0.25,
}


def normalize_scores(
    bundles: dict[str, StockScoreBundle],
    *,
    regime: MacroRegime | None = None,
) -> dict[str, StockScoreBundle]:
    """Attach z-scores to each bundle. Returns a NEW dict (immutable models).

    For each component, computes mean/std across the universe, then sets the
    corresponding ``z_*`` field per bundle. Bundles missing the raw value get
    z=0 (neutral) -- they don't penalize the universe stats either.
    """
    if not bundles:
        return {}

    # Compute mu/sigma per component from the values that are present.
    stats: dict[str, tuple[float, float]] = {}
    for raw_attr, _z_attr in _NORMALIZED_COMPONENTS:
        values = [
            getattr(b, raw_attr) for b in bundles.values()
            if getattr(b, raw_attr) is not None
        ]
        if not values:
            stats[raw_attr] = (0.0, 0.0)
            continue
        mu = statistics.fmean(values)
        sigma = statistics.pstdev(values) if len(values) > 1 else 0.0
        stats[raw_attr] = (mu, sigma)

    # Build the updated bundles.
    out: dict[str, StockScoreBundle] = {}
    for ticker, b in bundles.items():
        updates: dict[str, float | None] = {}
        for raw_attr, z_attr in _NORMALIZED_COMPONENTS:
            raw = getattr(b, raw_attr)
            mu, sigma = stats[raw_attr]
            if raw is None or sigma == 0.0:
                updates[z_attr] = 0.0
            else:
                updates[z_attr] = (raw - mu) / sigma
        # Composite z = weighted sum, then mapped to a 0-100 score
        # (z=+2 -> 100, z=-2 -> 0). The 0-100 axis keeps it comparable
        # with the raw composite_score for reporting.
        updates["composite_z"] = _composite_z(updates, regime)
        out[ticker] = b.model_copy(update=updates)
    return out


def _composite_z(z_values: dict[str, float | None], regime: MacroRegime | None) -> float:
    """Weighted z-composite, mapped from [-2, +2] z -> [0, 100]."""
    risk_off = regime is not None and (
        regime.financial_conditions == "tightening"
        or regime.cycle_phase == "recession"
    )
    weights = _RISK_OFF_WEIGHTS_Z if risk_off else _RISK_ON_WEIGHTS_Z
    weighted_z = sum(
        weights[k] * (z_values.get(k) or 0.0) for k in weights
    )
    score = 50.0 + 25.0 * weighted_z  # z=+2 -> 100, z=-2 -> 0
    return max(0.0, min(100.0, score))
