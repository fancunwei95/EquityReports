from __future__ import annotations

"""Steps 3.5 + 3.6: focus list + long/short selection with constraints.

Focus list (3.5): top 15 / bottom 15 by composite_z (falls back to
composite_score), plus an "outliers" bucket of names with extreme single-
component z-scores even if their composite is mid-pack.

Selection (3.6): from focus, pick 5 longs + 5 shorts respecting:
* Sector neutrality   -- max ``max_per_sector`` longs/shorts per GICS sector
* Turnover dampening  -- if a name was in last week's book and is still in
                         the focus list, prefer keeping it (avoids whipsaw)
* Beta neutralization -- assign basket-relative weights so long_beta ~= -short_beta

Earnings exclusion and short-interest filter from plan.md Step 3.6 are
stubbed via the ``earnings_exclude`` and ``short_interest_max`` kwargs;
v1 default is "no exclusion" since we don't yet pull earnings calendars
or shortPercentOfFloat reliably. The orchestrator can layer them on once
those data sources are wired in Stage 4+.
"""

from collections import defaultdict
from datetime import date

from weekly_strategy.data.schemas import (
    FocusList,
    FocusListItem,
    Portfolio,
    PortfolioPosition,
    StockScoreBundle,
)


TOP_K_LONGS = 15
TOP_K_SHORTS = 15
OUTLIER_THRESHOLD_Z = 1.5
DEFAULT_POSITION_COUNT = 5
DEFAULT_MAX_PER_SECTOR = 2
TURNOVER_RANK_THRESHOLD = 10  # keep held name unless it dropped past rank 10


# ---------------------------------------------------------------------------
# Focus list (3.5)
# ---------------------------------------------------------------------------


def _composite_for_rank(b: StockScoreBundle) -> float:
    """Prefer z-composite (cross-sectional), fall back to raw 0-100 composite."""
    if b.composite_z is not None:
        return b.composite_z
    if b.composite_score is not None:
        return b.composite_score
    return 50.0


def _item_from_bundle(b: StockScoreBundle, *, rank: int, outlier_reason: str | None = None) -> FocusListItem:
    return FocusListItem(
        ticker=b.ticker,
        composite_score=b.composite_score,
        composite_z=b.composite_z,
        quality_score=b.quality_score,
        valuation_score=b.valuation_score,
        momentum_score=b.momentum_score,
        sector=None,  # filled by caller if dossier lookup is available
        beta=None,
        rank=rank,
        outlier_reason=outlier_reason,
    )


def build_focus_list(
    bundles: dict[str, StockScoreBundle],
    *,
    week_ending: date,
    top_k_longs: int = TOP_K_LONGS,
    top_k_shorts: int = TOP_K_SHORTS,
    outlier_threshold_z: float = OUTLIER_THRESHOLD_Z,
    sector_lookup: dict[str, str | None] | None = None,
    beta_lookup: dict[str, float | None] | None = None,
) -> FocusList:
    """Rank the universe, return top-K / bottom-K + outlier bucket."""
    sector_lookup = sector_lookup or {}
    beta_lookup = beta_lookup or {}

    ranked = sorted(bundles.values(), key=_composite_for_rank, reverse=True)

    def enrich(item: FocusListItem) -> FocusListItem:
        return item.model_copy(update={
            "sector": sector_lookup.get(item.ticker),
            "beta": beta_lookup.get(item.ticker),
        })

    long_candidates = [
        enrich(_item_from_bundle(b, rank=i + 1))
        for i, b in enumerate(ranked[:top_k_longs])
    ]
    # Shorts: reverse order from the tail, rank counts from the WORST (1 = worst).
    short_candidates = [
        enrich(_item_from_bundle(b, rank=len(ranked) - i))
        for i, b in enumerate(reversed(ranked[-top_k_shorts:]))
    ]

    # Outliers: extreme single-component z across the universe, even if
    # the composite is mid-pack. Useful for "weird this week" surfacing.
    outliers: list[FocusListItem] = []
    picked: set[str] = set()
    for component, label in (
        ("z_news_sentiment", "extreme news sentiment"),
        ("z_quality", "extreme quality"),
        ("z_valuation", "extreme valuation"),
        ("z_momentum", "extreme momentum"),
    ):
        sorted_by_abs = sorted(
            (b for b in bundles.values() if getattr(b, component) is not None),
            key=lambda b: abs(getattr(b, component) or 0),
            reverse=True,
        )
        for b in sorted_by_abs[:2]:  # top 2 per component
            if b.ticker in picked:
                continue
            zval = getattr(b, component)
            if zval is None or abs(zval) < outlier_threshold_z:
                break
            picked.add(b.ticker)
            outliers.append(
                enrich(_item_from_bundle(
                    b, rank=ranked.index(b) + 1,
                    outlier_reason=f"{label} (z={zval:+.2f})",
                ))
            )

    return FocusList(
        week_ending=week_ending,
        long_candidates=long_candidates,
        short_candidates=short_candidates,
        outliers=outliers,
    )


# ---------------------------------------------------------------------------
# Selection with constraints (3.6)
# ---------------------------------------------------------------------------


def select_positions(
    focus: FocusList,
    *,
    week_ending: date,
    position_count: int = DEFAULT_POSITION_COUNT,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    held_long_tickers: set[str] | None = None,
    held_short_tickers: set[str] | None = None,
    earnings_exclude: set[str] | None = None,
    short_interest: dict[str, float] | None = None,
    short_interest_max: float | None = 0.25,
    shortable_filter: set[str] | None = None,
) -> Portfolio:
    """Pick ``position_count`` longs and ``position_count`` shorts subject to
    sector neutrality, optional earnings/short-interest exclusion, and
    turnover dampening for held names.
    """
    held_long_tickers = held_long_tickers or set()
    held_short_tickers = held_short_tickers or set()
    earnings_exclude = earnings_exclude or set()
    short_interest = short_interest or {}

    rejected: list[dict] = []

    def passes_long(item: FocusListItem, sector_counts: dict[str, int]) -> tuple[bool, str | None]:
        if item.ticker in earnings_exclude:
            return False, "earnings within exclusion window"
        if item.sector and sector_counts[item.sector] >= max_per_sector:
            return False, f"sector cap ({item.sector})"
        return True, None

    def passes_short(item: FocusListItem, sector_counts: dict[str, int]) -> tuple[bool, str | None]:
        if item.ticker in earnings_exclude:
            return False, "earnings within exclusion window"
        if item.sector and sector_counts[item.sector] >= max_per_sector:
            return False, f"sector cap ({item.sector})"
        if shortable_filter is not None and item.ticker not in shortable_filter:
            return False, "not in shortable list"
        si = short_interest.get(item.ticker)
        if (short_interest_max is not None and si is not None
                and si > short_interest_max):
            return False, f"short interest {si*100:.1f}% > {short_interest_max*100:.0f}%"
        return True, None

    # Turnover dampening: lift any held name still in the focus list
    # (rank <= TURNOVER_RANK_THRESHOLD) to the front of its side.
    def reorder_for_turnover(items: list[FocusListItem], held: set[str]) -> list[FocusListItem]:
        held_in_focus = [
            it for it in items
            if it.ticker in held and (it.rank or 1) <= TURNOVER_RANK_THRESHOLD
        ]
        fresh = [it for it in items if it.ticker not in held or (it.rank or 1) > TURNOVER_RANK_THRESHOLD]
        return held_in_focus + fresh

    longs: list[PortfolioPosition] = []
    sector_long: dict[str, int] = defaultdict(int)
    for item in reorder_for_turnover(focus.long_candidates, held_long_tickers):
        if len(longs) >= position_count:
            break
        ok, reason = passes_long(item, sector_long)
        if not ok:
            rejected.append({"ticker": item.ticker, "side": "long", "reason": reason})
            continue
        longs.append(_to_position(item, "long", held=(item.ticker in held_long_tickers)))
        if item.sector:
            sector_long[item.sector] += 1

    shorts: list[PortfolioPosition] = []
    sector_short: dict[str, int] = defaultdict(int)
    for item in reorder_for_turnover(focus.short_candidates, held_short_tickers):
        if len(shorts) >= position_count:
            break
        ok, reason = passes_short(item, sector_short)
        if not ok:
            rejected.append({"ticker": item.ticker, "side": "short", "reason": reason})
            continue
        shorts.append(_to_position(item, "short", held=(item.ticker in held_short_tickers)))
        if item.sector:
            sector_short[item.sector] += 1

    # Beta neutralization: equal-weight within each basket, then rescale shorts
    # so |short_beta| matches long_beta. If betas are missing we just
    # equal-weight both sides and let net beta drift.
    longs, shorts, long_beta, short_beta, net_beta = _beta_neutralize(longs, shorts)

    return Portfolio(
        week_ending=week_ending,
        longs=longs,
        shorts=shorts,
        long_basket_beta=long_beta,
        short_basket_beta=short_beta,
        net_beta=net_beta,
        rejected=rejected,
    )


def _to_position(item: FocusListItem, side: str, *, held: bool) -> PortfolioPosition:
    reason = "turnover_dampening" if held else "rank"
    return PortfolioPosition(
        ticker=item.ticker,
        direction=side,
        composite_score=item.composite_score,
        composite_z=item.composite_z,
        sector=item.sector,
        beta=item.beta,
        weight=1.0 / max(1, DEFAULT_POSITION_COUNT),  # equal-weight default
        selection_reason=reason,
    )


def _beta_neutralize(
    longs: list[PortfolioPosition],
    shorts: list[PortfolioPosition],
) -> tuple[list[PortfolioPosition], list[PortfolioPosition], float | None, float | None, float | None]:
    """Equal-weight inside each basket; rescale short basket so |short_beta| ~ long_beta.

    If any basket lacks beta data, return basket betas as None and skip
    rescaling -- net beta is then unknown. Both lists are returned as
    new objects since PortfolioPosition is frozen.
    """
    def basket_beta(positions: list[PortfolioPosition]) -> float | None:
        if not positions:
            return None
        betas = [p.beta for p in positions if p.beta is not None]
        if len(betas) < len(positions):
            return None  # incomplete data; refuse to compute
        return sum(p.weight * p.beta for p in positions)

    long_beta = basket_beta(longs)
    short_beta = basket_beta(shorts)

    if long_beta is None or short_beta is None or short_beta == 0:
        net = (long_beta or 0.0) - (short_beta or 0.0) if (long_beta is not None or short_beta is not None) else None
        return longs, shorts, long_beta, short_beta, net

    # Rescale shorts so weighted short_beta matches long_beta in magnitude.
    scale = abs(long_beta) / abs(short_beta) if short_beta else 1.0
    scaled_shorts = [p.model_copy(update={"weight": p.weight * scale}) for p in shorts]
    new_short_beta = basket_beta(scaled_shorts)
    net_beta = long_beta - (new_short_beta or 0.0)
    return longs, scaled_shorts, long_beta, new_short_beta, net_beta
