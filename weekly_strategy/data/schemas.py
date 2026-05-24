from __future__ import annotations

"""Pydantic schemas for typed records that flow through SQLite.

Prices stay as ``pandas.DataFrame`` and dossiers/reports stay as ``dict`` —
those don't pass through a row-oriented store, so they don't need a row schema.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


_SENTIMENT_VALUES = ("POSITIVE", "NEUTRAL", "NEGATIVE")
_MATERIALITY_VALUES = ("HIGH", "MEDIUM", "LOW")
_THEME_VALUES = (
    "earnings", "guidance", "product", "regulation", "litigation",
    "M&A", "management", "macro", "competitive", "analyst",
    "technical", "other",
)


class NewsClassification(BaseModel):
    """LLM verdict on a single headline."""

    model_config = ConfigDict(frozen=True)

    sentiment: str   # one of _SENTIMENT_VALUES (validated via field_validator below)
    materiality: str # one of _MATERIALITY_VALUES
    theme: str       # one of _THEME_VALUES (defaults to 'other' on unknown)
    rationale: str | None = None

    @classmethod
    def coerce(cls, raw: dict) -> "NewsClassification":
        """Best-effort: clamp unknown values to safe defaults so a bad LLM
        token doesn't poison a 200-item batch."""
        s = str(raw.get("sentiment", "")).upper()
        m = str(raw.get("materiality", "")).upper()
        t = str(raw.get("theme", "")).lower()
        return cls(
            sentiment=s if s in _SENTIMENT_VALUES else "NEUTRAL",
            materiality=m if m in _MATERIALITY_VALUES else "LOW",
            theme=t if t in _THEME_VALUES else "other",
            rationale=(raw.get("rationale") or None),
        )

    # Numeric mappings used by aggregation. Kept here so the mapping table
    # has one definition, not one per consumer.
    @property
    def sentiment_value(self) -> int:
        return {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}[self.sentiment]

    @property
    def materiality_weight(self) -> float:
        return {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.2}[self.materiality]


_AREA_VALUES = (
    "revenue", "gross_margin", "operating_margin", "fcf_margin",
    "operating_leverage", "working_capital", "leverage", "share_count",
    "tax_rate", "other",
)
_DIRECTION_VALUES = ("POSITIVE", "NEGATIVE", "UNCLEAR")
_MAGNITUDE_VALUES = ("small", "medium", "large")  # <1%, 1-5%, >5% of line item
_HORIZON_VALUES = ("Q", "FY25", "FY26", "FY27", "multi-year")


class FundamentalImpact(BaseModel):
    """Second-pass LLM read of a news item: how does this hit the financials?

    Only items already classified HIGH or MEDIUM materiality flow through
    this agent (the cascade in signals/news_sentiment.py). The output feeds
    the thesis writer in Step 1.8 so it can say *"FY26 revenue tailwind"*
    instead of *"news was positive"*.
    """

    model_config = ConfigDict(frozen=True)

    areas: list[str] = Field(default_factory=list)  # subset of _AREA_VALUES
    direction: str = "UNCLEAR"
    magnitude: str = "small"
    horizon: str = "Q"
    implication: str | None = None

    @classmethod
    def coerce(cls, raw: dict) -> "FundamentalImpact":
        raw_areas = raw.get("areas") or []
        if isinstance(raw_areas, str):
            raw_areas = [raw_areas]
        areas = [a for a in (str(x).lower() for x in raw_areas) if a in _AREA_VALUES]
        d = str(raw.get("direction", "")).upper()
        m = str(raw.get("magnitude", "")).lower()
        h = str(raw.get("horizon", "")).strip()
        return cls(
            areas=areas or ["other"],
            direction=d if d in _DIRECTION_VALUES else "UNCLEAR",
            magnitude=m if m in _MAGNITUDE_VALUES else "small",
            horizon=h if h in _HORIZON_VALUES else "Q",
            implication=raw.get("implication") or None,
        )

    @property
    def magnitude_weight(self) -> float:
        """Numeric scale for magnitude. Used by aggregator if surfaced later."""
        return {"small": 0.2, "medium": 0.5, "large": 1.0}[self.magnitude]


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    title: str | None = None
    source: str | None = None
    url: str
    published_at: datetime | None = None
    snippet: str | None = None
    # Populated by Step 1.5 pass 1; None for un-classified items.
    classification: NewsClassification | None = None
    # Populated by Step 1.5 pass 2; only HIGH+MEDIUM items get one.
    fundamental_impact: FundamentalImpact | None = None


class RedditPost(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    subreddit: str | None = None
    post_id: str
    title: str | None = None
    score: int = 0
    num_comments: int = Field(default=0, ge=0)
    created_at: datetime | None = None
    url: str | None = None


_FED_POSTURE_VALUES = (
    "HAWKISH", "SLIGHTLY_HAWKISH", "NEUTRAL", "SLIGHTLY_DOVISH", "DOVISH",
)


_RATE_REGIME_VALUES = ("cutting", "on_hold", "hiking_or_holding", "tightening_market", "easing_market", "unknown")
_FIN_COND_VALUES = ("easing", "stable", "tightening", "unknown")
_CYCLE_PHASE_VALUES = ("early", "mid", "late", "recession", "unknown")


class FocusListItem(BaseModel):
    """One name in the weekly focus list (Step 3.5)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    composite_score: float | None = None
    composite_z: float | None = None
    quality_score: float | None = None
    valuation_score: float | None = None
    momentum_score: float | None = None
    sector: str | None = None
    beta: float | None = None
    rank: int | None = None
    # Why this name made the outlier bucket, if applicable.
    outlier_reason: str | None = None


class FocusList(BaseModel):
    model_config = ConfigDict(frozen=True)

    week_ending: date
    long_candidates: list[FocusListItem] = Field(default_factory=list)
    short_candidates: list[FocusListItem] = Field(default_factory=list)
    outliers: list[FocusListItem] = Field(default_factory=list)


class PortfolioPosition(BaseModel):
    """A selected name in the weekly book (Step 3.6)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    direction: str             # "long" | "short"
    composite_score: float | None = None
    composite_z: float | None = None
    sector: str | None = None
    beta: float | None = None
    weight: float = 0.20       # fraction of basket (5 equal-weight = 0.20)
    selection_reason: str = "rank"  # "rank" | "thematic" | "turnover_dampening"


class Portfolio(BaseModel):
    """The weekly L/S book."""

    model_config = ConfigDict(frozen=True)

    week_ending: date
    longs: list[PortfolioPosition] = Field(default_factory=list)
    shorts: list[PortfolioPosition] = Field(default_factory=list)
    long_basket_beta: float | None = None
    short_basket_beta: float | None = None
    net_beta: float | None = None
    rejected: list[dict] = Field(default_factory=list)  # {ticker, side, reason}


class UniverseEntry(BaseModel):
    """A single name in the Stage 3 universe, with the metadata that put it there."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    beta: float | None = None
    included_reason: str = "cap_rank"  # cap_rank | thematic | manual


class Universe(BaseModel):
    """Stage 3 ticker universe, refreshed quarterly."""

    model_config = ConfigDict(frozen=True)

    quarter: str  # "2026Q2"
    constructed_at: datetime
    entries: list[UniverseEntry] = Field(default_factory=list)
    sector_counts: dict[str, int] = Field(default_factory=dict)
    n_thematic_added: int = 0

    @property
    def tickers(self) -> list[str]:
        return [e.ticker for e in self.entries]

    def get(self, ticker: str) -> UniverseEntry | None:
        for e in self.entries:
            if e.ticker == ticker.upper():
                return e
        return None


class CrossStockImplications(BaseModel):
    """Step 2.5 (deferred Step 1.5 pass 3): cross-stock / sectoral read.

    One LLM call ingests a ticker's material news + its sector context +
    the macro regime, then articulates implications going both directions
    (this stock's news -> sector implications, and sector context -> this
    stock implications). Other tickers that could be affected get named.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    week_ending: date
    sector_etf: str | None = None
    implications_for_sector: list[str] = Field(default_factory=list)
    implications_from_sector: list[str] = Field(default_factory=list)
    related_tickers: list[str] = Field(default_factory=list)
    summary: str | None = None

    @classmethod
    def coerce(
        cls,
        *,
        ticker: str,
        week_ending: date,
        sector_etf: str | None,
        raw: dict,
    ) -> "CrossStockImplications":
        return cls(
            ticker=ticker.upper(),
            week_ending=week_ending,
            sector_etf=sector_etf,
            implications_for_sector=_string_list(raw.get("implications_for_sector")),
            implications_from_sector=_string_list(raw.get("implications_from_sector")),
            related_tickers=[
                str(t).upper()[:6] for t in (raw.get("related_tickers") or [])
                if isinstance(t, str)
            ][:8],
            summary=(raw.get("summary") or None),
        )


class MacroRegime(BaseModel):
    """Stage 2 regime label.

    ``rate_regime`` and ``financial_conditions`` are deterministic functions
    of the MacroSnapshot + FedPosture inputs. ``cycle_phase`` and
    ``narrative`` come from a single LLM synthesis call.
    """

    model_config = ConfigDict(frozen=True)

    week_ending: date
    rate_regime: str = "unknown"
    financial_conditions: str = "unknown"
    cycle_phase: str = "unknown"
    # Mirrored from the underlying MacroSnapshot so downstream code (e.g.,
    # sector_score) doesn't need to thread both objects everywhere.
    hy_regime: str = "unknown"
    vix_regime: str = "unknown"
    narrative: str | None = None
    risks: list[str] = Field(default_factory=list)

    @classmethod
    def coerce_llm(
        cls,
        *,
        week_ending: date,
        rate_regime: str,
        financial_conditions: str,
        hy_regime: str = "unknown",
        vix_regime: str = "unknown",
        raw: dict,
    ) -> "MacroRegime":
        cp = str(raw.get("cycle_phase", "")).lower()
        return cls(
            week_ending=week_ending,
            rate_regime=rate_regime if rate_regime in _RATE_REGIME_VALUES else "unknown",
            financial_conditions=(
                financial_conditions if financial_conditions in _FIN_COND_VALUES else "unknown"
            ),
            hy_regime=hy_regime,
            vix_regime=vix_regime,
            cycle_phase=cp if cp in _CYCLE_PHASE_VALUES else "unknown",
            narrative=(raw.get("narrative") or None),
            risks=_string_list(raw.get("risks")),
        )


class SectorMetrics(BaseModel):
    """Per-sector momentum + volume profile vs SPY."""

    model_config = ConfigDict(frozen=True)

    etf: str
    sector: str
    return_1w: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    rel_1m: float | None = None       # sector_1m - spy_1m
    rel_3m: float | None = None
    volume_vs_20d: float | None = None  # last-5-day avg vol / trailing 20d avg


class SectorSnapshot(BaseModel):
    """Cross-sector snapshot built from XL* ETF flows."""

    model_config = ConfigDict(frozen=True)

    week_ending: date
    sectors: dict[str, SectorMetrics] = Field(default_factory=dict)
    leadership_ranking: list[str] = Field(default_factory=list)  # ETF tickers, best -> worst by rel_1m
    breadth: str = "unknown"  # "broad" | "narrow" | "unknown"

    def metrics_for_sector(self, sector_name: str) -> SectorMetrics | None:
        """Lookup helper -- get_basic_info returns 'sector' as a string label."""
        target = sector_name.lower().replace("&", "and") if sector_name else ""
        for m in self.sectors.values():
            if m.sector.lower().replace("&", "and") == target:
                return m
        return None


class FedPosture(BaseModel):
    """LLM read of Fed communication this week."""

    model_config = ConfigDict(frozen=True)

    week_ending: date
    posture: str = "NEUTRAL"            # one of _FED_POSTURE_VALUES
    summary: str | None = None
    policy_hints: list[str] = Field(default_factory=list)
    key_speakers: list[str] = Field(default_factory=list)
    n_items: int = 0

    @classmethod
    def coerce(cls, *, week_ending: date, n_items: int, raw: dict) -> "FedPosture":
        p = str(raw.get("posture", "")).upper().replace(" ", "_").replace("-", "_")
        return cls(
            week_ending=week_ending,
            posture=p if p in _FED_POSTURE_VALUES else "NEUTRAL",
            summary=(raw.get("summary") or None),
            policy_hints=_string_list(raw.get("policy_hints")),
            key_speakers=_string_list(raw.get("key_speakers")),
            n_items=n_items,
        )


def _string_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None][:8]
    return [str(raw)]


class MacroSnapshot(BaseModel):
    """Universe-level rates / credit / vol / inflation / FX snapshot.

    Populated once per weekly run (not per ticker). The Stage 2 thesis writer
    consumes this so a stock's view incorporates the macro backdrop.
    """

    model_config = ConfigDict(frozen=True)

    week_ending: date

    # Rates state -- all in percent (5.25 means 5.25%).
    yield_10y: float | None = None
    yield_10y_wow_change_bps: float | None = None
    yield_2y: float | None = None
    yield_2y_wow_change_bps: float | None = None
    real_yield_10y: float | None = None
    real_yield_10y_wow_change_bps: float | None = None
    fed_funds: float | None = None
    mortgage_30y: float | None = None

    # Curve
    curve_2s10s_bps: float | None = None
    curve_inverted: bool = False

    # Credit -- in basis points (350 means 350 bps OAS).
    hy_oas_bps: float | None = None
    hy_oas_wow_change_bps: float | None = None
    ig_oas_bps: float | None = None
    hy_regime: str = "unknown"      # tight | normal | wide | stressed | unknown

    # Vol
    vix_level: float | None = None
    vix_wow_change: float | None = None
    vix_regime: str = "unknown"     # complacent | normal | elevated | panic | unknown

    # Inflation expectations
    breakeven_10y: float | None = None
    breakeven_10y_wow_change_bps: float | None = None

    # Currency
    dxy_level: float | None = None
    dxy_wow_change_pct: float | None = None

    # Recent data prints (last 7 days). Free-form dicts so we can include
    # CPI / payrolls / claims surprises without a rigid schema in v1.
    recent_prints: list[dict] = Field(default_factory=list)


class WeeklyThesis(BaseModel):
    """Structured analyst note produced by the LLM thesis writer (Step 1.8)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    week_ending: date
    bottom_line: str
    what_changed: list[str] = Field(default_factory=list)
    quality_and_valuation: str | None = None
    risks: list[str] = Field(default_factory=list)
    watch_items: list[str] = Field(default_factory=list)
    # Side metadata so we can audit cost without re-running:
    model: str | None = None
    cost_usd: float | None = None


class StockScoreBundle(BaseModel):
    """Numeric weekly snapshot per ticker. Stage 1: standalone; Stage 2 adds
    macro/sector overlays; Stage 3 adds cross-sectional ranks."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    week_ending: date

    # Composite scores in [0, 100]. Higher is "more attractive" for a long.
    quality_score: float
    valuation_score: float
    momentum_score: float

    # Raw returns used by momentum, surfaced for the report.
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None

    # Cross-references to other signals so the thesis writer + report
    # can read everything from one bundle. None until the orchestrator
    # populates them.
    news_sentiment_score: float | None = None  # [-1, +1] from NewsScore
    news_noise_ratio: float | None = None
    reddit_sentiment: float | None = None      # [-1, +1] from RedditScore
    reddit_is_crowded: bool = False

    # Stage 2 additions -- macro / sector overlays. None when Stage 1 only.
    macro_regime_score: float | None = None    # 0-100 universe-level
    sector_score_value: float | None = None    # 0-100 per stock
    sector_etf: str | None = None              # which ETF was used
    sector_rank: int | None = None             # 1 = best sector this week
    sector_in_favor: bool = False              # regime fit >= 60
    composite_score: float | None = None       # weighted final 0-100

    # Stage 3 additions -- cross-sectional z-scores vs the working universe.
    # Populated by normalize_scores() in batch_scoring.py. None when running a
    # single-ticker / Stage 1-2 pipeline.
    z_quality: float | None = None
    z_valuation: float | None = None
    z_momentum: float | None = None
    z_news_sentiment: float | None = None
    z_sector: float | None = None
    composite_z: float | None = None  # weighted z-composite for ranking


class Dossier(BaseModel):
    """Per-ticker fundamental snapshot built from EDGAR + yfinance.

    Many fields are ``| None`` because either the company doesn't disclose
    them (e.g., banks have no gross profit), the concept tag is missing from
    their XBRL filings, or the value requires consensus data we don't have
    in v1 (forward PE, earnings surprises).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None

    # Flag: financials report fundamentally differently (no GP/COGS, leverage
    # is the business model, etc). Downstream scoring should branch on this.
    is_financials: bool = False

    # Quality block
    revenue_latest_fy: float | None = None
    revenue_3y_cagr: float | None = None
    revenue_yoy_latest: float | None = None
    revenue_qoq_latest: float | None = None
    gross_margin_current: float | None = None
    gross_margin_trend: list[float] = Field(default_factory=list)
    operating_margin_current: float | None = None
    fcf_margin_current: float | None = None
    roic: float | None = None
    roe: float | None = None
    share_count_change_yoy: float | None = None
    net_debt_to_ebitda: float | None = None
    fcf_conversion: float | None = None

    # Growth / momentum
    next_earnings_date: date | None = None
    last_4_surprises: list[dict] = Field(default_factory=list)  # TODO Step 1.x

    # Valuation (computed against current price)
    current_price: float | None = None
    market_cap: float | None = None
    pe_trailing: float | None = None
    pe_forward: float | None = None  # requires consensus; v1: None
    ps_ratio: float | None = None
    fcf_yield: float | None = None
    ev_to_ebitda: float | None = None

    # Catalyst (free-form; populated later)
    upcoming_events: list[str] = Field(default_factory=list)
    recent_8k_summary: str | None = None

    # Metadata
    last_updated: datetime
    last_filing_date: date | None = None


class ThemeCount(BaseModel):
    model_config = ConfigDict(frozen=True)
    theme: str
    count: int


class TopItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    url: str
    title: str | None = None
    source: str | None = None
    sentiment: str
    materiality: str
    contribution: float  # signed: sentiment_value * materiality_weight


class NewsScore(BaseModel):
    """Aggregated weekly news signal for a single ticker."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    window_start: datetime
    window_end: datetime
    n_total: int
    n_classified: int
    # Materiality-weighted average of sentiment values in [-1, +1].
    sentiment_score: float
    top_themes: list[ThemeCount] = Field(default_factory=list)
    top_items: list[TopItem] = Field(default_factory=list)
    # Share of items classified as LOW materiality. High noise_ratio means
    # a "quiet week" of routine coverage.
    noise_ratio: float


class RedditScore(BaseModel):
    """Post-LLM scoring of a Reddit snapshot."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    window_hours: int = 24
    mention_count: int = 0
    mention_count_prior_window: int = 0
    mention_delta_pct: float = 0.0
    # LLM-derived sentiment in [-1, +1]. None when we didn't bother (low volume).
    llm_sentiment: float | None = None
    llm_themes: list[str] = Field(default_factory=list)
    # Crowding heuristic: True if mention spike is unusual (delta > +100% or count > 50).
    is_crowded: bool = False


class RedditSnapshot(BaseModel):
    """Aggregated Reddit signal for a ticker over a rolling window.

    ``mention_count_prior_window`` looks at the previous window from our own
    SQLite buffer (we don't re-query Reddit for older data).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    window_hours: int = 24
    mention_count: int = 0
    mention_count_prior_window: int = 0
    mention_delta_pct: float = 0.0
    top_posts: list[RedditPost] = Field(default_factory=list)
    # Populated later by the LLM sentiment step (Step 1.6); kept optional here
    # so the schema accepts snapshots produced before sentiment is wired in.
    sentiment_proxy: dict | None = None
