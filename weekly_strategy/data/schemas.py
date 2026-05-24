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
