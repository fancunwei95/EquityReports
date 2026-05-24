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


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    title: str | None = None
    source: str | None = None
    url: str
    published_at: datetime | None = None
    snippet: str | None = None
    # Populated by Step 1.5; None for un-classified items.
    classification: NewsClassification | None = None


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
