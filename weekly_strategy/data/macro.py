from __future__ import annotations

"""FRED macro data fetcher + MacroSnapshot assembler.

Caches each series to ``data_cache/macro/{series_id}.parquet`` so re-runs
within a week don't re-hit FRED. FRED has no rate limits to speak of, but
caching also makes development iteration cheap.

Auth: set FRED_API_KEY in .env. Without it, get_fred_series raises a
clear error -- caller can choose to skip the macro layer entirely.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd

from weekly_strategy.config import settings
from weekly_strategy.data.schemas import FedPosture, MacroSnapshot, NewsItem
from weekly_strategy.llm import client, prompts


# ---------------------------------------------------------------------------
# Series catalog
# ---------------------------------------------------------------------------

# (series_id, friendly_label) per category. The fetcher pulls all of them
# in a single get_macro_snapshot call.
FRED_SERIES: dict[str, str] = {
    # Rates
    "DGS2":             "2y Treasury yield",
    "DGS10":            "10y Treasury yield",
    "T10Y2Y":           "10y-2y curve",
    "T10YIE":           "10y breakeven inflation",
    "DFII10":           "10y real yield",
    "DFF":              "Effective fed funds (daily)",
    "FEDFUNDS":         "Effective fed funds (monthly)",
    "MORTGAGE30US":     "30y mortgage rate",
    # Credit (OAS in percent; we convert to bps in the snapshot)
    "BAMLH0A0HYM2":     "HY OAS",
    "BAMLC0A0CM":       "IG OAS",
    "BAMLH0A0HYM2EY":   "HY effective yield",
    # Macro prints
    "CPIAUCSL":         "CPI (all urban)",
    "PCEPI":            "PCE price index",
    "PAYEMS":           "Nonfarm payrolls",
    "UNRATE":           "Unemployment",
    "INDPRO":           "Industrial production",
    "UMCSENT":          "U.Mich consumer sentiment",
    "ICSA":             "Initial claims",
    # Market
    "VIXCLS":           "VIX",
    "DTWEXBGS":         "Broad dollar index",
}

_MACRO_CACHE = settings.DATA_CACHE_DIR / "macro"
_MACRO_CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-series fetch
# ---------------------------------------------------------------------------


class FredAuthError(RuntimeError):
    """Raised when FRED_API_KEY is missing. Caller can degrade gracefully."""


def _require_key() -> str:
    key = settings.FRED_API_KEY
    if not key:
        raise FredAuthError(
            "FRED_API_KEY not set. Register at fred.stlouisfed.org and add "
            "FRED_API_KEY=... to your .env."
        )
    return key


def _cache_path(series_id: str) -> Path:
    return _MACRO_CACHE / f"{series_id}.parquet"


def get_fred_series(
    series_id: str,
    *,
    lookback_days: int = 365,
    force_refresh: bool = False,
) -> pd.Series:
    """Fetch one FRED series (cached). Returns a pd.Series indexed by date."""
    cache = _cache_path(series_id)
    if cache.exists() and not force_refresh:
        df = pd.read_parquet(cache)
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]).values, name=series_id)
        return s

    from fredapi import Fred  # imported lazily so tests can mock at module level

    fred = Fred(api_key=_require_key())
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    s = fred.get_series(series_id, observation_start=start)
    s.name = series_id
    s = s.dropna()
    # Cache as a flat (date, value) table to sidestep parquet/index quirks.
    out = pd.DataFrame({"date": pd.to_datetime(s.index), "value": s.values})
    out.to_parquet(cache, index=False)
    return s


# ---------------------------------------------------------------------------
# Snapshot assembler
# ---------------------------------------------------------------------------


def _latest_and_wow(s: pd.Series) -> tuple[float | None, float | None]:
    """Return (latest_value, 7d-ago value) -- None if not available."""
    if s is None or len(s) == 0:
        return None, None
    s = s.sort_index()
    latest = float(s.iloc[-1])
    # find a value ~7 days before the latest
    cutoff = s.index[-1] - pd.Timedelta(days=7)
    prior_window = s.loc[:cutoff]
    if prior_window.empty:
        return latest, None
    prior = float(prior_window.iloc[-1])
    return latest, prior


def _wow_bps(latest: float | None, prior: float | None) -> float | None:
    """Week-over-week change converted to basis points (for percent series)."""
    if latest is None or prior is None:
        return None
    return (latest - prior) * 100.0


def _classify_hy(oas_bps: float | None) -> str:
    if oas_bps is None:
        return "unknown"
    if oas_bps < 350:
        return "tight"
    if oas_bps < 500:
        return "normal"
    if oas_bps < 700:
        return "wide"
    return "stressed"


def _classify_vix(vix: float | None) -> str:
    if vix is None:
        return "unknown"
    if vix < 15:
        return "complacent"
    if vix < 20:
        return "normal"
    if vix < 30:
        return "elevated"
    return "panic"


def get_macro_snapshot(*, week_ending: date | None = None) -> MacroSnapshot:
    """Pull every FRED_SERIES and assemble a MacroSnapshot.

    Designed to keep working when individual series are temporarily missing
    -- a None on any single field is preferable to a hard fail mid-run.
    """
    we = week_ending or date.today()

    def pull(series_id: str) -> pd.Series | None:
        try:
            return get_fred_series(series_id)
        except Exception:
            return None

    series = {sid: pull(sid) for sid in FRED_SERIES}

    def latest_wow(sid: str) -> tuple[float | None, float | None]:
        s = series.get(sid)
        if s is None or s.empty:
            return None, None
        return _latest_and_wow(s)

    y10, y10_prior = latest_wow("DGS10")
    y2, y2_prior = latest_wow("DGS2")
    real10, real10_prior = latest_wow("DFII10")
    fedfunds, _ = latest_wow("DFF")
    mortgage, _ = latest_wow("MORTGAGE30US")
    curve_2s10s, _ = latest_wow("T10Y2Y")
    hy_oas_pct, hy_oas_prior = latest_wow("BAMLH0A0HYM2")
    ig_oas_pct, _ = latest_wow("BAMLC0A0CM")
    vix, vix_prior = latest_wow("VIXCLS")
    be10, be10_prior = latest_wow("T10YIE")
    dxy, dxy_prior = latest_wow("DTWEXBGS")

    hy_oas_bps = hy_oas_pct * 100.0 if hy_oas_pct is not None else None
    ig_oas_bps = ig_oas_pct * 100.0 if ig_oas_pct is not None else None
    hy_oas_wow_bps = (
        (hy_oas_pct - hy_oas_prior) * 100.0
        if hy_oas_pct is not None and hy_oas_prior is not None
        else None
    )

    return MacroSnapshot(
        week_ending=we,
        yield_10y=y10,
        yield_10y_wow_change_bps=_wow_bps(y10, y10_prior),
        yield_2y=y2,
        yield_2y_wow_change_bps=_wow_bps(y2, y2_prior),
        real_yield_10y=real10,
        real_yield_10y_wow_change_bps=_wow_bps(real10, real10_prior),
        fed_funds=fedfunds,
        mortgage_30y=mortgage,
        curve_2s10s_bps=(curve_2s10s * 100.0) if curve_2s10s is not None else None,
        curve_inverted=(curve_2s10s is not None and curve_2s10s < 0),
        hy_oas_bps=hy_oas_bps,
        hy_oas_wow_change_bps=hy_oas_wow_bps,
        ig_oas_bps=ig_oas_bps,
        hy_regime=_classify_hy(hy_oas_bps),
        vix_level=vix,
        vix_wow_change=(vix - vix_prior) if (vix is not None and vix_prior is not None) else None,
        vix_regime=_classify_vix(vix),
        breakeven_10y=be10,
        breakeven_10y_wow_change_bps=_wow_bps(be10, be10_prior),
        dxy_level=dxy,
        dxy_wow_change_pct=(
            (dxy / dxy_prior - 1.0) if (dxy is not None and dxy_prior not in (None, 0)) else None
        ),
        recent_prints=_recent_prints(series),
    )


# ---------------------------------------------------------------------------
# Step 2.2 -- Fed communication tracker
# ---------------------------------------------------------------------------

FED_NEWS_QUERY = '"Powell" OR "FOMC" OR "Federal Reserve"'


def fetch_fed_speak_news(*, days: int = 7) -> list[NewsItem]:
    """Pull Fed-related headlines from Google News for the past ``days`` days.

    Trusted-source filter and title dedup are inherited from the same machinery
    used by per-ticker news. The ``ticker`` field on each NewsItem is the
    sentinel "FED" so these don't pollute per-stock SQLite queries.
    """
    from weekly_strategy.data.fetchers import (
        _is_trusted, _normalize_title, _source_name_and_host,
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(FED_NEWS_QUERY)}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(url)
    seen: set[str] = set()
    items: list[NewsItem] = []
    for entry in feed.entries:
        title = entry.get("title") or ""
        norm = _normalize_title(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        source_name, host = _source_name_and_host(entry)
        if not _is_trusted(source_name, host):
            continue
        published = entry.get("published_parsed")
        published_dt = datetime(*published[:6]) if published else None
        items.append(
            NewsItem(
                ticker="FED",
                title=title,
                source=source_name or host or None,
                url=entry.get("link") or "",
                published_at=published_dt,
                snippet=(entry.get("summary") or "")[:500] or None,
            )
        )
    return items


def classify_fed_posture(
    items: list[NewsItem],
    *,
    week_ending: date,
    model: str = client.MODEL_SONNET,
) -> FedPosture:
    """Single Sonnet call. Synthesises hawkish/dovish lean from the week's headlines."""
    if not items:
        return FedPosture(week_ending=week_ending, posture="NEUTRAL", n_items=0)

    formatted = "\n".join(
        f"{i}. [{(it.source or '?'):<18}] {it.title}"
        for i, it in enumerate(items, 1)
    )
    user_prompt = prompts.FED_POSTURE_USER_TEMPLATE.format(
        n=len(items), items=formatted,
    )
    try:
        parsed, _resp = client.ask_json(
            user_prompt, model=model, system_prompt=prompts.FED_POSTURE_SYSTEM,
        )
    except client.ClaudeCliError:
        # Macro layer should never crash a weekly run.
        return FedPosture(week_ending=week_ending, posture="NEUTRAL", n_items=len(items))

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        parsed = {}
    return FedPosture.coerce(week_ending=week_ending, n_items=len(items), raw=parsed)


def get_fed_posture(
    *, week_ending: date | None = None, days: int = 7,
    model: str = client.MODEL_SONNET,
) -> FedPosture:
    """Fetch + classify in one call. Convenience entry point for the orchestrator."""
    we = week_ending or date.today()
    items = fetch_fed_speak_news(days=days)
    return classify_fed_posture(items, week_ending=we, model=model)


def _recent_prints(series_map: dict[str, pd.Series | None]) -> list[dict]:
    """Top-of-mind data prints in the last 7 days: CPI / payrolls / claims / etc."""
    monthly_or_weekly = ("CPIAUCSL", "PCEPI", "PAYEMS", "UNRATE", "INDPRO", "UMCSENT", "ICSA")
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)
    out: list[dict] = []
    for sid in monthly_or_weekly:
        s = series_map.get(sid)
        if s is None or s.empty:
            continue
        s = s.sort_index()
        latest_ts = pd.Timestamp(s.index[-1])
        if latest_ts < cutoff:
            continue
        latest = float(s.iloc[-1])
        prior = float(s.iloc[-2]) if len(s) >= 2 else None
        change = (latest - prior) if prior is not None else None
        out.append({
            "series": sid,
            "label": FRED_SERIES.get(sid, sid),
            "date": latest_ts.date().isoformat(),
            "value": latest,
            "prior": prior,
            "change": change,
        })
    return out
