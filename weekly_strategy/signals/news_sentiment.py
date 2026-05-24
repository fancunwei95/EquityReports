from __future__ import annotations

"""LLM-based news sentiment + fundamental scoring (Step 1.5).

Two-agent cascade:

* **Pass 1 - sentiment** (Haiku, every item). ``classify_batch`` packs
  items into a numbered list; the model returns a JSON array of
  ``NewsClassification`` records.

* **Pass 2 - fundamental impact** (Sonnet, HIGH+MEDIUM items only).
  ``analyze_fundamental_batch`` re-reads the material items and returns a
  ``FundamentalImpact`` per item: which line items move, in what
  direction, by how much, over what horizon. This is the structured input
  the thesis writer (Step 1.8) needs to write *"FY26 revenue tailwind"*
  rather than *"news was positive"*.

* **Pass 3 - sectoral / cross-stock** is deferred to Step 2.5 once we
  have other tickers' dossiers and sector ETF context.

``score_news(ticker, since, until=None)`` is the public entry point. It
loads news from SQLite, runs whichever passes haven't already touched
each item (idempotent), persists per-item judgments, then aggregates.

Aggregation is pure math: materiality-weighted sentiment, top themes,
top items by |contribution|, noise ratio. No LLM in the aggregator --
that's the project's "code does math, LLM does language" rule.
"""

import json
from collections import Counter
from datetime import datetime
from typing import Iterable

from weekly_strategy.data import storage
from weekly_strategy.data.schemas import (
    FundamentalImpact,
    NewsClassification,
    NewsItem,
    NewsScore,
    ThemeCount,
    TopItem,
)
from weekly_strategy.llm import client, prompts


DEFAULT_BATCH_SIZE = 50  # 50 items / call keeps the prompt under ~10K tokens
FUNDAMENTAL_BATCH_SIZE = 20  # fundamental analysis prompt is denser; smaller chunks
TOP_THEME_K = 3
TOP_ITEM_K = 5
_MATERIAL_LEVELS = frozenset({"HIGH", "MEDIUM"})


# ---------------------------------------------------------------------------
# Classifier (LLM-facing)
# ---------------------------------------------------------------------------


def classify_batch(
    ticker: str,
    items: list[NewsItem],
    *,
    model: str = client.MODEL_HAIKU,
    company_name: str | None = None,
) -> list[NewsClassification]:
    """Classify ``items`` in a single LLM call. Returns aligned list."""
    if not items:
        return []
    user_prompt = prompts.format_news_batch_user_prompt(
        ticker,
        [_item_payload(it) for it in items],
        company_name=company_name,
    )
    parsed, _resp = client.ask_json(
        user_prompt,
        model=model,
        system_prompt=prompts.NEWS_SENTIMENT_SYSTEM,
    )
    return _align_classifications(parsed, expected=len(items))


def _item_payload(item: NewsItem) -> dict:
    return {
        "title": item.title,
        "source": item.source,
        "snippet": item.snippet,
        "published_at": item.published_at.isoformat() if item.published_at else None,
    }


def _align_classifications(parsed, expected: int) -> list[NewsClassification]:
    """Coerce LLM output into ``expected`` classifications, in order.

    Robust to: model returning fewer/more entries than asked; model
    returning a dict with an ``items`` key; ``i`` field out of order.
    Missing slots fill with NEUTRAL/LOW/other.
    """
    if isinstance(parsed, dict):
        # Some models wrap the array.
        for key in ("items", "classifications", "data", "results"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        # Bail with all-neutral so the pipeline keeps moving.
        return [_default_classification() for _ in range(expected)]

    # Map by 'i' if present; otherwise positional.
    by_index: dict[int, dict] = {}
    for pos, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            continue
        i = entry.get("i")
        if isinstance(i, int) and 0 <= i < expected:
            by_index[i] = entry
        elif pos < expected:
            by_index.setdefault(pos, entry)

    return [
        NewsClassification.coerce(by_index[i]) if i in by_index else _default_classification()
        for i in range(expected)
    ]


def _default_classification() -> NewsClassification:
    return NewsClassification(
        sentiment="NEUTRAL", materiality="LOW", theme="other",
        rationale="classifier returned no entry for this item",
    )


# ---------------------------------------------------------------------------
# Pass 2 -- fundamental impact (Sonnet)
# ---------------------------------------------------------------------------


def analyze_fundamental_batch(
    ticker: str,
    items: list[NewsItem],
    *,
    model: str = client.MODEL_SONNET,
    company_name: str | None = None,
) -> list[FundamentalImpact]:
    """Same alignment machinery as ``classify_batch``, different prompt + schema."""
    if not items:
        return []
    user_prompt = prompts.format_fundamental_user_prompt(
        ticker,
        [_item_payload(it) for it in items],
        company_name=company_name,
    )
    parsed, _resp = client.ask_json(
        user_prompt,
        model=model,
        system_prompt=prompts.FUNDAMENTAL_IMPACT_SYSTEM,
    )
    return _align_fundamentals(parsed, expected=len(items))


def _align_fundamentals(parsed, expected: int) -> list[FundamentalImpact]:
    if isinstance(parsed, dict):
        for key in ("items", "impacts", "data", "results"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        return [_default_fundamental() for _ in range(expected)]

    by_index: dict[int, dict] = {}
    for pos, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            continue
        i = entry.get("i")
        if isinstance(i, int) and 0 <= i < expected:
            by_index[i] = entry
        elif pos < expected:
            by_index.setdefault(pos, entry)

    return [
        FundamentalImpact.coerce(by_index[i]) if i in by_index else _default_fundamental()
        for i in range(expected)
    ]


def _default_fundamental() -> FundamentalImpact:
    return FundamentalImpact(
        areas=["other"], direction="UNCLEAR", magnitude="small", horizon="Q",
        implication="analyzer returned no entry for this item",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def score_news(
    ticker: str,
    since: datetime,
    until: datetime | None = None,
    *,
    sentiment_model: str = client.MODEL_HAIKU,
    fundamental_model: str = client.MODEL_SONNET,
    company_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fundamental_batch_size: int = FUNDAMENTAL_BATCH_SIZE,
    skip_fundamental: bool = False,
) -> NewsScore:
    """Cascade: sentiment (all) -> fundamental (HIGH+MED only). Idempotent."""
    until = until or datetime.utcnow()
    items = storage.get_news(ticker, since=since, until=until)
    if not items:
        return _empty_score(ticker, since, until)

    # Pass 1: sentiment for any unclassified items.
    unclassified = [it for it in items if it.classification is None]
    if unclassified:
        for chunk in _chunked(unclassified, batch_size):
            classifications = classify_batch(
                ticker, chunk, model=sentiment_model, company_name=company_name
            )
            for item, cls in zip(chunk, classifications):
                storage.update_news_classification(item.url, cls)
        items = storage.get_news(ticker, since=since, until=until)

    # Pass 2: fundamental analysis on classified-as-material items only.
    if not skip_fundamental:
        needs_fundamental = [
            it for it in items
            if it.classification is not None
            and it.classification.materiality in _MATERIAL_LEVELS
            and it.fundamental_impact is None
        ]
        if needs_fundamental:
            for chunk in _chunked(needs_fundamental, fundamental_batch_size):
                impacts = analyze_fundamental_batch(
                    ticker, chunk, model=fundamental_model, company_name=company_name
                )
                for item, impact in zip(chunk, impacts):
                    storage.update_news_fundamental(item.url, impact)
            items = storage.get_news(ticker, since=since, until=until)

    return aggregate(ticker, items, since, until)


def _chunked(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _empty_score(ticker: str, since: datetime, until: datetime) -> NewsScore:
    return NewsScore(
        ticker=ticker.upper(),
        window_start=since,
        window_end=until,
        n_total=0,
        n_classified=0,
        sentiment_score=0.0,
        top_themes=[],
        top_items=[],
        noise_ratio=0.0,
    )


# ---------------------------------------------------------------------------
# Aggregator (pure)
# ---------------------------------------------------------------------------


def aggregate(
    ticker: str,
    items: Iterable[NewsItem],
    window_start: datetime,
    window_end: datetime,
) -> NewsScore:
    items = list(items)
    classified = [it for it in items if it.classification is not None]
    n_total = len(items)
    n_classified = len(classified)

    if not classified:
        return _empty_score(ticker, window_start, window_end)

    # Materiality-weighted sentiment.
    weight_sum = sum(it.classification.materiality_weight for it in classified)
    if weight_sum == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = sum(
            it.classification.sentiment_value * it.classification.materiality_weight
            for it in classified
        ) / weight_sum

    # Top themes by HIGH+MEDIUM item count.
    theme_counter: Counter = Counter()
    for it in classified:
        if it.classification.materiality in ("HIGH", "MEDIUM"):
            theme_counter[it.classification.theme] += 1
    top_themes = [
        ThemeCount(theme=th, count=c)
        for th, c in theme_counter.most_common(TOP_THEME_K)
    ]

    # Top items by |signed contribution|.
    scored: list[tuple[float, NewsItem]] = [
        (
            it.classification.sentiment_value * it.classification.materiality_weight,
            it,
        )
        for it in classified
    ]
    scored.sort(key=lambda pair: abs(pair[0]), reverse=True)
    top_items = [
        TopItem(
            url=it.url,
            title=it.title,
            source=it.source,
            sentiment=it.classification.sentiment,
            materiality=it.classification.materiality,
            contribution=contrib,
        )
        for contrib, it in scored[:TOP_ITEM_K]
    ]

    n_low = sum(1 for it in classified if it.classification.materiality == "LOW")
    noise_ratio = n_low / n_classified if n_classified else 0.0

    return NewsScore(
        ticker=ticker.upper(),
        window_start=window_start,
        window_end=window_end,
        n_total=n_total,
        n_classified=n_classified,
        sentiment_score=sentiment_score,
        top_themes=top_themes,
        top_items=top_items,
        noise_ratio=noise_ratio,
    )
