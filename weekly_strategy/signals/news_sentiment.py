from __future__ import annotations

"""LLM-based news sentiment scoring (Step 1.5).

Flow:

* ``classify_batch(ticker, items)`` -- ONE ``claude -p`` call. Packs all
  items into a numbered list; the model returns a JSON array. We validate
  length, coerce each entry through ``NewsClassification.coerce``, and
  return a list aligned with ``items``.

* ``score_news(ticker, since, until=None)`` -- the public entry point.
  Loads news from SQLite, classifies any items that don't already have a
  ``classification`` (idempotent re-runs), persists classifications back
  via ``storage.update_news_classification``, then aggregates.

Aggregation is pure math: materiality-weighted sentiment, top themes by
HIGH+MEDIUM count, top items by |contribution|, noise ratio. No LLM
involved in the aggregator -- that's the project's "code does math, LLM
does language" rule from plan.md.
"""

import json
from collections import Counter
from datetime import datetime
from typing import Iterable

from weekly_strategy.data import storage
from weekly_strategy.data.schemas import (
    NewsClassification,
    NewsItem,
    NewsScore,
    ThemeCount,
    TopItem,
)
from weekly_strategy.llm import client, prompts


DEFAULT_BATCH_SIZE = 50  # 50 items / call keeps the prompt under ~10K tokens
TOP_THEME_K = 3
TOP_ITEM_K = 5


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
# Orchestrator
# ---------------------------------------------------------------------------


def score_news(
    ticker: str,
    since: datetime,
    until: datetime | None = None,
    *,
    model: str = client.MODEL_HAIKU,
    company_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> NewsScore:
    """Classify all unclassified news in the window, persist, aggregate."""
    until = until or datetime.utcnow()
    items = storage.get_news(ticker, since=since, until=until)
    if not items:
        return _empty_score(ticker, since, until)

    unclassified = [it for it in items if it.classification is None]
    if unclassified:
        for chunk in _chunked(unclassified, batch_size):
            classifications = classify_batch(
                ticker, chunk, model=model, company_name=company_name
            )
            for item, cls in zip(chunk, classifications):
                storage.update_news_classification(item.url, cls)
        # Reload so the aggregator sees the freshly-attached classifications.
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
