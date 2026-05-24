from __future__ import annotations

"""Centralized prompt templates.

Kept here, not interleaved with code, so prompt iteration is a
single-file diff and so we can later run prompt-diff tests.
"""

# ---------------------------------------------------------------------------
# Step 1.5 pass 2 -- fundamental implication of a (material) news item
# ---------------------------------------------------------------------------

FUNDAMENTAL_IMPACT_SYSTEM = (
    "You are an equity research analyst. For each news item you read, "
    "identify which fundamental line items it could move and by how much, "
    "from the perspective of the named stock only. You return ONLY a JSON "
    "array. No prose, no markdown fences, no trailing commentary."
)


FUNDAMENTAL_IMPACT_USER_TEMPLATE = """\
Stock: {ticker}{company_suffix}

For each of the {n} material news items below, output a JSON object.

Allowed values:
- areas (multi-select, list of strings): revenue | gross_margin | operating_margin
        | fcf_margin | operating_leverage | working_capital | leverage
        | share_count | tax_rate | other
- direction : POSITIVE | NEGATIVE | UNCLEAR
- magnitude : small (<1% of relevant line item) | medium (1-5%) | large (>5%)
- horizon   : Q | FY25 | FY26 | FY27 | multi-year

Return a JSON array of exactly {n} objects in the SAME order as the items.
Each object is shaped like:
{{"i": <0-based index>, "areas": ["..."], "direction": "...", "magnitude": "...", "horizon": "...", "implication": "<1-2 sentence link between news and line item>"}}

Items:
{items}
"""


def format_fundamental_user_prompt(
    ticker: str,
    items: list[dict],
    *,
    company_name: str | None = None,
) -> str:
    """Same item formatting as the sentiment prompt; different instructions."""
    return _format_items_template(
        FUNDAMENTAL_IMPACT_USER_TEMPLATE, ticker, items, company_name=company_name
    )


def _format_items_template(
    template: str,
    ticker: str,
    items: list[dict],
    *,
    company_name: str | None = None,
) -> str:
    lines: list[str] = []
    for i, it in enumerate(items):
        title = (it.get("title") or "").strip()
        source = (it.get("source") or "?").strip()
        snippet = (it.get("snippet") or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        published = it.get("published_at") or ""
        head = f"{i}. [{source} | {published[:10]}] {title}"
        if snippet:
            head += f"\n     {snippet}"
        lines.append(head)
    company_suffix = f" ({company_name})" if company_name else ""
    return template.format(
        ticker=ticker.upper(),
        company_suffix=company_suffix,
        n=len(items),
        items="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Step 1.5 pass 1 -- news sentiment classification
# ---------------------------------------------------------------------------

NEWS_SENTIMENT_SYSTEM = (
    "You are a sell-side equity analyst classifying news items for a single "
    "stock. You return ONLY a JSON array. No prose, no markdown fences, no "
    "trailing commentary. Each element must have keys: i, sentiment, "
    "materiality, theme, rationale. Sentiment is from THE STOCK's "
    "perspective: a competitor's win is NEGATIVE for our stock."
)


NEWS_SENTIMENT_USER_TEMPLATE = """\
Stock: {ticker}{company_suffix}

Classify each of the {n} items below.

Enums:
- sentiment   : POSITIVE | NEUTRAL | NEGATIVE
- materiality : HIGH (could move the stock 2%+) | MEDIUM (worth noting) | LOW (background noise)
- theme       : earnings | guidance | product | regulation | litigation | M&A | management | macro | competitive | analyst | technical | other

For each item, also include a one-sentence "rationale".

Return a JSON array of exactly {n} objects in the SAME order as the items,
each shaped like:
{{"i": <0-based index>, "sentiment": "...", "materiality": "...", "theme": "...", "rationale": "..."}}

Items:
{items}
"""


def format_news_batch_user_prompt(
    ticker: str,
    items: list[dict],
    *,
    company_name: str | None = None,
) -> str:
    """Format the user prompt for one batch of headlines.

    ``items`` is a list of dicts with keys 'title', 'source', 'snippet',
    'published_at' (ISO string). Snippet is truncated to 240 chars to keep
    each item compact in the prompt without losing the lede sentence.
    """
    return _format_items_template(
        NEWS_SENTIMENT_USER_TEMPLATE, ticker, items, company_name=company_name
    )
