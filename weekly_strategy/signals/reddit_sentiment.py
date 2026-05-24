from __future__ import annotations

"""Reddit sentiment scoring (Step 1.6).

Lighter-touch than the news pipeline. Reddit is primarily a *crowding*
signal -- a sudden spike in mentions usually means a trade is getting
crowded, not that any particular fundamental has changed. So we keep the
LLM out of the per-post loop and only invoke it once per ticker per week
on the top few posts (and only if mention count exceeds a threshold).

Math vs LLM split, consistent with the project rule:
* Math: mention count, mention delta, crowding flag.
* LLM (Haiku, one call): an overall sentiment read on the top 3-5 posts
  and 1-3 themes driving them.
"""

import json
from datetime import datetime

from weekly_strategy.data.schemas import RedditPost, RedditScore, RedditSnapshot
from weekly_strategy.llm import client


# Only invoke the LLM when there's enough chatter to bother. Below this,
# Reddit is just noise and we return llm_sentiment=None.
MIN_MENTIONS_FOR_LLM = 5
TOP_POSTS_FOR_LLM = 5
# Crowding requires either an unusual absolute level, or a spike off a non-trivial
# baseline. A 1->3 mention pickup doubles the prior window but isn't "crowded".
CROWDING_DELTA_THRESHOLD = 1.0
CROWDING_ABS_THRESHOLD = 50
CROWDING_DELTA_MIN_COUNT = 10


REDDIT_SENTIMENT_SYSTEM = (
    "You read a short list of Reddit posts about a single stock and return ONE "
    "JSON object summarizing crowd sentiment. No prose, no markdown fences."
)


REDDIT_SENTIMENT_USER_TEMPLATE = """\
Stock: {ticker}

The {n} highest-engagement Reddit posts about {ticker} in the last 24h:
{posts}

Return a single JSON object:
{{"sentiment": float in [-1, +1], "themes": ["short tag", ...]}}

Sentiment is the *crowd's* view of the stock. Themes are 1-3 short tags
describing what the chatter is about (e.g., "earnings beat", "AI hype",
"short squeeze").
"""


def _format_posts(posts: list[RedditPost]) -> str:
    lines: list[str] = []
    for i, p in enumerate(posts, 1):
        title = (p.title or "").strip()
        lines.append(
            f"{i}. r/{p.subreddit or '?'} ({p.score} upvotes, "
            f"{p.num_comments} comments) -- {title}"
        )
    return "\n".join(lines)


def score_reddit_snapshot(
    snapshot: RedditSnapshot,
    *,
    model: str = client.MODEL_HAIKU,
) -> RedditScore:
    """Turn a RedditSnapshot into a RedditScore. One LLM call if material."""
    n = snapshot.mention_count
    is_crowded = (
        n >= CROWDING_ABS_THRESHOLD
        or (
            snapshot.mention_delta_pct >= CROWDING_DELTA_THRESHOLD
            and n >= CROWDING_DELTA_MIN_COUNT
        )
    )

    llm_sentiment: float | None = None
    llm_themes: list[str] = []

    if n >= MIN_MENTIONS_FOR_LLM and snapshot.top_posts:
        top = snapshot.top_posts[:TOP_POSTS_FOR_LLM]
        prompt = REDDIT_SENTIMENT_USER_TEMPLATE.format(
            ticker=snapshot.ticker.upper(),
            n=len(top),
            posts=_format_posts(top),
        )
        try:
            parsed, _resp = client.ask_json(
                prompt, model=model, system_prompt=REDDIT_SENTIMENT_SYSTEM
            )
            llm_sentiment, llm_themes = _parse_reddit_payload(parsed)
        except (client.ClaudeCliError, ValueError, json.JSONDecodeError):
            # Reddit signal is nice-to-have; degrade silently to numeric-only.
            llm_sentiment, llm_themes = None, []

    return RedditScore(
        ticker=snapshot.ticker.upper(),
        window_hours=snapshot.window_hours,
        mention_count=n,
        mention_count_prior_window=snapshot.mention_count_prior_window,
        mention_delta_pct=snapshot.mention_delta_pct,
        llm_sentiment=llm_sentiment,
        llm_themes=llm_themes,
        is_crowded=is_crowded,
    )


def _parse_reddit_payload(parsed) -> tuple[float | None, list[str]]:
    # Tolerate dict wrappers and case-mangling.
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return None, []
    raw_sent = parsed.get("sentiment")
    sentiment: float | None
    try:
        sentiment = float(raw_sent) if raw_sent is not None else None
    except (TypeError, ValueError):
        sentiment = None
    if sentiment is not None:
        # Clamp to [-1, +1] -- LLMs sometimes return 1.5 or -2.0.
        sentiment = max(-1.0, min(1.0, sentiment))
    themes_raw = parsed.get("themes") or []
    if isinstance(themes_raw, str):
        themes_raw = [themes_raw]
    themes = [str(t)[:60] for t in themes_raw if t][:3]
    return sentiment, themes
