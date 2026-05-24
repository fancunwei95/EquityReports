from __future__ import annotations

"""Pydantic schemas for typed records that flow through SQLite.

Prices stay as ``pandas.DataFrame`` and dossiers/reports stay as ``dict`` —
those don't pass through a row-oriented store, so they don't need a row schema.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    title: str | None = None
    source: str | None = None
    url: str
    published_at: datetime | None = None
    snippet: str | None = None


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
