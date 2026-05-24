from __future__ import annotations

"""Tests for signals/reddit_sentiment.py. LLM is mocked."""

import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def reddit_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import weekly_strategy.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings_mod, "SQLITE_PATH", tmp_path / "cache" / "weekly.db")
    settings_mod.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import weekly_strategy.signals.reddit_sentiment as mod
    mod = importlib.reload(mod)
    return mod


@dataclass
class _FakeResp:
    text: str = ""
    cost_usd: float = 0.0
    session_id: str = "sess"
    raw: dict = None  # type: ignore[assignment]


def _stub_llm(monkeypatch, mod, payload):
    captured = {"calls": 0}

    def fake_ask_json(prompt, model, system_prompt=None, timeout_s=120.0):
        captured["calls"] += 1
        captured["prompt"] = prompt
        return payload, _FakeResp()

    monkeypatch.setattr(mod.client, "ask_json", fake_ask_json)
    return captured


def _post(ticker="AAPL", post_id="p1", score=100, n=20, sub="stocks", title="x"):
    from weekly_strategy.data.schemas import RedditPost
    return RedditPost(
        ticker=ticker, subreddit=sub, post_id=post_id, title=title,
        score=score, num_comments=n,
        created_at=datetime.utcnow() - timedelta(hours=2),
        url="https://reddit.com/x",
    )


def _snapshot(n=10, prior=5, top_posts=None):
    from weekly_strategy.data.schemas import RedditSnapshot
    return RedditSnapshot(
        ticker="AAPL",
        window_hours=24,
        mention_count=n,
        mention_count_prior_window=prior,
        mention_delta_pct=(n - prior) / max(prior, 1),
        top_posts=top_posts or [],
    )


def test_skips_llm_when_too_few_mentions(reddit_mod, monkeypatch):
    captured = _stub_llm(monkeypatch, reddit_mod, payload={
        "sentiment": 0.8, "themes": ["should not run"],
    })
    snap = _snapshot(n=3, prior=1, top_posts=[_post()])
    score = reddit_mod.score_reddit_snapshot(snap)
    assert captured["calls"] == 0
    assert score.llm_sentiment is None
    assert score.llm_themes == []
    assert score.is_crowded is False  # delta=2.0 but absolute mentions still small


def test_invokes_llm_above_threshold_and_parses(reddit_mod, monkeypatch):
    posts = [_post(post_id=f"p{i}", score=100 - i*10, title=f"AAPL post {i}") for i in range(6)]
    snap = _snapshot(n=20, prior=10, top_posts=posts)
    captured = _stub_llm(monkeypatch, reddit_mod, payload={
        "sentiment": 0.62, "themes": ["AI hype", "earnings beat", "extra ignored", "x"],
    })
    score = reddit_mod.score_reddit_snapshot(snap)
    assert captured["calls"] == 1
    assert score.llm_sentiment == pytest.approx(0.62)
    assert score.llm_themes == ["AI hype", "earnings beat", "extra ignored"]
    # Top-5 cap: prompt includes 5 of the 6 posts.
    assert "AAPL post 4" in captured["prompt"]
    assert "AAPL post 5" not in captured["prompt"]


def test_clamps_out_of_range_sentiment(reddit_mod, monkeypatch):
    _stub_llm(monkeypatch, reddit_mod, payload={"sentiment": 1.7, "themes": []})
    snap = _snapshot(n=10, top_posts=[_post()])
    score = reddit_mod.score_reddit_snapshot(snap)
    assert score.llm_sentiment == pytest.approx(1.0)


def test_crowding_flag_on_high_delta(reddit_mod, monkeypatch):
    _stub_llm(monkeypatch, reddit_mod, payload={"sentiment": 0.0, "themes": []})
    snap = _snapshot(n=15, prior=5, top_posts=[_post()])  # delta = 2.0
    score = reddit_mod.score_reddit_snapshot(snap)
    assert score.is_crowded is True


def test_crowding_flag_on_high_absolute_count(reddit_mod, monkeypatch):
    _stub_llm(monkeypatch, reddit_mod, payload={"sentiment": 0.1, "themes": []})
    snap = _snapshot(n=60, prior=55, top_posts=[_post()])
    score = reddit_mod.score_reddit_snapshot(snap)
    assert score.is_crowded is True


def test_llm_failure_degrades_to_numeric_only(reddit_mod, monkeypatch):
    def boom(*_a, **_kw):
        raise reddit_mod.client.ClaudeCliError("simulated outage")
    monkeypatch.setattr(reddit_mod.client, "ask_json", boom)
    snap = _snapshot(n=10, top_posts=[_post()])
    score = reddit_mod.score_reddit_snapshot(snap)
    assert score.llm_sentiment is None
    assert score.llm_themes == []
    assert score.mention_count == 10
