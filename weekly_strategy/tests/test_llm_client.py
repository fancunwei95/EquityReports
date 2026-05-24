from __future__ import annotations

"""Tests for the headless-Claude subprocess wrapper.

Every call is mocked via subprocess.run -- no real claude invocation here.
A separate real smoke test lives outside the pytest suite.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from weekly_strategy.llm import client


def _fake_envelope(result: str, cost: float = 0.0042, sid: str = "sess-1") -> str:
    return json.dumps(
        {"result": result, "session_id": sid, "total_cost_usd": cost, "duration_ms": 1234}
    )


def test_ask_returns_text_cost_and_session(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(
            returncode=0, stdout=_fake_envelope("hello world", cost=0.01), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    resp = client.ask("say hi", model="claude-haiku-4-5-20251001")
    assert resp.text == "hello world"
    assert resp.cost_usd == pytest.approx(0.01)
    assert resp.session_id == "sess-1"
    # The flags we depend on are present (and `--bare` is NOT, because it
    # disables OAuth auth and we don't have an API key).
    cmd = captured["cmd"]
    assert "--bare" not in cmd
    assert "-p" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--model" in cmd and "claude-haiku-4-5-20251001" in cmd
    assert "--system-prompt" in cmd
    assert "--disable-slash-commands" in cmd
    assert "--no-session-persistence" in cmd
    assert "--tools" in cmd


def test_ask_strips_api_key_from_child_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-ignored")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-also-ignored")
    captured: dict = {}

    def fake_run(cmd, capture_output, text, timeout, env, check):  # noqa: ARG001
        captured["env"] = env
        return SimpleNamespace(returncode=0, stdout=_fake_envelope("ok"), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client.ask("ping")
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]


def test_ask_json_parses_fenced_response(monkeypatch):
    payload = '```json\n{"sentiment": 0.7, "theme": "earnings"}\n```'

    def fake_run(*_a, **_kw):
        return SimpleNamespace(returncode=0, stdout=_fake_envelope(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    parsed, resp = client.ask_json("classify this")
    assert parsed == {"sentiment": 0.7, "theme": "earnings"}
    assert resp.text == payload  # raw text preserved


def test_ask_json_parses_bare_json(monkeypatch):
    payload = '{"a": 1, "b": [2, 3]}'

    def fake_run(*_a, **_kw):
        return SimpleNamespace(returncode=0, stdout=_fake_envelope(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    parsed, _ = client.ask_json("ping")
    assert parsed == {"a": 1, "b": [2, 3]}


def test_ask_raises_on_nonzero_exit(monkeypatch):
    def fake_run(*_a, **_kw):
        return SimpleNamespace(returncode=2, stdout="", stderr="auth failure: no credential")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(client.ClaudeCliError, match="auth failure"):
        client.ask("ping")


def test_ask_raises_on_malformed_envelope(monkeypatch):
    def fake_run(*_a, **_kw):
        return SimpleNamespace(returncode=0, stdout="not json at all", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(client.ClaudeCliError, match="non-JSON"):
        client.ask("ping")


def test_ask_raises_on_missing_result_field(monkeypatch):
    def fake_run(*_a, **_kw):
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"session_id": "x"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(client.ClaudeCliError, match="missing string 'result'"):
        client.ask("ping")


def test_ask_raises_when_binary_missing(monkeypatch):
    def fake_run(*_a, **_kw):
        raise FileNotFoundError("no claude")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(client.ClaudeCliError, match="not found on PATH"):
        client.ask("ping")


def test_ask_raises_on_timeout(monkeypatch):
    def fake_run(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(client.ClaudeCliError, match="timed out"):
        client.ask("ping", timeout_s=1.0)
