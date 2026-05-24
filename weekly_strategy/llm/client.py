from __future__ import annotations

"""Thin wrapper around `claude` in headless mode.

We do not hold an Anthropic API key. Every LLM call shells out to the
``claude`` CLI in non-interactive mode against the user's Pro/Max
subscription.

Subtleties learned the hard way
-------------------------------
* **Do NOT use ``--bare``.** Its help text says it skips CLAUDE.md /
  auto-memory / hooks, which sounds appealing -- but it *also* says
  "Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper; OAuth and
  keychain are never read." With no API key, ``--bare`` cannot authenticate
  at all.
* **Always pass ``--system-prompt``.** Without it, the default system prompt
  loads CLAUDE.md, the auto-memory MEMORY.md tree, env info, and Claude
  Code's own scaffolding -- 25K+ tokens per call. With a custom system
  prompt the per-machine sections are skipped and a "ping" call drops from
  ~$0.010 to ~$0.0015.
* **Disable tools and slash commands.** ``--tools ""`` and
  ``--disable-slash-commands`` keep the model in single-turn prompt-in /
  text-out mode rather than entering an agent loop.

Auth invariant
--------------
``claude`` resolves auth in this order: ANTHROPIC_API_KEY >
ANTHROPIC_AUTH_TOKEN > CLAUDE_CODE_OAUTH_TOKEN > stored subscription OAuth.
We *defensively* strip the env-var paths so the subscription wins even if
the user accidentally exports a key.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from weekly_strategy.config import settings


# Model IDs that the project will actually use. Resolving names here gives us a
# single place to bump them when a new family ships.
MODEL_HAIKU = settings.MODEL_HAIKU
MODEL_SONNET = settings.MODEL_SONNET
MODEL_OPUS = settings.MODEL_OPUS


class ClaudeCliError(RuntimeError):
    """Raised when `claude -p` exits non-zero or returns a malformed envelope."""


@dataclass(frozen=True)
class ClaudeResponse:
    text: str               # the model's free-form text (the "result" field)
    cost_usd: float         # total_cost_usd reported by claude
    session_id: str | None  # claude assigns one per call in bare mode
    raw: dict               # full envelope, in case callers want more fields

    def as_json(self) -> Any:
        """Parse self.text as JSON. Tolerates ```json fences and trailing prose."""
        return _loads_with_fence_strip(self.text)


def _child_env() -> dict[str, str]:
    """Subprocess env with key-based auth stripped so subscription OAuth wins."""
    env = os.environ.copy()
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env


def _loads_with_fence_strip(s: str) -> Any:
    """Best-effort JSON parse: strip ```json fences and leading/trailing prose."""
    t = s.strip()
    if t.startswith("```"):
        # Drop the opening fence line (e.g., ```json) and a closing fence.
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return json.loads(t)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a focused, single-turn assistant. Follow the user's instructions "
    "exactly. If the user asks for JSON, return only valid JSON with no "
    "markdown fences and no commentary."
)


def ask(
    prompt: str,
    model: str = MODEL_HAIKU,
    *,
    system_prompt: str | None = None,
    timeout_s: float = 120.0,
) -> ClaudeResponse:
    """Single-shot ``claude -p`` call. Returns the model's text + cost.

    We pass a custom ``--system-prompt`` to bypass the default Claude Code
    scaffolding (CLAUDE.md, auto-memory, env info) and disable tools / slash
    commands so the call stays a clean prompt-in / text-out. Callers should
    pass a task-specific ``system_prompt`` when the task warrants one.
    """
    cmd = [
        settings.CLAUDE_BIN,
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system_prompt or _DEFAULT_SYSTEM_PROMPT,
        "--tools", "",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_child_env(),
            check=False,
        )
    except FileNotFoundError as e:
        raise ClaudeCliError(
            f"`{settings.CLAUDE_BIN}` not found on PATH. Install Claude Code "
            "or set CLAUDE_BIN in .env."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ClaudeCliError(f"claude call timed out after {timeout_s}s") from e

    if proc.returncode != 0:
        raise ClaudeCliError(
            f"claude exited {proc.returncode}.\nstderr:\n{proc.stderr.strip()}"
        )

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCliError(
            f"claude returned non-JSON on stdout:\n{proc.stdout[:500]!r}"
        ) from e

    text = envelope.get("result")
    if not isinstance(text, str):
        raise ClaudeCliError(
            f"claude envelope missing string 'result' field: {envelope!r}"
        )

    return ClaudeResponse(
        text=text,
        cost_usd=float(envelope.get("total_cost_usd") or 0.0),
        session_id=envelope.get("session_id"),
        raw=envelope,
    )


def ask_json(
    prompt: str,
    model: str = MODEL_HAIKU,
    *,
    system_prompt: str | None = None,
    timeout_s: float = 120.0,
) -> tuple[Any, ClaudeResponse]:
    """Ask the model and parse its text response as JSON.

    The prompt should explicitly instruct the model to return JSON only; this
    wrapper tolerates ```json fences but is not a JSON-mode hard constraint.
    Returns (parsed_value, full_response) so callers can still inspect cost.
    """
    resp = ask(prompt, model=model, system_prompt=system_prompt, timeout_s=timeout_s)
    return resp.as_json(), resp
