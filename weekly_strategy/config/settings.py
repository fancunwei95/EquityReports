from __future__ import annotations

"""Project settings. Loads API keys from .env and exposes paths/constants.

API keys are read from the environment; never hardcode. See .env.example at repo root.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")

# --- Paths ---
# Writable data root. Set WEEKLY_STRATEGY_DATA_ROOT in .env (or the shell)
# to redirect ALL caches + reports off the project tree -- useful when the
# project lives on a flaky shared mount and you want writes to land on a
# stable local disk. Default keeps the original in-tree layout for tests.
DATA_ROOT_ENV = os.getenv("WEEKLY_STRATEGY_DATA_ROOT")
if DATA_ROOT_ENV:
    _DATA_ROOT = Path(DATA_ROOT_ENV).expanduser().resolve()
    DATA_CACHE_DIR = _DATA_ROOT / "data_cache"
    REPORTS_DIR = _DATA_ROOT / "reports"
    DAILY_REPORTS_DIR = _DATA_ROOT / "daily_reports"
    DOSSIER_DIR = _DATA_ROOT / "dossier_data"
else:
    DATA_CACHE_DIR = PACKAGE_ROOT / "data_cache"
    REPORTS_DIR = PACKAGE_ROOT / "reports"           # per-ticker single-stock notes
    # Daily portfolio HTML/MD lands under docs/ so GitHub Pages picks it up.
    DAILY_REPORTS_DIR = REPO_ROOT / "docs" / "portfolio"
    DOSSIER_DIR = PACKAGE_ROOT / "dossier" / "data"

# Public site root (GitHub Pages serves this). Holds index.html + per-day reports.
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

TICKERS_FILE = PACKAGE_ROOT / "config" / "tickers.json"
SQLITE_PATH = DATA_CACHE_DIR / "weekly.db"

for _d in (DATA_CACHE_DIR, REPORTS_DIR, DAILY_REPORTS_DIR, DOSSIER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API keys (all free tier) ---
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")

# LLM access: we use `claude` CLI in headless mode against the user's
# subscription, not the Anthropic API. CLAUDE_BIN overrides the binary path;
# default works as long as `claude` is on PATH.
CLAUDE_BIN: str = os.getenv("CLAUDE_BIN", "claude")

# SEC EDGAR requires a contact email in the User-Agent (no key, just identification).
SEC_USER_AGENT: str = os.getenv(
    "SEC_USER_AGENT", "weekly-strategy cunwei.fan@tophash.io"
)

# Reddit uses public JSON endpoints; no key. Just a polite UA.
REDDIT_USER_AGENT: str = os.getenv(
    "REDDIT_USER_AGENT", "weekly-strategy/0.1 by cunwei.fan@tophash.io"
)

# --- LLM model defaults (per project memory: always use latest released family) ---
MODEL_HAIKU = "claude-haiku-4-5-20251001"  # batch classification
MODEL_SONNET = "claude-sonnet-4-6"          # thesis + conviction
MODEL_OPUS = "claude-opus-4-7"              # monthly meta-review only
