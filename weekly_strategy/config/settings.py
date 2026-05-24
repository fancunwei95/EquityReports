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
DATA_CACHE_DIR = PACKAGE_ROOT / "data_cache"
REPORTS_DIR = PACKAGE_ROOT / "reports"
DOSSIER_DIR = PACKAGE_ROOT / "dossier" / "data"
TICKERS_FILE = PACKAGE_ROOT / "config" / "tickers.json"
SQLITE_PATH = DATA_CACHE_DIR / "weekly.db"

for _d in (DATA_CACHE_DIR, REPORTS_DIR, DOSSIER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API keys (all free tier) ---
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")

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
