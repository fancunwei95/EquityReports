from __future__ import annotations

"""Stage 1 orchestrator.

For each ticker in ``config/tickers.json[stage1]`` this script walks the
full per-stock pipeline:

    prices -> news fetch+store -> dossier -> news cascade ->
    reddit fetch+score -> score bundle -> thesis -> JSON report
    -> Markdown report

Per-ticker errors are caught and surfaced in the final summary so one
failure doesn't kill the run. The script is deliberately a flat
top-level sequence -- no class hierarchy -- because that's easier to
read when something goes wrong on a Sunday evening.
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from weekly_strategy.config import settings
from weekly_strategy.data import fetchers, storage
from weekly_strategy.data.schemas import Dossier
from weekly_strategy.dossier import builder as dossier_builder
from weekly_strategy.llm.thesis import write_thesis
from weekly_strategy.reporting import single_stock
from weekly_strategy.signals import fundamental as fnd
from weekly_strategy.signals import news_sentiment, reddit_sentiment


# ---------------------------------------------------------------------------
# Per-ticker result for the final summary
# ---------------------------------------------------------------------------


@dataclass
class TickerResult:
    ticker: str
    status: str = "ok"            # "ok" or "failed"
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    quality: float | None = None
    valuation: float | None = None
    momentum: float | None = None
    news_n: int = 0
    news_sentiment: float | None = None
    reddit_mentions: int = 0
    report_path: Path | None = None
    error: str | None = None
    error_step: str | None = None


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------


def run_one(
    ticker: str,
    *,
    week_ending: date,
    news_days: int = 7,
    price_lookback_days: int = 365,
    reddit_hours: int = 24,
    skip_reddit: bool = False,
    log=print,
) -> TickerResult:
    """Full single-stock pipeline. Returns a TickerResult either way."""
    res = TickerResult(ticker=ticker.upper())
    t_start = time.time()
    current_step = "init"

    try:
        # --- 1. Prices ---
        current_step = "prices"
        log(f"[{ticker}] fetching prices ({price_lookback_days}d)...")
        fetchers.get_price_history(ticker, lookback_days=price_lookback_days)

        # --- 2. Basic info (sector/industry/name) ---
        current_step = "basic_info"
        info = fetchers.get_basic_info(ticker)
        company_name = info.get("name")

        # --- 3. News fetch + store ---
        current_step = "news_fetch"
        log(f"[{ticker}] fetching news ({news_days}d)...")
        news_items = fetchers.fetch_news(ticker, days=news_days)
        if news_items:
            storage.insert_news(news_items)

        # --- 4. Dossier ---
        current_step = "dossier"
        log(f"[{ticker}] building dossier from EDGAR...")
        dossier = dossier_builder.build_dossier(ticker)

        # --- 5. News cascade (sentiment + fundamental) ---
        current_step = "news_score"
        since = datetime.combine(week_ending, datetime.min.time()) - timedelta(days=news_days)
        log(f"[{ticker}] classifying + analyzing news (cascade)...")
        news_score = news_sentiment.score_news(
            ticker, since=since, company_name=company_name,
        )
        res.news_n = news_score.n_total
        res.news_sentiment = news_score.sentiment_score
        items_for_thesis = storage.get_news(ticker, since=since)

        # --- 6. Reddit (optional) ---
        reddit_score = None
        if not skip_reddit:
            current_step = "reddit"
            log(f"[{ticker}] fetching reddit ({reddit_hours}h)...")
            posts = fetchers.fetch_reddit_recent(ticker, hours=reddit_hours)
            if posts:
                storage.insert_reddit(posts)
            snap = fetchers.build_reddit_snapshot(
                ticker, current_posts=posts, window_hours=reddit_hours,
            )
            reddit_score = reddit_sentiment.score_reddit_snapshot(snap)
            res.reddit_mentions = reddit_score.mention_count

        # --- 7. Score bundle ---
        current_step = "score_bundle"
        prices = storage.load_prices(ticker)
        bundle = fnd.build_score_bundle(
            ticker=ticker, week_ending=week_ending,
            dossier=dossier, prices=prices,
            news_sentiment_score=news_score.sentiment_score,
            news_noise_ratio=news_score.noise_ratio,
            reddit_sentiment=reddit_score.llm_sentiment if reddit_score else None,
            reddit_is_crowded=reddit_score.is_crowded if reddit_score else False,
        )
        res.quality = bundle.quality_score
        res.valuation = bundle.valuation_score
        res.momentum = bundle.momentum_score

        # --- 8. Thesis (Sonnet) ---
        current_step = "thesis"
        log(f"[{ticker}] writing thesis (Sonnet)...")
        thesis = write_thesis(
            ticker=ticker, week_ending=week_ending,
            dossier=dossier, scores=bundle,
            news_score=news_score, news_items=items_for_thesis,
            reddit_score=reddit_score, prices=prices,
            company_name=company_name,
        )
        res.cost_usd = thesis.cost_usd or 0.0

        # --- 9. Persist JSON + Markdown ---
        current_step = "save"
        storage.save_weekly_report(
            ticker, week_ending, thesis.model_dump(mode="json"),
        )
        md = single_stock.render_markdown(
            thesis,
            dossier=dossier, scores=bundle,
            news_score=news_score, news_items=items_for_thesis,
            reddit_score=reddit_score,
        )
        res.report_path = single_stock.save_markdown(thesis, md)

        res.elapsed_s = time.time() - t_start
        log(f"[{ticker}] done in {res.elapsed_s:.1f}s -> {res.report_path}")
        return res

    except Exception as e:  # broad on purpose -- per-ticker isolation
        res.status = "failed"
        res.error = f"{type(e).__name__}: {e}"
        res.error_step = current_step
        res.elapsed_s = time.time() - t_start
        log(f"[{ticker}] FAILED at step {current_step}: {res.error}")
        traceback.print_exc(file=sys.stderr)
        return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_tickers(path: Path | None = None) -> list[str]:
    p = path or settings.TICKERS_FILE
    raw = json.loads(p.read_text())
    tickers = raw.get("stage1") or []
    if not tickers:
        raise SystemExit(f"No 'stage1' tickers found in {p}")
    return [t.upper() for t in tickers]


def _print_summary(results: list[TickerResult]) -> None:
    print()
    print("=" * 76)
    print(f"{'Ticker':<7} {'Status':<8} {'Q':>5} {'V':>5} {'M':>5} "
          f"{'News':>5} {'NewsS':>6} {'Time':>7} {'Cost':>8}")
    print("-" * 76)
    total_cost = 0.0
    for r in results:
        if r.status == "ok":
            line = (
                f"{r.ticker:<7} {r.status:<8} "
                f"{r.quality or 0:>5.1f} {r.valuation or 0:>5.1f} "
                f"{r.momentum or 0:>5.1f} "
                f"{r.news_n:>5d} "
                f"{(r.news_sentiment or 0):>+6.2f} "
                f"{r.elapsed_s:>6.1f}s "
                f"${r.cost_usd:>6.4f}"
            )
        else:
            line = (
                f"{r.ticker:<7} {r.status:<8} "
                f"failed at {r.error_step}: {r.error}"
            )
        print(line)
        total_cost += r.cost_usd
    print("-" * 76)
    print(f"Total thesis cost: ${total_cost:.4f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 weekly run for the Stage 1 ticker list.")
    parser.add_argument(
        "--tickers", nargs="*", help="Override the configured ticker list.",
    )
    parser.add_argument(
        "--week-ending", default=None,
        help="Week-ending date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument("--skip-reddit", action="store_true",
                        help="Skip the Reddit step (useful when Reddit is rate-limiting).")
    parser.add_argument("--news-days", type=int, default=7)
    parser.add_argument("--price-days", type=int, default=365)
    args = parser.parse_args(argv)

    storage.init_db()  # idempotent
    tickers = args.tickers or load_tickers()
    week_ending = (
        date.fromisoformat(args.week_ending) if args.week_ending else date.today()
    )

    print(f"Stage 1 run: tickers={tickers}, week_ending={week_ending}")
    results: list[TickerResult] = []
    for t in tickers:
        results.append(
            run_one(
                t,
                week_ending=week_ending,
                news_days=args.news_days,
                price_lookback_days=args.price_days,
                skip_reddit=args.skip_reddit,
            )
        )

    _print_summary(results)
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
