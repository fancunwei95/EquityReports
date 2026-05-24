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
from weekly_strategy.data import fetchers, macro as macro_mod, sector as sector_mod, storage
from weekly_strategy.data.schemas import (
    CrossStockImplications, Dossier, FedPosture, MacroRegime, MacroSnapshot, SectorSnapshot,
)
from weekly_strategy.dossier import builder as dossier_builder
from weekly_strategy.llm.thesis import write_thesis
from weekly_strategy.reporting import single_stock
from weekly_strategy.signals import fundamental as fnd
from weekly_strategy.signals import macro_regime as regime_mod
from weekly_strategy.signals import news_sentiment, reddit_sentiment, sector_score


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
    sector: float | None = None
    composite: float | None = None
    news_n: int = 0
    news_sentiment: float | None = None
    reddit_mentions: int = 0
    report_path: Path | None = None
    error: str | None = None
    error_step: str | None = None


@dataclass
class MacroContext:
    """Universe-level inputs pulled once per run and shared across tickers."""

    macro_snapshot: MacroSnapshot | None = None
    fed_posture: FedPosture | None = None
    macro_regime: MacroRegime | None = None
    sector_snap: SectorSnapshot | None = None
    cost_usd: float = 0.0          # Sonnet calls in Fed + regime synthesis
    warnings: list[str] = field(default_factory=list)


def build_macro_context(
    *, week_ending: date, skip_macro: bool, skip_sector: bool, log=print,
) -> MacroContext:
    """Pull macro / sector / Fed / regime once, log any soft failures."""
    ctx = MacroContext()

    if not skip_sector:
        try:
            log("[macro] fetching sector ETF prices...")
            sector_mod.fetch_sector_prices()
            ctx.sector_snap = sector_mod.get_sector_snapshot(week_ending=week_ending)
        except Exception as e:
            ctx.warnings.append(f"sector snapshot failed: {e}")
            log(f"[macro] sector snapshot FAILED: {e}")

    if not skip_macro:
        try:
            log("[macro] fetching FRED series + assembling macro snapshot...")
            ctx.macro_snapshot = macro_mod.get_macro_snapshot(week_ending=week_ending)
        except macro_mod.FredAuthError as e:
            ctx.warnings.append("FRED_API_KEY missing -- macro snapshot skipped")
            log(f"[macro] FRED key not set; continuing without macro snapshot")
        except Exception as e:
            ctx.warnings.append(f"macro snapshot failed: {e}")
            log(f"[macro] macro snapshot FAILED: {e}")

        try:
            log("[macro] classifying Fed posture (Sonnet)...")
            ctx.fed_posture = macro_mod.get_fed_posture(week_ending=week_ending)
        except Exception as e:
            ctx.warnings.append(f"Fed posture failed: {e}")
            log(f"[macro] Fed posture FAILED: {e}")

        if ctx.macro_snapshot is not None:
            try:
                log("[macro] synthesising macro regime (Sonnet)...")
                ctx.macro_regime = regime_mod.classify_regime(
                    ctx.macro_snapshot, fed=ctx.fed_posture,
                )
            except Exception as e:
                ctx.warnings.append(f"regime classification failed: {e}")
                log(f"[macro] regime classification FAILED: {e}")

    return ctx


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
    skip_cross_stock: bool = False,
    macro_ctx: MacroContext | None = None,
    log=print,
) -> TickerResult:
    """Full single-stock pipeline. Returns a TickerResult either way.

    If ``macro_ctx`` is provided, the Stage-2 overlays (sector_score,
    cross_stock_implications, macro context) are wired into the score
    bundle, thesis prompt, and Markdown report.
    """
    res = TickerResult(ticker=ticker.upper())
    t_start = time.time()
    current_step = "init"
    macro_ctx = macro_ctx or MacroContext()

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

        # --- 7a. Sector score (Stage 2) ---
        current_step = "sector_score"
        sector_info = sector_score.sector_score(
            dossier=dossier,
            sector_snap=macro_ctx.sector_snap,
            regime=macro_ctx.macro_regime,
        )

        # --- 7b. Cross-stock implications (Stage 2; LLM) ---
        cross_stock: CrossStockImplications | None = None
        if not skip_cross_stock:
            current_step = "cross_stock"
            log(f"[{ticker}] cross-stock implications (Sonnet)...")
            cross_stock = sector_score.cross_stock_implications(
                ticker,
                dossier=dossier,
                news_items=items_for_thesis,
                sector_snap=macro_ctx.sector_snap,
                regime=macro_ctx.macro_regime,
                week_ending=week_ending,
                company_name=company_name,
            )

        # --- 7c. Score bundle (with Stage 2 overlays) ---
        current_step = "score_bundle"
        prices = storage.load_prices(ticker)
        bundle = fnd.build_score_bundle(
            ticker=ticker, week_ending=week_ending,
            dossier=dossier, prices=prices,
            news_sentiment_score=news_score.sentiment_score,
            news_noise_ratio=news_score.noise_ratio,
            reddit_sentiment=reddit_score.llm_sentiment if reddit_score else None,
            reddit_is_crowded=reddit_score.is_crowded if reddit_score else False,
            macro_regime=macro_ctx.macro_regime,
            sector_score_info=sector_info,
        )
        res.quality = bundle.quality_score
        res.valuation = bundle.valuation_score
        res.momentum = bundle.momentum_score
        res.sector = bundle.sector_score_value
        res.composite = bundle.composite_score

        # --- 8. Thesis (Sonnet) ---
        current_step = "thesis"
        log(f"[{ticker}] writing thesis (Sonnet)...")
        thesis = write_thesis(
            ticker=ticker, week_ending=week_ending,
            dossier=dossier, scores=bundle,
            news_score=news_score, news_items=items_for_thesis,
            reddit_score=reddit_score, prices=prices,
            macro_snapshot=macro_ctx.macro_snapshot,
            macro_regime=macro_ctx.macro_regime,
            sector_snap=macro_ctx.sector_snap,
            cross_stock=cross_stock,
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
            macro_snapshot=macro_ctx.macro_snapshot,
            macro_regime=macro_ctx.macro_regime,
            fed_posture=macro_ctx.fed_posture,
            sector_snap=macro_ctx.sector_snap,
            cross_stock=cross_stock,
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


def _print_summary(results: list[TickerResult], macro_ctx: MacroContext | None = None) -> None:
    print()
    if macro_ctx is not None and (macro_ctx.macro_regime or macro_ctx.fed_posture or macro_ctx.sector_snap):
        print("=" * 92)
        print("MACRO CONTEXT")
        r = macro_ctx.macro_regime
        if r:
            print(f"  regime  : rate={r.rate_regime}  conditions={r.financial_conditions}  cycle={r.cycle_phase}")
            if r.narrative:
                print(f"  narrative: {r.narrative}")
        if macro_ctx.fed_posture and macro_ctx.fed_posture.n_items:
            f = macro_ctx.fed_posture
            print(f"  fed     : {f.posture} ({f.n_items} items)" + (f" -- {f.summary}" if f.summary else ""))
        if macro_ctx.sector_snap and macro_ctx.sector_snap.leadership_ranking:
            s = macro_ctx.sector_snap
            print(f"  sectors : breadth={s.breadth}  leaders={','.join(s.leadership_ranking[:3])}"
                  f"  laggards={','.join(s.leadership_ranking[-3:])}")
        if macro_ctx.warnings:
            print("  warnings:")
            for w in macro_ctx.warnings:
                print(f"    - {w}")
        print()

    print("=" * 92)
    print(f"{'Ticker':<7} {'Status':<8} {'Q':>5} {'V':>5} {'M':>5} {'Sec':>5} {'Comp':>5} "
          f"{'News':>5} {'NewsS':>6} {'Time':>7} {'Cost':>8}")
    print("-" * 92)
    total_cost = (macro_ctx.cost_usd if macro_ctx else 0.0)
    for r in results:
        if r.status == "ok":
            line = (
                f"{r.ticker:<7} {r.status:<8} "
                f"{r.quality or 0:>5.1f} {r.valuation or 0:>5.1f} "
                f"{r.momentum or 0:>5.1f} {(r.sector or 0):>5.1f} "
                f"{(r.composite or 0):>5.1f} "
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
    print("-" * 92)
    print(f"Total LLM cost (thesis + macro/regime/cross-stock): ${total_cost:.4f}")
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
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip the macro layer (FRED + Fed + regime).")
    parser.add_argument("--skip-sector", action="store_true",
                        help="Skip the sector ETF snapshot.")
    parser.add_argument("--skip-cross-stock", action="store_true",
                        help="Skip the cross-stock LLM agent per ticker.")
    parser.add_argument("--news-days", type=int, default=7)
    parser.add_argument("--price-days", type=int, default=365)
    args = parser.parse_args(argv)

    storage.init_db()  # idempotent
    tickers = args.tickers or load_tickers()
    week_ending = (
        date.fromisoformat(args.week_ending) if args.week_ending else date.today()
    )

    print(f"Stage 2 run: tickers={tickers}, week_ending={week_ending}")

    macro_ctx = build_macro_context(
        week_ending=week_ending,
        skip_macro=args.skip_macro,
        skip_sector=args.skip_sector,
    )

    results: list[TickerResult] = []
    for t in tickers:
        results.append(
            run_one(
                t,
                week_ending=week_ending,
                news_days=args.news_days,
                price_lookback_days=args.price_days,
                skip_reddit=args.skip_reddit,
                skip_cross_stock=args.skip_cross_stock,
                macro_ctx=macro_ctx,
            )
        )

    _print_summary(results, macro_ctx=macro_ctx)
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
