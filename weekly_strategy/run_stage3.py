from __future__ import annotations

"""Stage 3 orchestrator: universe-wide weekly portfolio run.

Flow:

    1. Load (or construct) universe
    2. Pre-build dossiers for any tickers missing one (optional flag)
    3. Macro / Fed / regime / sector context (once, shared)
    4. batch_weekly_scoring across the universe
    5. normalize_scores -> z-scores + composite_z
    6. build_focus_list (top 15 longs / bottom 15 shorts / outliers)
    7. select_positions (5+5 with sector neutrality, beta neutralization,
       optional earnings + short-interest filters, turnover dampening
       against last week's book)
    8. filter_by_conviction (Sonnet adversarial pass per position;
       LOWs are dropped and substituted)
    9. Render + save portfolio Markdown report

For fast smoke runs use ``--skip-news --skip-reddit --skip-conviction``
to bypass LLM-bound steps. The structural plumbing still gets exercised.
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from weekly_strategy.config import settings
from weekly_strategy.data import fetchers, predictions, storage
from weekly_strategy.data.schemas import Dossier, Universe, UniverseEntry
from weekly_strategy.dossier import builder as dossier_builder
from weekly_strategy.dossier import universe as universe_mod
from weekly_strategy.llm import conviction as conviction_mod
from weekly_strategy.reporting import portfolio as portfolio_mod
from weekly_strategy.run_stage1 import build_macro_context
from weekly_strategy.signals import batch_scoring, news_sentiment, reddit_sentiment, selection


def _adhoc_universe(tickers: list[str]) -> Universe:
    """Build a Universe-shaped object from a CLI ticker list (smoke runs).

    Sector / market_cap are pulled live via yfinance so downstream selection
    has enough metadata to enforce sector neutrality.
    """
    meta = universe_mod._fetch_metadata(tickers)
    entries: list[UniverseEntry] = []
    for t in tickers:
        info = meta.get(t, {})
        entries.append(UniverseEntry(
            ticker=t, sector=info.get("sector"), industry=info.get("industry"),
            market_cap=info.get("market_cap"), beta=info.get("beta"),
            included_reason="manual",
        ))
    return Universe(
        quarter="adhoc",
        constructed_at=datetime.utcnow(),
        entries=entries,
    )


def _load_dossier(ticker: str) -> Dossier | None:
    blob = storage.load_dossier(ticker)
    if blob is None:
        return None
    return Dossier(**blob["data"])


def _ensure_dossiers(universe: Universe, *, force_refresh: bool = False, log=print) -> None:
    """Build any missing dossiers (or all of them if force_refresh)."""
    missing = []
    for ticker in universe.tickers:
        if force_refresh or storage.load_dossier(ticker) is None:
            missing.append(ticker)
    if not missing:
        return
    log(f"[stage3] dossiers needed for {len(missing)} tickers: "
        f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    res = universe_mod.batch_build_dossiers(
        universe.model_copy(update={
            "entries": [e for e in universe.entries if e.ticker in missing]
        }),
        force_refresh=force_refresh, log=log,
    )
    if res.failed:
        log(f"[stage3] {len(res.failed)} dossier(s) failed; continuing")


def _load_previous_book(as_of: date) -> tuple[set[str], set[str]]:
    """Most-recent persisted portfolio prior to ``as_of``, for turnover dampening."""
    return predictions.load_previous_book(as_of)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3 weekly portfolio run.",
    )
    parser.add_argument("--tickers", nargs="*",
                        help="Override universe with a CLI ticker list (smoke runs).")
    parser.add_argument("--week-ending", default=None,
                        help="Week-ending date YYYY-MM-DD. Default: today.")
    parser.add_argument("--build-dossiers", action="store_true",
                        help="Build any missing dossiers before scoring.")
    parser.add_argument("--refresh-dossiers", action="store_true",
                        help="Force-refresh all dossiers (re-pull companyfacts).")
    parser.add_argument(
        "--position-count", type=int, default=10,
        help="Final selection size per side (post-rerank if --rerank-after-enrich). "
             "Default 10 -- top 5 shown prominently, next 5 folded.",
    )
    parser.add_argument("--max-per-sector", type=int, default=4,
                        help="Cap positions per GICS sector (default 4; with "
                             "position_count=10 across ~6 sectors this needs headroom).")
    parser.add_argument(
        "--focus-list-size", type=int, default=30,
        help="Top-K longs / bottom-K shorts pulled into the focus list. "
             "Default 30 -- with both sides this is the enrichment pool of 60 "
             "unique tickers (~$0.50/day equiv).",
    )
    parser.add_argument(
        "--top-shown", type=int, default=5,
        help="Number of positions per side rendered as full cards in the report. "
             "The remainder are folded into a collapsible 'Show N more' section. "
             "Default 5.",
    )
    parser.add_argument(
        "--no-rerank-after-enrich", action="store_true",
        help="Disable the two-stage rerank. By default, after enriching the "
             "focus list with news + reddit, the system re-normalizes those "
             "bundles (now including news sentiment) and re-selects positions. "
             "Pass this flag to keep the original ranking and use news/reddit "
             "as display-only.",
    )
    parser.add_argument("--news-days", type=int, default=7)
    parser.add_argument("--price-days", type=int, default=365)
    parser.add_argument("--reddit-hours", type=int, default=24)
    parser.add_argument("--skip-news", action="store_true",
                        help="Skip the per-ticker news cascade (fast smoke).")
    parser.add_argument("--skip-reddit", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-sector", action="store_true")
    parser.add_argument("--skip-conviction", action="store_true",
                        help="Skip the post-selection LLM conviction pass.")
    parser.add_argument("--skip-enrich-selected", action="store_true",
                        help="Skip news/reddit fetch + sentiment on the 10 selected "
                             "positions. By default we enrich the selected book even "
                             "in cheap mode, so the report has news/sentiment per "
                             "position (~$0.30/day, ~5 min added).")
    args = parser.parse_args(argv)

    storage.init_db()
    week_ending = (
        date.fromisoformat(args.week_ending) if args.week_ending else date.today()
    )

    # --- 1. Universe ---
    if args.tickers:
        universe = _adhoc_universe([t.upper() for t in args.tickers])
        print(f"[stage3] using ad-hoc universe of {len(universe.tickers)} tickers")
    else:
        universe = storage.load_current_universe()
        if universe is None:
            print("[stage3] no universe on file; constructing from candidate pool...")
            universe = universe_mod.construct_universe()
            storage.save_universe(universe)
            print(f"[stage3] saved universe {universe.quarter} "
                  f"({len(universe.tickers)} names)")

    # --- 2. Dossiers ---
    if args.build_dossiers or args.refresh_dossiers:
        _ensure_dossiers(universe, force_refresh=args.refresh_dossiers)

    # --- 3. Macro context (shared across all tickers) ---
    macro_ctx = build_macro_context(
        week_ending=week_ending,
        skip_macro=args.skip_macro,
        skip_sector=args.skip_sector,
    )

    # --- 4. Batch scoring ---
    print(f"[stage3] scoring {len(universe.tickers)} tickers...")
    score_res = batch_scoring.batch_weekly_scoring(
        universe,
        week_ending=week_ending,
        macro_snapshot=macro_ctx.macro_snapshot,
        macro_regime=macro_ctx.macro_regime,
        sector_snap=macro_ctx.sector_snap,
        news_days=args.news_days,
        price_lookback_days=args.price_days,
        reddit_hours=args.reddit_hours,
        skip_news=args.skip_news,
        skip_reddit=args.skip_reddit,
    )
    bundles = score_res.bundles

    # --- 5. Cross-sectional normalization ---
    bundles = batch_scoring.normalize_scores(
        bundles, regime=macro_ctx.macro_regime,
    )

    # --- 6. Focus list ---
    sector_lookup: dict[str, str | None] = {}
    beta_lookup: dict[str, float | None] = {}
    dossier_cache: dict[str, Dossier] = {}
    for t in bundles:
        d = _load_dossier(t)
        if d is not None:
            dossier_cache[t] = d
            sector_lookup[t] = d.sector
        beta_lookup[t] = universe.get(t).beta if universe.get(t) else None

    focus = selection.build_focus_list(
        bundles,
        week_ending=week_ending,
        top_k_longs=args.focus_list_size,
        top_k_shorts=args.focus_list_size,
        sector_lookup=sector_lookup,
        beta_lookup=beta_lookup,
    )

    # --- 7. Selection (with constraints + optional turnover dampening) ---
    prev_longs, prev_shorts = _load_previous_book(week_ending)
    portfolio = selection.select_positions(
        focus,
        week_ending=week_ending,
        position_count=args.position_count,
        max_per_sector=args.max_per_sector,
        held_long_tickers=prev_longs,
        held_short_tickers=prev_shorts,
    )

    # --- 7d. Enrich selected positions with news + reddit (cheap-mode default) ---
    # Even when scoring skipped LLM cascades (--skip-news/--skip-reddit), we still
    # want news + reddit context on the ~10 selected positions for the report.
    # Costs ~$0.30/day and ~5 min, but produces a meaningfully richer per-position
    # reasoning card.
    # Two-stage: enrich the FOCUS LIST (not just final selection) so news +
    # reddit can shape the final rank. Pool = top-K longs + bottom-K shorts.
    # With focus_list_size=30 that's up to 60 unique names.
    if args.skip_enrich_selected:
        enrich_pool: list[str] = []
    else:
        enrich_pool = list(dict.fromkeys(
            [it.ticker for it in focus.long_candidates]
            + [it.ticker for it in focus.short_candidates]
        ))

    news_score_by_ticker: dict[str, float] = {}
    selected_reddit_scores: dict = {}
    if enrich_pool:
        print(f"[stage3] enriching focus list ({len(enrich_pool)} unique tickers) "
              f"with news + reddit...")
        since_dt = datetime.combine(week_ending, datetime.min.time()) - timedelta(days=args.news_days)
        for tk in enrich_pool:
            try:
                info = fetchers.get_basic_info(tk)
                items = fetchers.fetch_news(tk, days=args.news_days)
                if items:
                    storage.insert_news(items)
                ns = news_sentiment.score_news(
                    tk, since=since_dt, company_name=info.get("name"),
                )
                news_score_by_ticker[tk] = ns.sentiment_score
                print(f"   [{tk}] news: {ns.n_classified}/{ns.n_total} classified, "
                      f"sent={ns.sentiment_score:+.2f}")
            except Exception as e:
                print(f"   [{tk}] news enrich FAILED: {e}")
            try:
                posts = fetchers.fetch_reddit_recent(tk, hours=args.reddit_hours)
                if posts:
                    storage.insert_reddit(posts)
                snap = fetchers.build_reddit_snapshot(
                    tk, current_posts=posts, window_hours=args.reddit_hours,
                )
                selected_reddit_scores[tk] = reddit_sentiment.score_reddit_snapshot(snap)
            except Exception as e:
                print(f"   [{tk}] reddit enrich FAILED: {e}")

    # Step 2: re-rank within the enriched pool with news + reddit included.
    if news_score_by_ticker and not args.no_rerank_after_enrich:
        print(f"[stage3] re-ranking {len(enrich_pool)} enriched names "
              f"with news + reddit in composite...")
        enriched_bundles: dict = {}
        for tk in enrich_pool:
            if tk not in bundles:
                continue
            updates: dict = {}
            if tk in news_score_by_ticker:
                updates["news_sentiment_score"] = news_score_by_ticker[tk]
            if tk in selected_reddit_scores:
                rs = selected_reddit_scores[tk]
                updates["reddit_sentiment"] = rs.llm_sentiment
                updates["reddit_is_crowded"] = rs.is_crowded
            enriched_bundles[tk] = bundles[tk].model_copy(update=updates)

        # Re-normalize within the enriched subset only -- otherwise non-enriched
        # tickers (z_news=0 by fallback) dilute the news signal.
        enriched_bundles = batch_scoring.normalize_scores(
            enriched_bundles, regime=macro_ctx.macro_regime,
        )

        focus = selection.build_focus_list(
            enriched_bundles,
            week_ending=week_ending,
            top_k_longs=args.position_count,
            top_k_shorts=args.position_count,
            sector_lookup=sector_lookup,
            beta_lookup=beta_lookup,
        )
        portfolio = selection.select_positions(
            focus,
            week_ending=week_ending,
            position_count=args.position_count,
            max_per_sector=args.max_per_sector,
            held_long_tickers=prev_longs,
            held_short_tickers=prev_shorts,
        )
        bundles = enriched_bundles

    # --- 8. Conviction (Sonnet, one call per selected position) ---
    checks: dict = {}
    if not args.skip_conviction:
        print(f"[stage3] conviction check on "
              f"{len(portfolio.longs) + len(portfolio.shorts)} positions...")
        # Reuse news fetches: get_news from storage with the same window.
        news_items_by_ticker = {
            t: storage.get_news(
                t, since=datetime.combine(week_ending - timedelta(days=args.news_days), datetime.min.time()),
            )
            for t in (
                [p.ticker for p in portfolio.longs]
                + [p.ticker for p in portfolio.shorts]
            )
        }
        company_names = {
            t: (fetchers.get_basic_info(t).get("name") or t)
            for t in news_items_by_ticker
        }
        portfolio, checks = conviction_mod.filter_by_conviction(
            portfolio,
            focus=focus,
            bundles=bundles,
            dossiers=dossier_cache,
            news_items_by_ticker=news_items_by_ticker,
            company_names=company_names,
            macro_regime=macro_ctx.macro_regime,
            sector_snap=macro_ctx.sector_snap,
            max_per_sector=args.max_per_sector,
        )

    # --- 9. Render + save portfolio report ---
    # Pull dossiers and any cached news items for the selected positions
    # so per-position reasoning includes fundamentals + (when available) news.
    selected_tickers = [p.ticker for p in portfolio.longs + portfolio.shorts]
    selected_dossiers = {t: dossier_cache[t] for t in selected_tickers if t in dossier_cache}
    news_for_positions = {}
    news_window_start = datetime.combine(
        week_ending - timedelta(days=args.news_days), datetime.min.time()
    )
    for t in selected_tickers:
        items = storage.get_news(t, since=news_window_start)
        if items:
            news_for_positions[t] = items

    md = portfolio_mod.render_portfolio_markdown(
        portfolio,
        bundles=bundles,
        convictions=checks,
        focus=focus,
        macro_snapshot=macro_ctx.macro_snapshot,
        macro_regime=macro_ctx.macro_regime,
        fed_posture=macro_ctx.fed_posture,
        sector_snap=macro_ctx.sector_snap,
        previous_longs=prev_longs,
        previous_shorts=prev_shorts,
        dossiers=selected_dossiers,
        news_items_by_ticker=news_for_positions,
        reddit_scores=selected_reddit_scores,
        top_shown=args.top_shown,
    )
    path = portfolio_mod.save_portfolio_markdown(portfolio, md)
    html_path = portfolio_mod.save_portfolio_html(portfolio, md)
    print(f"[stage3] saved report -> {path}")
    print(f"[stage3] saved html   -> {html_path}")

    # --- 10. Persist daily prediction snapshot (Stage 5 calibration substrate) ---
    snap_dir = predictions.save_daily_snapshot(
        as_of=week_ending,
        bundles=bundles,
        portfolio=portfolio,
        macro_regime=macro_ctx.macro_regime,
        sector_snap=macro_ctx.sector_snap,
        universe_tickers=universe.tickers,
    )
    print(f"[stage3] saved snapshot -> {snap_dir}")

    _print_summary(score_res, portfolio, checks)

    return 0 if score_res.bundles else 1


def _print_summary(score_res, portfolio, checks) -> None:
    n_ok = len(score_res.bundles)
    n_fail = len(score_res.failed)
    print()
    print("=" * 80)
    print(f"Stage 3 run: {n_ok} scored / {n_fail} failed; "
          f"{len(portfolio.longs)} longs + {len(portfolio.shorts)} shorts")
    if portfolio.longs:
        print("Longs:")
        for p in portfolio.longs:
            tag = ""
            if p.ticker in checks:
                tag = f" [{checks[p.ticker].confidence}]"
            print(f"  {p.ticker:<6} sector={p.sector or '?':<22} "
                  f"comp_z={(p.composite_z or 0):+.2f}{tag}")
    if portfolio.shorts:
        print("Shorts:")
        for p in portfolio.shorts:
            tag = ""
            if p.ticker in checks:
                tag = f" [{checks[p.ticker].confidence}]"
            print(f"  {p.ticker:<6} sector={p.sector or '?':<22} "
                  f"comp_z={(p.composite_z or 0):+.2f}{tag}")
    print(f"Net beta: {portfolio.net_beta if portfolio.net_beta is not None else 'n/a'}")
    print("=" * 80)


if __name__ == "__main__":
    raise SystemExit(main())
