# Calibration plan — after 30+ daily snapshots

This document outlines the calibration loop that runs once we have
roughly a month (~22 trading days) of daily prediction snapshots
persisted under `weekly_strategy/data_cache/predictions/{date}/`.

The goal at month-1 is **not** to pick a final model. It's to find out
which signals are doing useful work and which are noise, so we can
delete or down-weight the noise before paper trading.

---

## What gets persisted each day

By every Stage 3 run via `predictions.save_daily_snapshot`:

- `bundles.parquet` — one row per ticker × component score. Includes the
  raw 0-100 components (quality, valuation, momentum, news_sentiment,
  sector_score, macro_regime_score), the composite scores
  (`composite_score`, `composite_z`), the cross-sectional z-scores
  (`z_quality`, `z_valuation`, `z_momentum`, `z_news`, `z_sector`), and
  trailing returns at the snapshot date.
- `portfolio.json` — the selected 5+5 book, macro regime, sector
  breadth + leadership.
- `prices.parquet` — close + adj_close at `as_of` for every universe
  ticker. Pinned at snapshot time so calibration doesn't get mugged by
  later corporate actions or splits in the live price cache.

After 22 sessions we'll have ~22 × 100 = ~2,200 (ticker, day) score
rows + matching prices. That's enough for an honest first cut.

---

## Calibration metrics (to compute, in order of importance)

### 1. Information coefficient (IC) per component

For each component score `s ∈ {quality, valuation, momentum, news,
sector, composite_z}` and each forward window `h ∈ {1d, 5d, 21d}`:

```
IC(s, h) = corr(score_on_day_t, forward_return_t_to_t+h) over all (t, ticker)
```

Spearman rank-IC is more robust than Pearson at the component level
(absolute scale matters less than ordering). Compute both.

**Acceptance bar:** `|rank-IC|` for the composite at 5d ≥ 0.03 across
the full month. If composite IC is around zero, the signal is noise
right now and paper-trading will burn money — **stop and re-engineer
before going live.**

### 2. IC stability across days

The 22 daily ICs (one per day) shouldn't be all-positive on some
weeks and all-negative on others. If the time series of IC swings
wildly, the signal is regime-dependent (it works in some macro states
and not others) — that's fine but we need to know which regime turns
it on/off.

Plot: daily IC vs `macro_regime.cycle_phase` and `financial_conditions`
to see if any pattern jumps out.

### 3. Long-short decile spread

Sort the universe by `composite_z` each day into deciles. Compute the
average forward 5d return of decile 10 (top) minus decile 1 (bottom).

**Acceptance bar:** positive average spread with a t-stat ≥ 1.5 over
22 days. Higher is better. Negative means the model is anti-predictive
(short the longs, long the shorts, and your problems go away).

### 4. Per-component contribution

Fit a simple OLS:

```
forward_5d_return ~ z_quality + z_valuation + z_momentum + z_news + z_sector + sector_dummies
```

The coefficients tell us which components carry the predictive load.
At month-1 these will be noisy (small sample), but they'll cleanly
flag components that contribute *zero* — those are deletion candidates.

### 5. Selection-vs-universe lift

Did the actual selected 5+5 book outperform a random 5+5 draw from the
universe? Specifically:

- Average 5d realized return of selected longs minus selected shorts
- vs. the average 5d return of a uniformly-random L/S draw

If the L/S spread isn't materially above random, the selection
constraints (sector caps, turnover dampening, beta neutralization) are
washing out the signal — investigate.

### 6. Conviction-grade lift

Did positions graded `HIGH` by the conviction check actually
outperform `MEDIUM` and `LOW`? If the grading is randomly correlated
with realized returns, the conviction LLM pass is overhead.

---

## Decisions the month-1 calibration enables

1. **Component weights**: rebalance `_RISK_ON_WEIGHTS` /
   `_RISK_OFF_WEIGHTS` in `signals/fundamental.py` and the parallel
   z-weights in `signals/batch_scoring.py` based on per-component IC.
2. **Component pruning**: drop a component entirely if its IC is
   indistinguishable from zero. Today's score has 7 components;
   we may need 4.
3. **Threshold tuning** in `quality_score` / `valuation_score`. If a
   component has good IC but the top-of-scale never fires, the linear
   anchors are too generous.
4. **Sector-regime fit table** in `signals/sector_score.py`. The
   `regime_fit_for_etf` table is "conventional wisdom" today —
   month-1 IC by sector × regime tells us whether it actually fires
   the right way.
5. **Materiality cascade utility**: did the per-headline fundamental
   pass (Step 1.5 pass 2) actually shift composite ranks vs sentiment
   alone? If not, that's $$$ of Sonnet calls per week we can drop.

---

## What this calibration is NOT

- It's **not** a backtest. 22 days of forward returns from one regime
  is a sliver. Real backtest (Stage 5) needs months-to-years of data
  AND point-in-time dossiers, which we don't yet have.
- It's **not** a strategy approval. Even if month-1 IC is good, that's
  one regime. Paper trading still gates real capital.
- It's **not** a tuning loop. Don't refit 50 hyperparameters off 22
  samples — you'll overfit the regime. Pick 2-3 high-impact changes
  and re-evaluate at month-2.

---

## Implementation sketch (Stage 5)

When we're ready to actually compute the above, this would be roughly
~300-500 LOC under `weekly_strategy/calibration/`:

- `forward_returns.py` — join `predictions.load_snapshot_history()` with
  same-ticker prices `h` days later. Handle weekends + missing days
  conservatively.
- `ic.py` — Spearman + Pearson IC per component × horizon. Decile
  bucketing + long-short spread.
- `attribution.py` — OLS coefficient table per component.
- `report.py` — Markdown summary of the above.

But all of that only makes sense once snapshots exist. **The daily
collection loop is the prerequisite**; calibration code can wait.
