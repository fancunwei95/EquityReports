# Daily prediction loop — runbook

After the 2026Q2 thematic universe is bootstrapped, this is how you
collect month-1 calibration data. Read top-to-bottom on the first run;
the routine after that is just one command.

---

## One-time setup (done)

```
python -m weekly_strategy.scripts.bootstrap_thematic_universe
```

Loaded the 100-ticker universe from `config/thematic_universe.json`
into `data_cache/universe/2026Q2.json`. Re-run only when the curated
list changes.

To swap out JNPR / CFLT (the two with missing metadata at bootstrap),
edit `config/thematic_universe.json` and re-bootstrap. Or just re-run
the bootstrap once — yfinance flakiness on a single ticker usually
resolves on the next call.

---

## Per-day kickoff

The cheap loop (sector + dossier scoring + selection; no news cascade,
no conviction). One LLM call per day for macro regime synthesis (Fed
posture skips since FRED key is not set):

```
python -m weekly_strategy.run_stage3 \
    --skip-news --skip-reddit --skip-conviction --skip-macro \
    --week-ending $(date -u +%F)
```

Cost: ~0$/day (no Sonnet calls when --skip-macro and --skip-conviction).
Runtime: ~30s once dossiers are built.

The full loop (news cascade + reddit + cross-stock + conviction):

```
python -m weekly_strategy.run_stage3 \
    --build-dossiers --week-ending $(date -u +%F)
```

Cost: ~$3-5/day (100 tickers × ~$0.02 Sonnet thesis + 10 × ~$0.03
conviction). Runtime: hours (100 × ~60s per ticker for Sonnet calls).
Pro's $20/mo headless credit will run out around mid-month at this
rate. Drop to the cheap loop unless you're testing the full pipeline.

**Recommended for month-1 data collection**: cheap loop on weekdays,
full loop on weekends to seed a deeper read. That gives ~20 cheap + ~8
full snapshots over 4 weeks — enough variety to calibrate.

---

## What gets saved each run

Inside `weekly_strategy/data_cache/predictions/{YYYY-MM-DD}/`:

* `bundles.parquet` — one row per ticker × all score components +
  z-scores + returns. **This is the feature matrix calibration regresses
  against.**
* `portfolio.json` — selected 5+5 book + macro/sector context.
* `prices.parquet` — close prices for every universe ticker at run time
  (pinned snapshot so calibration doesn't rely on the live price cache
  weeks later).

Also under `weekly_strategy/reports/`:

* `portfolio_{date}.md` — the human-readable weekly note.

These directories are gitignored by default; nothing else clutters the
repo.

---

## Cadence options

### A. Local cron (recommended)

Edit your crontab (`crontab -e`):

```
# Daily 4:30pm ET (after US market close) -- adjust TZ
30 16 * * 1-5 cd /mnt/thdcan01/data/cfan/projects/202605_equityReports && \
    .venv/bin/python -m weekly_strategy.run_stage3 \
        --skip-news --skip-reddit --skip-conviction --skip-macro \
        --week-ending $(date -u +\%F) >> daily.log 2>&1
```

### B. Claude Code `/schedule`

Claude Code can schedule recurring runs via `/schedule`. Run that slash
command at the prompt and configure a daily routine that invokes the
same command above. (`/schedule` is a user-triggered skill; I can't set
it up for you, but it's a one-liner once you invoke it.)

### C. Manual

Just run it daily. The script is idempotent: re-running with the same
`--week-ending` overwrites the snapshot in place. Missing days are
genuinely missing — no implicit backfill.

---

## Stop conditions before month-1 review

Watch for these in the daily output and abort the loop if you see
them:

* **Multiple snapshot dirs with empty `bundles.parquet`** — scoring is
  failing silently. Open one of the directories and grep for FAIL in
  the run logs.
* **Composite_z spread shrinking week-over-week** — if every ticker
  ends up at z ≈ 0 across the universe, the cross-sectional
  normalization is degenerate (universe might be too homogeneous).
* **All selected positions in the same sector** — sector neutrality
  isn't binding because two GICS sectors dominate the universe.
  Tighten `--max-per-sector` to 1.

---

## At month-1: invoke calibration

We haven't built the calibration code yet (Stage 5). See
`docs/calibration_plan.md` for the formal plan. The substrate is
already there:

```python
from weekly_strategy.data import predictions
hist = predictions.load_snapshot_history()
# hist is a DataFrame with one row per (ticker, day) and every
# component score + z-score + return alongside.
```

From there it's joining `hist` against forward returns and computing
ICs per component × horizon. ~300-500 LOC implementation. We do that
once we have data; don't pre-implement and overfit ergonomics to
imagined needs.
