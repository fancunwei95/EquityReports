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

In `daily_reports/` at the repo root:

* `{YYYY-MM-DD}.md`   — Markdown portfolio note (one per day)
* `{YYYY-MM-DD}.html` — styled standalone HTML for browser viewing

Daemon logs go to `daily_loop_logs/loop_{startup-date}.log` plus stdout
inside the `screen` session.

All these directories are gitignored by default; nothing clutters the repo.

---

## Cadence options

### A. Long-running Python daemon in `screen` (recommended for this setup)

```
# Start a detachable session
screen -S daily-loop

# Inside the session: launch the daemon
cd /mnt/thdcan01/data/cfan/projects/202605_equityReports
.venv/bin/python -m weekly_strategy.scripts.daily_loop \
    --time-et 08:30 \
    --mode cheap \
    --weekdays-only

# Detach (leave it running): Ctrl-A then D
# Re-attach later:
screen -r daily-loop
```

The daemon sleeps until ``--time-et`` each day (default behavior when
neither flag is set: 21:30 UTC = 5:30 PM ET DST / 4:30 PM ET standard).
``--time-et 08:30`` runs at 8:30 AM US Eastern year-round; the UTC offset
adjusts automatically across DST. Use ``--time-utc HH:MM`` instead if
you'd rather pin to UTC. Then it invokes ``run_stage3`` with the right
flag set for the chosen ``--mode``:

* ``cheap``     -- ``--skip-news --skip-reddit --skip-conviction --skip-macro``.
                   No Sonnet calls. ~50s/run. Recommended for month-1
                   data collection.
* ``standard``  -- ``--skip-conviction``. Adds Haiku news sentiment + Sonnet
                   fundamental cascade + macro layer. ~2-5 min/run; ~$0.05/day.
* ``full``      -- everything including the conviction pass. ~hours/run; not
                   recommended for daily.

Single-day crashes log and continue -- the loop survives one bad day.
Logs go to stdout AND ``daily_loop_logs/loop_{date}.log``.

Smoke test before committing to the loop:

```
.venv/bin/python -m weekly_strategy.scripts.daily_loop --once --mode cheap
```

### B. Local cron

Edit your crontab (`crontab -e`):

```
# Daily 4:30pm ET (after US market close) -- adjust TZ
30 16 * * 1-5 cd /mnt/thdcan01/data/cfan/projects/202605_equityReports && \
    .venv/bin/python -m weekly_strategy.run_stage3 \
        --skip-news --skip-reddit --skip-conviction --skip-macro \
        --week-ending $(date -u +\%F) >> daily.log 2>&1
```

### C. Claude Code `/schedule`

Claude Code can schedule recurring runs via `/schedule`. Run that slash
command at the prompt and configure a daily routine that invokes the
same command above. (`/schedule` is a user-triggered skill; I can't set
it up for you, but it's a one-liner once you invoke it.)

### D. Manual

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
