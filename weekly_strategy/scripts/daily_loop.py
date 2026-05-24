from __future__ import annotations

"""Long-running daily prediction loop.

Designed to be launched inside a ``screen`` (or ``tmux``) session so it
survives ssh disconnects:

    screen -S daily-loop
    python -m weekly_strategy.scripts.daily_loop

    # Detach: Ctrl-A then D
    # Re-attach: screen -r daily-loop

The loop sleeps until the configured time-of-day (default 21:30 UTC =
5:30 PM ET, after US close), then invokes ``weekly_strategy.run_stage3``
as a subprocess. Mode flags control how much of the LLM cascade fires:

    --mode cheap     (default) skip news + reddit + conviction + macro
    --mode standard  include news cascade (Haiku + Sonnet pass 2)
    --mode full      everything (news + reddit + macro + conviction)

Each run writes:
    daily_reports/{YYYY-MM-DD}.md
    daily_reports/{YYYY-MM-DD}.html
    weekly_strategy/data_cache/predictions/{YYYY-MM-DD}/...

Logs to stdout AND ``daily_loop_logs/loop_{YYYY-MM-DD}.log``. Single-day
crashes don't kill the loop -- they log and continue to the next slot.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "daily_loop_logs"

DEFAULT_TIME_UTC = "21:30"          # 5:30 PM ET, after US close
_ET = ZoneInfo("America/New_York")  # DST-aware US Eastern

MODE_FLAGS: dict[str, list[str]] = {
    # Cheap: no Sonnet calls except macro regime synthesis if --skip-macro
    # is NOT passed. Default skips everything LLM-bound.
    "cheap":    ["--skip-news", "--skip-reddit", "--skip-conviction", "--skip-macro"],
    # Standard: news cascade (Haiku per-headline sentiment + Sonnet per-batch
    # fundamental). Macro layer included.
    "standard": ["--skip-conviction"],
    # Full: everything including the adversarial conviction pass.
    "full":     [],
}


def _parse_hhmm(s: str) -> tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected HH:MM, got {s!r}")
    return int(parts[0]), int(parts[1])


def _next_run_time(now: datetime, hh: int, mm: int) -> datetime:
    """Next occurrence of HH:MM UTC at or strictly after ``now``."""
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _next_run_time_et(now: datetime, hh: int, mm: int) -> datetime:
    """Next occurrence of HH:MM US/Eastern at or strictly after ``now``.

    Returns a UTC datetime, but the wall-clock target in ET is fixed --
    so DST transitions are handled transparently. The actual UTC offset
    will swing between -5h and -4h across the year.
    """
    now_et = now.astimezone(_ET)
    candidate_et = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate_et <= now_et:
        candidate_et += timedelta(days=1)
    return candidate_et.astimezone(timezone.utc)


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daily_loop")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / f"loop_{date.today().isoformat()}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _python_bin() -> str:
    """Use the same Python interpreter that's running this script.

    If we're running under ``.venv/bin/python -m weekly_strategy.scripts.daily_loop``
    then sys.executable points at .venv/bin/python; subprocess inherits it.
    """
    return sys.executable


def run_once(*, run_date: date, mode: str, logger: logging.Logger) -> int:
    """Invoke run_stage3 as a subprocess. Returns the process return code."""
    flags = MODE_FLAGS.get(mode)
    if flags is None:
        logger.error("unknown mode %r; valid: %s", mode, sorted(MODE_FLAGS))
        return 2
    cmd = [
        _python_bin(), "-m", "weekly_strategy.run_stage3",
        *flags,
        "--week-ending", run_date.isoformat(),
    ]
    logger.info("invoking: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        # No timeout: full-mode runs can take 1-3 hours on 100 names.
    )
    elapsed = time.time() - t0
    # Last 2K of stdout is usually the summary block.
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    logger.info("rc=%d  elapsed=%.1fs\nstdout-tail:\n%s", proc.returncode, elapsed, tail)
    if proc.returncode != 0:
        err_tail = "\n".join(proc.stderr.splitlines()[-30:])
        logger.error("non-zero exit; stderr-tail:\n%s", err_tail)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Daily prediction loop daemon. Run inside a screen session.",
    )
    parser.add_argument(
        "--time-utc", default=None,
        help="Time of day to fire, HH:MM UTC. Mutually exclusive with --time-et. "
             "Default when neither is set: 21:30 UTC (5:30 PM ET DST / 4:30 PM ET ST).",
    )
    parser.add_argument(
        "--time-et", default=None,
        help="Time of day to fire, HH:MM in US/Eastern (DST-aware). "
             "Example: 08:30 = 8:30 AM ET year-round; UTC offset auto-adjusts "
             "with daylight saving. Mutually exclusive with --time-utc.",
    )
    parser.add_argument(
        "--mode", choices=sorted(MODE_FLAGS), default="cheap",
        help="LLM coverage. 'cheap' (default) skips all expensive LLM passes "
             "and just produces ranked composite scores from dossier + price "
             "data. 'standard' adds news cascade. 'full' adds conviction.",
    )
    parser.add_argument(
        "--weekdays-only", action="store_true",
        help="Only run Mon-Fri (US trading days; skip Sat/Sun).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once immediately and exit (smoke test).",
    )
    args = parser.parse_args(argv)

    if args.time_utc and args.time_et:
        parser.error("--time-utc and --time-et are mutually exclusive")

    logger = _setup_logging()

    use_et = args.time_et is not None
    if use_et:
        hh, mm = _parse_hhmm(args.time_et)
        tz_label = f"{hh:02d}:{mm:02d} ET (DST-aware)"
    else:
        hh, mm = _parse_hhmm(args.time_utc or DEFAULT_TIME_UTC)
        tz_label = f"{hh:02d}:{mm:02d} UTC"

    if args.once:
        logger.info("--once: running immediately, mode=%s", args.mode)
        return run_once(run_date=date.today(), mode=args.mode, logger=logger)

    logger.info(
        "daily loop started; time=%s, mode=%s, weekdays_only=%s, pid=%d",
        tz_label, args.mode, args.weekdays_only, os.getpid(),
    )

    while True:
        now = datetime.now(timezone.utc)
        target = (
            _next_run_time_et(now, hh, mm) if use_et
            else _next_run_time(now, hh, mm)
        )
        wait_s = (target - now).total_seconds()
        target_et = target.astimezone(_ET)
        logger.info(
            "next run at %s UTC / %s ET  (sleep %.0fs / %.1fh)",
            target.isoformat(timespec="seconds"),
            target_et.isoformat(timespec="seconds"),
            wait_s, wait_s / 3600,
        )
        try:
            time.sleep(wait_s)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt; exiting cleanly")
            return 0

        # Use the ET *wall-clock* date for both the run-date and the
        # weekend check. Otherwise running at 8:30 AM ET Sunday could be
        # 12:30 UTC Sunday and still skipped -- which is what we want --
        # but running at 9:00 PM ET Friday would be 01:00 UTC Saturday,
        # falsely flagged as "weekend".
        run_date = target_et.date()
        if args.weekdays_only and target_et.weekday() >= 5:
            logger.info("weekend (ET dow=%d); skipping", target_et.weekday())
            continue

        try:
            rc = run_once(run_date=run_date, mode=args.mode, logger=logger)
            logger.info("run for %s done (rc=%d)", run_date, rc)
        except Exception:
            logger.exception("run crashed; loop continues")


if __name__ == "__main__":
    raise SystemExit(main())
