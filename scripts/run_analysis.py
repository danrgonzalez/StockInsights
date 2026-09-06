#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis Runner -- the one entry point that regenerates every output.

Each run rebuilds everything downstream of ``data/StockData.xlsx`` and the
newest quotes file, in this order:

    1. index         scripts/indexer.py        -> data/StockData_Indexed.xlsx
    2. export        export_dashboard_data.py  -> data/exports/*.json, *.csv
    3. acquisitions  acquisition_profile.py    -> data/exports/acquisition_profiles.csv
    4. earnings      fit_earnings_dates.py     -> report only (accuracy of the
                                                 modelled dates), logged
    5. backtest      core.backtesting          -> data/exports/backtest_*.csv and
                                                 config/ticker_strategy_mapping.json
    6. manifest                                -> data/exports/manifest.json

Every step's console output is saved under ``data/exports/logs/`` and the
manifest records what ran, from which inputs (with hashes), and how long it
took. Nothing here is hand-edited: delete ``data/exports/`` and re-run to get
it all back.

The dashboard is launched at the end unless ``--no-dashboard`` is given.

Usage:
    python scripts/run_analysis.py                     # everything, then dashboard
    python scripts/run_analysis.py --no-dashboard      # regenerate outputs only
    python scripts/run_analysis.py --skip-backtest     # skip the slowest step
    python scripts/run_analysis.py --fetch-earnings-dates
                                                       # also refresh the SEC
                                                       # registry (network)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Every script here resolves data paths against the repo root; the indexer
# still uses relative ones, so run from there regardless of where we started.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from core.enums import FilePaths  # noqa: E402

RAW_FILE = Path(FilePaths.RAW_DATA_FILE)
INDEXED_FILE = Path(FilePaths.DATA_FILE)
MAPPING_FILE = Path(FilePaths.STRATEGY_MAPPING_FILE)
REGISTRY_FILE = Path(FilePaths.CONFIG_DIR) / "earnings_dates.json"
STATUS_FILE = Path(FilePaths.CONFIG_DIR) / "ticker_status.json"

REQUIRED_PACKAGES = ["pandas", "numpy", "plotly", "streamlit", "openpyxl"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    """Path, size, mtime and hash -- enough to tell whether an input changed."""
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.exists():
        return {"path": str(path.relative_to(REPO_ROOT)), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "sha256": sha256(path),
    }


def git_state() -> dict:
    """Commit and dirty flag, so an output can be traced to the code that made it."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def check_dependencies() -> bool:
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"ERROR: missing packages: {', '.join(missing)}")
        print("Activate the stockinsights env first:  source setup_env.sh")
        return False
    return True


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


class Pipeline:
    """Runs named steps, saves each one's output, and collects the manifest."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_dir = output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict] = []
        self.started = utc_now()

    def run_script(
        self, name: str, script: str, args: list[str], expect: list[Path] = ()
    ) -> bool:
        """Run one of the scripts/ tools as a subprocess and log its output.

        ``expect`` lists files the step must have (re)written; a script that
        swallows its own exception and exits 0 -- the indexer does -- is caught
        by checking that they were touched after the step began.
        """
        banner(f"STEP {len(self.steps) + 1}: {name}")
        command = [sys.executable, script, *args]
        print("$", " ".join(command))
        start = time.time()
        log_path = self.log_dir / f"{name}.log"

        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout + (
            ("\n[stderr]\n" + result.stderr) if result.stderr else ""
        )
        log_path.write_text(output)

        ok = result.returncode == 0
        stale = [p for p in expect if not p.exists() or p.stat().st_mtime < start]
        if stale:
            ok = False
            print("ERROR: step did not produce " + ", ".join(str(p) for p in stale))

        self._finish(name, ok, start, log_path, self._tail(output))
        return ok

    def run_callable(self, name: str, func) -> tuple[bool, object]:
        """Run an in-process step with stdout captured to its log file."""
        banner(f"STEP {len(self.steps) + 1}: {name}")
        start = time.time()
        log_path = self.log_dir / f"{name}.log"
        value, ok, error = None, True, None
        with open(log_path, "w") as log, contextlib.redirect_stdout(log):
            try:
                value = func()
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                ok, error = False, f"{type(exc).__name__}: {exc}"
                print(error)
        self._finish(name, ok, start, log_path, error or "")
        return ok, value

    def _finish(self, name, ok, start, log_path, tail):
        seconds = round(time.time() - start, 1)
        self.steps.append(
            {
                "step": name,
                "status": "ok" if ok else "failed",
                "seconds": seconds,
                "log": str(log_path.relative_to(REPO_ROOT)),
            }
        )
        print(f"{'OK' if ok else 'FAILED'} in {seconds}s  (log: {log_path})")
        if not ok and tail:
            print(tail)

    @staticmethod
    def _tail(text: str, lines: int = 15) -> str:
        return "\n".join(text.strip().splitlines()[-lines:])

    @property
    def failed(self) -> list[str]:
        return [s["step"] for s in self.steps if s["status"] != "ok"]


# ---------------------------------------------------------------------------
# Steps that need more than a subprocess call
# ---------------------------------------------------------------------------


def price_overrides(output_dir: Path) -> dict:
    """Which latest-row prices the indexer replaced from the quotes file.

    The indexer prints this and then it is gone, yet it changes every latest-
    quarter ratio. Recomputed here from the two workbooks rather than parsed
    from the log, so it stays correct if the indexer's wording changes.
    """
    import pandas as pd

    raw = pd.read_excel(RAW_FILE)
    indexed = pd.read_excel(INDEXED_FILE)
    raw = raw.drop(columns=[c for c in raw.columns if "Unnamed:" in str(c)])

    latest_pos = indexed.groupby("Ticker")["Index"].idxmax()
    rows = []
    for ticker, pos in latest_pos.items():
        old, new = raw.loc[pos, "Price"], indexed.loc[pos, "Price"]
        changed = not (pd.isna(old) and pd.isna(new)) and old != new
        rows.append(
            {
                "Ticker": ticker,
                "Report": indexed.loc[pos, "Report"],
                "WorkbookPrice": old,
                "IndexedPrice": new,
                "Overridden": bool(changed),
                "ChangePct": (
                    round((new - old) / old * 100, 2)
                    if changed and old not in (0, None) and not pd.isna(old)
                    else None
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("Ticker")
    path = output_dir / "price_overrides.csv"
    frame.to_csv(path, index=False)

    not_overridden = sorted(frame.loc[~frame["Overridden"], "Ticker"])
    print(f"{int(frame['Overridden'].sum())} of {len(frame)} latest prices overridden")
    if not_overridden:
        print("kept workbook price:", ", ".join(not_overridden))
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "overridden": int(frame["Overridden"].sum()),
        "kept_workbook_price": not_overridden,
    }


def load_stats() -> dict:
    """What the loader drops on the way in: duplicates and excluded tickers."""
    from core.data_processing import load_stock_data_with_stats

    df, stats = load_stock_data_with_stats(FilePaths.DATA_FILE)
    if df is None:
        raise RuntimeError(f"could not load {FilePaths.DATA_FILE}: {stats['error']}")
    everything, _ = load_stock_data_with_stats(
        FilePaths.DATA_FILE, include_excluded=True
    )
    result = {
        "rows_total": int(len(everything)),
        "tickers_total": int(everything["Ticker"].nunique()),
        "rows_active": int(len(df)),
        "tickers_active": int(df["Ticker"].nunique()),
        "duplicates_dropped": int(stats["duplicates_dropped"]),
        "excluded": {k: int(v) for k, v in stats["excluded"].items()},
    }
    print(json.dumps(result, indent=2))
    return result


def backtest(output_dir: Path) -> dict:
    """Score every strategy on every ticker and keep all of it.

    The backtest's own ``__main__`` keeps only the winner per ticker and pulls
    in streamlit through the dashboard loader. This writes the per-strategy
    scores and the per-period detail as well, and still refreshes the mapping
    file, which stays a research artifact predictions do not read.
    """
    import pandas as pd

    from core.backtesting import run_multi_ticker_backtest, save_ticker_strategy_mapping
    from core.data_processing import calculate_qoq_changes, load_stock_data

    df = calculate_qoq_changes(load_stock_data(FilePaths.DATA_FILE))
    results, best = run_multi_ticker_backtest(df)

    summary_rows, detail_frames = [], []
    for ticker, by_strategy in results.items():
        for strategy, res in by_strategy.items():
            summary_rows.append(
                {
                    "Ticker": ticker,
                    "Strategy": strategy,
                    "IsBest": strategy == best.get(ticker),
                    **{k: v for k, v in res.items() if k != "results_detail"},
                }
            )
            detail = res["results_detail"].copy()
            detail.insert(0, "Strategy", strategy)
            detail.insert(0, "Ticker", ticker)
            detail_frames.append(detail)

    summary_path = output_dir / "backtest_results.csv"
    detail_path = output_dir / "backtest_detail.csv"
    pd.DataFrame(summary_rows).sort_values(["Ticker", "accuracy_score"]).to_csv(
        summary_path, index=False
    )
    pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False)
    save_ticker_strategy_mapping(best)

    return {
        "tickers_tested": len(results),
        "results": str(summary_path.relative_to(REPO_ROOT)),
        "detail": str(detail_path.relative_to(REPO_ROOT)),
        "mapping": str(MAPPING_FILE.relative_to(REPO_ROOT)),
    }


def launch_dashboard() -> None:
    banner("Launching Streamlit dashboard")
    print("Press Ctrl+C to stop it. URL is usually http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate every StockInsights output, then launch the dashboard."
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports",
        help="Where the exports, logs and manifest go (default: data/exports)",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Regenerate outputs and stop."
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the strategy backtest, the slowest step.",
    )
    parser.add_argument(
        "--fetch-earnings-dates",
        action="store_true",
        help="Refresh config/earnings_dates.json from SEC EDGAR first (needs network).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()

    banner("StockInsights pipeline")
    print(f"Repo   : {REPO_ROOT}")
    print(f"Python : {sys.executable}")
    print(f"Output : {output_dir}")

    if not check_dependencies():
        return 1
    if not RAW_FILE.exists():
        print(f"ERROR: {RAW_FILE} not found. See README.md, 'Data'.")
        return 1

    from scripts.indexer import get_latest_csv_file

    quotes_file = get_latest_csv_file()
    if quotes_file is None:
        print("WARNING: no data/quotes/*-Quote.csv; prices stay as in the workbook")

    pipe = Pipeline(output_dir)
    manifest: dict = {
        "generated_at_utc": pipe.started,
        "generated_by": "scripts/run_analysis.py",
        "git": git_state(),
        "python": platform.python_version(),
        "inputs": {
            "raw_workbook": file_record(RAW_FILE),
            "quotes_file": file_record(Path(quotes_file)) if quotes_file else None,
            "ticker_status": file_record(STATUS_FILE),
            "earnings_dates_registry": file_record(REGISTRY_FILE),
        },
        "outputs": {},
        "steps": pipe.steps,
    }

    # 1. Index. Everything else reads its output, so a failure here stops the run.
    if not pipe.run_script("index", "scripts/indexer.py", [], expect=[INDEXED_FILE]):
        print("\nIndexer failed; nothing downstream can run.")
        return 1
    manifest["outputs"]["indexed_workbook"] = file_record(INDEXED_FILE)

    ok, overrides = pipe.run_callable(
        "price_overrides", lambda: price_overrides(output_dir)
    )
    manifest["outputs"]["price_overrides"] = overrides if ok else None

    ok, stats = pipe.run_callable("load_stats", load_stats)
    manifest["universe"] = stats if ok else None

    # 2. The dashboard's numbers, on disk.
    pipe.run_script(
        "export",
        "scripts/export_dashboard_data.py",
        ["--output-dir", str(output_dir)],
        expect=[output_dir / "stock_analysis_export.json"],
    )

    # 3. Acquired tickers, aligned on their announcement dates.
    pipe.run_script(
        "acquisitions",
        "scripts/acquisition_profile.py",
        ["--output-dir", str(output_dir)],
        expect=[output_dir / "acquisition_profiles.csv"],
    )

    # 4. Earnings dates: optionally refresh the SEC registry, always report how
    #    accurate the modelled remainder is.
    if args.fetch_earnings_dates:
        pipe.run_script(
            "fetch_earnings_dates",
            "scripts/fetch_earnings_dates.py",
            [],
            expect=[REGISTRY_FILE],
        )
        manifest["inputs"]["earnings_dates_registry"] = file_record(REGISTRY_FILE)
    pipe.run_script("earnings_dates_fit", "scripts/fit_earnings_dates.py", [])

    # 5. Strategy backtest, every score kept.
    if args.skip_backtest:
        pipe.steps.append(
            {"step": "backtest", "status": "skipped", "seconds": 0, "log": None}
        )
    else:
        ok, result = pipe.run_callable("backtest", lambda: backtest(output_dir))
        manifest["outputs"]["backtest"] = result if ok else None

    # 6. Manifest.
    export_files = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in output_dir.iterdir()
        if p.is_file() and p.name != "manifest.json"
    )
    manifest["outputs"]["export_files"] = export_files
    manifest["finished_at_utc"] = utc_now()
    manifest["status"] = "ok" if not pipe.failed else "failed"
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    banner("Summary")
    for step in pipe.steps:
        print(f"  {step['step']:<22} {step['status']:<8} {step['seconds']:>6}s")
    print(f"\nManifest: {manifest_path}")

    if pipe.failed:
        print(f"\nFAILED steps: {', '.join(pipe.failed)}. Dashboard not launched.")
        return 1

    if not args.no_dashboard:
        launch_dashboard()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
