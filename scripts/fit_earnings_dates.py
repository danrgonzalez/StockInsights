#!/usr/bin/env python3
"""
Fill earnings dates that SEC EDGAR cannot supply, and report how good the fit is.

``scripts/fetch_earnings_dates.py`` covers ~99% of the panel from real filings.
The rest genuinely cannot be sourced -- quarters before a company's IPO, or
either side of a reorganisation -- so they are modelled here instead.

The model is deliberately simple, because it only has to interpolate a reporting
cadence that barely moves: for each ticker, a straight line through the real
dates against the row index, plus a per-fiscal-quarter offset. Companies report
on a stable rhythm, so a linear fit with seasonal offsets captures nearly all of
it.

Accuracy is measured by leave-one-out on the real dates: hold each one out, fit
on the rest, predict it back. That is the honest estimate of how wrong a
*modelled* date is, and it is reported every run rather than being a claim made
once and never rechecked.

Usage:
    python scripts/fit_earnings_dates.py                 # report only
    python scripts/fit_earnings_dates.py --apply         # write to the workbook
    python scripts/fit_earnings_dates.py --apply --output data/StockData.xlsx
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from core.enums import Column, FilePaths  # noqa: E402

REGISTRY = REPO_ROOT / "config" / "earnings_dates.json"

# Quarters per year, in days. A calendar year is 365.25 days, so a quarterly
# reporting cadence averages this. Used only as a fallback slope when a ticker
# has too few real dates to fit one.
DAYS_PER_QUARTER = 365.25 / 4


def find_date_column(panel: pd.DataFrame) -> str:
    """The earnings-date column, under either name it goes by.

    The raw workbook heads it "Earnings Report Date"; the indexer renames it to
    "EarningsDate". This script is useful against both, so it detects rather
    than assumes, and writes back under whatever name it found.
    """
    if Column.EARNINGS_DATE.value in panel.columns:
        return Column.EARNINGS_DATE.value
    for col in panel.columns:
        name = str(col).lower()
        if "earning" in name and "date" in name:
            return col
    raise SystemExit(
        f"No earnings-date column in {list(panel.columns)}. "
        "Expected 'EarningsDate' or 'Earnings Report Date'."
    )


def load_registry() -> dict:
    if not REGISTRY.exists():
        raise SystemExit(
            f"No registry at {REGISTRY}. Run scripts/fetch_earnings_dates.py first."
        )
    return json.loads(REGISTRY.read_text()).get("tickers", {})


def fiscal_quarter(report: str) -> int:
    """The 1-4 quarter number from a label like Q3'25."""
    try:
        return int(str(report)[1])
    except (IndexError, ValueError):
        return 0


def fit_ticker(known: pd.DataFrame) -> tuple[float, float, dict]:
    """Fit date = intercept + slope * index, plus per-quarter offsets.

    Args:
        known: rows with an ``index``, ``quarter`` and ordinal ``y``

    Returns:
        (intercept, slope, {quarter: offset})
    """
    x = known["index"].to_numpy(dtype=float)
    y = known["y"].to_numpy(dtype=float)

    if len(known) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        # Not enough spread to fit a slope; anchor on the single point.
        slope = DAYS_PER_QUARTER
        intercept = y[0] - slope * x[0]

    residual = y - (intercept + slope * x)
    offsets = {}
    for q in (1, 2, 3, 4):
        mask = known["quarter"].to_numpy() == q
        if mask.any():
            offsets[q] = float(residual[mask].mean())
    return float(intercept), float(slope), offsets


def predict(intercept: float, slope: float, offsets: dict, index, quarter) -> float:
    return intercept + slope * index + offsets.get(quarter, 0.0)


def leave_one_out(known: pd.DataFrame) -> list[float]:
    """Absolute error in days from predicting each real date without itself."""
    errors = []
    if len(known) < 4:
        return errors
    for i in range(len(known)):
        held = known.iloc[i]
        rest = known.drop(known.index[i])
        if rest.empty:
            continue
        intercept, slope, offsets = fit_ticker(rest)
        pred = predict(intercept, slope, offsets, held["index"], held["quarter"])
        errors.append(abs(pred - held["y"]))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=FilePaths.DATA_FILE)
    parser.add_argument(
        "--apply", action="store_true", help="write the merged dates back"
    )
    parser.add_argument("--output", help="where to write (default: overwrite --source)")
    args = parser.parse_args()

    registry = load_registry()
    panel = pd.read_excel(args.source)
    date_col = find_date_column(panel)
    ticker_col, report_col = Column.TICKER.value, Column.REPORT.value
    index_col = Column.INDEX.value if Column.INDEX.value in panel.columns else None
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")

    resolved, modelled, all_errors, per_ticker = {}, {}, [], []

    for ticker, rows in panel.groupby(ticker_col, sort=False):
        rows = rows.sort_values(index_col) if index_col else rows
        real = registry.get(ticker, {})

        frame = pd.DataFrame(
            {
                "row": rows.index,
                "report": rows[report_col].astype(str),
                "index": (
                    rows[index_col].to_numpy(dtype=float)
                    if index_col
                    else np.arange(len(rows), dtype=float)
                ),
            }
        )
        frame["quarter"] = frame["report"].map(fiscal_quarter)
        frame["real"] = frame["report"].map(
            lambda r: pd.Timestamp(real[r].split("|")[0]) if r in real else pd.NaT
        )

        known = frame.dropna(subset=["real"]).copy()
        known["y"] = known["real"].map(pd.Timestamp.toordinal).astype(float)

        for _, r in known.iterrows():
            resolved[r["row"]] = r["real"]

        missing = frame[frame["real"].isna()]
        if len(known) >= 2 and not missing.empty:
            intercept, slope, offsets = fit_ticker(known)
            for _, r in missing.iterrows():
                ordinal = predict(intercept, slope, offsets, r["index"], r["quarter"])
                modelled[r["row"]] = pd.Timestamp.fromordinal(int(round(ordinal)))

        errors = leave_one_out(known)
        if errors:
            all_errors += errors
            per_ticker.append((ticker, len(known), float(np.median(errors))))

    print("=" * 70)
    print("EARNINGS DATE COVERAGE")
    print("=" * 70)
    total = len(panel)
    print(f"  rows in panel            : {total:,}")
    real_pct = len(resolved) / total * 100
    model_pct = len(modelled) / total * 100
    print(f"  real, from SEC filings   : {len(resolved):,} ({real_pct:.1f}%)")
    print(f"  modelled by this fit     : {len(modelled):,} ({model_pct:.1f}%)")
    unfilled = total - len(resolved) - len(modelled)
    print(f"  still unfilled           : {unfilled:,}")

    if all_errors:
        e = pd.Series(all_errors)
        print()
        print("=" * 70)
        print("LEAVE-ONE-OUT ACCURACY OF THE FIT (days)")
        print("=" * 70)
        print(f"  observations : {len(e):,}")
        print(f"  median       : {e.median():.1f}")
        print(f"  mean         : {e.mean():.1f}")
        print(f"  p90          : {e.quantile(0.90):.1f}")
        print(f"  within 7d    : {(e <= 7).mean()*100:.1f}%")
        print(f"  within 14d   : {(e <= 14).mean()*100:.1f}%")
        worst = sorted(per_ticker, key=lambda t: -t[2])[:5]
        print("  least predictable tickers:")
        for ticker, n, med in worst:
            print(f"    {ticker:6s} median {med:5.1f}d over {n} real dates")

    if not args.apply:
        print("\n(report only — pass --apply to write these dates back)")
        return

    changed = 0
    for row, value in {**resolved, **modelled}.items():
        if panel.at[row, date_col] != value:
            changed += 1
        panel.at[row, date_col] = value

    out = args.output or args.source
    panel.to_excel(out, index=False)
    print(f"\nwrote {out} — {changed:,} dates changed")


if __name__ == "__main__":
    main()
