#!/usr/bin/env python3
"""
Pre-acquisition profiles: what the numbers looked like before a buyout.

Acquired tickers are held out of the active analysis universe but kept in the
data, so their final quarters can be lined up on the announcement date and
compared across deals.

Usage:
    python scripts/acquisition_profile.py
    python scripts/acquisition_profile.py --metric Multiple
    python scripts/acquisition_profile.py --ticker EA
    python scripts/acquisition_profile.py --output-dir data/exports
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from core.acquisitions import (  # noqa: E402
    PROFILE_METRICS,
    QUARTER_OFFSET,
    aligned_metric,
    all_profiles,
    build_profile,
)
from core.data_processing import (  # noqa: E402
    calculate_qoq_changes,
    load_stock_data,
)
from core.enums import Column, FilePaths  # noqa: E402
from core.ticker_status import get_acquired_tickers  # noqa: E402

RULE = "=" * 78


def load_panel(source: str) -> pd.DataFrame:
    """Load with acquired tickers included, then derive metrics."""
    df = load_stock_data(source, include_excluded=True)
    if df is None:
        raise SystemExit(f"Could not load {source}")
    return calculate_qoq_changes(df)


def print_summary(panel: pd.DataFrame) -> None:
    profiles = all_profiles(panel)
    if profiles.empty:
        print("No acquired tickers configured.")
        return

    print(RULE)
    print("ACQUISITIONS")
    print(RULE)
    for _, row in profiles.iterrows():
        print(f"\n{row['ticker']} — {row['acquirer']}")
        print(f"  type          : {row['type']}")
        print(f"  announced     : {row['announced']}")
        print(f"  completed     : {row['completed']}")
        if row["price_per_share"]:
            print(f"  price/share   : ${row['price_per_share']:,.2f}")
        if row["deal_value_usd_bn"]:
            print(f"  deal value    : ${row['deal_value_usd_bn']:,.2f}B")
        print(
            f"  history       : {row['quarters_before_announcement']} quarters "
            f"to announcement, {row['quarters_while_pending']} while pending"
        )
        last_price = row["price_at_last_report"]
        if last_price:
            print(
                f"  last report   : {row['last_report_before_announcement']} "
                f"at ${last_price:,.2f}"
            )
        gap = row["deal_price_vs_last_report_pct"]
        if gap is not None:
            print(
                f"  deal price vs that report: {gap:+.1f}%  "
                "(not the announced premium — the report can be a quarter old)"
            )


def print_runup(panel: pd.DataFrame) -> None:
    profiles = all_profiles(panel)
    cols = [c for c in profiles.columns if c.endswith(("_mean_4q", "_mean_8q"))]
    if not cols:
        return
    print()
    print(RULE)
    print("RUN-UP — mean QoQ % over the quarters into the announcement")
    print(RULE)
    print(profiles[["ticker"] + cols].to_string(index=False))


def print_aligned(panel: pd.DataFrame, metric: str, lo: int, hi: int) -> None:
    frame = aligned_metric(panel, metric)
    if frame.empty:
        print(f"\nNo data for metric {metric!r}.")
        return
    window = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
    print()
    print(RULE)
    print(f"{metric.upper()} aligned on quarters to announcement")
    print("(0 = last report before the announcement; positive = deal pending)")
    print(RULE)
    print(window.round(2).to_string())


def print_ticker(panel: pd.DataFrame, ticker: str) -> None:
    profile = build_profile(panel, ticker)
    if profile is None:
        acquired = ", ".join(sorted(get_acquired_tickers())) or "none configured"
        raise SystemExit(f"{ticker} is not an acquired ticker. Available: {acquired}")

    cols = [Column.REPORT.value, Column.EARNINGS_DATE.value, QUARTER_OFFSET]
    cols += [m for m in PROFILE_METRICS if m in profile.columns]
    print(RULE)
    print(f"{ticker} — full quarterly history")
    print(RULE)
    print(profile[cols].round(2).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile what acquired companies looked like before the deal."
    )
    parser.add_argument("--source", default=FilePaths.DATA_FILE)
    parser.add_argument(
        "--metric",
        default="Multiple",
        help="Metric to align across deals (default: Multiple)",
    )
    parser.add_argument("--ticker", help="Show one ticker's full history instead")
    parser.add_argument("--from-quarter", type=int, default=-12)
    parser.add_argument("--to-quarter", type=int, default=4)
    parser.add_argument("--output-dir", help="Also write acquisition_profiles.csv here")
    args = parser.parse_args()

    panel = load_panel(args.source)

    if args.ticker:
        print_ticker(panel, args.ticker.upper())
        return

    print_summary(panel)
    print_runup(panel)
    print_aligned(panel, args.metric, args.from_quarter, args.to_quarter)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "acquisition_profiles.csv"
        all_profiles(panel).to_csv(path, index=False)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
