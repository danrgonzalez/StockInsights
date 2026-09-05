#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Data Exporter

Runs the exact same calculation pipeline the Streamlit dashboard runs against
data/StockData_Indexed.xlsx, then writes every resulting number to files that
are easy to read without the dashboard - and easy to hand to another agent.

Outputs (default directory: data/exports/):
    stock_analysis_export.json  Self-describing bundle: metadata, a field-by-field
                                data dictionary, a per-ticker snapshot (latest
                                values, QoQ summaries, rolling averages, peer ranks,
                                next-quarter prediction, data-quality flags) and
                                sector aggregates. Hand this one to an agent.
    quarterly_metrics.csv       Full panel - one row per ticker per quarter with
                                every base and derived column.
    ticker_snapshot.csv         Flattened one-row-per-ticker view of the JSON
                                snapshot.
    sector_summary.csv          Sector aggregates.
    README.md                   Short human/agent-readable guide to the above.

Usage:
    python scripts/export_dashboard_data.py
    python scripts/export_dashboard_data.py --output-dir data/exports --include-history
    python scripts/export_dashboard_data.py --input data/StockData_Indexed.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# core.backtesting resolves config/ticker_strategy_mapping.json relative to the
# working directory, so predictions silently change strategy if we run from
# anywhere else. Anchor to the repo root before importing anything from core.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from core.backtesting import load_ticker_strategy_mapping  # noqa: E402
from core.data_processing import (  # noqa: E402
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    load_stock_data,
    report_range,
)
from core.enums import (  # noqa: E402
    Column,
    DerivedMetric,
    FilePaths,
    Metric,
    RollingWindow,
    Strategy,
)
from core.predictions import predict_next_eps  # noqa: E402

try:
    from core.classifications import get_stock_classification

    CLASSIFICATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover - classifications ship with the repo
    CLASSIFICATIONS_AVAILABLE = False

# Metrics whose latest value the snapshot reports (dashboard summary cards +
# everything else calculate_qoq_changes produces).
LATEST_VALUE_METRICS = [
    Metric.EPS.value,
    Metric.REVENUE.value,
    Metric.PRICE.value,
    Metric.DIVIDEND_AMOUNT.value,
    DerivedMetric.EPS_TTM.value,
    DerivedMetric.REVENUE_TTM.value,
    DerivedMetric.MULTIPLE.value,
    DerivedMetric.DIVIDEND_YIELD.value,
    DerivedMetric.DIVIDEND_YIELD_ANNUAL.value,
    DerivedMetric.PAYOUT_RATIO.value,
    DerivedMetric.PEG_RATIO.value,
    DerivedMetric.PEGY_RATIO.value,
    DerivedMetric.DIVIDEND_GROWTH_RATE.value,
    DerivedMetric.DIVIDEND_INCREASE_FREQ.value,
    DerivedMetric.AVG_DIVIDEND_INCREASE.value,
    DerivedMetric.EPS_MOMENTUM.value,
    DerivedMetric.PRICE_VOLATILITY.value,
    DerivedMetric.REVENUE_CONSISTENCY.value,
    DerivedMetric.DOWNSIDE_CAPTURE.value,
]

# Base metrics that actually get a *_QoQ series (see DerivedMetric.qoq_metrics).
QOQ_BASE_METRICS = DerivedMetric.qoq_metrics()

ROLLING_WINDOWS = {
    "4Q": RollingWindow.SHORT,
    "8Q": RollingWindow.LONG,
    "12Q": RollingWindow.EXTENDED,
}

# Metrics ranked against peers: the direction that counts as "better", and the
# unit, which decides how the gap to the peer average is expressed. A metric
# already measured in percent is compared as a difference in percentage points;
# a ratio against an average percentage explodes whenever that average sits near
# zero, which it routinely does for QoQ growth.
PEER_METRICS = {
    DerivedMetric.EPS_TTM.value: ("higher_is_better", "level"),
    DerivedMetric.REVENUE_TTM.value: ("higher_is_better", "level"),
    DerivedMetric.EPS_QOQ.value: ("higher_is_better", "percent"),
    DerivedMetric.REVENUE_QOQ.value: ("higher_is_better", "percent"),
    DerivedMetric.PRICE_QOQ.value: ("higher_is_better", "percent"),
    DerivedMetric.EPS_TTM_QOQ.value: ("higher_is_better", "percent"),
    DerivedMetric.REVENUE_TTM_QOQ.value: ("higher_is_better", "percent"),
    DerivedMetric.DIVIDEND_YIELD_ANNUAL.value: ("higher_is_better", "percent"),
    DerivedMetric.DIVIDEND_GROWTH_RATE.value: ("higher_is_better", "percent"),
    DerivedMetric.EPS_MOMENTUM.value: ("higher_is_better", "percent"),
    DerivedMetric.MULTIPLE.value: ("lower_is_better", "level"),
    DerivedMetric.PEG_RATIO.value: ("lower_is_better", "level"),
    DerivedMetric.PEGY_RATIO.value: ("lower_is_better", "level"),
    DerivedMetric.PRICE_VOLATILITY.value: ("lower_is_better", "percent"),
    DerivedMetric.DOWNSIDE_CAPTURE.value: ("lower_is_better", "percent"),
}

# Valuation metrics are undefined when earnings are negative: a P/E of -50 is
# not cheaper than a P/E of 12. Non-positive values are left out of these ranks
# entirely rather than sorted to the top.
POSITIVE_ONLY_METRICS = {
    DerivedMetric.MULTIPLE.value,
    DerivedMetric.PEG_RATIO.value,
    DerivedMetric.PEGY_RATIO.value,
}

# Metrics the dashboard's Sector Analysis table averages per sector.
SECTOR_AVG_METRICS = [
    Metric.EPS.value,
    Metric.REVENUE.value,
    DerivedMetric.EPS_TTM.value,
    DerivedMetric.REVENUE_TTM.value,
    Metric.PRICE.value,
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_panel(input_path: str) -> pd.DataFrame:
    """Load the indexed workbook and run the dashboard's calculation pipeline."""
    df = load_stock_data(input_path)
    if df is None:
        raise SystemExit(f"ERROR: could not load {input_path}")

    df = add_classifications(df)

    # Same order as dashboard/app.py main()
    df = calculate_qoq_changes(df)
    df = calculate_sector_rankings(df)
    df = calculate_outperformance_ratios(df)
    df = calculate_downside_capture(df)

    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value
    return df.sort_values([ticker_col, index_col]).reset_index(drop=True)


def add_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Sector/Industry/Sub_Industry so sector-relative metrics resolve.

    The dashboard looks classifications up per ticker inside its Rolling
    Averages tab but never merges them into the main frame, which leaves its
    sector rankings and sector outperformance columns empty. Merging them here
    means those columns are populated in the export.
    """
    ticker_col = Column.TICKER.value
    unknown = "Unclassified"

    sectors, industries, sub_industries = [], [], []
    for ticker in df[ticker_col]:
        classification = (
            get_stock_classification(ticker) if CLASSIFICATIONS_AVAILABLE else None
        )
        if classification is None:
            sectors.append(unknown)
            industries.append(unknown)
            sub_industries.append(unknown)
        else:
            sectors.append(classification.sector.value)
            industries.append(classification.industry.value)
            sub_industries.append(classification.sub_industry.value)

    df = df.copy()
    df[Column.SECTOR.value] = sectors
    df[Column.INDUSTRY.value] = industries
    df[Column.SUB_INDUSTRY.value] = sub_industries
    return df


# ---------------------------------------------------------------------------
# Per-ticker snapshot
# ---------------------------------------------------------------------------


# A latest value more than this many quarters behind the ticker's newest
# quarter is treated as stale: reported, but kept out of peer ranking.
MAX_STALE_QUARTERS = 1


def latest_valid(data: pd.DataFrame, metric: str, latest_index) -> tuple:
    """Last non-null value of a metric, and how many quarters behind it is.

    Matches the dashboard's .dropna().iloc[-1], which reaches back through blank
    quarters - so a company that stopped paying dividends six years ago still
    reports a dividend yield. The age is returned alongside so the caller can
    tell a current value from a fossil.
    """
    valid = data[[Column.INDEX.value, metric]].dropna()
    if valid.empty:
        return None, None
    value = clean(valid[metric].iloc[-1])
    age = int(latest_index - valid[Column.INDEX.value].iloc[-1])
    return value, age


def clean(value):
    """Convert numpy/pandas scalars to JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else value.strftime("%Y-%m-%d")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NaT or (isinstance(value, str) and value == "nan"):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_ticker_snapshot(
    panel: pd.DataFrame, strategy_mapping: dict | None
) -> list[dict]:
    """One record per ticker with every number the dashboard surfaces."""
    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value
    report_col = Column.REPORT.value
    earnings_col = Column.EARNINGS_DATE.value

    records = []
    for ticker in sorted(panel[ticker_col].unique()):
        data = panel[panel[ticker_col] == ticker].sort_values(index_col)
        last_row = data.iloc[-1]

        record: dict = {
            "ticker": ticker,
            "classification": {
                "sector": clean(last_row.get(Column.SECTOR.value)),
                "industry": clean(last_row.get(Column.INDUSTRY.value)),
                "sub_industry": clean(last_row.get(Column.SUB_INDUSTRY.value)),
            },
            "coverage": {
                "quarters": int(len(data)),
                "first_report": clean(data[report_col].iloc[0]),
                "latest_report": clean(last_row[report_col]),
                "latest_index": clean(last_row[index_col]),
                "latest_earnings_date": clean(last_row.get(earnings_col)),
            },
            "latest": {},
            "latest_stale_quarters": {},
            "qoq_summary": build_qoq_summary(data),
            "rolling_qoq_avg": build_rolling_averages(data),
            "peer_comparison": {},  # filled by attach_peer_comparison
            "prediction": build_prediction(panel, ticker, strategy_mapping),
        }
        for metric in LATEST_VALUE_METRICS:
            if metric not in data.columns:
                continue
            value, age = latest_valid(data, metric, last_row[index_col])
            if value is None:
                continue
            record["latest"][metric] = value
            if age > MAX_STALE_QUARTERS:
                record["latest_stale_quarters"][metric] = age

        record["data_flags"] = build_data_flags(record)
        records.append(record)

    attach_peer_comparison(records)
    return records


def build_data_flags(record: dict) -> dict:
    """Conditions that make some of this ticker's metrics unusable.

    Saves a downstream reader from having to know which formulas break on which
    inputs - the reasons are spelled out in `warnings`.
    """
    latest = record["latest"]
    eps_ttm = latest.get(DerivedMetric.EPS_TTM.value)
    div_amt = latest.get(Metric.DIVIDEND_AMOUNT.value)
    quarters = record["coverage"]["quarters"]

    negative_eps_ttm = eps_ttm is not None and eps_ttm <= 0
    pays_dividend = bool(div_amt)
    short_history = quarters < RollingWindow.EXTENDED

    warnings = []
    if negative_eps_ttm:
        warnings.append(
            "EPS_TTM is not positive, so Multiple, PEGRatio, PEGYRatio and "
            "PayoutRatio are meaningless for this ticker and it is left out of "
            "the valuation peer ranks."
        )
    if not pays_dividend:
        warnings.append(
            "No dividend in the latest quarter; dividend metrics are zero or null."
        )
    if short_history:
        warnings.append(
            f"Only {quarters} quarters of history, so 8Q and 12Q figures are "
            "computed over fewer periods than their label suggests."
        )
    stale = record.get("latest_stale_quarters", {})
    if stale:
        oldest = max(stale.values())
        warnings.append(
            f"{len(stale)} metric(s) carry a value from an earlier quarter "
            f"(up to {oldest} quarters back): "
            f"{', '.join(sorted(stale))}. They are reported but excluded from "
            "peer ranks. See latest_stale_quarters."
        )
    if record["prediction"] is None:
        warnings.append("No forecast: not enough usable EPS history for any strategy.")

    return {
        "negative_eps_ttm": negative_eps_ttm,
        "pays_dividend": pays_dividend,
        "short_history": short_history,
        "has_stale_metrics": bool(stale),
        "has_prediction": record["prediction"] is not None,
        "warnings": warnings,
    }


def build_qoq_summary(data: pd.DataFrame) -> dict:
    """Latest QoQ %, mean QoQ % and growth-period counts (dashboard tab 1)."""
    summary = {}
    for metric in QOQ_BASE_METRICS:
        qoq_col = f"{metric}_QoQ"
        if qoq_col not in data.columns:
            continue
        values = data[qoq_col].dropna()
        if values.empty:
            continue
        summary[metric] = {
            "latest_pct": clean(values.iloc[-1]),
            "average_pct": clean(values.mean()),
            "periods_positive": int((values > 0).sum()),
            "periods_total": int(len(values)),
        }
    return summary


def build_rolling_averages(data: pd.DataFrame) -> dict:
    """Latest 4Q/8Q/12Q rolling means of each QoQ series (dashboard tab 3).

    Uses min_periods=1, the same as the dashboard, so a ticker with fewer than
    12 quarters still reports a 12Q value computed over what it has.
    """
    averages = {}
    for metric in QOQ_BASE_METRICS:
        qoq_col = f"{metric}_QoQ"
        if qoq_col not in data.columns:
            continue
        values = data[qoq_col].dropna()
        if values.empty:
            continue
        averages[metric] = {
            label: clean(values.rolling(window=window, min_periods=1).mean().iloc[-1])
            for label, window in ROLLING_WINDOWS.items()
        }
    return averages


def peer_value(record: dict, metric: str):
    """The snapshot value a peer comparison ranks, for base and QoQ metrics.

    Stale values are withheld: ranking a dividend yield last paid six years ago
    against yields paid this quarter would be nonsense.
    """
    if metric.endswith("_QoQ"):
        base = metric[: -len("_QoQ")]
        return record["qoq_summary"].get(base, {}).get("latest_pct")
    if metric in record.get("latest_stale_quarters", {}):
        return None
    return record["latest"].get(metric)


def attach_peer_comparison(snapshot: list[dict]) -> None:
    """Rank each ticker against its sector and the whole panel, in place.

    The pipeline's own *_SectorRank / *_Outperf columns rank a row against every
    ticker-quarter row in the sector across all history, which is not a peer
    comparison. This ranks each ticker's latest reported value against the other
    tickers' latest reported values, which is what the numbers are read as.
    """
    for record in snapshot:
        record["peer_comparison"] = {
            "sector": record["classification"]["sector"],
            "metrics": {},
        }

    for metric, (direction, unit) in PEER_METRICS.items():
        values = {
            record["ticker"]: peer_value(record, metric)
            for record in snapshot
            if rankable(peer_value(record, metric), metric)
        }
        if not values:
            continue

        market_series = pd.Series(values)
        market_ranks = rank_series(market_series, direction)
        market_avg = float(market_series.mean())

        sector_ranks, sector_avgs = {}, {}
        sectors = {r["ticker"]: r["classification"]["sector"] for r in snapshot}
        for sector in set(sectors.values()):
            members = [t for t in values if sectors.get(t) == sector]
            if not members:
                continue
            sector_series = market_series[members]
            sector_ranks.update(rank_series(sector_series, direction).to_dict())
            sector_avgs[sector] = float(sector_series.mean())

        for record in snapshot:
            ticker = record["ticker"]
            if ticker not in values:
                continue
            value = values[ticker]
            sector = sectors[ticker]
            sector_peers = sum(1 for t in values if sectors.get(t) == sector)
            sector_avg = sector_avgs.get(sector)

            entry = {
                "value": clean(value),
                "better_is": direction,
                "sector_rank": int(sector_ranks[ticker]),
                "sector_peers": sector_peers,
                "sector_percentile": percentile(sector_ranks[ticker], sector_peers),
                "market_rank": int(market_ranks[ticker]),
                "market_peers": len(values),
            }
            if unit == "percent":
                entry["vs_sector_avg_pp"] = gap(value, sector_avg)
                entry["vs_market_avg_pp"] = gap(value, market_avg)
            else:
                entry["vs_sector_avg_pct"] = ratio_pct(value, sector_avg)
                entry["vs_market_avg_pct"] = ratio_pct(value, market_avg)
            record["peer_comparison"]["metrics"][metric] = entry


def rankable(value, metric: str) -> bool:
    """Whether a value belongs in this metric's peer ranking."""
    if value is None:
        return False
    if metric in POSITIVE_ONLY_METRICS and value <= 0:
        return False
    return True


def rank_series(series: pd.Series, direction: str) -> pd.Series:
    """Rank a ticker->value series so that 1 is always the best value."""
    return series.rank(method="min", ascending=(direction == "lower_is_better"))


def percentile(rank: float, peers: int) -> float | None:
    """Share of peers this ticker beats, 0-100 (100 = best in group)."""
    if peers < 2:
        return None
    return clean((peers - rank) / (peers - 1) * 100)


def ratio_pct(value: float, average: float | None) -> float | None:
    """Value as a percentage of an average. Only meaningful for level metrics
    whose peer average is positive, so anything else returns None rather than an
    exploded ratio."""
    if average is None or math.isnan(average) or average <= 0:
        return None
    return clean(value / average * 100)


def gap(value: float, average: float | None) -> float | None:
    """Difference from an average, in the metric's own units (percentage points
    for percent metrics)."""
    if average is None or math.isnan(average):
        return None
    return clean(value - average)


def build_prediction(
    panel: pd.DataFrame, ticker: str, strategy_mapping: dict | None
) -> dict | None:
    """Next-quarter EPS/TTM/price forecast for a ticker."""
    prediction = predict_next_eps(panel, ticker)
    if prediction is None:
        return None

    result = {key: clean(value) for key, value in prediction.items()}
    mapping = strategy_mapping or {}
    result["strategy"] = mapping.get(ticker, Strategy.WEIGHTED_GROWTH.value)
    result["strategy_source"] = (
        "backtested_optimal" if ticker in mapping else "default_fallback"
    )
    return result


# ---------------------------------------------------------------------------
# Sector aggregates
# ---------------------------------------------------------------------------


def build_sector_summary(snapshot: list[dict]) -> list[dict]:
    """Average 4Q/8Q rolling QoQ growth per sector (dashboard tab 3)."""
    by_sector: dict[str, list[dict]] = {}
    for record in snapshot:
        sector = record["classification"]["sector"]
        if sector in (None, "Unknown", "Unclassified", "N/A"):
            continue
        by_sector.setdefault(sector, []).append(record)

    summary = []
    for sector, records in sorted(by_sector.items()):
        entry = {"sector": sector, "ticker_count": len(records)}

        # Average rolling QoQ growth, as the dashboard's Sector Analysis table
        for metric in SECTOR_AVG_METRICS:
            for label in ("4Q", "8Q"):
                values = [
                    r["rolling_qoq_avg"].get(metric, {}).get(label)
                    for r in records
                    if r["rolling_qoq_avg"].get(metric, {}).get(label) is not None
                ]
                entry[f"{metric}_{label}_avg_pct"] = (
                    clean(float(np.mean(values))) if values else None
                )

        # Mean and median of the latest value of each peer-ranked metric, so a
        # reader can place any ticker without recomputing the sector themselves
        for metric in PEER_METRICS:
            values = [
                peer_value(r, metric)
                for r in records
                if rankable(peer_value(r, metric), metric)
            ]
            entry[f"{metric}_latest_avg"] = (
                clean(float(np.mean(values))) if values else None
            )
            entry[f"{metric}_latest_median"] = (
                clean(float(np.median(values))) if values else None
            )
        summary.append(entry)

    return summary


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------


def build_data_dictionary() -> dict:
    """Field-by-field explanation so a downstream reader needs no source code."""
    return {
        "source_columns": {
            "Ticker": "Stock symbol, normalized (e.g. BRK.B).",
            "Index": "Sequential quarter counter per ticker; higher = more recent.",
            "Report": "Fiscal quarter label, e.g. Q3'25.",
            "EarningsDate": "Date the quarter was reported (many are estimates).",
            "EPS": "Reported earnings per share for the quarter, USD.",
            "Revenue": "Reported revenue for the quarter, USD millions.",
            "Price": "Share price at/near the report, USD.",
            "DivAmt": "Dividend paid per share for the quarter, USD.",
        },
        "derived_columns": {
            "EPS_TTM": "Sum of the last 4 quarterly EPS values (needs all 4).",
            "Revenue_TTM": "Sum of the last 4 quarterly Revenue values.",
            "Multiple": "P/E multiple = Price / EPS_TTM.",
            "DivYield": "Quarterly dividend yield = DivAmt / Price * 100, %.",
            "DivYieldAnnual": "Annualized yield = DivAmt * 4 / Price * 100, %.",
            "PayoutRatio": "DivAmt * 4 / EPS_TTM * 100, %.",
            "PEGRatio": (
                "Multiple / |annualized 4Q average EPS QoQ growth|. Uses the "
                "absolute growth value, so shrinking earnings produce a "
                "positive PEG - read it alongside EPS_QoQ."
            ),
            "PEGYRatio": "PEGRatio / DivYieldAnnual, only where the yield is > 0.",
            "DivGrowthRate": (
                "Annualized CAGR between the first and last detected dividend "
                "change; 0.0 for tickers with fewer than two changes."
            ),
            "DivIncreaseFreq": "Dividend increases per year over the history.",
            "AvgDivIncrease": "Mean size of a dividend increase, %.",
            "EPSMomentum": "4Q mean EPS_QoQ minus 8Q mean EPS_QoQ, percentage points.",
            "PriceVolatility": "Rolling 8Q standard deviation of Price_QoQ, %.",
            "RevenueConsistency": (
                "100 - (8Q std of Revenue_QoQ / |8Q mean of Revenue_QoQ| * 100). "
                "Unbounded below and unstable when the mean growth is near zero; "
                "large negative values mean noisy growth, not negative growth."
            ),
            "DownsideCapture": (
                "Ticker's mean Price_QoQ during quarters when the market average "
                "was negative, as a % of the market's mean move. Under 100 = "
                "falls less than the market."
            ),
            "<metric>_QoQ": (
                "Percent change vs the ticker's previous quarter for that metric. "
                "Computed for: " + ", ".join(QOQ_BASE_METRICS) + "."
            ),
            "<metric>_SectorRank": (
                "Raw pipeline output, quarterly_metrics.csv only: rank of this "
                "row among EVERY ticker-quarter row in the sector across all "
                "history - a history-wide row rank, not a peer rank. Use the "
                "JSON snapshot's peer_comparison to compare tickers."
            ),
            "<metric>_MarketOutperf": (
                "Raw pipeline output: value as a % of the mean across every row "
                "in the panel (all tickers, all quarters, so a long-history "
                "ticker weighs more)."
            ),
            "<metric>_SectorOutperf": (
                "Raw pipeline output: value as a % of the mean across every row "
                "in that sector, all quarters included."
            ),
        },
        "snapshot_fields": {
            "latest.<metric>": (
                "Most recent non-null value of that metric for the ticker. Reaches "
                "back through blank quarters, so it can predate "
                "coverage.latest_report."
            ),
            "latest_stale_quarters.<metric>": (
                "Only present for metrics whose latest value is more than one "
                "quarter behind coverage.latest_index; the number is how many "
                "quarters back it came from. Those values are excluded from "
                "peer_comparison."
            ),
            "qoq_summary.<metric>.latest_pct": "Most recent quarter-over-quarter %.",
            "qoq_summary.<metric>.average_pct": "Mean QoQ % across all history.",
            "qoq_summary.<metric>.periods_positive": "Quarters with a positive QoQ.",
            "qoq_summary.<metric>.periods_total": "Quarters with a QoQ value at all.",
            "rolling_qoq_avg.<metric>.{4Q,8Q,12Q}": (
                "Latest rolling mean of that metric's QoQ series over the last "
                "4/8/12 quarters, min_periods=1."
            ),
            "peer_comparison.sector": (
                "Sector whose peers this ticker is ranked against."
            ),
            "peer_comparison.metrics.<metric>.value": (
                "The value being ranked - the same number as latest.<metric>, or "
                "qoq_summary.<base>.latest_pct for a _QoQ metric."
            ),
            "peer_comparison.metrics.<metric>.better_is": (
                "higher_is_better or lower_is_better - which end of the range "
                "rank 1 sits at."
            ),
            "peer_comparison.metrics.<metric>.sector_rank / sector_peers": (
                "Rank among the tickers in the same sector that report this "
                "metric; 1 is best in the direction given by better_is."
            ),
            "peer_comparison.metrics.<metric>.sector_percentile": (
                "Share of sector peers this ticker beats, 0-100 (100 = best)."
            ),
            "peer_comparison.metrics.<metric>.market_rank / market_peers": (
                "Same rank across all 137 tickers rather than just the sector."
            ),
            "peer_comparison.metrics.<metric>.vs_sector_avg_pct": (
                "Level metrics (EPS_TTM, Revenue_TTM, Multiple, PEGRatio, "
                "PEGYRatio) only: value as a % of the sector average, 100 = "
                "exactly average. Null when that average is not positive."
            ),
            "peer_comparison.metrics.<metric>.vs_market_avg_pct": (
                "Same, against the all-ticker average."
            ),
            "peer_comparison.metrics.<metric>.vs_sector_avg_pp": (
                "Percent metrics only: gap to the sector average in percentage "
                "points. A ratio is not used here because the average QoQ growth "
                "of a sector sits near zero and would explode it."
            ),
            "peer_comparison.metrics.<metric>.vs_market_avg_pp": (
                "Same, against the all-ticker average."
            ),
        },
        "data_flags": {
            "negative_eps_ttm": (
                "Latest EPS_TTM is zero or negative. Multiple, PEGRatio, "
                "PEGYRatio and PayoutRatio are then meaningless (a P/E of -50 is "
                "not cheap), and the ticker is excluded from those peer ranks."
            ),
            "pays_dividend": "A dividend was paid in the latest quarter.",
            "has_stale_metrics": (
                "At least one latest.<metric> comes from an earlier quarter than "
                "the ticker's newest one; latest_stale_quarters says which and "
                "how far back."
            ),
            "short_history": "Fewer than 12 quarters, so 12Q figures are partial.",
            "has_prediction": "A next-quarter forecast could be produced.",
            "warnings": "Plain-text explanation of each flag that is set.",
        },
        "prediction_fields": {
            "strategy": "Backtested per-ticker strategy used for the forecast.",
            "strategy_source": (
                "backtested_optimal when config/ticker_strategy_mapping.json "
                "names a strategy for the ticker, else default_fallback "
                "(weighted_growth)."
            ),
            "methodology": "Human-readable description of the strategy.",
            "latest_eps": "Most recent actual quarterly EPS, the forecast base.",
            "predicted_eps": "Base-case next-quarter EPS.",
            "best_case_eps": "Base case plus one volatility band.",
            "worst_case_eps": "Base case minus one volatility band.",
            "predicted_growth / best_case_growth / worst_case_growth": (
                "Implied QoQ EPS growth for each scenario, %."
            ),
            "current_eps_ttm": "Latest actual EPS_TTM.",
            "predicted_eps_ttm": "Last 3 actual quarters plus the predicted quarter.",
            "best_case_eps_ttm / worst_case_eps_ttm": "Same for the other scenarios.",
            "predicted_eps_ttm_growth": "Growth of predicted vs current EPS_TTM, %.",
            "current_price": "Latest actual price.",
            "current_multiple": "Latest actual P/E multiple, held constant.",
            "predicted_price": "predicted_eps_ttm * current_multiple.",
            "best_case_price / worst_case_price": "Same for the other scenarios.",
            "predicted_price_growth": "Implied price move vs current_price, %.",
            "volatility": "Volatility used to size the scenario bands, %.",
            "confidence": "High / Medium / Low, from available data points.",
            "data_points": "Number of EPS_QoQ observations behind the forecast.",
            "next_index": "Index value the forecast quarter would carry.",
        },
        "caveats": [
            "Forecasts extrapolate the ticker's own EPS history. They assume a "
            "constant P/E multiple and no macro or company-specific news.",
            "PEGRatio uses |growth|, so a company with falling earnings can show "
            "an attractive-looking PEG. Check EPS_QoQ before using it.",
            "RevenueConsistency divides by the mean growth rate and blows up when "
            "that mean is near zero.",
            "peer_comparison ranks each ticker's latest reported quarter against "
            "other tickers' latest reported quarters. Those are not all the same "
            "calendar quarter, so a ticker that reports late is compared on "
            "slightly older figures.",
            "Tickers listed in config/excluded_tickers.json are dropped before "
            "any calculation.",
        ],
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def flatten(record: dict, parent: str = "", separator: str = ".") -> dict:
    """Flatten nested snapshot dicts into single-level CSV columns."""
    flat = {}
    for key, value in record.items():
        name = f"{parent}{separator}{key}" if parent else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, name, separator))
        else:
            flat[name] = value
    return flat


def write_outputs(
    panel: pd.DataFrame,
    snapshot: list[dict],
    sector_summary: list[dict],
    output_dir: Path,
    input_path: str,
    include_history: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Full quarterly panel
    panel_path = output_dir / "quarterly_metrics.csv"
    panel.to_csv(panel_path, index=False)
    paths["quarterly_metrics.csv"] = panel_path

    # 2. Flat ticker snapshot
    snapshot_path = output_dir / "ticker_snapshot.csv"
    pd.DataFrame([flatten(record) for record in snapshot]).to_csv(
        snapshot_path, index=False
    )
    paths["ticker_snapshot.csv"] = snapshot_path

    # 3. Sector aggregates
    sector_path = output_dir / "sector_summary.csv"
    pd.DataFrame(sector_summary).to_csv(sector_path, index=False)
    paths["sector_summary.csv"] = sector_path

    # 4. Self-describing JSON bundle
    bundle = {
        "meta": build_meta(panel, snapshot, input_path, include_history),
        "data_dictionary": build_data_dictionary(),
        "sector_summary": sector_summary,
        "tickers": snapshot,
    }

    if include_history:
        history = build_history(panel)
        for record in bundle["tickers"]:
            record["history"] = history.get(record["ticker"], [])

    json_path = output_dir / "stock_analysis_export.json"
    with open(json_path, "w") as handle:
        json.dump(bundle, handle, indent=2, allow_nan=False)
    paths["stock_analysis_export.json"] = json_path

    # 5. Guide
    readme_path = output_dir / "README.md"
    readme_path.write_text(build_readme(bundle["meta"], paths))
    paths["README.md"] = readme_path

    return paths


def build_meta(
    panel: pd.DataFrame, snapshot: list[dict], input_path: str, include_history: bool
) -> dict:
    report_col = Column.REPORT.value
    predictions = sum(1 for record in snapshot if record["prediction"] is not None)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/export_dashboard_data.py",
        "source_file": input_path,
        "source_file_modified_utc": datetime.fromtimestamp(
            Path(input_path).stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": [
            "load_stock_data (clean, dedupe, drop excluded tickers)",
            "attach Sector / Industry / Sub_Industry classifications",
            "calculate_qoq_changes (TTM, Multiple, yields, QoQ, PEG/PEGY, momentum)",
            "calculate_sector_rankings",
            "calculate_outperformance_ratios",
            "calculate_downside_capture",
            "predict_next_eps per ticker",
        ],
        "tickers": len(snapshot),
        "quarterly_rows": int(len(panel)),
        "columns": int(panel.shape[1]),
        "report_range": report_range(panel[report_col]),
        "tickers_with_prediction": predictions,
        "history_embedded": include_history,
        "currency": "USD; Revenue in millions; ratios and *_QoQ in percent",
    }


def build_history(panel: pd.DataFrame) -> dict[str, list[dict]]:
    """Per-ticker quarterly rows, JSON-safe, for --include-history."""
    ticker_col = Column.TICKER.value
    history: dict[str, list[dict]] = {}
    for ticker, data in panel.groupby(ticker_col):
        rows = []
        for _, row in data.iterrows():
            rows.append(
                {
                    key: clean(value)
                    for key, value in row.items()
                    if key != ticker_col and clean(value) is not None
                }
            )
        history[ticker] = rows
    return history


def build_readme(meta: dict, paths: dict[str, Path]) -> str:
    """Short guide written alongside the data files."""
    # Markdown table rows have to stay on one line each, so they are assembled
    # from pieces rather than written inline in the template below.
    rows = [
        (
            "`stock_analysis_export.json`",
            "**Start here.** Metadata, a full data dictionary, and a per-ticker "
            "snapshot: latest values, QoQ summaries, 4Q/8Q/12Q rolling averages, "
            "peer ranks within sector and market, a next-quarter forecast, and "
            "data-quality flags. Plus sector aggregates.",
        ),
        (
            "`quarterly_metrics.csv`",
            "Full panel: one row per ticker per quarter, "
            f"{meta['columns']} columns, every base and derived metric.",
        ),
        (
            "`ticker_snapshot.csv`",
            "The JSON's per-ticker snapshot flattened for spreadsheet use.",
        ),
        (
            "`sector_summary.csv`",
            "Per-sector averages: rolling QoQ growth, plus the mean and median "
            "of every peer-ranked metric.",
        ),
    ]
    table = "\n".join(f"| {name} | {description} |" for name, description in rows)
    json_name = paths["stock_analysis_export.json"].name

    return f"""# StockInsights Dashboard Export

Generated {meta['generated_at_utc']} by `{meta['generated_by']}` from
`{meta['source_file']}` - {meta['tickers']} tickers, {meta['quarterly_rows']:,}
quarterly rows, {meta['report_range'][0]} to {meta['report_range'][1]}.

Everything the Streamlit dashboard computes is in these files, so nothing here
needs the dashboard or the source code to read.

## Files

| File | What it holds |
| --- | --- |
{table}

## Reading the JSON

```python
import json

with open("{json_name}") as f:
    export = json.load(f)

export["meta"]                     # provenance and pipeline steps
export["data_dictionary"]          # what every field means
export["sector_summary"]           # per-sector aggregates

by_ticker = {{t["ticker"]: t for t in export["tickers"]}}
by_ticker["AAPL"]["prediction"]    # next-quarter EPS, TTM and price scenarios
by_ticker["AAPL"]["peer_comparison"]["metrics"]["Multiple"]
```

## Before using the numbers

Every field is defined in `export["data_dictionary"]`. Two things to read first:

- `data_dictionary.caveats` - what `prediction`, `PEGRatio` and
  `RevenueConsistency` do and do not mean.
- each ticker's `data_flags.warnings` - plain-text notes on that specific
  ticker, such as negative trailing earnings making its P/E meaningless, or a
  metric whose latest value is years old.

Forecasts extrapolate a ticker's own EPS history at a constant P/E multiple.
They know nothing about guidance, macro conditions or news, and are not
investment advice.

## Units

{meta['currency']}. Missing values are `null` in JSON and empty in CSV.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export all dashboard-computed metrics to JSON and CSV."
    )
    parser.add_argument(
        "--input",
        default=FilePaths.DATA_FILE,
        help=f"Indexed Excel workbook (default: {FilePaths.DATA_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports",
        help="Directory for the exported files (default: data/exports)",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help=(
            "Embed every quarterly row inside the JSON as well. Much larger "
            "file; the same data is always written to quarterly_metrics.csv."
        ),
    )
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    print("=" * 60)
    print("StockInsights Dashboard Data Export")
    print("=" * 60)
    print(f"Source : {input_path}")

    panel = build_panel(input_path)
    print(
        f"Panel  : {len(panel):,} quarterly rows x {panel.shape[1]} columns, "
        f"{panel[Column.TICKER.value].nunique()} tickers"
    )

    strategy_mapping = load_ticker_strategy_mapping()
    if strategy_mapping is None:
        print(
            "Warning: config/ticker_strategy_mapping.json not loaded - "
            "every forecast falls back to weighted_growth."
        )

    snapshot = build_ticker_snapshot(panel, strategy_mapping)
    forecasts = sum(1 for record in snapshot if record["prediction"] is not None)
    print(f"Snapshot: {len(snapshot)} tickers, {forecasts} with a forecast")

    sector_summary = build_sector_summary(snapshot)
    print(f"Sectors : {len(sector_summary)}")

    paths = write_outputs(
        panel=panel,
        snapshot=snapshot,
        sector_summary=sector_summary,
        output_dir=Path(args.output_dir),
        input_path=input_path,
        include_history=args.include_history,
    )

    print("-" * 60)
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:<30} {size_kb:>10,.1f} KB")
    print("-" * 60)
    print(f"Done. Hand {paths['stock_analysis_export.json']} to another agent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
