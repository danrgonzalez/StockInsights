"""
Pre-acquisition profiling.

What does a company's reporting look like in the quarters before someone buys
it? Acquired tickers are kept in the data (see ``core/ticker_status.py``) rather
than deleted, so their final quarters can be lined up and compared.

Quarters are indexed relative to the **announcement** date, not the completion
date: the announcement is when the market learns, and everything after it is a
company trading on deal terms rather than fundamentals. Quarter ``0`` is the
last report on or before the announcement, ``-1`` the one before that, and
positive offsets are reports filed while the deal was pending.
"""

import numpy as np
import pandas as pd

from core.enums import Column, DerivedMetric, Metric
from core.ticker_status import get_acquired_tickers, get_acquisition

# Column added to a profile frame: quarters relative to the announcement.
QUARTER_OFFSET = "QuartersToAnnouncement"

# Metrics whose run-up is worth looking at.
PROFILE_METRICS = [
    Metric.EPS.value,
    Metric.REVENUE.value,
    Metric.PRICE.value,
    DerivedMetric.EPS_TTM.value,
    DerivedMetric.REVENUE_TTM.value,
    DerivedMetric.MULTIPLE.value,
]


def build_profile(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    One acquired ticker's quarters, indexed relative to the announcement.

    Args:
        df: Stock data including acquired tickers -- load with
            ``include_excluded=True``, and run ``calculate_qoq_changes`` first
            if you want the derived metrics populated.
        ticker: The acquired ticker

    Returns:
        The ticker's rows with a ``QuartersToAnnouncement`` column, or None if
        the ticker was not acquired or is absent from ``df``.
    """
    acquisition = get_acquisition(ticker)
    if acquisition is None:
        return None

    announced = acquisition.get("announced")
    if not announced:
        return None

    ticker_col = Column.TICKER.value
    date_col = Column.EARNINGS_DATE.value
    rows = df[df[ticker_col] == ticker].copy()
    if rows.empty or date_col not in rows.columns:
        return None

    rows = rows.sort_values(Column.INDEX.value)
    announced_ts = pd.Timestamp(announced)

    # Quarter 0 is the last report on or before the announcement.
    reported = rows[date_col].notna()
    before = rows[reported & (rows[date_col] <= announced_ts)]
    if before.empty:
        return None

    anchor_index = before[Column.INDEX.value].iloc[-1]
    rows[QUARTER_OFFSET] = rows[Column.INDEX.value] - anchor_index
    return rows


def summarize_profile(df: pd.DataFrame, ticker: str) -> dict | None:
    """
    Headline numbers for one acquisition.

    ``deal_price_vs_last_report_pct`` is deliberately NOT called a premium. It
    compares the deal price against the share price at the last report before
    the announcement, which can be up to a quarter earlier -- by then a leaked
    or anticipated deal may already be in the stock. Skechers computes to 0.5%
    here against an announced move of roughly 24%, which shows the difference.
    Read it as "how much of the takeout price was not yet in the stock at the
    last report", not as what shareholders gained.

    Args:
        df: Stock data including acquired tickers
        ticker: The acquired ticker

    Returns:
        Summary dict, or None if the ticker was not acquired.
    """
    profile = build_profile(df, ticker)
    if profile is None:
        return None

    acquisition = get_acquisition(ticker)
    pre = profile[profile[QUARTER_OFFSET] <= 0]
    last_pre = pre.iloc[-1] if not pre.empty else None

    summary = {
        "ticker": ticker,
        "acquirer": acquisition.get("acquirer"),
        "type": acquisition.get("type"),
        "announced": acquisition.get("announced"),
        "completed": acquisition.get("completed"),
        "price_per_share": acquisition.get("price_per_share"),
        "deal_value_usd_bn": acquisition.get("deal_value_usd_bn"),
        "quarters_of_history": len(profile),
        "quarters_before_announcement": int((profile[QUARTER_OFFSET] <= 0).sum()),
        "quarters_while_pending": int((profile[QUARTER_OFFSET] > 0).sum()),
        "last_report_before_announcement": (
            str(last_pre[Column.REPORT.value]) if last_pre is not None else None
        ),
    }

    price_col = Metric.PRICE.value
    last_price = (
        last_pre[price_col]
        if last_pre is not None and price_col in profile.columns
        else None
    )
    summary["price_at_last_report"] = (
        float(last_price) if last_price is not None and pd.notna(last_price) else None
    )

    deal_price = acquisition.get("price_per_share")
    if deal_price and summary["price_at_last_report"]:
        gap = (deal_price / summary["price_at_last_report"] - 1) * 100
        summary["deal_price_vs_last_report_pct"] = round(gap, 1)
    else:
        # All-stock deals carry no per-share cash price to compare against.
        summary["deal_price_vs_last_report_pct"] = None

    # Trajectory into the deal: mean QoQ over the 4 and 8 quarters up to
    # announcement, which is where a decelerating business shows up.
    for metric in (Metric.EPS.value, Metric.REVENUE.value, Metric.PRICE.value):
        qoq_col = f"{metric}_QoQ"
        if qoq_col not in profile.columns:
            continue
        for window in (4, 8):
            window_rows = pre.tail(window)
            values = window_rows[qoq_col].dropna()
            summary[f"{metric}_QoQ_mean_{window}q"] = (
                round(float(values.mean()), 2) if len(values) else None
            )

    return summary


def all_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per acquired ticker, summarising each deal and its run-up.

    Args:
        df: Stock data including acquired tickers

    Returns:
        DataFrame of summaries, ordered by announcement date. Empty if none.
    """
    summaries = [
        summary
        for ticker in sorted(get_acquired_tickers())
        if (summary := summarize_profile(df, ticker)) is not None
    ]
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame(summaries).sort_values("announced").reset_index(drop=True)


def aligned_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    One metric for every acquired ticker, aligned on quarters-to-announcement.

    This is the comparison view: each column is a ticker, each row an offset,
    so the quarters before a buyout line up across deals.

    Args:
        df: Stock data including acquired tickers
        metric: Column to extract, e.g. "EPS" or "Multiple"

    Returns:
        DataFrame indexed by QuartersToAnnouncement, one column per ticker.
    """
    series = {}
    for ticker in sorted(get_acquired_tickers()):
        profile = build_profile(df, ticker)
        if profile is None or metric not in profile.columns:
            continue
        values = profile.set_index(QUARTER_OFFSET)[metric]
        series[ticker] = values[~values.index.duplicated(keep="last")]

    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index().replace([np.inf, -np.inf], np.nan)
