"""
Core data processing module for stock financial analysis.

This module contains pure data processing logic without any UI dependencies,
making it suitable for use in any application (Streamlit, CLI, API, etc.).
"""

import numpy as np
import pandas as pd

from core.enums import (
    Column,
    DataQuality,
    DerivedMetric,
    Metric,
    RankingSuffix,
    RollingWindow,
    StrategyWeights,
    Thresholds,
)


def latest_with_age(
    ticker_data: pd.DataFrame, column: str
) -> tuple[float | None, str | None, int | None]:
    """Most recent non-null value for one ticker's column, with its age.

    ``dropna().iloc[-1]`` returns the last value that exists, which is not the
    same as the value for the latest quarter. When a company stops paying a
    dividend the column goes blank but the old figure keeps being reported as
    current -- DAL's card showed the dividend it last paid in Q4'19.

    Args:
        ticker_data: Rows for a single ticker.
        column: Column to read.

    Returns:
        ``(value, report_label, quarters_stale)``, where ``quarters_stale`` is 0
        when the value belongs to the ticker's latest row. ``(None, None, None)``
        when the column is missing or entirely blank.
    """
    if column not in ticker_data.columns or ticker_data.empty:
        return (None, None, None)

    index_col = Column.INDEX.value
    ordered = (
        ticker_data.sort_values(index_col)
        if index_col in ticker_data.columns
        else ticker_data
    )

    present = ordered[ordered[column].notna()]
    if present.empty:
        return (None, None, None)

    source = present.iloc[-1]
    report_col = Column.REPORT.value
    label = str(source[report_col]) if report_col in ordered.columns else None

    if index_col in ordered.columns:
        stale = int(ordered[index_col].iloc[-1] - source[index_col])
    else:
        stale = len(ordered) - 1 - ordered.index.get_loc(source.name)

    return (source[column], label, stale)


def report_sort_key(label: str) -> tuple[int, int]:
    """Sort key for a fiscal quarter label such as ``Q3'25``.

    Report labels sort wrongly as plain strings -- the quarter leads, so
    ``Q1'26`` lands below ``Q4'10``. This returns ``(year, quarter)`` instead.

    Two-digit years are read as 20xx, which covers the whole panel (2010
    through the fiscal-year-ahead 2027 labels). Anything unparseable sorts
    last rather than raising, so a stray label cannot break a sidebar.
    """
    text = str(label).strip()
    try:
        quarter = int(text[1])
        year = int(text.split("'")[1])
    except (IndexError, ValueError):
        return (9999, 9)
    return (2000 + year if year < 100 else year, quarter)


def report_range(reports: pd.Series) -> tuple[str | None, str | None]:
    """Earliest and latest fiscal quarter label, ordered by (year, quarter)."""
    labels = sorted({str(label) for label in reports.dropna()}, key=report_sort_key)
    return (labels[0], labels[-1]) if labels else (None, None)


def load_stock_data_with_stats(
    file_path: str,
    include_excluded: bool = False,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Load and clean stock data, reporting what was dropped along the way.

    This is the single implementation; ``load_stock_data`` is the plain
    wrapper. The stats let a UI surface the same cleaning decisions the batch
    path makes silently, without a second copy of the loader.

    Args:
        file_path: Path to the Excel file
        include_excluded: Keep tickers held out of the active universe --
            acquired companies and data-quality holds. Off by default so the
            active analysis is unaffected; on for acquisition profiling.

    Returns:
        (DataFrame or None, stats) where stats carries ``duplicates_dropped``
        (int), ``excluded`` (ticker -> row count) and ``error``
        (None, "not_found", or the exception message).
    """
    stats: dict = {"duplicates_dropped": 0, "excluded": {}, "error": None}

    try:
        df = pd.read_excel(file_path)

        # Clean numeric columns - convert non-numeric values to NaN
        numeric_columns = Metric.numeric_columns()
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse the quarterly earnings report date as a datetime
        earnings_date_col = Column.EARNINGS_DATE.value
        if earnings_date_col in df.columns:
            df[earnings_date_col] = pd.to_datetime(
                df[earnings_date_col], errors="coerce"
            )

        # Clean and normalize ticker symbols
        ticker_col = Column.TICKER.value
        if ticker_col in df.columns:
            try:
                from core.classifications import normalize_symbol

                df[ticker_col] = df[ticker_col].astype(str).apply(normalize_symbol)
            except ImportError:
                df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()

        # Clean other text columns
        report_col = Column.REPORT.value
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in [ticker_col, report_col]:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("", np.nan)

        # Remove duplicates - keep the newest row when a label is repeated
        if ticker_col in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[ticker_col, report_col], keep="last")
            stats["duplicates_dropped"] = before - len(df)

        # Mark lifecycle status, then drop anything held out of the active
        # universe. Acquired tickers are excluded by default but retained when
        # include_excluded is set, so their final quarters stay available for
        # pre-acquisition profiling.
        from core.ticker_status import attach_status, filter_excluded

        df = attach_status(df)
        if include_excluded:
            stats["excluded"] = {}
        else:
            df, dropped = filter_excluded(df)
            stats["excluded"] = dropped

        return df, stats
    except FileNotFoundError:
        stats["error"] = "not_found"
        return None, stats
    except Exception as exc:
        stats["error"] = str(exc)
        return None, stats


def load_stock_data(
    file_path: str, include_excluded: bool = False
) -> pd.DataFrame | None:
    """
    Load and clean stock data from an Excel file.

    Args:
        file_path: Path to the Excel file
        include_excluded: Keep acquired tickers and data-quality holds

    Returns:
        Cleaned DataFrame or None if loading fails
    """
    df, _ = load_stock_data_with_stats(file_path, include_excluded=include_excluded)
    return df


def calculate_qoq_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Quarter-over-Quarter percent changes for each ticker.

    Computes:
    - TTM (Trailing Twelve Months) for EPS and Revenue
    - P/E Multiple
    - Dividend yields (quarterly and annual)
    - Payout ratio
    - QoQ changes for all metrics
    - Advanced metrics: EPS Momentum, Price Volatility, Revenue Consistency
    - PEG and PEGY ratios
    - Dividend growth rate

    Args:
        df: Stock data DataFrame with Ticker, Index, EPS, Revenue, Price,
            DivAmt

    Returns:
        DataFrame with all calculated metrics added
    """
    df_with_qoq = df.copy()
    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value
    df_with_qoq = df_with_qoq.sort_values([ticker_col, index_col])

    for ticker in df_with_qoq[ticker_col].unique():
        ticker_mask = df_with_qoq[ticker_col] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()

        # Calculate TTM (Trailing Twelve Months) values
        for metric in [Metric.EPS.value, Metric.REVENUE.value]:
            if metric in ticker_data.columns:
                ttm_values = (
                    ticker_data[metric]
                    .rolling(window=RollingWindow.TTM, min_periods=RollingWindow.TTM)
                    .sum()
                )
                ttm_col = f"{metric}_TTM"
                df_with_qoq.loc[ticker_mask, ttm_col] = ttm_values

        # Calculate P/E Multiple
        price_col = Metric.PRICE.value
        eps_ttm_col = DerivedMetric.EPS_TTM.value
        multiple_col = DerivedMetric.MULTIPLE.value
        if (
            price_col in ticker_data.columns
            and eps_ttm_col in df_with_qoq.loc[ticker_mask].columns
        ):
            price_data = ticker_data[price_col]
            eps_ttm_data = df_with_qoq.loc[ticker_mask, eps_ttm_col]
            # A P/E on negative earnings is not a small multiple, it is an
            # undefined one. Leaving it signed sorted money-losing companies to
            # rank 1 wherever "lower is better".
            positive_eps_ttm = eps_ttm_data.where(eps_ttm_data > 0)
            multiple = price_data / positive_eps_ttm
            multiple = multiple.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, multiple_col] = multiple

        # Calculate dividend yields
        div_col = Metric.DIVIDEND_AMOUNT.value
        if price_col in ticker_data.columns and div_col in ticker_data.columns:
            price_data = ticker_data[price_col]
            div_data = ticker_data[div_col]

            # Quarterly dividend yield
            dividend_yield = (div_data / price_data) * 100
            dividend_yield = dividend_yield.replace([np.inf, -np.inf], np.nan)
            div_yield_col = DerivedMetric.DIVIDEND_YIELD.value
            df_with_qoq.loc[ticker_mask, div_yield_col] = dividend_yield

            # Annualized dividend yield
            dividend_yield_annual = (
                div_data * RollingWindow.QUARTERS_PER_YEAR / price_data
            ) * 100
            dividend_yield_annual = dividend_yield_annual.replace(
                [np.inf, -np.inf], np.nan
            )
            div_yield_ann_col = DerivedMetric.DIVIDEND_YIELD_ANNUAL.value
            df_with_qoq.loc[ticker_mask, div_yield_ann_col] = dividend_yield_annual

        # Calculate Payout Ratio
        if (
            div_col in ticker_data.columns
            and eps_ttm_col in df_with_qoq.loc[ticker_mask].columns
        ):
            div_data = ticker_data[div_col]
            eps_ttm_data = df_with_qoq.loc[ticker_mask, eps_ttm_col]
            annual_div = div_data * RollingWindow.QUARTERS_PER_YEAR
            # Undefined against non-positive earnings, same as Multiple: the
            # ratio ranged -8,600 to 5,800 before this mask.
            positive_eps_ttm = eps_ttm_data.where(eps_ttm_data > 0)
            payout_ratio = (annual_div / positive_eps_ttm) * 100
            payout_ratio = payout_ratio.replace([np.inf, -np.inf], np.nan)
            payout_col = DerivedMetric.PAYOUT_RATIO.value
            df_with_qoq.loc[ticker_mask, payout_col] = payout_ratio

    # Calculate QoQ changes for all metrics
    df_with_qoq = df_with_qoq.sort_values([ticker_col, index_col])

    for ticker in df_with_qoq[ticker_col].unique():
        ticker_mask = df_with_qoq[ticker_col] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()

        # Use the enum method for QoQ metrics list
        for metric in DerivedMetric.qoq_metrics():
            if metric in ticker_data.columns:
                qoq_change = ticker_data[metric].pct_change(fill_method=None) * 100
                # A percent change off a zero base is infinite, not a very large
                # growth rate. Left in, it poisoned every rolling mean built on
                # the series. The level ratios above already do this.
                qoq_change = qoq_change.replace([np.inf, -np.inf], np.nan)
                df_with_qoq.loc[ticker_mask, f"{metric}_QoQ"] = qoq_change

        # Calculate advanced metrics requiring QoQ data
        ticker_data_with_qoq = df_with_qoq[ticker_mask].copy()

        # EPS Growth Momentum
        eps_qoq_col = DerivedMetric.EPS_QOQ.value
        if eps_qoq_col in ticker_data_with_qoq.columns:
            eps_qoq_values = ticker_data_with_qoq[eps_qoq_col].dropna()
            if len(eps_qoq_values) >= RollingWindow.LONG:
                rolling_4q = eps_qoq_values.rolling(
                    window=RollingWindow.SHORT, min_periods=RollingWindow.SHORT
                ).mean()
                rolling_8q = eps_qoq_values.rolling(
                    window=RollingWindow.LONG, min_periods=RollingWindow.LONG
                ).mean()
                eps_momentum = rolling_4q - rolling_8q
                momentum_col = DerivedMetric.EPS_MOMENTUM.value
                df_with_qoq.loc[ticker_mask, momentum_col] = eps_momentum

        # Price Volatility
        price_qoq_col = DerivedMetric.PRICE_QOQ.value
        if price_qoq_col in ticker_data_with_qoq.columns:
            price_qoq_values = ticker_data_with_qoq[price_qoq_col].dropna()
            if len(price_qoq_values) >= RollingWindow.SHORT:
                price_volatility = price_qoq_values.rolling(
                    window=RollingWindow.LONG, min_periods=RollingWindow.SHORT
                ).std()
                volatility_col = DerivedMetric.PRICE_VOLATILITY.value
                df_with_qoq.loc[ticker_mask, volatility_col] = price_volatility

        # Revenue Consistency
        rev_qoq_col = DerivedMetric.REVENUE_QOQ.value
        if rev_qoq_col in ticker_data_with_qoq.columns:
            revenue_qoq_values = ticker_data_with_qoq[rev_qoq_col].dropna()
            if len(revenue_qoq_values) >= RollingWindow.SHORT:
                rolling_std = revenue_qoq_values.rolling(
                    window=RollingWindow.LONG, min_periods=RollingWindow.SHORT
                ).std()
                # The old definition divided by |rolling_mean|, which is near
                # zero for a normal company, so the score exploded: dataset
                # median -318, min -1,860,112. Score the dispersion directly
                # instead. Bounded (0, 100]: 100 is a perfectly steady revenue
                # line, and the score halves for every
                # CONSISTENCY_VOLATILITY_SCALE percentage points of typical
                # quarterly swing.
                revenue_consistency = 100 / (
                    1 + rolling_std.abs() / Thresholds.CONSISTENCY_VOLATILITY_SCALE
                )
                revenue_consistency = revenue_consistency.replace(
                    [np.inf, -np.inf], np.nan
                )
                consistency_col = DerivedMetric.REVENUE_CONSISTENCY.value
                df_with_qoq.loc[ticker_mask, consistency_col] = revenue_consistency

        # Dividend Growth Rate
        if div_col in ticker_data_with_qoq.columns:
            df_with_qoq = _calculate_dividend_growth(
                df_with_qoq, ticker_mask, ticker_data_with_qoq
            )

        # PEG Ratio
        if (
            multiple_col in ticker_data_with_qoq.columns
            and eps_qoq_col in ticker_data_with_qoq.columns
        ):
            multiple_data = ticker_data_with_qoq[multiple_col]
            eps_qoq_values = ticker_data_with_qoq[eps_qoq_col].dropna()
            if len(eps_qoq_values) >= RollingWindow.SHORT:
                eps_growth_4q = eps_qoq_values.rolling(
                    window=RollingWindow.SHORT, min_periods=RollingWindow.SHORT
                ).mean()
                eps_growth_annual = (
                    (1 + eps_growth_4q / 100) ** RollingWindow.QUARTERS_PER_YEAR - 1
                ) * 100
                # .abs() here made -30% and +30% annual EPS growth produce the
                # same PEG, so shrinking companies looked cheap. PEG is only
                # defined for positive growth.
                positive_growth = eps_growth_annual.where(eps_growth_annual > 0)
                peg_ratio = multiple_data / positive_growth
                peg_ratio = peg_ratio.replace([np.inf, -np.inf], np.nan)
                peg_col = DerivedMetric.PEG_RATIO.value
                df_with_qoq.loc[ticker_mask, peg_col] = peg_ratio

        # PEGY Ratio
        peg_col = DerivedMetric.PEG_RATIO.value
        div_yield_ann_col = DerivedMetric.DIVIDEND_YIELD_ANNUAL.value
        if (
            peg_col in df_with_qoq.loc[ticker_mask].columns
            and div_yield_ann_col in df_with_qoq.loc[ticker_mask].columns
        ):
            peg_data = df_with_qoq.loc[ticker_mask, peg_col]
            div_yield_annual_data = df_with_qoq.loc[ticker_mask, div_yield_ann_col]
            pegy_ratio = np.where(
                div_yield_annual_data > 0, peg_data / div_yield_annual_data, np.nan
            )
            pegy_col = DerivedMetric.PEGY_RATIO.value
            df_with_qoq.loc[ticker_mask, pegy_col] = pegy_ratio

    return df_with_qoq


def _calculate_dividend_growth(
    df_with_qoq: pd.DataFrame,
    ticker_mask: pd.Series,
    ticker_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate dividend growth rate for a ticker."""
    div_col = Metric.DIVIDEND_AMOUNT.value
    div_amounts = ticker_data[div_col].dropna()
    if len(div_amounts) < RollingWindow.SHORT:
        return df_with_qoq

    # Find dividend change points
    div_changes = []
    current_div = None
    last_change_idx = 0

    for i, (idx, div_amt) in enumerate(div_amounts.items()):
        if current_div is None:
            current_div = div_amt
            last_change_idx = i
        elif abs(div_amt - current_div) > StrategyWeights.DIVIDEND_CHANGE_THRESHOLD:
            periods_since_last_change = i - last_change_idx
            if current_div > 0:
                growth_rate = ((div_amt - current_div) / current_div) * 100
                div_changes.append(
                    {
                        "from_amount": current_div,
                        "to_amount": div_amt,
                        "growth_rate": growth_rate,
                        "periods": periods_since_last_change,
                        "end_idx": idx,
                    }
                )
            current_div = div_amt
            last_change_idx = i

    # Calculate overall dividend growth metrics
    if len(div_changes) >= 2:
        first_div = div_changes[0]["from_amount"]
        last_div = div_changes[-1]["to_amount"]
        total_periods = len(div_amounts)

        if first_div > 0 and total_periods > RollingWindow.SHORT:
            years = total_periods / RollingWindow.QUARTERS_PER_YEAR
            annual_growth = ((last_div / first_div) ** (1 / years) - 1) * 100
            growth_col = DerivedMetric.DIVIDEND_GROWTH_RATE.value
            df_with_qoq.loc[ticker_mask, growth_col] = annual_growth

        increases = [change for change in div_changes if change["growth_rate"] > 0]
        if len(increases) > 0:
            avg_increase_rate = np.mean([inc["growth_rate"] for inc in increases])
            freq_col = DerivedMetric.DIVIDEND_INCREASE_FREQ.value
            avg_col = DerivedMetric.AVG_DIVIDEND_INCREASE.value
            df_with_qoq.loc[ticker_mask, freq_col] = len(increases) / (
                total_periods / RollingWindow.QUARTERS_PER_YEAR
            )
            df_with_qoq.loc[ticker_mask, avg_col] = avg_increase_rate
    else:
        # Fewer than two detected dividend changes means growth cannot be
        # measured. Leave it NaN -- 0.0 here was indistinguishable from a real
        # zero-growth dividend, and made current payers look like they had
        # frozen their dividend.
        growth_col = DerivedMetric.DIVIDEND_GROWTH_RATE.value
        df_with_qoq.loc[ticker_mask, growth_col] = np.nan

    return df_with_qoq


def attach_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Add Sector, Industry and Sub_Industry columns, looked up per ticker.

    ``get_stock_classification`` has always worked, but nothing joined its
    output onto the frame, so ``calculate_sector_rankings`` returned
    immediately for want of a Sector column and the whole sector feature was
    dead. Tickers with no classification are marked "Unclassified", which
    ``DataQuality.invalid_values()`` excludes from ranking.

    Args:
        df: Stock data with a Ticker column

    Returns:
        DataFrame with the three classification columns added
    """
    from core.classifications import get_stock_classification

    ticker_col = Column.TICKER.value
    if ticker_col not in df.columns:
        return df

    unknown = "Unclassified"
    lookup = {}
    for ticker in df[ticker_col].dropna().unique():
        classification = get_stock_classification(ticker)
        if classification is None:
            lookup[ticker] = (unknown, unknown, unknown)
        else:
            lookup[ticker] = (
                classification.sector.value,
                classification.industry.value,
                classification.sub_industry.value,
            )

    df = df.copy()
    tickers = df[ticker_col]
    df[Column.SECTOR.value] = tickers.map(lambda t: lookup.get(t, (unknown,) * 3)[0])
    df[Column.INDUSTRY.value] = tickers.map(lambda t: lookup.get(t, (unknown,) * 3)[1])
    df[Column.SUB_INDUSTRY.value] = tickers.map(
        lambda t: lookup.get(t, (unknown,) * 3)[2]
    )
    return df


def latest_row_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: the highest Index, i.e. its latest reported quarter.

    Peer comparisons are between companies, so they need one observation each.
    Ranking every ticker-quarter row instead lets a ticker with 60 quarters of
    history outvote one with 20, and produces "ranks" far larger than the
    number of companies.
    """
    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value
    if ticker_col not in df.columns:
        return df
    if index_col not in df.columns:
        return df.groupby(ticker_col, sort=False).tail(1)
    return df.loc[df.groupby(ticker_col, sort=False)[index_col].idxmax()]


def calculate_sector_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each ticker against its sector peers on key metrics.

    The rank is a property of the company, computed from its latest reported
    quarter and against the other tickers' latest quarters. It is then repeated
    on every row of that ticker, so it reads the same wherever it is joined --
    it is deliberately not a per-quarter time series.

    Args:
        df: Stock data with calculated metrics, including a Sector column

    Returns:
        DataFrame with sector ranking columns added
    """
    sector_col = Column.SECTOR.value
    ticker_col = Column.TICKER.value
    if sector_col not in df.columns or ticker_col not in df.columns:
        return df

    df_with_rankings = df.copy()
    latest = latest_row_per_ticker(df_with_rankings)

    positive_metrics = DerivedMetric.positive_ranking_metrics()
    negative_metrics = DerivedMetric.negative_ranking_metrics()
    invalid_sectors = DataQuality.invalid_values()

    for metric in positive_metrics + negative_metrics:
        if metric not in df.columns:
            continue

        ranking_col = RankingSuffix.SECTOR_RANK.column_name(metric)
        ranks_by_ticker = {}

        for sector in latest[sector_col].dropna().unique():
            if sector in invalid_sectors:
                continue

            peers = latest[latest[sector_col] == sector]
            values = peers.set_index(ticker_col)[metric].dropna()
            if len(values) <= 1:
                continue

            ranks = values.rank(method="min", ascending=metric in negative_metrics)
            ranks_by_ticker.update(ranks.to_dict())

        df_with_rankings[ranking_col] = df_with_rankings[ticker_col].map(
            ranks_by_ticker
        )

    return df_with_rankings


def calculate_outperformance_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each ticker against its sector and the whole market.

    Like the sector ranks, the benchmark is built from one row per ticker (its
    latest quarter), not from every historical row -- a row-weighted average
    lets long-history tickers dominate and is not the "average ticker" the
    figure is read as.

    Metrics already expressed in percent are reported as a percentage-point
    difference (``_MarketGapPP`` / ``_SectorGapPP``). Level metrics keep the
    ratio form (``_MarketOutperf`` / ``_SectorOutperf``), where 100 means
    average.

    Args:
        df: Stock data with calculated metrics

    Returns:
        DataFrame with outperformance columns added
    """
    df_with_outperf = df.copy()
    sector_col = Column.SECTOR.value
    ticker_col = Column.TICKER.value

    if ticker_col not in df.columns:
        return df_with_outperf

    metrics = [
        DerivedMetric.PRICE_QOQ.value,
        DerivedMetric.EPS_QOQ.value,
        DerivedMetric.REVENUE_QOQ.value,
        DerivedMetric.EPS_TTM.value,
        DerivedMetric.REVENUE_TTM.value,
    ]
    percent_metrics = DerivedMetric.percent_unit_metrics()
    invalid_sectors = DataQuality.invalid_values()

    latest = latest_row_per_ticker(df_with_outperf)

    for metric in metrics:
        if metric not in df.columns:
            continue

        as_pp = metric in percent_metrics
        market_suffix = (
            RankingSuffix.MARKET_GAP_PP if as_pp else RankingSuffix.MARKET_OUTPERF
        )
        sector_suffix = (
            RankingSuffix.SECTOR_GAP_PP if as_pp else RankingSuffix.SECTOR_OUTPERF
        )

        values = latest.set_index(ticker_col)[metric].dropna()
        if values.empty:
            continue

        def compare(series, benchmark):
            if pd.isna(benchmark):
                return None
            if as_pp:
                return series - benchmark
            if benchmark == 0:
                return None
            return (series / benchmark) * 100

        market_result = compare(values, values.mean())
        market_col = market_suffix.column_name(metric)
        df_with_outperf[market_col] = (
            df_with_outperf[ticker_col].map(market_result)
            if market_result is not None
            else np.nan
        )

        sector_col_name = sector_suffix.column_name(metric)
        df_with_outperf[sector_col_name] = np.nan
        if sector_col not in latest.columns:
            continue

        sector_result = {}
        sectors = latest.set_index(ticker_col)[sector_col]
        for sector in sectors.dropna().unique():
            if sector in invalid_sectors:
                continue
            members = [t for t in values.index if sectors.get(t) == sector]
            if not members:
                continue
            peer_values = values[members]
            compared = compare(peer_values, peer_values.mean())
            if compared is not None:
                sector_result.update(compared.to_dict())

        df_with_outperf[sector_col_name] = df_with_outperf[ticker_col].map(
            sector_result
        )

    return df_with_outperf


def calculate_downside_capture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate downside capture ratio for each ticker.

    Measures how much a stock falls during market downturns.

    Args:
        df: Stock data with Price_QoQ calculated

    Returns:
        DataFrame with downside capture column added
    """
    price_qoq_col = DerivedMetric.PRICE_QOQ.value
    if price_qoq_col not in df.columns:
        return df

    df_with_downside = df.copy()
    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value
    downside_col = DerivedMetric.DOWNSIDE_CAPTURE.value

    market_performance = df.groupby(index_col)[price_qoq_col].mean().dropna()
    negative_periods = market_performance[market_performance < 0]

    if len(negative_periods) == 0:
        df_with_downside[downside_col] = np.nan
        return df_with_downside

    df_with_downside[downside_col] = np.nan
    min_periods = Thresholds.MIN_DOWNSIDE_PERIODS

    for ticker in df[ticker_col].unique():
        ticker_mask = df_with_downside[ticker_col] == ticker
        ticker_data = df_with_downside[ticker_mask].copy()

        ticker_downside_periods = []
        market_downside_periods = []

        for index, market_return in negative_periods.items():
            ticker_return = ticker_data[ticker_data[index_col] == index][price_qoq_col]
            if not ticker_return.empty and not pd.isna(ticker_return.iloc[0]):
                ticker_downside_periods.append(ticker_return.iloc[0])
                market_downside_periods.append(market_return)

        if len(ticker_downside_periods) >= min_periods:
            ticker_avg_down = np.mean(ticker_downside_periods)
            market_avg_down = np.mean(market_downside_periods)

            if market_avg_down != 0:
                downside_capture = (ticker_avg_down / market_avg_down) * 100
                df_with_downside.loc[ticker_mask, downside_col] = downside_capture

    return df_with_downside
