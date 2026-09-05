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


def load_stock_data(file_path: str) -> pd.DataFrame | None:
    """
    Load and clean stock data from an Excel file.

    Args:
        file_path: Path to the Excel file

    Returns:
        Cleaned DataFrame or None if loading fails
    """
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
            df = df.drop_duplicates(subset=[ticker_col, report_col], keep="last")

        # Drop tickers flagged in config/excluded_tickers.json
        from core.exclusions import filter_excluded

        df, _ = filter_excluded(df)

        return df
    except FileNotFoundError:
        return None
    except Exception:
        return None


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
            multiple = price_data / eps_ttm_data
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
            payout_ratio = (annual_div / eps_ttm_data) * 100
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
                rolling_mean = revenue_qoq_values.rolling(
                    window=RollingWindow.LONG, min_periods=RollingWindow.SHORT
                ).mean()
                rolling_std = revenue_qoq_values.rolling(
                    window=RollingWindow.LONG, min_periods=RollingWindow.SHORT
                ).std()
                revenue_consistency = 100 - ((rolling_std / rolling_mean.abs()) * 100)
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
                peg_ratio = multiple_data / eps_growth_annual.abs()
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


def calculate_sector_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sector rankings for each ticker based on key metrics.

    Args:
        df: Stock data with calculated metrics

    Returns:
        DataFrame with sector ranking columns added
    """
    sector_col = Column.SECTOR.value
    if sector_col not in df.columns:
        return df

    df_with_rankings = df.copy()

    positive_metrics = DerivedMetric.positive_ranking_metrics()
    negative_metrics = DerivedMetric.negative_ranking_metrics()

    invalid_sectors = DataQuality.invalid_values()

    for metric in positive_metrics + negative_metrics:
        if metric in df.columns:
            ranking_col = RankingSuffix.SECTOR_RANK.column_name(metric)
            df_with_rankings[ranking_col] = np.nan

            for sector in df[sector_col].unique():
                if sector in invalid_sectors:
                    continue

                sector_mask = df_with_rankings[sector_col] == sector
                sector_data = df_with_rankings[sector_mask][metric].dropna()

                if len(sector_data) > 1:
                    if metric in positive_metrics:
                        ranks = sector_data.rank(method="min", ascending=False)
                    else:
                        ranks = sector_data.rank(method="min", ascending=True)

                    df_with_rankings.loc[sector_mask, ranking_col] = ranks.reindex(
                        df_with_rankings[sector_mask].index
                    )

    return df_with_rankings


def calculate_outperformance_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate outperformance ratios vs sector and overall market averages.

    Args:
        df: Stock data with calculated metrics

    Returns:
        DataFrame with outperformance ratio columns added
    """
    df_with_outperf = df.copy()
    sector_col = Column.SECTOR.value

    metrics = [
        DerivedMetric.PRICE_QOQ.value,
        DerivedMetric.EPS_QOQ.value,
        DerivedMetric.REVENUE_QOQ.value,
        DerivedMetric.EPS_TTM.value,
        DerivedMetric.REVENUE_TTM.value,
    ]

    invalid_sectors = DataQuality.invalid_values()

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Market outperformance
        market_avg = df[metric].mean()
        market_outperf_col = RankingSuffix.MARKET_OUTPERF.column_name(metric)
        df_with_outperf[market_outperf_col] = (
            (df[metric] / market_avg) * 100 if market_avg != 0 else np.nan
        )

        # Sector outperformance
        if sector_col in df.columns:
            sector_outperf_col = RankingSuffix.SECTOR_OUTPERF.column_name(metric)
            df_with_outperf[sector_outperf_col] = np.nan

            for sector in df[sector_col].unique():
                if sector in invalid_sectors:
                    continue

                sector_mask = df_with_outperf[sector_col] == sector
                sector_avg = df_with_outperf[sector_mask][metric].mean()

                if sector_avg != 0 and not pd.isna(sector_avg):
                    sector_outperf = (
                        df_with_outperf.loc[sector_mask, metric] / sector_avg
                    ) * 100
                    df_with_outperf.loc[sector_mask, sector_outperf_col] = (
                        sector_outperf
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
