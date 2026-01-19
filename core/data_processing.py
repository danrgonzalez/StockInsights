"""
Core data processing module for stock financial analysis.

This module contains pure data processing logic without any UI dependencies,
making it suitable for use in any application (Streamlit, CLI, API, etc.).
"""

import numpy as np
import pandas as pd


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
        numeric_columns = ["EPS", "Revenue", "Price", "DivAmt", "Index"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean and normalize ticker symbols
        if "Ticker" in df.columns:
            try:
                from stock_classifications import normalize_symbol

                df["Ticker"] = df["Ticker"].astype(str).apply(normalize_symbol)
            except ImportError:
                df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

        # Clean other text columns
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in ["Ticker", "Report"]:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("", np.nan)

        # Remove duplicates
        if "Ticker" in df.columns:
            df = df.drop_duplicates(subset=["Ticker", "Report"], keep="first")

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
        df: Stock data DataFrame with Ticker, Index, EPS, Revenue, Price, DivAmt

    Returns:
        DataFrame with all calculated metrics added
    """
    df_with_qoq = df.copy()
    df_with_qoq = df_with_qoq.sort_values(["Ticker", "Index"])

    for ticker in df_with_qoq["Ticker"].unique():
        ticker_mask = df_with_qoq["Ticker"] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()

        # Calculate TTM (Trailing Twelve Months) values
        for metric in ["EPS", "Revenue"]:
            if metric in ticker_data.columns:
                ttm_values = ticker_data[metric].rolling(window=4, min_periods=4).sum()
                df_with_qoq.loc[ticker_mask, f"{metric}_TTM"] = ttm_values

        # Calculate P/E Multiple
        if (
            "Price" in ticker_data.columns
            and "EPS_TTM" in df_with_qoq.loc[ticker_mask].columns
        ):
            price_data = ticker_data["Price"]
            eps_ttm_data = df_with_qoq.loc[ticker_mask, "EPS_TTM"]
            multiple = price_data / eps_ttm_data
            multiple = multiple.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, "Multiple"] = multiple

        # Calculate dividend yields
        if "Price" in ticker_data.columns and "DivAmt" in ticker_data.columns:
            price_data = ticker_data["Price"]
            div_data = ticker_data["DivAmt"]

            # Quarterly dividend yield
            dividend_yield = (div_data / price_data) * 100
            dividend_yield = dividend_yield.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, "DivYield"] = dividend_yield

            # Annualized dividend yield
            dividend_yield_annual = (div_data * 4 / price_data) * 100
            dividend_yield_annual = dividend_yield_annual.replace(
                [np.inf, -np.inf], np.nan
            )
            df_with_qoq.loc[ticker_mask, "DivYieldAnnual"] = dividend_yield_annual

        # Calculate Payout Ratio
        if (
            "DivAmt" in ticker_data.columns
            and "EPS_TTM" in df_with_qoq.loc[ticker_mask].columns
        ):
            div_data = ticker_data["DivAmt"]
            eps_ttm_data = df_with_qoq.loc[ticker_mask, "EPS_TTM"]
            annual_div = div_data * 4
            payout_ratio = (annual_div / eps_ttm_data) * 100
            payout_ratio = payout_ratio.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, "PayoutRatio"] = payout_ratio

    # Calculate QoQ changes for all metrics
    df_with_qoq = df_with_qoq.sort_values(["Ticker", "Index"])

    for ticker in df_with_qoq["Ticker"].unique():
        ticker_mask = df_with_qoq["Ticker"] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()

        for metric in [
            "EPS",
            "Revenue",
            "Price",
            "EPS_TTM",
            "Revenue_TTM",
            "Multiple",
            "DivAmt",
            "DivYield",
            "DivYieldAnnual",
            "PayoutRatio",
        ]:
            if metric in ticker_data.columns:
                qoq_change = ticker_data[metric].pct_change(fill_method=None) * 100
                df_with_qoq.loc[ticker_mask, f"{metric}_QoQ"] = qoq_change

        # Calculate advanced metrics requiring QoQ data
        ticker_data_with_qoq = df_with_qoq[ticker_mask].copy()

        # EPS Growth Momentum
        if "EPS_QoQ" in ticker_data_with_qoq.columns:
            eps_qoq_values = ticker_data_with_qoq["EPS_QoQ"].dropna()
            if len(eps_qoq_values) >= 8:
                rolling_4q = eps_qoq_values.rolling(window=4, min_periods=4).mean()
                rolling_8q = eps_qoq_values.rolling(window=8, min_periods=8).mean()
                eps_momentum = rolling_4q - rolling_8q
                df_with_qoq.loc[ticker_mask, "EPSMomentum"] = eps_momentum

        # Price Volatility
        if "Price_QoQ" in ticker_data_with_qoq.columns:
            price_qoq_values = ticker_data_with_qoq["Price_QoQ"].dropna()
            if len(price_qoq_values) >= 4:
                price_volatility = price_qoq_values.rolling(
                    window=8, min_periods=4
                ).std()
                df_with_qoq.loc[ticker_mask, "PriceVolatility"] = price_volatility

        # Revenue Consistency
        if "Revenue_QoQ" in ticker_data_with_qoq.columns:
            revenue_qoq_values = ticker_data_with_qoq["Revenue_QoQ"].dropna()
            if len(revenue_qoq_values) >= 4:
                rolling_mean = revenue_qoq_values.rolling(
                    window=8, min_periods=4
                ).mean()
                rolling_std = revenue_qoq_values.rolling(window=8, min_periods=4).std()
                revenue_consistency = 100 - ((rolling_std / rolling_mean.abs()) * 100)
                revenue_consistency = revenue_consistency.replace(
                    [np.inf, -np.inf], np.nan
                )
                df_with_qoq.loc[ticker_mask, "RevenueConsistency"] = revenue_consistency

        # Dividend Growth Rate
        if "DivAmt" in ticker_data_with_qoq.columns:
            df_with_qoq = _calculate_dividend_growth(
                df_with_qoq, ticker_mask, ticker_data_with_qoq
            )

        # PEG Ratio
        if (
            "Multiple" in ticker_data_with_qoq.columns
            and "EPS_QoQ" in ticker_data_with_qoq.columns
        ):
            multiple_data = ticker_data_with_qoq["Multiple"]
            eps_qoq_values = ticker_data_with_qoq["EPS_QoQ"].dropna()
            if len(eps_qoq_values) >= 4:
                eps_growth_4q = eps_qoq_values.rolling(window=4, min_periods=4).mean()
                eps_growth_annual = ((1 + eps_growth_4q / 100) ** 4 - 1) * 100
                peg_ratio = multiple_data / eps_growth_annual.abs()
                peg_ratio = peg_ratio.replace([np.inf, -np.inf], np.nan)
                df_with_qoq.loc[ticker_mask, "PEGRatio"] = peg_ratio

        # PEGY Ratio
        if (
            "PEGRatio" in df_with_qoq.loc[ticker_mask].columns
            and "DivYieldAnnual" in df_with_qoq.loc[ticker_mask].columns
        ):
            peg_data = df_with_qoq.loc[ticker_mask, "PEGRatio"]
            div_yield_annual_data = df_with_qoq.loc[ticker_mask, "DivYieldAnnual"]
            pegy_ratio = np.where(
                div_yield_annual_data > 0, peg_data / div_yield_annual_data, np.nan
            )
            df_with_qoq.loc[ticker_mask, "PEGYRatio"] = pegy_ratio

    return df_with_qoq


def _calculate_dividend_growth(
    df_with_qoq: pd.DataFrame,
    ticker_mask: pd.Series,
    ticker_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate dividend growth rate for a ticker."""
    div_amounts = ticker_data["DivAmt"].dropna()
    if len(div_amounts) < 4:
        return df_with_qoq

    # Find dividend change points
    div_changes = []
    current_div = None
    last_change_idx = 0

    for i, (idx, div_amt) in enumerate(div_amounts.items()):
        if current_div is None:
            current_div = div_amt
            last_change_idx = i
        elif abs(div_amt - current_div) > 0.001:
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

        if first_div > 0 and total_periods > 4:
            years = total_periods / 4
            annual_growth = ((last_div / first_div) ** (1 / years) - 1) * 100
            df_with_qoq.loc[ticker_mask, "DivGrowthRate"] = annual_growth

        increases = [change for change in div_changes if change["growth_rate"] > 0]
        if len(increases) > 0:
            avg_increase_rate = np.mean([inc["growth_rate"] for inc in increases])
            df_with_qoq.loc[ticker_mask, "DivIncreaseFreq"] = len(increases) / (
                total_periods / 4
            )
            df_with_qoq.loc[ticker_mask, "AvgDivIncrease"] = avg_increase_rate
    else:
        df_with_qoq.loc[ticker_mask, "DivGrowthRate"] = 0.0

    return df_with_qoq


def calculate_sector_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sector rankings for each ticker based on key metrics.

    Args:
        df: Stock data with calculated metrics

    Returns:
        DataFrame with sector ranking columns added
    """
    if "Sector" not in df.columns:
        return df

    df_with_rankings = df.copy()

    positive_metrics = [
        "EPS_TTM",
        "Revenue_TTM",
        "DivYield",
        "DivYieldAnnual",
        "RevenueConsistency",
        "EPSMomentum",
    ]
    negative_metrics = ["Multiple", "PriceVolatility", "PEGRatio", "PEGYRatio"]

    for metric in positive_metrics + negative_metrics:
        if metric in df.columns:
            ranking_col = f"{metric}_SectorRank"
            df_with_rankings[ranking_col] = np.nan

            for sector in df["Sector"].unique():
                if sector in ["Unknown", "Unclassified", "N/A", None]:
                    continue

                sector_mask = df_with_rankings["Sector"] == sector
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

    metrics = ["Price_QoQ", "EPS_QoQ", "Revenue_QoQ", "EPS_TTM", "Revenue_TTM"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Market outperformance
        market_avg = df[metric].mean()
        market_outperf_col = f"{metric}_MarketOutperf"
        df_with_outperf[market_outperf_col] = (
            (df[metric] / market_avg) * 100 if market_avg != 0 else np.nan
        )

        # Sector outperformance
        if "Sector" in df.columns:
            sector_outperf_col = f"{metric}_SectorOutperf"
            df_with_outperf[sector_outperf_col] = np.nan

            for sector in df["Sector"].unique():
                if sector in ["Unknown", "Unclassified", "N/A", None]:
                    continue

                sector_mask = df_with_outperf["Sector"] == sector
                sector_avg = df_with_outperf[sector_mask][metric].mean()

                if sector_avg != 0 and not pd.isna(sector_avg):
                    sector_outperf = (
                        df_with_outperf.loc[sector_mask, metric] / sector_avg
                    ) * 100
                    df_with_outperf.loc[
                        sector_mask, sector_outperf_col
                    ] = sector_outperf

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
    if "Price_QoQ" not in df.columns:
        return df

    df_with_downside = df.copy()

    market_performance = df.groupby("Index")["Price_QoQ"].mean().dropna()
    negative_periods = market_performance[market_performance < 0]

    if len(negative_periods) == 0:
        df_with_downside["DownsideCapture"] = np.nan
        return df_with_downside

    df_with_downside["DownsideCapture"] = np.nan

    for ticker in df["Ticker"].unique():
        ticker_mask = df_with_downside["Ticker"] == ticker
        ticker_data = df_with_downside[ticker_mask].copy()

        ticker_downside_periods = []
        market_downside_periods = []

        for index, market_return in negative_periods.items():
            ticker_return = ticker_data[ticker_data["Index"] == index]["Price_QoQ"]
            if not ticker_return.empty and not pd.isna(ticker_return.iloc[0]):
                ticker_downside_periods.append(ticker_return.iloc[0])
                market_downside_periods.append(market_return)

        if len(ticker_downside_periods) >= 3:
            ticker_avg_down = np.mean(ticker_downside_periods)
            market_avg_down = np.mean(market_downside_periods)

            if market_avg_down != 0:
                downside_capture = (ticker_avg_down / market_avg_down) * 100
                df_with_downside.loc[ticker_mask, "DownsideCapture"] = downside_capture

    return df_with_downside
