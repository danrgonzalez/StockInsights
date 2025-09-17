import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_data(file_path):
    """Load the processed stock data with caching"""
    try:
        df = pd.read_excel(file_path)
        # Clean numeric columns - convert non-numeric values to NaN
        numeric_columns = ["EPS", "Revenue", "Price", "DivAmt", "Index"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Clean and normalize ticker symbols to prevent duplicates
        if "Ticker" in df.columns:
            # Import normalization function if available
            try:
                from stock_classifications import normalize_symbol

                df["Ticker"] = df["Ticker"].astype(str).apply(normalize_symbol)
            except ImportError:
                # Fallback to basic cleaning
                df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

        # Clean other text columns
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in ["Ticker", "Report"]:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("", np.nan)

        # Check for and warn about duplicate tickers after cleaning
        if "Ticker" in df.columns:
            original_count = len(df)
            df_clean = df.drop_duplicates(subset=["Ticker", "Report"], keep="first")
            dropped_count = original_count - len(df_clean)
            if dropped_count > 0:
                st.warning(
                    f"Removed {dropped_count} duplicate ticker/report "
                    "combinations during data cleaning"
                )
                df = df_clean
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def calculate_qoq_changes(df):
    """Calculate Quarter-over-Quarter percent changes for each ticker"""
    df_with_qoq = df.copy()
    df_with_qoq = df_with_qoq.sort_values(["Ticker", "Index"])
    for ticker in df_with_qoq["Ticker"].unique():
        ticker_mask = df_with_qoq["Ticker"] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()
        for metric in ["EPS", "Revenue"]:
            if metric in ticker_data.columns:
                ttm_values = ticker_data[metric].rolling(window=4, min_periods=4).sum()
                df_with_qoq.loc[ticker_mask, f"{metric}_TTM"] = ttm_values
        if (
            "Price" in ticker_data.columns
            and "EPS_TTM" in df_with_qoq.loc[ticker_mask].columns
        ):
            price_data = ticker_data["Price"]
            eps_ttm_data = df_with_qoq.loc[ticker_mask, "EPS_TTM"]
            multiple = price_data / eps_ttm_data
            multiple = multiple.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, "Multiple"] = multiple

        # Calculate dividend yield (DivAmt/Price * 100) - Quarterly
        if "Price" in ticker_data.columns and "DivAmt" in ticker_data.columns:
            price_data = ticker_data["Price"]
            div_data = ticker_data["DivAmt"]
            dividend_yield = (div_data / price_data) * 100
            dividend_yield = dividend_yield.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, "DivYield"] = dividend_yield

            # Calculate annualized dividend yield (DivAmt * 4 / Price * 100)
            dividend_yield_annual = (div_data * 4 / price_data) * 100
            dividend_yield_annual = dividend_yield_annual.replace(
                [np.inf, -np.inf], np.nan
            )
            df_with_qoq.loc[ticker_mask, "DivYieldAnnual"] = dividend_yield_annual

        # Calculate Payout Ratio (Annual DivAmt / EPS_TTM * 100)
        if (
            "DivAmt" in ticker_data.columns
            and "EPS_TTM" in df_with_qoq.loc[ticker_mask].columns
        ):
            div_data = ticker_data["DivAmt"]
            eps_ttm_data = df_with_qoq.loc[ticker_mask, "EPS_TTM"]
            # Annualize dividend by multiplying quarterly dividend by 4
            annual_div = div_data * 4
            payout_ratio = (annual_div / eps_ttm_data) * 100
            payout_ratio = payout_ratio.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, "PayoutRatio"] = payout_ratio
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

        # EPS Growth Momentum (EPS_4Q_Avg - EPS_8Q_Avg)
        if "EPS_QoQ" in ticker_data_with_qoq.columns:
            eps_qoq_values = ticker_data_with_qoq["EPS_QoQ"].dropna()
            if len(eps_qoq_values) >= 8:
                rolling_4q = eps_qoq_values.rolling(window=4, min_periods=4).mean()
                rolling_8q = eps_qoq_values.rolling(window=8, min_periods=8).mean()
                eps_momentum = rolling_4q - rolling_8q
                df_with_qoq.loc[ticker_mask, "EPSMomentum"] = eps_momentum

        # Price Volatility (Standard deviation of Price_QoQ over 8Q window)
        if "Price_QoQ" in ticker_data_with_qoq.columns:
            price_qoq_values = ticker_data_with_qoq["Price_QoQ"].dropna()
            if len(price_qoq_values) >= 4:
                price_volatility = price_qoq_values.rolling(
                    window=8, min_periods=4
                ).std()
                df_with_qoq.loc[ticker_mask, "PriceVolatility"] = price_volatility

        # Revenue Consistency (Coefficient of variation for Revenue_QoQ)
        if "Revenue_QoQ" in ticker_data_with_qoq.columns:
            revenue_qoq_values = ticker_data_with_qoq["Revenue_QoQ"].dropna()
            if len(revenue_qoq_values) >= 4:
                rolling_mean = revenue_qoq_values.rolling(
                    window=8, min_periods=4
                ).mean()
                rolling_std = revenue_qoq_values.rolling(window=8, min_periods=4).std()
                # Coefficient of variation = std / mean * 100,
                # inverted for consistency score
                revenue_consistency = 100 - ((rolling_std / rolling_mean.abs()) * 100)
                revenue_consistency = revenue_consistency.replace(
                    [np.inf, -np.inf], np.nan
                )
                df_with_qoq.loc[ticker_mask, "RevenueConsistency"] = revenue_consistency

        # Dividend Growth Rate - Based on actual dividend changes over time
        # Find when dividends actually increased/decreased, then calculate growth rate
        if "DivAmt" in ticker_data_with_qoq.columns:
            div_amounts = ticker_data_with_qoq["DivAmt"].dropna()
            if len(div_amounts) >= 4:
                # Find dividend change points (where amount actually changed)
                div_changes = []
                current_div = None

                for i, (idx, div_amt) in enumerate(div_amounts.items()):
                    if current_div is None:
                        current_div = div_amt
                        last_change_idx = i
                    elif (
                        abs(div_amt - current_div) > 0.001
                    ):  # Tolerance for floating point comparison
                        # Dividend changed!
                        periods_since_last_change = i - last_change_idx
                        if current_div > 0:  # Avoid division by zero
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
                    # Calculate annualized growth rate from first to last change
                    first_div = div_changes[0]["from_amount"]
                    last_div = div_changes[-1]["to_amount"]
                    total_periods = len(div_amounts)

                    if first_div > 0 and total_periods > 4:
                        # Annualized dividend growth rate
                        years = total_periods / 4  # Assuming quarterly data
                        annual_growth = (
                            (last_div / first_div) ** (1 / years) - 1
                        ) * 100

                        # Assign this growth rate to recent periods
                        df_with_qoq.loc[ticker_mask, "DivGrowthRate"] = annual_growth

                    # Also track frequency of increases
                    increases = [
                        change for change in div_changes if change["growth_rate"] > 0
                    ]
                    if len(increases) > 0:
                        avg_increase_rate = np.mean(
                            [inc["growth_rate"] for inc in increases]
                        )
                        df_with_qoq.loc[ticker_mask, "DivIncreaseFreq"] = len(
                            increases
                        ) / (
                            total_periods / 4
                        )  # increases per year
                        df_with_qoq.loc[
                            ticker_mask, "AvgDivIncrease"
                        ] = avg_increase_rate
                else:
                    # Not enough dividend changes to calculate meaningful growth
                    df_with_qoq.loc[ticker_mask, "DivGrowthRate"] = 0.0

        # PEG Ratio (P/E Multiple / EPS Growth Rate)
        if (
            "Multiple" in ticker_data_with_qoq.columns
            and "EPS_QoQ" in ticker_data_with_qoq.columns
        ):
            multiple_data = ticker_data_with_qoq["Multiple"]
            eps_qoq_values = ticker_data_with_qoq["EPS_QoQ"].dropna()
            if len(eps_qoq_values) >= 4:
                # Use 4Q rolling average of EPS growth for PEG calculation
                eps_growth_4q = eps_qoq_values.rolling(window=4, min_periods=4).mean()
                # Annualize the quarterly growth rate: (1 + quarterly_rate/100)^4 - 1
                eps_growth_annual = ((1 + eps_growth_4q / 100) ** 4 - 1) * 100
                peg_ratio = (
                    multiple_data / eps_growth_annual.abs()
                )  # Use absolute value to handle negative growth
                peg_ratio = peg_ratio.replace([np.inf, -np.inf], np.nan)
                df_with_qoq.loc[ticker_mask, "PEGRatio"] = peg_ratio

        # PEGY Ratio (PEG Ratio / Annualized Dividend Yield)
        if (
            "PEGRatio" in df_with_qoq.loc[ticker_mask].columns
            and "DivYieldAnnual" in df_with_qoq.loc[ticker_mask].columns
        ):
            peg_data = df_with_qoq.loc[ticker_mask, "PEGRatio"]
            div_yield_annual_data = df_with_qoq.loc[ticker_mask, "DivYieldAnnual"]
            # Only calculate PEGY if annualized dividend yield > 0
            pegy_ratio = np.where(
                div_yield_annual_data > 0, peg_data / div_yield_annual_data, np.nan
            )
            df_with_qoq.loc[ticker_mask, "PEGYRatio"] = pegy_ratio

    return df_with_qoq


def predict_next_eps(df, ticker):
    """
    Predict next quarter EPS using ticker-specific optimal strategies.

    Uses individual backtesting results to select the best strategy for each ticker.

    Args:
        df (pandas.DataFrame): Stock data with QoQ calculations
        ticker (str): Stock ticker symbol

    Returns:
        dict or None: Prediction results with comprehensive scenarios
    """
    from multi_ticker_backtest import get_ticker_strategy
    from strategies import get_strategy

    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")

    # Get the optimal strategy for this specific ticker
    optimal_strategy_name = get_ticker_strategy(
        ticker, default_strategy="weighted_growth"
    )
    optimal_strategy_func = get_strategy(optimal_strategy_name)

    # Get basic prediction from optimal strategy
    basic_prediction = optimal_strategy_func(ticker_data)

    if basic_prediction is None:
        return None

    # Enhance the prediction with EPS_TTM and price calculations
    prediction = basic_prediction.copy()

    # Calculate predicted EPS_TTM scenarios
    # EPS_TTM = sum of last 4 quarters, so we replace the oldest quarter with predicted
    current_eps_ttm = None
    predicted_eps_ttm = None
    best_case_eps_ttm = None
    worst_case_eps_ttm = None

    eps_data = ticker_data["EPS"].dropna()

    if "EPS_TTM" in ticker_data.columns and len(eps_data) >= 4:
        # Get current EPS_TTM
        eps_ttm_data = ticker_data["EPS_TTM"].dropna()
        if len(eps_ttm_data) > 0:
            current_eps_ttm = eps_ttm_data.iloc[-1]

            # Get the last 4 quarters of EPS (including latest)
            recent_eps = eps_data.tail(4)
            if len(recent_eps) >= 4:
                # Replace the oldest quarter (first in the 4Q window) with predicted
                last_3_quarters = recent_eps.iloc[-3:].sum()  # Most recent 3 quarters

                predicted_eps_ttm = last_3_quarters + prediction["predicted_eps"]
                best_case_eps_ttm = last_3_quarters + prediction["best_case_eps"]
                worst_case_eps_ttm = last_3_quarters + prediction["worst_case_eps"]
            elif len(recent_eps) >= 1:
                # If we don't have 4 full quarters, use what we have plus prediction
                actual_quarters_sum = recent_eps.sum()
                # Estimate missing quarters (conservative approach)
                avg_quarter = actual_quarters_sum / len(recent_eps)
                missing_quarters = 4 - len(recent_eps)
                estimated_missing = avg_quarter * missing_quarters

                predicted_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + prediction["predicted_eps"]
                    - prediction["latest_eps"]
                )
                best_case_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + prediction["best_case_eps"]
                    - prediction["latest_eps"]
                )
                worst_case_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + prediction["worst_case_eps"]
                    - prediction["latest_eps"]
                )

    # Calculate EPS_TTM growth rates if we have both current and predicted
    predicted_eps_ttm_growth = None
    best_case_eps_ttm_growth = None
    worst_case_eps_ttm_growth = None

    if (
        current_eps_ttm is not None
        and predicted_eps_ttm is not None
        and current_eps_ttm != 0
    ):
        predicted_eps_ttm_growth = (
            (predicted_eps_ttm - current_eps_ttm) / abs(current_eps_ttm)
        ) * 100
        best_case_eps_ttm_growth = (
            (best_case_eps_ttm - current_eps_ttm) / abs(current_eps_ttm)
        ) * 100
        worst_case_eps_ttm_growth = (
            (worst_case_eps_ttm - current_eps_ttm) / abs(current_eps_ttm)
        ) * 100

    # Calculate predicted Price scenarios using current Multiple
    # Price = EPS_TTM × Multiple, so predicted price = predicted EPS_TTM ×
    # current Multiple
    current_price = None
    current_multiple = None
    predicted_price = None
    best_case_price = None
    worst_case_price = None
    predicted_price_growth = None
    best_case_price_growth = None
    worst_case_price_growth = None

    if "Price" in ticker_data.columns and "Multiple" in ticker_data.columns:
        # Get current price and multiple
        price_data = ticker_data["Price"].dropna()
        multiple_data = ticker_data["Multiple"].dropna()

        if len(price_data) > 0:
            current_price = price_data.iloc[-1]

        if len(multiple_data) > 0:
            current_multiple = multiple_data.iloc[-1]

            # Only calculate price predictions if we have both
            # current multiple and predicted EPS_TTM
            if (
                current_multiple is not None
                and not pd.isna(current_multiple)
                and current_multiple > 0
                and predicted_eps_ttm is not None
            ):
                predicted_price = predicted_eps_ttm * current_multiple
                best_case_price = best_case_eps_ttm * current_multiple
                worst_case_price = worst_case_eps_ttm * current_multiple

                # Calculate price growth rates
                if current_price is not None and current_price != 0:
                    predicted_price_growth = (
                        (predicted_price - current_price) / abs(current_price)
                    ) * 100
                    best_case_price_growth = (
                        (best_case_price - current_price) / abs(current_price)
                    ) * 100
                    worst_case_price_growth = (
                        (worst_case_price - current_price) / abs(current_price)
                    ) * 100

    # Add the enhanced predictions to the result
    prediction.update(
        {
            "current_eps_ttm": current_eps_ttm,
            "predicted_eps_ttm": predicted_eps_ttm,
            "best_case_eps_ttm": best_case_eps_ttm,
            "worst_case_eps_ttm": worst_case_eps_ttm,
            "predicted_eps_ttm_growth": predicted_eps_ttm_growth,
            "best_case_eps_ttm_growth": best_case_eps_ttm_growth,
            "worst_case_eps_ttm_growth": worst_case_eps_ttm_growth,
            "current_price": current_price,
            "current_multiple": current_multiple,
            "predicted_price": predicted_price,
            "best_case_price": best_case_price,
            "worst_case_price": worst_case_price,
            "predicted_price_growth": predicted_price_growth,
            "best_case_price_growth": best_case_price_growth,
            "worst_case_price_growth": worst_case_price_growth,
            "next_index": ticker_data["Index"].max() + 1,
            "methodology": (
                f"Ticker-specific {optimal_strategy_name.replace('_', ' ').title()} "
                "(backtested optimal)"
            ),
        }
    )

    return prediction


def calculate_sector_rankings(df):
    """
    Calculate sector rankings for each ticker based on key metrics.

    Args:
        df (pandas.DataFrame): Stock data with calculated metrics

    Returns:
        pandas.DataFrame: DataFrame with sector ranking columns added
    """
    if "Sector" not in df.columns:
        return df

    df_with_rankings = df.copy()

    # Metrics to rank (higher is better)
    positive_metrics = [
        "EPS_TTM",
        "Revenue_TTM",
        "DivYield",
        "DivYieldAnnual",
        "RevenueConsistency",
        "EPSMomentum",
    ]
    # Metrics to rank (lower is better)
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
                        # Higher values get better ranks (1 = best)
                        ranks = sector_data.rank(method="min", ascending=False)
                    else:
                        # Lower values get better ranks (1 = best)
                        ranks = sector_data.rank(method="min", ascending=True)

                    df_with_rankings.loc[sector_mask, ranking_col] = ranks.reindex(
                        df_with_rankings[sector_mask].index
                    )

    return df_with_rankings


def calculate_outperformance_ratios(df):
    """
    Calculate outperformance ratios vs sector and overall market averages.

    Args:
        df (pandas.DataFrame): Stock data with calculated metrics

    Returns:
        pandas.DataFrame: DataFrame with outperformance ratio columns added
    """
    df_with_outperf = df.copy()

    # Metrics to calculate outperformance for
    metrics = ["Price_QoQ", "EPS_QoQ", "Revenue_QoQ", "EPS_TTM", "Revenue_TTM"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Calculate market (overall) average
        market_avg = df[metric].mean()
        market_outperf_col = f"{metric}_MarketOutperf"
        df_with_outperf[market_outperf_col] = (
            (df[metric] / market_avg) * 100 if market_avg != 0 else np.nan
        )

        # Calculate sector outperformance if sector data available
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


def calculate_downside_capture(df):
    """
    Calculate downside capture ratio - how much a stock falls during market downturns.

    Args:
        df (pandas.DataFrame): Stock data with Price_QoQ calculated

    Returns:
        pandas.DataFrame: DataFrame with downside capture column added
    """
    if "Price_QoQ" not in df.columns:
        return df

    df_with_downside = df.copy()

    # Calculate market average price performance for each period
    market_performance = df.groupby("Index")["Price_QoQ"].mean().dropna()

    # Identify negative market periods (market down quarters)
    negative_periods = market_performance[market_performance < 0]

    if len(negative_periods) == 0:
        df_with_downside["DownsideCapture"] = np.nan
        return df_with_downside

    # Calculate downside capture for each ticker
    df_with_downside["DownsideCapture"] = np.nan

    for ticker in df["Ticker"].unique():
        ticker_mask = df_with_downside["Ticker"] == ticker
        ticker_data = df_with_downside[ticker_mask].copy()

        # Get ticker performance during negative market periods
        ticker_downside_periods = []
        market_downside_periods = []

        for index, market_return in negative_periods.items():
            ticker_return = ticker_data[ticker_data["Index"] == index]["Price_QoQ"]
            if not ticker_return.empty and not pd.isna(ticker_return.iloc[0]):
                ticker_downside_periods.append(ticker_return.iloc[0])
                market_downside_periods.append(market_return)

        if (
            len(ticker_downside_periods) >= 3
        ):  # Need at least 3 down periods for meaningful calculation
            # Calculate average downside capture
            ticker_avg_down = np.mean(ticker_downside_periods)
            market_avg_down = np.mean(market_downside_periods)

            if market_avg_down != 0:
                downside_capture = (ticker_avg_down / market_avg_down) * 100
                # Populate all rows for this ticker with the same downside capture ratio
                df_with_downside.loc[ticker_mask, "DownsideCapture"] = downside_capture

    return df_with_downside
