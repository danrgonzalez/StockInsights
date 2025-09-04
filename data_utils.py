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
    df_with_qoq = df_with_qoq.sort_values(["Ticker", "Index"])
    for ticker in df_with_qoq["Ticker"].unique():
        ticker_mask = df_with_qoq["Ticker"] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()
        for metric in ["EPS", "Revenue", "Price", "EPS_TTM", "Revenue_TTM", "Multiple"]:
            if metric in ticker_data.columns:
                qoq_change = ticker_data[metric].pct_change(fill_method=None) * 100
                df_with_qoq.loc[ticker_mask, f"{metric}_QoQ"] = qoq_change
    return df_with_qoq


def predict_next_eps(df, ticker):
    """
    Predict next quarter EPS using the best performing strategy from backtesting.

    Uses seasonal strategy (year-over-year patterns) which performed best on historical data.

    Args:
        df (pandas.DataFrame): Stock data with QoQ calculations
        ticker (str): Stock ticker symbol

    Returns:
        dict or None: Prediction results with comprehensive scenarios
    """
    from strategies import seasonal_strategy

    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")

    # Get basic prediction from seasonal strategy
    basic_prediction = seasonal_strategy(ticker_data)

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
    # Price = EPS_TTM × Multiple, so predicted price = predicted EPS_TTM × current Multiple
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

            # Only calculate price predictions if we have both current multiple and predicted EPS_TTM
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
            "methodology": "Year-over-year seasonal (backtest winner)",
        }
    )

    return prediction
