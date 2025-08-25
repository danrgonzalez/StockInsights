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
    Predict next quarter EPS using growth rate approach.

    Uses weighted average of recent QoQ growth rates to project forward.

    Args:
        df (pandas.DataFrame): Stock data with QoQ calculations
        ticker (str): Stock ticker symbol

    Returns:
        dict or None: Prediction results with keys:
            - predicted_eps: Predicted EPS value
            - latest_eps: Current latest EPS
            - predicted_growth: Expected growth rate %
            - confidence: Confidence level (High/Medium/Low)
            - data_points: Number of QoQ data points used
            - growth_4q: Recent 4Q average growth
            - growth_8q: Recent 8Q average growth (if available)
    """
    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")

    if "EPS" not in ticker_data.columns or "EPS_QoQ" not in ticker_data.columns:
        return None

    # Get recent EPS and QoQ data
    eps_data = ticker_data["EPS"].dropna()
    qoq_data = ticker_data["EPS_QoQ"].dropna()

    # Need at least 4 quarters of data for meaningful prediction
    if len(eps_data) < 4 or len(qoq_data) < 3:
        return None

    # Get latest EPS value
    latest_eps = eps_data.iloc[-1]

    # Calculate recent growth rates
    recent_qoq_4q = qoq_data.tail(4)
    recent_qoq_8q = qoq_data.tail(8) if len(qoq_data) >= 8 else recent_qoq_4q

    # Remove extreme outliers (beyond ±200% growth)
    recent_qoq_4q_clean = recent_qoq_4q[
        (recent_qoq_4q >= -200) & (recent_qoq_4q <= 200)
    ]
    recent_qoq_8q_clean = recent_qoq_8q[
        (recent_qoq_8q >= -200) & (recent_qoq_8q <= 200)
    ]

    if len(recent_qoq_4q_clean) < 2:
        return None

    # Calculate average growth rates
    avg_growth_4q = recent_qoq_4q_clean.mean()
    avg_growth_8q = (
        recent_qoq_8q_clean.mean() if len(recent_qoq_8q_clean) >= 4 else avg_growth_4q
    )

    # Weighted average: favor recent performance but consider longer term
    if len(qoq_data) >= 8:
        predicted_growth = (0.7 * avg_growth_4q) + (0.3 * avg_growth_8q)
        confidence = "High" if len(recent_qoq_4q_clean) == 4 else "Medium"
    else:
        predicted_growth = avg_growth_4q
        confidence = "Medium" if len(recent_qoq_4q_clean) >= 3 else "Low"

    # Calculate historical volatility for scenario analysis
    growth_std_4q = recent_qoq_4q_clean.std()
    growth_std_8q = (
        recent_qoq_8q_clean.std() if len(recent_qoq_8q_clean) >= 4 else growth_std_4q
    )

    # Use weighted standard deviation (similar to growth rate weighting)
    if len(qoq_data) >= 8:
        predicted_volatility = (0.7 * growth_std_4q) + (0.3 * growth_std_8q)
    else:
        predicted_volatility = growth_std_4q

    # Handle cases with very low volatility (set minimum threshold)
    predicted_volatility = max(predicted_volatility, 5.0)  # Minimum 5% volatility

    # Calculate scenario growth rates (±1 standard deviation)
    best_case_growth = predicted_growth + predicted_volatility
    worst_case_growth = predicted_growth - predicted_volatility

    # Apply growth to latest EPS for all scenarios
    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_case_eps = latest_eps * (1 + best_case_growth / 100)
    worst_case_eps = latest_eps * (1 + worst_case_growth / 100)

    # Ensure predictions are reasonable (not negative for positive companies)
    if latest_eps > 0:
        if predicted_eps < 0:
            predicted_growth = max(predicted_growth, -50)  # Cap at -50%
            predicted_eps = latest_eps * (1 + predicted_growth / 100)
            confidence = "Low"

        if worst_case_eps < 0:
            worst_case_growth = max(worst_case_growth, -75)  # Cap worst case at -75%
            worst_case_eps = latest_eps * (1 + worst_case_growth / 100)

        # Best case shouldn't be unrealistically high (cap at +200%)
        if best_case_growth > 200:
            best_case_growth = 200
            best_case_eps = latest_eps * (1 + best_case_growth / 100)

    # Calculate predicted EPS_TTM scenarios
    # EPS_TTM = sum of last 4 quarters, so we replace the oldest quarter with predicted
    current_eps_ttm = None
    predicted_eps_ttm = None
    best_case_eps_ttm = None
    worst_case_eps_ttm = None

    if "EPS_TTM" in ticker_data.columns:
        # Get current EPS_TTM
        eps_ttm_data = ticker_data["EPS_TTM"].dropna()
        if len(eps_ttm_data) > 0:
            current_eps_ttm = eps_ttm_data.iloc[-1]

            # Get the last 4 quarters of EPS (including latest)
            recent_eps = eps_data.tail(4)
            if len(recent_eps) >= 4:
                # Replace the oldest quarter (first in the 4Q window) with predicted
                last_3_quarters = recent_eps.iloc[-3:].sum()  # Most recent 3 quarters

                predicted_eps_ttm = last_3_quarters + predicted_eps
                best_case_eps_ttm = last_3_quarters + best_case_eps
                worst_case_eps_ttm = last_3_quarters + worst_case_eps
            elif len(recent_eps) >= 1:
                # If we don't have 4 full quarters, use what we have plus prediction
                actual_quarters_sum = recent_eps.sum()
                # Estimate missing quarters (conservative approach)
                avg_quarter = actual_quarters_sum / len(recent_eps)
                missing_quarters = 4 - len(recent_eps)
                estimated_missing = avg_quarter * missing_quarters

                predicted_eps_ttm = (
                    actual_quarters_sum + estimated_missing + predicted_eps - latest_eps
                )
                best_case_eps_ttm = (
                    actual_quarters_sum + estimated_missing + best_case_eps - latest_eps
                )
                worst_case_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + worst_case_eps
                    - latest_eps
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

    return {
        "predicted_eps": predicted_eps,
        "best_case_eps": best_case_eps,
        "worst_case_eps": worst_case_eps,
        "latest_eps": latest_eps,
        "predicted_growth": predicted_growth,
        "best_case_growth": best_case_growth,
        "worst_case_growth": worst_case_growth,
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
        "volatility": predicted_volatility,
        "confidence": confidence,
        "data_points": len(qoq_data),
        "growth_4q": avg_growth_4q,
        "growth_8q": avg_growth_8q if len(qoq_data) >= 8 else None,
        "std_4q": growth_std_4q,
        "std_8q": growth_std_8q if len(qoq_data) >= 8 else None,
        "next_index": ticker_data["Index"].max() + 1,
        "methodology": "Weighted average with ±1σ volatility bands",
    }
