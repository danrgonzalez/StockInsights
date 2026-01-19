"""
EPS prediction module with multiple strategies and ticker-specific optimization.

This module provides earnings prediction functionality without UI dependencies.
"""

import pandas as pd


def predict_next_eps(df: pd.DataFrame, ticker: str) -> dict | None:
    """
    Predict next quarter EPS using ticker-specific optimal strategies.

    Uses individual backtesting results to select the best strategy for each ticker.

    Args:
        df: Stock data with QoQ calculations
        ticker: Stock ticker symbol

    Returns:
        Prediction results with comprehensive scenarios, or None if insufficient data
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
    current_eps_ttm = None
    predicted_eps_ttm = None
    best_case_eps_ttm = None
    worst_case_eps_ttm = None

    eps_data = ticker_data["EPS"].dropna()

    if "EPS_TTM" in ticker_data.columns and len(eps_data) >= 4:
        eps_ttm_data = ticker_data["EPS_TTM"].dropna()
        if len(eps_ttm_data) > 0:
            current_eps_ttm = eps_ttm_data.iloc[-1]

            recent_eps = eps_data.tail(4)
            if len(recent_eps) >= 4:
                last_3_quarters = recent_eps.iloc[-3:].sum()

                predicted_eps_ttm = last_3_quarters + prediction["predicted_eps"]
                best_case_eps_ttm = last_3_quarters + prediction["best_case_eps"]
                worst_case_eps_ttm = last_3_quarters + prediction["worst_case_eps"]
            elif len(recent_eps) >= 1:
                actual_quarters_sum = recent_eps.sum()
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

    # Calculate EPS_TTM growth rates
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
    current_price = None
    current_multiple = None
    predicted_price = None
    best_case_price = None
    worst_case_price = None
    predicted_price_growth = None
    best_case_price_growth = None
    worst_case_price_growth = None

    if "Price" in ticker_data.columns and "Multiple" in ticker_data.columns:
        price_data = ticker_data["Price"].dropna()
        multiple_data = ticker_data["Multiple"].dropna()

        if len(price_data) > 0:
            current_price = price_data.iloc[-1]

        if len(multiple_data) > 0:
            current_multiple = multiple_data.iloc[-1]

            if (
                current_multiple is not None
                and not pd.isna(current_multiple)
                and current_multiple > 0
                and predicted_eps_ttm is not None
            ):
                predicted_price = predicted_eps_ttm * current_multiple
                best_case_price = best_case_eps_ttm * current_multiple
                worst_case_price = worst_case_eps_ttm * current_multiple

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
