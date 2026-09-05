"""
EPS prediction module with ticker-specific optimization.

This module provides earnings prediction functionality without UI dependencies.
"""

import pandas as pd

from core.enums import (
    Column,
    DerivedMetric,
    Metric,
    PredictionKey,
    StrategyPolicy,
)


def _eps_crosses_zero(ticker_data: pd.DataFrame, quarters: int = 12) -> bool:
    """Whether recent EPS changes sign.

    Growth rates across a sign change are not interpretable, and the ±200%
    outlier filter discards most of them, so any forecast for such a ticker
    rests on very little.
    """
    eps = ticker_data[Metric.EPS.value].dropna().tail(quarters)
    if len(eps) < 2:
        return False
    signs = eps.gt(0)
    return bool(signs.nunique() > 1)


def predict_next_eps(df: pd.DataFrame, ticker: str) -> dict | None:
    """
    Predict next quarter EPS using ticker-specific optimal strategies.

    Uses individual backtesting results to select the best strategy for each
    ticker.

    Args:
        df: Stock data with QoQ calculations
        ticker: Stock ticker symbol

    Returns:
        Prediction results with comprehensive scenarios, or None if
        insufficient data
    """
    from core.backtesting import (
        STRATEGY_SOURCE_BACKTESTED,
        STRATEGY_SOURCE_GLOBAL_DEFAULT,
        STRATEGY_SOURCE_MAPPING_UNAVAILABLE,
        get_ticker_strategy_with_source,
    )
    from core.strategies import get_strategy

    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value

    ticker_data = df[df[ticker_col] == ticker].copy()
    ticker_data = ticker_data.sort_values(index_col)

    # Choose the strategy for this ticker
    optimal_strategy_name, strategy_source = get_ticker_strategy_with_source(
        ticker, default_strategy=StrategyPolicy.GLOBAL_DEFAULT
    )
    basic_prediction = get_strategy(optimal_strategy_name)(ticker_data)

    # One strategy failing is not the same as a ticker being unpredictable.
    # INTC's EPS crosses zero, which only the seasonal strategy cannot fit --
    # while it was INTC's mapped strategy, the ticker silently had no forecast
    # at all even though four other strategies handle it fine.
    fallback_from = None
    if basic_prediction is None:
        for candidate in StrategyPolicy.FALLBACK_ORDER:
            if candidate == optimal_strategy_name:
                continue
            basic_prediction = get_strategy(candidate)(ticker_data)
            if basic_prediction is not None:
                fallback_from = optimal_strategy_name
                optimal_strategy_name = candidate
                break

    # Genuinely not predictable: no strategy could fit this ticker.
    if basic_prediction is None:
        return None

    # Enhance the prediction with EPS_TTM and price calculations
    prediction = basic_prediction.copy()

    # Calculate predicted EPS_TTM scenarios
    current_eps_ttm = None
    predicted_eps_ttm = None
    best_case_eps_ttm = None
    worst_case_eps_ttm = None

    eps_col = Metric.EPS.value
    eps_ttm_col = DerivedMetric.EPS_TTM.value
    eps_data = ticker_data[eps_col].dropna()

    if eps_ttm_col in ticker_data.columns and len(eps_data) >= 4:
        eps_ttm_data = ticker_data[eps_ttm_col].dropna()
        if len(eps_ttm_data) > 0:
            current_eps_ttm = eps_ttm_data.iloc[-1]

            recent_eps = eps_data.tail(4)
            if len(recent_eps) >= 4:
                last_3_quarters = recent_eps.iloc[-3:].sum()

                predicted_eps_ttm = (
                    last_3_quarters + prediction[PredictionKey.PREDICTED_EPS]
                )
                best_case_eps_ttm = (
                    last_3_quarters + prediction[PredictionKey.BEST_CASE_EPS]
                )
                worst_case_eps_ttm = (
                    last_3_quarters + prediction[PredictionKey.WORST_CASE_EPS]
                )
            elif len(recent_eps) >= 1:
                actual_quarters_sum = recent_eps.sum()
                avg_quarter = actual_quarters_sum / len(recent_eps)
                missing_quarters = 4 - len(recent_eps)
                estimated_missing = avg_quarter * missing_quarters

                latest_eps = prediction[PredictionKey.LATEST_EPS]
                predicted_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + prediction[PredictionKey.PREDICTED_EPS]
                    - latest_eps
                )
                best_case_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + prediction[PredictionKey.BEST_CASE_EPS]
                    - latest_eps
                )
                worst_case_eps_ttm = (
                    actual_quarters_sum
                    + estimated_missing
                    + prediction[PredictionKey.WORST_CASE_EPS]
                    - latest_eps
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

    price_col = Metric.PRICE.value
    multiple_col = DerivedMetric.MULTIPLE.value

    has_price = price_col in ticker_data.columns
    has_multiple = multiple_col in ticker_data.columns
    if has_price and has_multiple:
        price_data = ticker_data[price_col].dropna()
        multiple_data = ticker_data[multiple_col].dropna()

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

    # Build methodology string. It has to say where the strategy came from: the
    # old string claimed "backtested optimal" unconditionally, including when
    # the mapping had not loaded at all.
    strategy_display = optimal_strategy_name.replace("_", " ").title()
    if fallback_from is not None:
        # The label belongs to the strategy that was chosen, not the one that
        # actually ran, so describe the substitution rather than mislabelling it.
        chosen_display = fallback_from.replace("_", " ").title()
        chosen_origin = (
            "global default"
            if strategy_source == STRATEGY_SOURCE_GLOBAL_DEFAULT
            else "selected"
        )
        methodology = (
            f"{strategy_display} — fell back from {chosen_display} "
            f"({chosen_origin}), which could not fit this ticker"
        )
    elif strategy_source == STRATEGY_SOURCE_BACKTESTED:
        methodology = f"Ticker-specific {strategy_display} (backtested optimal)"
    elif strategy_source == STRATEGY_SOURCE_MAPPING_UNAVAILABLE:
        methodology = (
            f"{strategy_display} (default; backtested strategy mapping unavailable)"
        )
    elif strategy_source == STRATEGY_SOURCE_GLOBAL_DEFAULT:
        methodology = f"{strategy_display} (global default)"
    else:
        methodology = (
            f"{strategy_display} (default; no backtested strategy for {ticker})"
        )

    # EPS changing sign makes every growth rate in these strategies unstable:
    # percent change across zero is not meaningful, so the number is reported
    # but should not be leaned on.
    crosses_zero = _eps_crosses_zero(ticker_data)
    if crosses_zero:
        methodology += " — caution: EPS crosses zero, growth rates are unstable"

    # Add the enhanced predictions to the result
    prediction.update(
        {
            PredictionKey.CURRENT_EPS_TTM: current_eps_ttm,
            PredictionKey.PREDICTED_EPS_TTM: predicted_eps_ttm,
            PredictionKey.BEST_CASE_EPS_TTM: best_case_eps_ttm,
            PredictionKey.WORST_CASE_EPS_TTM: worst_case_eps_ttm,
            PredictionKey.PREDICTED_EPS_TTM_GROWTH: predicted_eps_ttm_growth,
            PredictionKey.BEST_CASE_EPS_TTM_GROWTH: best_case_eps_ttm_growth,
            PredictionKey.WORST_CASE_EPS_TTM_GROWTH: worst_case_eps_ttm_growth,
            PredictionKey.CURRENT_PRICE: current_price,
            PredictionKey.CURRENT_MULTIPLE: current_multiple,
            PredictionKey.PREDICTED_PRICE: predicted_price,
            PredictionKey.BEST_CASE_PRICE: best_case_price,
            PredictionKey.WORST_CASE_PRICE: worst_case_price,
            PredictionKey.PREDICTED_PRICE_GROWTH: predicted_price_growth,
            PredictionKey.BEST_CASE_PRICE_GROWTH: best_case_price_growth,
            PredictionKey.WORST_CASE_PRICE_GROWTH: worst_case_price_growth,
            PredictionKey.NEXT_INDEX: ticker_data[index_col].max() + 1,
            PredictionKey.METHODOLOGY: methodology,
            PredictionKey.STRATEGY: optimal_strategy_name,
            PredictionKey.STRATEGY_SOURCE: strategy_source,
            PredictionKey.EPS_CROSSES_ZERO: crosses_zero,
        }
    )

    return prediction
