"""
EPS prediction strategies module.

This module provides multiple prediction algorithms for forecasting next quarter
EPS based on historical data patterns.
"""

import numpy as np

from core.enums import (
    Confidence,
    DerivedMetric,
    Metric,
    PredictionKey,
    Strategy,
    Thresholds,
)


def weighted_growth_strategy(ticker_data):
    """
    Original weighted growth rate strategy.

    Uses weighted average of recent QoQ growth rates to project forward.
    70% weight on recent 4Q, 30% weight on recent 8Q (if available).

    Args:
        ticker_data: Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if (
        Metric.EPS.value not in ticker_data.columns
        or DerivedMetric.EPS_QOQ.value not in ticker_data.columns
    ):
        return None

    # Get recent EPS and QoQ data
    eps_data = ticker_data[Metric.EPS.value].dropna()
    qoq_data = ticker_data[DerivedMetric.EPS_QOQ.value].dropna()

    # Need at least 4 quarters of data for meaningful prediction
    if (
        len(eps_data) < Thresholds.MIN_QUARTERS_FOR_PREDICTION
        or len(qoq_data) < Thresholds.MIN_QOQ_DATA_POINTS
    ):
        return None

    # Get latest EPS value
    latest_eps = eps_data.iloc[-1]

    # Calculate recent growth rates
    recent_qoq_4q = qoq_data.tail(4)
    recent_qoq_8q = qoq_data.tail(8) if len(qoq_data) >= 8 else recent_qoq_4q

    # Remove extreme outliers (beyond ±200% growth)
    outlier_min = Thresholds.GROWTH_OUTLIER_MIN
    outlier_max = Thresholds.GROWTH_OUTLIER_MAX
    recent_qoq_4q_clean = recent_qoq_4q[
        (recent_qoq_4q >= outlier_min) & (recent_qoq_4q <= outlier_max)
    ]
    recent_qoq_8q_clean = recent_qoq_8q[
        (recent_qoq_8q >= outlier_min) & (recent_qoq_8q <= outlier_max)
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
        confidence = (
            Confidence.HIGH.value
            if len(recent_qoq_4q_clean) == 4
            else Confidence.MEDIUM.value
        )
    else:
        predicted_growth = avg_growth_4q
        confidence = (
            Confidence.MEDIUM.value
            if len(recent_qoq_4q_clean) >= 3
            else Confidence.LOW.value
        )

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
    predicted_volatility = max(predicted_volatility, Thresholds.MIN_VOLATILITY)

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
            floor = Thresholds.PREDICTED_GROWTH_FLOOR
            predicted_growth = max(predicted_growth, floor)
            predicted_eps = latest_eps * (1 + predicted_growth / 100)
            confidence = Confidence.LOW.value

        if worst_case_eps < 0:
            worst_floor = Thresholds.WORST_CASE_FLOOR
            worst_case_growth = max(worst_case_growth, worst_floor)
            worst_case_eps = latest_eps * (1 + worst_case_growth / 100)

        # Best case shouldn't be unrealistically high
        if best_case_growth > Thresholds.BEST_CASE_CEILING:
            best_case_growth = Thresholds.BEST_CASE_CEILING
            best_case_eps = latest_eps * (1 + best_case_growth / 100)

    return {
        PredictionKey.PREDICTED_EPS: predicted_eps,
        PredictionKey.BEST_CASE_EPS: best_case_eps,
        PredictionKey.WORST_CASE_EPS: worst_case_eps,
        PredictionKey.LATEST_EPS: latest_eps,
        PredictionKey.PREDICTED_GROWTH: predicted_growth,
        PredictionKey.BEST_CASE_GROWTH: best_case_growth,
        PredictionKey.WORST_CASE_GROWTH: worst_case_growth,
        PredictionKey.VOLATILITY: predicted_volatility,
        PredictionKey.CONFIDENCE: confidence,
        PredictionKey.DATA_POINTS: len(qoq_data),
        PredictionKey.GROWTH_4Q: avg_growth_4q,
        PredictionKey.GROWTH_8Q: avg_growth_8q if len(qoq_data) >= 8 else None,
        PredictionKey.STD_4Q: growth_std_4q,
        PredictionKey.STD_8Q: growth_std_8q if len(qoq_data) >= 8 else None,
        PredictionKey.METHODOLOGY: "Weighted average with ±1σ volatility bands",
    }


def simple_average_strategy(ticker_data):
    """
    Simple average of last N quarters strategy.

    Uses simple mean of recent QoQ growth rates.

    Args:
        ticker_data: Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if (
        Metric.EPS.value not in ticker_data.columns
        or DerivedMetric.EPS_QOQ.value not in ticker_data.columns
    ):
        return None

    eps_data = ticker_data[Metric.EPS.value].dropna()
    qoq_data = ticker_data[DerivedMetric.EPS_QOQ.value].dropna()

    if (
        len(eps_data) < Thresholds.MIN_QUARTERS_FOR_PREDICTION
        or len(qoq_data) < Thresholds.MIN_QOQ_DATA_POINTS
    ):
        return None

    latest_eps = eps_data.iloc[-1]

    # Use last 4 quarters for prediction
    recent_qoq = qoq_data.tail(4)
    outlier_min = Thresholds.GROWTH_OUTLIER_MIN
    outlier_max = Thresholds.GROWTH_OUTLIER_MAX
    recent_qoq_clean = recent_qoq[
        (recent_qoq >= outlier_min) & (recent_qoq <= outlier_max)
    ]

    if len(recent_qoq_clean) < 2:
        return None

    predicted_growth = recent_qoq_clean.mean()
    predicted_volatility = max(recent_qoq_clean.std(), Thresholds.MIN_VOLATILITY)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_growth = predicted_growth + predicted_volatility
    worst_growth = predicted_growth - predicted_volatility
    best_case_eps = latest_eps * (1 + best_growth / 100)
    worst_case_eps = latest_eps * (1 + worst_growth / 100)

    # Apply same safety caps as original
    if latest_eps > 0 and predicted_eps < 0:
        floor = Thresholds.PREDICTED_GROWTH_FLOOR
        predicted_growth = max(predicted_growth, floor)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        PredictionKey.PREDICTED_EPS: predicted_eps,
        PredictionKey.BEST_CASE_EPS: best_case_eps,
        PredictionKey.WORST_CASE_EPS: worst_case_eps,
        PredictionKey.LATEST_EPS: latest_eps,
        PredictionKey.PREDICTED_GROWTH: predicted_growth,
        PredictionKey.BEST_CASE_GROWTH: best_growth,
        PredictionKey.WORST_CASE_GROWTH: worst_growth,
        PredictionKey.VOLATILITY: predicted_volatility,
        PredictionKey.CONFIDENCE: Confidence.MEDIUM.value,
        PredictionKey.DATA_POINTS: len(qoq_data),
        PredictionKey.METHODOLOGY: "Simple 4Q average",
    }


def momentum_strategy(ticker_data):
    """
    Momentum-based strategy that gives more weight to recent quarters.

    Uses exponentially weighted moving average with higher weights on recent
    data.

    Args:
        ticker_data: Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if (
        Metric.EPS.value not in ticker_data.columns
        or DerivedMetric.EPS_QOQ.value not in ticker_data.columns
    ):
        return None

    eps_data = ticker_data[Metric.EPS.value].dropna()
    qoq_data = ticker_data[DerivedMetric.EPS_QOQ.value].dropna()

    if (
        len(eps_data) < Thresholds.MIN_QUARTERS_FOR_PREDICTION
        or len(qoq_data) < Thresholds.MIN_QOQ_DATA_POINTS
    ):
        return None

    latest_eps = eps_data.iloc[-1]

    # Use last 6 quarters if available, otherwise last 4
    recent_qoq = qoq_data.tail(6) if len(qoq_data) >= 6 else qoq_data.tail(4)
    outlier_min = Thresholds.GROWTH_OUTLIER_MIN
    outlier_max = Thresholds.GROWTH_OUTLIER_MAX
    recent_qoq_clean = recent_qoq[
        (recent_qoq >= outlier_min) & (recent_qoq <= outlier_max)
    ]

    if len(recent_qoq_clean) < 2:
        return None

    # Apply exponential weights (most recent gets highest weight)
    weights = np.exp(np.arange(len(recent_qoq_clean)) * 0.3)
    weights = weights / weights.sum()  # Normalize to sum to 1

    predicted_growth = np.average(recent_qoq_clean, weights=weights)
    predicted_volatility = max(recent_qoq_clean.std(), Thresholds.MIN_VOLATILITY)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_growth = predicted_growth + predicted_volatility
    worst_growth = predicted_growth - predicted_volatility
    best_case_eps = latest_eps * (1 + best_growth / 100)
    worst_case_eps = latest_eps * (1 + worst_growth / 100)

    # Apply safety caps
    if latest_eps > 0 and predicted_eps < 0:
        floor = Thresholds.PREDICTED_GROWTH_FLOOR
        predicted_growth = max(predicted_growth, floor)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        PredictionKey.PREDICTED_EPS: predicted_eps,
        PredictionKey.BEST_CASE_EPS: best_case_eps,
        PredictionKey.WORST_CASE_EPS: worst_case_eps,
        PredictionKey.LATEST_EPS: latest_eps,
        PredictionKey.PREDICTED_GROWTH: predicted_growth,
        PredictionKey.BEST_CASE_GROWTH: best_growth,
        PredictionKey.WORST_CASE_GROWTH: worst_growth,
        PredictionKey.VOLATILITY: predicted_volatility,
        PredictionKey.CONFIDENCE: Confidence.MEDIUM.value,
        PredictionKey.DATA_POINTS: len(qoq_data),
        PredictionKey.METHODOLOGY: "Exponentially weighted momentum",
    }


def trend_analysis_strategy(ticker_data):
    """
    Trend analysis strategy using linear regression on recent QoQ growth.

    Fits a linear trend to recent QoQ data and extrapolates forward.

    Args:
        ticker_data: Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if (
        Metric.EPS.value not in ticker_data.columns
        or DerivedMetric.EPS_QOQ.value not in ticker_data.columns
    ):
        return None

    eps_data = ticker_data[Metric.EPS.value].dropna()
    qoq_data = ticker_data[DerivedMetric.EPS_QOQ.value].dropna()

    min_quarters = Thresholds.MIN_QUARTERS_FOR_PREDICTION
    if len(eps_data) < min_quarters or len(qoq_data) < min_quarters:
        return None

    latest_eps = eps_data.iloc[-1]

    # Use last 8 quarters if available, otherwise last 6 or 4
    n_quarters = min(8, len(qoq_data))
    recent_qoq = qoq_data.tail(n_quarters)
    outlier_min = Thresholds.GROWTH_OUTLIER_MIN
    outlier_max = Thresholds.GROWTH_OUTLIER_MAX
    recent_qoq_clean = recent_qoq[
        (recent_qoq >= outlier_min) & (recent_qoq <= outlier_max)
    ]

    if len(recent_qoq_clean) < 3:
        return None

    # Fit linear trend to QoQ data
    x = np.arange(len(recent_qoq_clean))
    y = recent_qoq_clean.values

    # Simple linear regression: y = mx + b
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n

    # Predict next quarter growth (x = len(recent_qoq_clean))
    predicted_growth = slope * len(recent_qoq_clean) + intercept

    # Calculate volatility from residuals
    predicted_values = slope * x + intercept
    residuals = y - predicted_values
    predicted_volatility = max(np.std(residuals), Thresholds.MIN_VOLATILITY)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_growth = predicted_growth + predicted_volatility
    worst_growth = predicted_growth - predicted_volatility
    best_case_eps = latest_eps * (1 + best_growth / 100)
    worst_case_eps = latest_eps * (1 + worst_growth / 100)

    # Apply safety caps
    if latest_eps > 0 and predicted_eps < 0:
        floor = Thresholds.PREDICTED_GROWTH_FLOOR
        predicted_growth = max(predicted_growth, floor)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        PredictionKey.PREDICTED_EPS: predicted_eps,
        PredictionKey.BEST_CASE_EPS: best_case_eps,
        PredictionKey.WORST_CASE_EPS: worst_case_eps,
        PredictionKey.LATEST_EPS: latest_eps,
        PredictionKey.PREDICTED_GROWTH: predicted_growth,
        PredictionKey.BEST_CASE_GROWTH: best_growth,
        PredictionKey.WORST_CASE_GROWTH: worst_growth,
        PredictionKey.VOLATILITY: predicted_volatility,
        PredictionKey.CONFIDENCE: Confidence.MEDIUM.value,
        PredictionKey.DATA_POINTS: len(qoq_data),
        PredictionKey.METHODOLOGY: "Linear trend analysis",
    }


def seasonal_strategy(ticker_data):
    """
    Seasonal strategy that looks at year-over-year patterns.

    Uses same quarter from previous year(s) to predict growth.

    Args:
        ticker_data: Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if Metric.EPS.value not in ticker_data.columns:
        return None

    eps_data = ticker_data[Metric.EPS.value].dropna()

    # Need at least 2 years of data
    if len(eps_data) < Thresholds.MIN_QUARTERS_FOR_SEASONAL:
        return None

    latest_eps = eps_data.iloc[-1]

    # Look at year-over-year growth patterns
    # Assuming quarterly data, look at quarters 4, 8, 12 quarters ago
    yoy_growths = []
    outlier_min = Thresholds.GROWTH_OUTLIER_MIN
    outlier_max = Thresholds.GROWTH_OUTLIER_MAX

    for year_back in [4, 8, 12]:
        if len(eps_data) > year_back:
            previous_year_eps = eps_data.iloc[-(year_back + 1)]
            if previous_year_eps != 0:
                yoy_growth = (
                    (latest_eps - previous_year_eps) / abs(previous_year_eps)
                ) * 100
                if outlier_min <= yoy_growth <= outlier_max:
                    yoy_growths.append(yoy_growth)

    if len(yoy_growths) == 0:
        return None

    # Average the year-over-year growth rates
    predicted_growth = np.mean(yoy_growths)
    predicted_volatility = max(np.std(yoy_growths), Thresholds.MIN_VOLATILITY)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_growth = predicted_growth + predicted_volatility
    worst_growth = predicted_growth - predicted_volatility
    best_case_eps = latest_eps * (1 + best_growth / 100)
    worst_case_eps = latest_eps * (1 + worst_growth / 100)

    # Apply safety caps
    if latest_eps > 0 and predicted_eps < 0:
        floor = Thresholds.PREDICTED_GROWTH_FLOOR
        predicted_growth = max(predicted_growth, floor)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        PredictionKey.PREDICTED_EPS: predicted_eps,
        PredictionKey.BEST_CASE_EPS: best_case_eps,
        PredictionKey.WORST_CASE_EPS: worst_case_eps,
        PredictionKey.LATEST_EPS: latest_eps,
        PredictionKey.PREDICTED_GROWTH: predicted_growth,
        PredictionKey.BEST_CASE_GROWTH: best_growth,
        PredictionKey.WORST_CASE_GROWTH: worst_growth,
        PredictionKey.VOLATILITY: predicted_volatility,
        PredictionKey.CONFIDENCE: Confidence.MEDIUM.value,
        PredictionKey.DATA_POINTS: len(eps_data),
        PredictionKey.METHODOLOGY: "Year-over-year seasonal",
    }


# Registry of all available strategies using enum
STRATEGIES = {
    Strategy.WEIGHTED_GROWTH.value: weighted_growth_strategy,
    Strategy.SIMPLE_AVERAGE.value: simple_average_strategy,
    Strategy.MOMENTUM.value: momentum_strategy,
    Strategy.TREND_ANALYSIS.value: trend_analysis_strategy,
    Strategy.SEASONAL.value: seasonal_strategy,
}


def get_strategy(strategy_name):
    """
    Get a strategy function by name.

    Args:
        strategy_name: Strategy name string or Strategy enum value

    Returns:
        Strategy function, defaults to weighted_growth if not found
    """
    if isinstance(strategy_name, Strategy):
        strategy_name = strategy_name.value
    return STRATEGIES.get(strategy_name, weighted_growth_strategy)


def get_all_strategies():
    """Get all available strategies."""
    return STRATEGIES
