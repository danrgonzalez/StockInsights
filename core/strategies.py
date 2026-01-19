import numpy as np


def weighted_growth_strategy(ticker_data):
    """
    Original weighted growth rate strategy.

    Uses weighted average of recent QoQ growth rates to project forward.
    70% weight on recent 4Q, 30% weight on recent 8Q (if available).

    Args:
        ticker_data (pandas.DataFrame): Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
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

    return {
        "predicted_eps": predicted_eps,
        "best_case_eps": best_case_eps,
        "worst_case_eps": worst_case_eps,
        "latest_eps": latest_eps,
        "predicted_growth": predicted_growth,
        "best_case_growth": best_case_growth,
        "worst_case_growth": worst_case_growth,
        "volatility": predicted_volatility,
        "confidence": confidence,
        "data_points": len(qoq_data),
        "growth_4q": avg_growth_4q,
        "growth_8q": avg_growth_8q if len(qoq_data) >= 8 else None,
        "std_4q": growth_std_4q,
        "std_8q": growth_std_8q if len(qoq_data) >= 8 else None,
        "methodology": "Weighted average with ±1σ volatility bands",
    }


def simple_average_strategy(ticker_data):
    """
    Simple average of last N quarters strategy.

    Uses simple mean of recent QoQ growth rates.

    Args:
        ticker_data (pandas.DataFrame): Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if "EPS" not in ticker_data.columns or "EPS_QoQ" not in ticker_data.columns:
        return None

    eps_data = ticker_data["EPS"].dropna()
    qoq_data = ticker_data["EPS_QoQ"].dropna()

    if len(eps_data) < 4 or len(qoq_data) < 3:
        return None

    latest_eps = eps_data.iloc[-1]

    # Use last 4 quarters for prediction
    recent_qoq = qoq_data.tail(4)
    recent_qoq_clean = recent_qoq[(recent_qoq >= -200) & (recent_qoq <= 200)]

    if len(recent_qoq_clean) < 2:
        return None

    predicted_growth = recent_qoq_clean.mean()
    predicted_volatility = max(recent_qoq_clean.std(), 5.0)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_case_eps = latest_eps * (1 + (predicted_growth + predicted_volatility) / 100)
    worst_case_eps = latest_eps * (1 + (predicted_growth - predicted_volatility) / 100)

    # Apply same safety caps as original
    if latest_eps > 0 and predicted_eps < 0:
        predicted_growth = max(predicted_growth, -50)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        "predicted_eps": predicted_eps,
        "best_case_eps": best_case_eps,
        "worst_case_eps": worst_case_eps,
        "latest_eps": latest_eps,
        "predicted_growth": predicted_growth,
        "best_case_growth": predicted_growth + predicted_volatility,
        "worst_case_growth": predicted_growth - predicted_volatility,
        "volatility": predicted_volatility,
        "confidence": "Medium",
        "data_points": len(qoq_data),
        "methodology": "Simple 4Q average",
    }


def momentum_strategy(ticker_data):
    """
    Momentum-based strategy that gives more weight to recent quarters.

    Uses exponentially weighted moving average with higher weights on recent data.

    Args:
        ticker_data (pandas.DataFrame): Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if "EPS" not in ticker_data.columns or "EPS_QoQ" not in ticker_data.columns:
        return None

    eps_data = ticker_data["EPS"].dropna()
    qoq_data = ticker_data["EPS_QoQ"].dropna()

    if len(eps_data) < 4 or len(qoq_data) < 3:
        return None

    latest_eps = eps_data.iloc[-1]

    # Use last 6 quarters if available, otherwise last 4
    recent_qoq = qoq_data.tail(6) if len(qoq_data) >= 6 else qoq_data.tail(4)
    recent_qoq_clean = recent_qoq[(recent_qoq >= -200) & (recent_qoq <= 200)]

    if len(recent_qoq_clean) < 2:
        return None

    # Apply exponential weights (most recent gets highest weight)
    weights = np.exp(np.arange(len(recent_qoq_clean)) * 0.3)  # Exponential decay
    weights = weights / weights.sum()  # Normalize to sum to 1

    predicted_growth = np.average(recent_qoq_clean, weights=weights)
    predicted_volatility = max(recent_qoq_clean.std(), 5.0)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_case_eps = latest_eps * (1 + (predicted_growth + predicted_volatility) / 100)
    worst_case_eps = latest_eps * (1 + (predicted_growth - predicted_volatility) / 100)

    # Apply safety caps
    if latest_eps > 0 and predicted_eps < 0:
        predicted_growth = max(predicted_growth, -50)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        "predicted_eps": predicted_eps,
        "best_case_eps": best_case_eps,
        "worst_case_eps": worst_case_eps,
        "latest_eps": latest_eps,
        "predicted_growth": predicted_growth,
        "best_case_growth": predicted_growth + predicted_volatility,
        "worst_case_growth": predicted_growth - predicted_volatility,
        "volatility": predicted_volatility,
        "confidence": "Medium",
        "data_points": len(qoq_data),
        "methodology": "Exponentially weighted momentum",
    }


def trend_analysis_strategy(ticker_data):
    """
    Trend analysis strategy using linear regression on recent QoQ growth.

    Fits a linear trend to recent QoQ data and extrapolates forward.

    Args:
        ticker_data (pandas.DataFrame): Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if "EPS" not in ticker_data.columns or "EPS_QoQ" not in ticker_data.columns:
        return None

    eps_data = ticker_data["EPS"].dropna()
    qoq_data = ticker_data["EPS_QoQ"].dropna()

    if len(eps_data) < 4 or len(qoq_data) < 4:
        return None

    latest_eps = eps_data.iloc[-1]

    # Use last 8 quarters if available, otherwise last 6 or 4
    n_quarters = min(8, len(qoq_data))
    recent_qoq = qoq_data.tail(n_quarters)
    recent_qoq_clean = recent_qoq[(recent_qoq >= -200) & (recent_qoq <= 200)]

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
    predicted_volatility = max(np.std(residuals), 5.0)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_case_eps = latest_eps * (1 + (predicted_growth + predicted_volatility) / 100)
    worst_case_eps = latest_eps * (1 + (predicted_growth - predicted_volatility) / 100)

    # Apply safety caps
    if latest_eps > 0 and predicted_eps < 0:
        predicted_growth = max(predicted_growth, -50)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        "predicted_eps": predicted_eps,
        "best_case_eps": best_case_eps,
        "worst_case_eps": worst_case_eps,
        "latest_eps": latest_eps,
        "predicted_growth": predicted_growth,
        "best_case_growth": predicted_growth + predicted_volatility,
        "worst_case_growth": predicted_growth - predicted_volatility,
        "volatility": predicted_volatility,
        "confidence": "Medium",
        "data_points": len(qoq_data),
        "methodology": "Linear trend analysis",
    }


def seasonal_strategy(ticker_data):
    """
    Seasonal strategy that looks at year-over-year patterns.

    Uses same quarter from previous year(s) to predict growth.

    Args:
        ticker_data (pandas.DataFrame): Stock data for single ticker, sorted by Index

    Returns:
        dict or None: Prediction results
    """
    if "EPS" not in ticker_data.columns:
        return None

    eps_data = ticker_data["EPS"].dropna()

    if len(eps_data) < 8:  # Need at least 2 years of data
        return None

    latest_eps = eps_data.iloc[-1]

    # Look at year-over-year growth patterns
    # Assuming quarterly data, look at quarters 4, 8, 12 quarters ago (1, 2, 3 years)
    yoy_growths = []

    for year_back in [4, 8, 12]:
        if len(eps_data) > year_back:
            previous_year_eps = eps_data.iloc[-(year_back + 1)]
            if previous_year_eps != 0:
                yoy_growth = (
                    (latest_eps - previous_year_eps) / abs(previous_year_eps)
                ) * 100
                if -200 <= yoy_growth <= 200:  # Filter outliers
                    yoy_growths.append(yoy_growth)

    if len(yoy_growths) == 0:
        return None

    # Average the year-over-year growth rates
    predicted_growth = np.mean(yoy_growths)
    predicted_volatility = max(np.std(yoy_growths), 5.0)

    predicted_eps = latest_eps * (1 + predicted_growth / 100)
    best_case_eps = latest_eps * (1 + (predicted_growth + predicted_volatility) / 100)
    worst_case_eps = latest_eps * (1 + (predicted_growth - predicted_volatility) / 100)

    # Apply safety caps
    if latest_eps > 0 and predicted_eps < 0:
        predicted_growth = max(predicted_growth, -50)
        predicted_eps = latest_eps * (1 + predicted_growth / 100)

    return {
        "predicted_eps": predicted_eps,
        "best_case_eps": best_case_eps,
        "worst_case_eps": worst_case_eps,
        "latest_eps": latest_eps,
        "predicted_growth": predicted_growth,
        "best_case_growth": predicted_growth + predicted_volatility,
        "worst_case_growth": predicted_growth - predicted_volatility,
        "volatility": predicted_volatility,
        "confidence": "Medium",
        "data_points": len(eps_data),
        "methodology": "Year-over-year seasonal",
    }


# Registry of all available strategies
STRATEGIES = {
    "weighted_growth": weighted_growth_strategy,
    "simple_average": simple_average_strategy,
    "momentum": momentum_strategy,
    "trend_analysis": trend_analysis_strategy,
    "seasonal": seasonal_strategy,
}


def get_strategy(strategy_name):
    """Get a strategy function by name."""
    return STRATEGIES.get(strategy_name, weighted_growth_strategy)


def get_all_strategies():
    """Get all available strategies."""
    return STRATEGIES
