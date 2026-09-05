"""
StockInsights Core Library

This package contains the core data processing and analysis logic for stock
financial analysis, separated from any UI dependencies.

The core library can be imported and used in any Python application:
- Streamlit dashboards
- Flask/FastAPI web services
- CLI tools
- Jupyter notebooks
- Batch processing scripts

Example usage:
    from core import (
        load_stock_data,
        calculate_qoq_changes,
        calculate_sector_rankings,
        predict_next_eps,
    )

    # Load and process data
    df = load_stock_data("data/StockData_Indexed.xlsx")
    df = calculate_qoq_changes(df)
    df = calculate_sector_rankings(df)

    # Get predictions
    prediction = predict_next_eps(df, "AAPL")
"""

# Backtesting
from core.backtesting import (
    backtest_strategy,
    get_best_strategy,
    get_ticker_strategy,
    load_ticker_strategy_mapping,
    run_backtest_comparison,
    run_multi_ticker_backtest,
    save_ticker_strategy_mapping,
)

# Classifications
from core.classifications import (
    Industry,
    Sector,
    StockClassification,
    StockSymbol,
    SubIndustry,
    get_stock_classification,
    get_stocks_by_industry,
    get_stocks_by_sector,
    normalize_symbol,
)
from core.data_processing import (
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    load_stock_data,
    load_stock_data_with_stats,
    report_range,
    report_sort_key,
)

# Enums and constants (consistent vocabulary)
from core.enums import (
    BacktestConfig,
    ChartDefaults,
    Column,
    Confidence,
    DataQuality,
    DefaultTickers,
    DerivedMetric,
    FilePaths,
    Metric,
    PredictionKey,
    RankingSuffix,
    RollingWindow,
    Scenario,
    Strategy,
    StrategyWeights,
    Thresholds,
    TimeWindow,
)
from core.predictions import predict_next_eps

# Strategies
from core.strategies import (
    get_all_strategies,
    get_strategy,
    momentum_strategy,
    seasonal_strategy,
    simple_average_strategy,
    trend_analysis_strategy,
    weighted_growth_strategy,
)

__all__ = [
    # Enums and constants (consistent vocabulary)
    "Strategy",
    "Confidence",
    "Scenario",
    "Metric",
    "DerivedMetric",
    "Column",
    "TimeWindow",
    "DataQuality",
    "RankingSuffix",
    "Thresholds",
    "PredictionKey",
    "RollingWindow",
    "BacktestConfig",
    "StrategyWeights",
    "ChartDefaults",
    "FilePaths",
    "DefaultTickers",
    # Data loading
    "load_stock_data",
    "load_stock_data_with_stats",
    "report_range",
    "report_sort_key",
    # Core calculations
    "calculate_qoq_changes",
    "calculate_sector_rankings",
    "calculate_outperformance_ratios",
    "calculate_downside_capture",
    # Predictions
    "predict_next_eps",
    # Strategies
    "get_all_strategies",
    "get_strategy",
    "weighted_growth_strategy",
    "simple_average_strategy",
    "momentum_strategy",
    "trend_analysis_strategy",
    "seasonal_strategy",
    # Backtesting
    "backtest_strategy",
    "run_backtest_comparison",
    "run_multi_ticker_backtest",
    "get_best_strategy",
    "get_ticker_strategy",
    "load_ticker_strategy_mapping",
    "save_ticker_strategy_mapping",
    # Classifications
    "Sector",
    "Industry",
    "SubIndustry",
    "StockSymbol",
    "StockClassification",
    "get_stock_classification",
    "get_stocks_by_sector",
    "get_stocks_by_industry",
    "normalize_symbol",
]

__version__ = "1.0.0"
