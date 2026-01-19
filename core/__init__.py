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
    df = load_stock_data("StockData_Indexed.xlsx")
    df = calculate_qoq_changes(df)
    df = calculate_sector_rankings(df)

    # Get predictions
    prediction = predict_next_eps(df, "AAPL")
"""

from core.data_processing import (
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    load_stock_data,
)
from core.predictions import predict_next_eps

__all__ = [
    # Data loading
    "load_stock_data",
    # Core calculations
    "calculate_qoq_changes",
    "calculate_sector_rankings",
    "calculate_outperformance_ratios",
    "calculate_downside_capture",
    # Predictions
    "predict_next_eps",
]

__version__ = "1.0.0"
