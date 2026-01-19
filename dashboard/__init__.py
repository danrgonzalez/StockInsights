"""
StockInsights Dashboard Package

This package contains the Streamlit dashboard components for stock visualization
and analysis. These components can be reused in other Streamlit applications.

Components:
- app.py: Main Streamlit dashboard entry point
- data_utils.py: Streamlit-cached data processing functions
- charts.py: Plotly chart creation functions
- ui_components.py: Reusable Streamlit UI components
"""

from dashboard.charts import (
    create_combined_peg_pegy_chart,
    create_comparison_chart,
    create_eps_prediction_chart,
    create_eps_ttm_prediction_chart,
    create_metric_chart,
    create_price_prediction_chart,
    create_qoq_chart,
)
from dashboard.data_utils import (
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    load_data,
    predict_next_eps,
)
from dashboard.ui_components import display_summary_stats

__all__ = [
    # Data utilities
    "load_data",
    "calculate_qoq_changes",
    "calculate_sector_rankings",
    "calculate_outperformance_ratios",
    "calculate_downside_capture",
    "predict_next_eps",
    # Charts
    "create_metric_chart",
    "create_qoq_chart",
    "create_comparison_chart",
    "create_eps_prediction_chart",
    "create_eps_ttm_prediction_chart",
    "create_price_prediction_chart",
    "create_combined_peg_pegy_chart",
    # UI Components
    "display_summary_stats",
]
