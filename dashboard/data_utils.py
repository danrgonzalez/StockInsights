"""
Data utilities module for the Streamlit dashboard.

This module provides Streamlit-compatible data processing functions with caching.
Most functionality is delegated to the core module for consistency.

For non-Streamlit applications (CLI, API, batch processing), import directly
from the core module:

    from core import (
        load_stock_data,
        calculate_qoq_changes,
        calculate_sector_rankings,
        predict_next_eps,
    )

The core module has no Streamlit dependencies and can be used in any
Python application.
"""

import streamlit as st

# Re-export core functions for backward compatibility
# These are the authoritative implementations
from core.data_processing import (  # noqa: F401
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    load_stock_data_with_stats,
)
from core.predictions import predict_next_eps  # noqa: F401


@st.cache_data
def load_data(file_path):
    """
    Load the processed stock data with Streamlit caching.

    A thin wrapper around core.load_stock_data_with_stats: the cleaning rules
    live in core so the dashboard and the batch paths cannot drift apart. This
    adds only caching and user-facing messages.

    Args:
        file_path: Path to the Excel file

    Returns:
        DataFrame or None if loading fails
    """
    df, stats = load_stock_data_with_stats(file_path)

    if stats["error"] == "not_found":
        st.error(f"File not found: {file_path}")
        return None
    if stats["error"] is not None:
        st.error(f"Error loading data: {stats['error']}")
        return None

    if stats["duplicates_dropped"]:
        st.warning(
            f"Removed {stats['duplicates_dropped']} duplicate ticker/report "
            "combinations during data cleaning"
        )

    if stats["excluded"]:
        summary = ", ".join(
            f"{ticker} ({count})" for ticker, count in sorted(stats["excluded"].items())
        )
        st.info(f"Excluded from the active universe (ticker_status.json): {summary}")

    return df
