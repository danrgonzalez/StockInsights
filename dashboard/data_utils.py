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

import numpy as np
import pandas as pd
import streamlit as st

# Re-export core functions for backward compatibility
# These are the authoritative implementations
from core.data_processing import (  # noqa: F401
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
)
from core.enums import Column, Metric
from core.predictions import predict_next_eps  # noqa: F401


@st.cache_data
def load_data(file_path):
    """
    Load the processed stock data with Streamlit caching.

    This is a Streamlit-specific wrapper around core.load_stock_data
    that adds caching and user-friendly error messages.

    Args:
        file_path: Path to the Excel file

    Returns:
        DataFrame or None if loading fails
    """
    try:
        df = pd.read_excel(file_path)

        # Clean numeric columns - convert non-numeric values to NaN
        numeric_columns = Metric.numeric_columns()
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean and normalize ticker symbols to prevent duplicates
        ticker_col = Column.TICKER.value
        report_col = Column.REPORT.value

        if ticker_col in df.columns:
            try:
                from core.classifications import normalize_symbol

                df[ticker_col] = df[ticker_col].astype(str).apply(normalize_symbol)
            except ImportError:
                # Fallback to basic cleaning
                df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()

        # Clean other text columns
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in [ticker_col, report_col]:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("", np.nan)

        # Check for and warn about duplicate tickers after cleaning
        if ticker_col in df.columns:
            original_count = len(df)
            df_clean = df.drop_duplicates(subset=[ticker_col, report_col], keep="first")
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
