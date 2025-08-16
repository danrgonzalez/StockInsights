import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data(file_path):
    """Load the processed stock data with caching"""
    try:
        df = pd.read_excel(file_path)
        # Clean numeric columns - convert non-numeric values to NaN
        numeric_columns = ['EPS', 'Revenue', 'Price', 'DivAmt', 'Index']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Clean and normalize ticker symbols to prevent duplicates
        if 'Ticker' in df.columns:
            # Import normalization function if available
            try:
                from stock_classifications import normalize_symbol
                df['Ticker'] = df['Ticker'].astype(str).apply(normalize_symbol)
            except ImportError:
                # Fallback to basic cleaning
                df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            
        # Clean other text columns
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['Ticker', 'Report']:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('', np.nan)
                
        # Check for and warn about duplicate tickers after cleaning
        if 'Ticker' in df.columns:
            original_count = len(df)
            df_clean = df.drop_duplicates(subset=['Ticker', 'Report'], keep='first')
            dropped_count = original_count - len(df_clean)
            if dropped_count > 0:
                st.warning(f"Removed {dropped_count} duplicate ticker/report combinations during data cleaning")
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
    df_with_qoq = df_with_qoq.sort_values(['Ticker', 'Index'])
    for ticker in df_with_qoq['Ticker'].unique():
        ticker_mask = df_with_qoq['Ticker'] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()
        for metric in ['EPS', 'Revenue']:
            if metric in ticker_data.columns:
                ttm_values = ticker_data[metric].rolling(window=4, min_periods=4).sum()
                df_with_qoq.loc[ticker_mask, f'{metric}_TTM'] = ttm_values
        if (
            'Price' in ticker_data.columns and
            'EPS_TTM' in df_with_qoq.loc[ticker_mask].columns
        ):
            price_data = ticker_data['Price']
            eps_ttm_data = df_with_qoq.loc[ticker_mask, 'EPS_TTM']
            multiple = price_data / eps_ttm_data
            multiple = multiple.replace([np.inf, -np.inf], np.nan)
            df_with_qoq.loc[ticker_mask, 'Multiple'] = multiple
    df_with_qoq = df_with_qoq.sort_values(['Ticker', 'Index'])
    for ticker in df_with_qoq['Ticker'].unique():
        ticker_mask = df_with_qoq['Ticker'] == ticker
        ticker_data = df_with_qoq[ticker_mask].copy()
        for metric in [
            'EPS', 'Revenue', 'Price', 'EPS_TTM', 'Revenue_TTM', 'Multiple'
        ]:
            if metric in ticker_data.columns:
                qoq_change = ticker_data[metric].pct_change(fill_method=None) * 100
                df_with_qoq.loc[ticker_mask, f'{metric}_QoQ'] = qoq_change
    return df_with_qoq