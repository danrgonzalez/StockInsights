"""
Ticker exclusion list.

Single source of truth for tickers with known data-quality problems, defined in
``config/excluded_tickers.json``. Excluded tickers are filtered out at data load
so the dashboard, predictions and backtests all ignore the same set.

The config path is resolved relative to this file, not the working directory, so
this behaves identically whether it is imported from the repo root, a notebook or
a service.
"""

import json
from pathlib import Path

import pandas as pd

from core.enums import Column

EXCLUSIONS_FILE = (
    Path(__file__).resolve().parents[1] / "config" / "excluded_tickers.json"
)


def _normalize(symbol: str) -> str:
    """Normalize a symbol the same way loaded data is normalized."""
    from core.classifications import normalize_symbol

    return normalize_symbol(symbol)


def load_exclusions(file_path: Path | str | None = None) -> dict:
    """
    Load the full exclusion config.

    Returns:
        Mapping of normalized ticker -> entry dict. Empty if the file is
        missing or unreadable.
    """
    path = Path(file_path) if file_path else EXCLUSIONS_FILE
    try:
        with open(path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {_normalize(k): v for k, v in config.get("tickers", {}).items()}


def get_excluded_tickers(file_path: Path | str | None = None) -> set[str]:
    """Return the set of normalized tickers marked ``exclude: true``."""
    return {
        ticker
        for ticker, entry in load_exclusions(file_path).items()
        if entry.get("exclude", False)
    }


def filter_excluded(
    df: pd.DataFrame, file_path: Path | str | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop excluded tickers from a loaded DataFrame.

    Args:
        df: Stock data with a Ticker column (already normalized)
        file_path: Override for the exclusion config location

    Returns:
        (filtered DataFrame, mapping of dropped ticker -> row count)
    """
    ticker_col = Column.TICKER.value
    if ticker_col not in df.columns:
        return df, {}

    excluded = get_excluded_tickers(file_path)
    if not excluded:
        return df, {}

    mask = df[ticker_col].isin(excluded)
    if not mask.any():
        return df, {}

    dropped = df.loc[mask, ticker_col].value_counts().to_dict()
    return df.loc[~mask].copy(), dropped
