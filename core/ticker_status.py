"""
Ticker lifecycle status and data-quality holds.

Single source of truth is ``config/ticker_status.json``. Two separate ideas live
there, and keeping them apart matters:

``status``
    A fact about the company. ``active`` means it still reports; ``acquired``
    means it was bought out, taken private or merged away and will never report
    again.

``exclude_from_active``
    Whether to drop the ticker from the *active analysis universe* at load.
    True for every acquired ticker -- a company frozen in 2019 does not belong
    in a 2026 sector benchmark -- and also true for tickers held back purely for
    data quality, which are still ``active`` companies.

Acquired tickers are excluded, not deleted. ``load_stock_data(...,
include_excluded=True)`` returns them so their final quarters can be studied as
a pre-acquisition profile.

The config path is resolved relative to this file, not the working directory, so
this behaves identically whether it is imported from the repo root, a notebook or
a service.
"""

import json
from pathlib import Path

import pandas as pd

from core.enums import Column

STATUS_FILE = Path(__file__).resolve().parents[1] / "config" / "ticker_status.json"

STATUS_ACTIVE = "active"
STATUS_ACQUIRED = "acquired"


def _normalize(symbol: str) -> str:
    """Normalize a symbol the same way loaded data is normalized."""
    from core.classifications import normalize_symbol

    return normalize_symbol(symbol)


def load_status(file_path: Path | str | None = None) -> dict:
    """
    Load the full ticker status config.

    Returns:
        Mapping of normalized ticker -> entry dict. Empty if the file is
        missing or unreadable.
    """
    path = Path(file_path) if file_path else STATUS_FILE
    try:
        with open(path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {_normalize(k): v for k, v in config.get("tickers", {}).items()}


def get_excluded_tickers(file_path: Path | str | None = None) -> set[str]:
    """Return tickers to drop from the active analysis universe."""
    return {
        ticker
        for ticker, entry in load_status(file_path).items()
        if entry.get("exclude_from_active", False)
    }


def get_acquired_tickers(file_path: Path | str | None = None) -> set[str]:
    """Return tickers whose company was acquired, taken private or merged."""
    return {
        ticker
        for ticker, entry in load_status(file_path).items()
        if entry.get("status") == STATUS_ACQUIRED
    }


def get_ticker_status(ticker: str, file_path: Path | str | None = None) -> str:
    """Status for one ticker. Anything not listed is assumed active."""
    entry = load_status(file_path).get(_normalize(ticker), {})
    return entry.get("status", STATUS_ACTIVE)


def get_acquisition(ticker: str, file_path: Path | str | None = None) -> dict | None:
    """Acquisition details for one ticker, or None if it was not acquired."""
    entry = load_status(file_path).get(_normalize(ticker), {})
    if entry.get("status") != STATUS_ACQUIRED:
        return None
    return entry.get("acquisition")


def attach_status(
    df: pd.DataFrame, file_path: Path | str | None = None
) -> pd.DataFrame:
    """
    Add a ``Status`` column so active and acquired rows can be told apart.

    Needed because acquired tickers are kept in the data for profiling; without
    a marker there is nothing distinguishing a company's last quarter from any
    other quarter.

    Args:
        df: Stock data with a Ticker column (already normalized)
        file_path: Override for the status config location

    Returns:
        DataFrame with a Status column added
    """
    ticker_col = Column.TICKER.value
    if ticker_col not in df.columns:
        return df

    acquired = get_acquired_tickers(file_path)
    df = df.copy()
    df[Column.STATUS.value] = [
        STATUS_ACQUIRED if ticker in acquired else STATUS_ACTIVE
        for ticker in df[ticker_col]
    ]
    return df


def filter_excluded(
    df: pd.DataFrame, file_path: Path | str | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop tickers held out of the active analysis universe.

    Args:
        df: Stock data with a Ticker column (already normalized)
        file_path: Override for the status config location

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
