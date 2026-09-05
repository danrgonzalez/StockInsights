import streamlit as st

from core.data_processing import latest_with_age


def _stale_note(label, quarter, stale):
    """Wording for a value that is not from the ticker's latest quarter."""
    if not stale:
        return None
    quarters = "quarter" if stale == 1 else "quarters"
    return (
        f"{label} is from {quarter}, {stale} {quarters} before the latest "
        "reported quarter, which has no value for it."
    )


def _read(ticker_data, column, label):
    """Latest value for a column plus a note when it is stale.

    Returns (value, note) with value None when the column is missing or blank.
    """
    value, quarter, stale = latest_with_age(ticker_data, column)
    if value is None:
        return None, None
    return value, _stale_note(label, quarter, stale)


def _render(title, value_text, helps, notes):
    """One st.metric, flagged when any value behind it is stale."""
    notes = [note for note in notes if note]
    parts = [part for part in helps if part]
    if notes:
        title = f"{title} ⚠️"
        parts.extend(notes)
    st.metric(title, value_text, help=" | ".join(parts) if parts else None)


def display_summary_stats(df, ticker):
    ticker_data = df[df["Ticker"] == ticker]
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Records", len(ticker_data))

    with col2:
        eps, eps_note = _read(ticker_data, "EPS", "EPS")
        eps_ttm, eps_ttm_note = _read(ticker_data, "EPS_TTM", "EPS TTM")
        if eps is None:
            st.metric("Latest EPS", "N/A")
        else:
            _render(
                "Latest EPS",
                f"{eps:.2f}",
                [f"TTM: {eps_ttm:.2f}" if eps_ttm is not None else None],
                [eps_note, eps_ttm_note],
            )

    with col3:
        revenue, rev_note = _read(ticker_data, "Revenue", "Revenue")
        revenue_ttm, rev_ttm_note = _read(ticker_data, "Revenue_TTM", "Revenue TTM")
        if revenue is None:
            st.metric("Latest Revenue", "N/A")
        else:
            _render(
                "Latest Revenue",
                f"${revenue:,.0f}M",
                [f"TTM: ${revenue_ttm:,.0f}M" if revenue_ttm is not None else None],
                [rev_note, rev_ttm_note],
            )

    with col4:
        price, price_note = _read(ticker_data, "Price", "Price")
        multiple, multiple_note = _read(ticker_data, "Multiple", "P/E Multiple")
        if price is None:
            st.metric("Latest Price", "N/A")
        else:
            _render(
                "Latest Price",
                f"${price:.2f}",
                [
                    (
                        f"P/E Multiple: {multiple:.1f}x"
                        if multiple is not None
                        else "P/E Multiple: n/a (earnings not positive)"
                    )
                ],
                [price_note, multiple_note],
            )

    with col5:
        yield_annual, ya_note = _read(
            ticker_data, "DivYieldAnnual", "Annual dividend yield"
        )
        yield_quarterly, yq_note = _read(
            ticker_data, "DivYield", "Quarterly dividend yield"
        )
        div_amt, amt_note = _read(ticker_data, "DivAmt", "Dividend amount")
        if yield_annual is None:
            st.metric("Dividend Yield (Annual)", "N/A")
        else:
            detail = []
            if yield_quarterly is not None:
                detail.append(f"Quarterly: {yield_quarterly:.2f}%")
            if div_amt is not None:
                detail.append(f"Amount: ${div_amt:.2f}")
            _render(
                "Dividend Yield (Annual)",
                f"{yield_annual:.2f}%",
                [" | ".join(detail) if detail else None],
                [ya_note, yq_note, amt_note],
            )

    with col6:
        peg, peg_note = _read(ticker_data, "PEGRatio", "PEG ratio")
        payout, payout_note = _read(ticker_data, "PayoutRatio", "Payout ratio")
        if peg is None:
            st.metric("PEG Ratio", "N/A")
        else:
            _render(
                "PEG Ratio",
                f"{peg:.2f}",
                [f"Payout Ratio: {payout:.1f}%" if payout is not None else None],
                [peg_note, payout_note],
            )
