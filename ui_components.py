import streamlit as st


def display_summary_stats(df, ticker):
    ticker_data = df[df["Ticker"] == ticker]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Records", len(ticker_data))
    with col2:
        latest_eps = (
            ticker_data["EPS"].dropna().iloc[-1]
            if not ticker_data["EPS"].dropna().empty
            else "N/A"
        )
        latest_eps_ttm = (
            ticker_data["EPS_TTM"].dropna().iloc[-1]
            if (
                "EPS_TTM" in ticker_data.columns
                and not ticker_data["EPS_TTM"].dropna().empty
            )
            else "N/A"
        )
        if latest_eps != "N/A" and latest_eps_ttm != "N/A":
            st.metric(
                "Latest EPS", f"{latest_eps:.2f}", help=f"TTM: {latest_eps_ttm:.2f}"
            )
        elif latest_eps != "N/A":
            st.metric("Latest EPS", f"{latest_eps:.2f}")
        else:
            st.metric("Latest EPS", "N/A")
    with col3:
        latest_revenue = (
            ticker_data["Revenue"].dropna().iloc[-1]
            if not ticker_data["Revenue"].dropna().empty
            else "N/A"
        )
        latest_revenue_ttm = (
            ticker_data["Revenue_TTM"].dropna().iloc[-1]
            if (
                "Revenue_TTM" in ticker_data.columns
                and not ticker_data["Revenue_TTM"].dropna().empty
            )
            else "N/A"
        )
        if latest_revenue != "N/A" and latest_revenue_ttm != "N/A":
            st.metric(
                "Latest Revenue",
                f"${latest_revenue:,.0f}M",
                help=f"TTM: ${latest_revenue_ttm:,.0f}M",
            )
        elif latest_revenue != "N/A":
            st.metric("Latest Revenue", f"${latest_revenue:,.0f}M")
        else:
            st.metric("Latest Revenue", "N/A")
    with col4:
        latest_price = (
            ticker_data["Price"].dropna().iloc[-1]
            if not ticker_data["Price"].dropna().empty
            else "N/A"
        )
        latest_multiple = (
            ticker_data["Multiple"].dropna().iloc[-1]
            if (
                "Multiple" in ticker_data.columns
                and not ticker_data["Multiple"].dropna().empty
            )
            else "N/A"
        )
        if latest_price != "N/A" and latest_multiple != "N/A":
            st.metric(
                "Latest Price",
                f"${latest_price:.2f}",
                help=f"P/E Multiple: {latest_multiple:.1f}x",
            )
        elif latest_price != "N/A":
            st.metric("Latest Price", f"${latest_price:.2f}")
        else:
            st.metric("Latest Price", "N/A")
    with col5:
        latest_div_yield_annual = (
            ticker_data["DivYieldAnnual"].dropna().iloc[-1]
            if (
                "DivYieldAnnual" in ticker_data.columns
                and not ticker_data["DivYieldAnnual"].dropna().empty
            )
            else "N/A"
        )
        latest_div_yield_quarterly = (
            ticker_data["DivYield"].dropna().iloc[-1]
            if (
                "DivYield" in ticker_data.columns
                and not ticker_data["DivYield"].dropna().empty
            )
            else "N/A"
        )
        latest_div_amt = (
            ticker_data["DivAmt"].dropna().iloc[-1]
            if not ticker_data["DivAmt"].dropna().empty
            else "N/A"
        )
        if latest_div_yield_annual != "N/A" and latest_div_yield_quarterly != "N/A":
            st.metric(
                "Dividend Yield (Annual)",
                f"{latest_div_yield_annual:.2f}%",
                help=(
                    f"Quarterly: {latest_div_yield_quarterly:.2f}% | "
                    f"Amount: ${latest_div_amt:.2f}"
                ),
            )
        elif latest_div_yield_annual != "N/A":
            st.metric("Dividend Yield (Annual)", f"{latest_div_yield_annual:.2f}%")
        else:
            st.metric("Dividend Yield (Annual)", "N/A")
    with col6:
        latest_peg_ratio = (
            ticker_data["PEGRatio"].dropna().iloc[-1]
            if (
                "PEGRatio" in ticker_data.columns
                and not ticker_data["PEGRatio"].dropna().empty
            )
            else "N/A"
        )
        latest_payout_ratio = (
            ticker_data["PayoutRatio"].dropna().iloc[-1]
            if (
                "PayoutRatio" in ticker_data.columns
                and not ticker_data["PayoutRatio"].dropna().empty
            )
            else "N/A"
        )
        if latest_peg_ratio != "N/A" and latest_payout_ratio != "N/A":
            st.metric(
                "PEG Ratio",
                f"{latest_peg_ratio:.2f}",
                help=f"Payout Ratio: {latest_payout_ratio:.1f}%",
            )
        elif latest_peg_ratio != "N/A":
            st.metric("PEG Ratio", f"{latest_peg_ratio:.2f}")
        else:
            st.metric("PEG Ratio", "N/A")
