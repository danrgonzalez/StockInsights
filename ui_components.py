import streamlit as st


def display_summary_stats(df, ticker):
    ticker_data = df[df["Ticker"] == ticker]
    col1, col2, col3, col4 = st.columns(4)
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
