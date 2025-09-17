import numpy as np
import pandas as pd
import streamlit as st

# Import the stock classifications module (without calling st.warning yet)
try:
    from stock_classifications import get_stock_classification

    CLASSIFICATIONS_AVAILABLE = True
    CLASSIFICATION_ERROR = None
except ImportError:
    CLASSIFICATIONS_AVAILABLE = False
    CLASSIFICATION_ERROR = (
        "Stock classifications module not found. "
        "Classification data will not be available."
    )

from charts import (
    create_combined_peg_pegy_chart,
    create_comparison_chart,
    create_eps_prediction_chart,
    create_eps_ttm_prediction_chart,
    create_metric_chart,
    create_price_prediction_chart,
    create_qoq_chart,
)
from data_utils import (
    calculate_downside_capture,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    load_data,
    predict_next_eps,
)
from ui_components import display_summary_stats

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Custom CSS to reduce padding between charts
    st.markdown(
        """
    <style>
    .stPlotlyChart {
        margin-bottom: -20px !important;
        margin-top: -10px !important;
    }
    .element-container {
        margin-bottom: 0px !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(.stPlotlyChart) {
        gap: 0.2rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("📈 Stock Data Dashboard")
    st.markdown("---")

    # Now display the classification warning if needed
    # (after st.set_page_config)
    if not CLASSIFICATIONS_AVAILABLE:
        st.warning(CLASSIFICATION_ERROR)

    # File upload or default file
    uploaded_file = st.file_uploader(
        "Upload your processed stock data file (Excel)",
        type=["xlsx"],
        help=("Upload the StockData_Indexed.xlsx file from the processing script"),
    )

    # Default file path
    default_file = "StockData_Indexed.xlsx"

    # Load data
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        # Clean numeric columns - convert non-numeric values to NaN
        numeric_columns = ["EPS", "Revenue", "Price", "DivAmt", "Index"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        st.info("Using default file: StockData_Indexed.xlsx")
        df = load_data(default_file)

    if df is None:
        st.stop()

    # Calculate QoQ changes
    df = calculate_qoq_changes(df)

    # Calculate advanced analytics
    df = calculate_sector_rankings(df)
    df = calculate_outperformance_ratios(df)
    df = calculate_downside_capture(df)

    # Display data info
    st.sidebar.header("📊 Data Overview")
    st.sidebar.write(f"**Total Records:** {len(df):,}")
    st.sidebar.write(f"**Unique Tickers:** {df['Ticker'].nunique()}")
    st.sidebar.write(f"**Date Range:** {df['Report'].min()} to {df['Report'].max()}")

    # Show classification status
    if CLASSIFICATIONS_AVAILABLE:
        st.sidebar.success("✅ Stock classifications loaded")
    else:
        st.sidebar.warning("⚠️ Stock classifications not available")

    # Sidebar - Ticker Selection
    st.sidebar.header("🎯 Select Ticker")
    available_tickers = sorted(df["Ticker"].unique())

    # Set default to AAPL if it exists, otherwise use first ticker
    default_ticker = "AAPL" if "AAPL" in available_tickers else available_tickers[0]
    selected_ticker = st.sidebar.selectbox(
        "Choose a ticker:",
        available_tickers,
        index=available_tickers.index(default_ticker),
    )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 Individual Analysis",
            "📊 Multi-Ticker Comparison",
            "📈 Rolling Averages Summary",
            "📚 Methodology",
        ]
    )

    with tab1:
        st.header(f"Analysis for {selected_ticker}")

        # Summary stats
        display_summary_stats(df, selected_ticker)
        st.markdown("---")

        # Charts for individual ticker
        st.subheader("📊 Financial Metrics (Last 5 Years)")

        # Two-column layout for all charts
        col1, col2 = st.columns(2)

        with col1:
            # EPS Chart
            eps_chart = create_metric_chart(
                df, selected_ticker, "EPS", "Earnings Per Share", height=400
            )
            if eps_chart:
                st.plotly_chart(eps_chart, use_container_width=True)
            else:
                st.info("No EPS data available for this ticker")

            eps_qoq_chart = create_qoq_chart(
                df, selected_ticker, "EPS", "EPS", height=240
            )
            st.plotly_chart(eps_qoq_chart, use_container_width=True)

            # Revenue Chart
            revenue_chart = create_metric_chart(
                df, selected_ticker, "Revenue", "Revenue", height=400
            )
            if revenue_chart:
                st.plotly_chart(revenue_chart, use_container_width=True)
            else:
                st.info("No Revenue data available for this ticker")

            revenue_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Revenue", "Revenue", height=240
            )
            st.plotly_chart(revenue_qoq_chart, use_container_width=True)

            # Price Chart
            price_chart = create_metric_chart(
                df, selected_ticker, "Price", "Stock Price", height=400
            )
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.info("No Price data available for this ticker")

            price_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Price", "Price", height=240
            )
            st.plotly_chart(price_qoq_chart, use_container_width=True)

            # Dividend Amount Chart
            div_chart = create_metric_chart(
                df, selected_ticker, "DivAmt", "Dividend Amount", height=400
            )
            if div_chart:
                st.plotly_chart(div_chart, use_container_width=True)
            else:
                st.info("No Dividend data available for this ticker")

            div_amt_qoq_chart = create_qoq_chart(
                df, selected_ticker, "DivAmt", "Dividend Amount", height=240
            )

            st.plotly_chart(div_amt_qoq_chart, use_container_width=True)

            # Revenue Consistency Chart
            revenue_consistency_chart = create_metric_chart(
                df,
                selected_ticker,
                "RevenueConsistency",
                "Revenue Consistency (% StdDev)",
                height=400,
            )
            if revenue_consistency_chart:
                st.plotly_chart(revenue_consistency_chart, use_container_width=True)
            else:
                st.info("No Revenue Consistency data available for this ticker")

            revenue_consistency_qoq_chart = create_qoq_chart(
                df,
                selected_ticker,
                "RevenueConsistency",
                "Revenue Consistency",
                height=240,
            )

            st.plotly_chart(revenue_consistency_qoq_chart, use_container_width=True)

            # EPS Growth Momentum Chart
            eps_momentum_chart = create_metric_chart(
                df,
                selected_ticker,
                "EPSMomentum",
                "EPS Growth Momentum (pp)",
                height=400,
            )
            if eps_momentum_chart:
                st.plotly_chart(eps_momentum_chart, use_container_width=True)
            else:
                st.info("No EPS Growth Momentum data available for this ticker")

            eps_momentum_qoq_chart = create_qoq_chart(
                df, selected_ticker, "EPSMomentum", "EPS Growth Momentum", height=240
            )

            st.plotly_chart(eps_momentum_qoq_chart, use_container_width=True)

            # Combined PEG & PEGY Ratios Chart
            combined_peg_pegy_chart = create_combined_peg_pegy_chart(
                df, selected_ticker, height=400
            )
            if combined_peg_pegy_chart:
                st.plotly_chart(combined_peg_pegy_chart, use_container_width=True)
            else:
                st.info("No PEG or PEGY Ratio data available for this ticker")

            # For QoQ, we'll show PEG QoQ since it's the primary metric
            peg_ratio_qoq_chart = create_qoq_chart(
                df, selected_ticker, "PEGRatio", "PEG Ratio", height=240
            )
            st.plotly_chart(peg_ratio_qoq_chart, use_container_width=True)

        with col2:
            # EPS TTM Chart
            eps_ttm_chart = create_metric_chart(
                df, selected_ticker, "EPS_TTM", "EPS - TTM", height=400
            )
            if eps_ttm_chart:
                st.plotly_chart(eps_ttm_chart, use_container_width=True)
            else:
                st.info("No EPS TTM data available for this ticker")

            eps_ttm_qoq_chart = create_qoq_chart(
                df, selected_ticker, "EPS_TTM", "EPS TTM", height=240
            )

            st.plotly_chart(eps_ttm_qoq_chart, use_container_width=True)

            # Revenue TTM Chart
            revenue_ttm_chart = create_metric_chart(
                df, selected_ticker, "Revenue_TTM", "Revenue - TTM", height=400
            )
            if revenue_ttm_chart:
                st.plotly_chart(revenue_ttm_chart, use_container_width=True)
            else:
                st.info("No Revenue TTM data available for this ticker")

            revenue_ttm_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Revenue_TTM", "Revenue TTM", height=240
            )

            st.plotly_chart(revenue_ttm_qoq_chart, use_container_width=True)

            # P/E Multiple Chart
            multiple_chart = create_metric_chart(
                df,
                selected_ticker,
                "Multiple",
                "P/E Multiple (Price / EPS TTM)",
                height=400,
            )
            if multiple_chart:
                st.plotly_chart(multiple_chart, use_container_width=True)
            else:
                st.info("No Multiple data available for this ticker")

            multiple_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Multiple", "P/E Multiple", height=240
            )

            st.plotly_chart(multiple_qoq_chart, use_container_width=True)

            # Dividend Yield Chart
            div_yield_chart = create_metric_chart(
                df,
                selected_ticker,
                "DivYield",
                "Dividend Yield (Dividend / Price)",
                height=400,
            )
            if div_yield_chart:
                st.plotly_chart(div_yield_chart, use_container_width=True)
            else:
                st.info("No Dividend Yield data available for this ticker")

            div_yield_qoq_chart = create_qoq_chart(
                df, selected_ticker, "DivYield", "Dividend Yield", height=240
            )

            st.plotly_chart(div_yield_qoq_chart, use_container_width=True)

            # Payout Ratio Chart
            payout_ratio_chart = create_metric_chart(
                df,
                selected_ticker,
                "PayoutRatio",
                "Payout Ratio (Dividend / EPS)",
                height=400,
            )
            if payout_ratio_chart:
                st.plotly_chart(payout_ratio_chart, use_container_width=True)
            else:
                st.info("No Payout Ratio data available for this ticker")

            payout_ratio_qoq_chart = create_qoq_chart(
                df, selected_ticker, "PayoutRatio", "Payout Ratio", height=240
            )

            st.plotly_chart(payout_ratio_qoq_chart, use_container_width=True)

            # Price Volatility Chart
            price_volatility_chart = create_metric_chart(
                df,
                selected_ticker,
                "PriceVolatility",
                "Price Volatility (% StdDev)",
                height=400,
            )
            if price_volatility_chart:
                st.plotly_chart(price_volatility_chart, use_container_width=True)
            else:
                st.info("No Price Volatility data available for this ticker")

            price_volatility_qoq_chart = create_qoq_chart(
                df, selected_ticker, "PriceVolatility", "Price Volatility", height=240
            )

            st.plotly_chart(price_volatility_qoq_chart, use_container_width=True)

        # QoQ Summary Section
        st.subheader("📊 QoQ Change Summary")
        st.info(
            "💡 QoQ charts are now paired with their corresponding absolute "
            "value charts above."
        )
        ticker_data = df[df["Ticker"] == selected_ticker]

        qoq_metrics = []
        for metric in [
            "EPS",
            "EPS_TTM",
            "Revenue",
            "Revenue_TTM",
            "Price",
            "Multiple",
            "DivAmt",
            "DivYield",
            "DivYieldAnnual",
            "PayoutRatio",
            "PEGRatio",
            "EPSMomentum",
            "PriceVolatility",
            "RevenueConsistency",
        ]:
            qoq_col = f"{metric}_QoQ"
            if qoq_col in ticker_data.columns:
                qoq_data = ticker_data[qoq_col].dropna()
                if not qoq_data.empty:
                    latest_qoq = qoq_data.iloc[-1] if len(qoq_data) > 0 else None
                    avg_qoq = qoq_data.mean()
                    qoq_metrics.append(
                        {
                            "Metric": metric,
                            "Latest QoQ %": (
                                f"{latest_qoq:.1f}%"
                                if latest_qoq is not None
                                else "N/A"
                            ),
                            "Average QoQ %": (
                                f"{avg_qoq:.1f}%" if not np.isnan(avg_qoq) else "N/A"
                            ),
                            "Periods with Growth": (
                                f"{(qoq_data > 0).sum()}/{len(qoq_data)}"
                            ),
                        }
                    )

        if qoq_metrics:
            qoq_df = pd.DataFrame(qoq_metrics)
            st.dataframe(qoq_df, use_container_width=True, hide_index=True)

        # EPS Prediction Section
        st.subheader("🔮 EPS Prediction")
        prediction = predict_next_eps(df, selected_ticker)
        if prediction:
            # All EPS prediction metrics in one compact row
            st.markdown("**Next Quarter EPS Scenarios:**")
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

            with col1:
                st.markdown(
                    f"<small>🔴 **Worst**<br>"
                    f"${prediction['worst_case_eps']:.2f}<br>"
                    f"{prediction['worst_case_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"<small>⭐ **Base**<br>"
                    f"${prediction['predicted_eps']:.2f}<br>"
                    f"{prediction['predicted_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"<small>🟢 **Best**<br>"
                    f"${prediction['best_case_eps']:.2f}<br>"
                    f"{prediction['best_case_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"<small>**Current**<br>${prediction['latest_eps']:.2f}</small>",
                    unsafe_allow_html=True,
                )

            with col5:
                confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                st.markdown(
                    f"<small>**Confidence**<br>{confidence_color.get(prediction['confidence'], '⚪')} "
                    f"{prediction['confidence']}</small>",
                    unsafe_allow_html=True,
                )

            with col6:
                st.markdown(
                    f"<small>**Volatility**<br>±{prediction['volatility']:.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col7:
                range_span = prediction["best_case_eps"] - prediction["worst_case_eps"]
                range_pct = (
                    (range_span / prediction["predicted_eps"]) * 100
                    if prediction["predicted_eps"] != 0
                    else 0
                )
                st.markdown(
                    f"<small>**Range**<br>${range_span:.2f}<br>({range_pct:.0f}%)</small>",
                    unsafe_allow_html=True,
                )

            with col8:
                st.markdown(
                    f"<small>**Method**<br>{prediction['methodology'].split('(')[0].strip()}</small>",
                    unsafe_allow_html=True,
                )

            # Enhanced EPS chart with all scenarios
            eps_pred_chart = create_eps_prediction_chart(df, selected_ticker)
            if eps_pred_chart:
                st.plotly_chart(eps_pred_chart, use_container_width=True)

            # Methodology details in expander
            with st.expander("📊 Prediction Methodology & Statistics"):
                col_method1, col_method2 = st.columns(2)

                with col_method1:
                    st.write("**Methodology:**")
                    st.write(f"• {prediction['methodology']}")
                    st.write(f"• Data Points: {prediction['data_points']} quarters")

                    # Handle different strategy keys gracefully
                    if "growth_4q" in prediction:
                        st.write(f"• Recent 4Q Growth: {prediction['growth_4q']:.1f}%")
                    if (
                        "growth_8q" in prediction
                        and prediction["growth_8q"] is not None
                    ):
                        st.write(f"• Recent 8Q Growth: {prediction['growth_8q']:.1f}%")
                        st.write("• Weighting: 70% recent + 30% long-term")

                with col_method2:
                    st.write("**Statistical Analysis:**")

                    # Handle different strategy keys gracefully
                    if "std_4q" in prediction:
                        st.write(f"• 4Q Std Dev: {prediction['std_4q']:.1f}%")
                    if "std_8q" in prediction and prediction["std_8q"] is not None:
                        st.write(f"• 8Q Std Dev: {prediction['std_8q']:.1f}%")

                    st.write(f"• Combined Volatility: {prediction['volatility']:.1f}%")
                    st.write("• Scenarios: Base ±1σ volatility")

                st.markdown("**Scenario Interpretation:**")
                st.markdown(
                    "• **Best Case**: Historical volatility suggests upside potential"
                )
                st.markdown(
                    "• **Base Case**: Most likely outcome based on recent trends"
                )
                st.markdown(
                    "• **Worst Case**: Downside risk based on historical volatility"
                )
        else:
            st.info(
                "Insufficient historical data for EPS prediction "
                "(requires at least 4 quarters)"
            )

        # EPS TTM Prediction Section
        if prediction and prediction["predicted_eps_ttm"] is not None:
            st.subheader("📊 EPS TTM Prediction")
            st.markdown("**Next Quarter Impact on TTM:**")

            # All TTM prediction metrics in one compact row
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

            with col1:
                st.markdown(
                    f"<small>🔴 **Worst TTM**<br>${prediction['worst_case_eps_ttm']:.2f}<br>{prediction['worst_case_eps_ttm_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"<small>⭐ **Base TTM**<br>${prediction['predicted_eps_ttm']:.2f}<br>{prediction['predicted_eps_ttm_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"<small>🟢 **Best TTM**<br>${prediction['best_case_eps_ttm']:.2f}<br>{prediction['best_case_eps_ttm_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"<small>**Current TTM**<br>${prediction['current_eps_ttm']:.2f}</small>",
                    unsafe_allow_html=True,
                )

            with col5:
                ttm_range_span = (
                    prediction["best_case_eps_ttm"] - prediction["worst_case_eps_ttm"]
                )
                ttm_range_pct = (
                    (ttm_range_span / prediction["predicted_eps_ttm"]) * 100
                    if prediction["predicted_eps_ttm"] != 0
                    else 0
                )
                st.markdown(
                    f"<small>**Range**<br>${ttm_range_span:.2f}<br>({ttm_range_pct:.0f}%)</small>",
                    unsafe_allow_html=True,
                )

            with col6:
                st.markdown(
                    f"<small>**Impact**<br>{prediction['predicted_eps_ttm_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col7:
                confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                st.markdown(
                    f"<small>**Confidence**<br>{confidence_color.get(prediction['confidence'], '⚪')} "
                    f"{prediction['confidence']}</small>",
                    unsafe_allow_html=True,
                )

            # Enhanced EPS TTM chart with all scenarios
            eps_ttm_pred_chart = create_eps_ttm_prediction_chart(df, selected_ticker)
            if eps_ttm_pred_chart:
                st.plotly_chart(eps_ttm_pred_chart, use_container_width=True)

            with st.expander("📊 EPS TTM Calculation Details"):
                st.markdown("**TTM Prediction Method:**")
                st.markdown("• TTM = Sum of last 4 quarters of EPS")
                st.markdown(
                    "• Prediction replaces oldest quarter with forecasted quarter"
                )
                st.markdown(
                    "• Shows annual earnings impact of next quarter performance"
                )
                st.markdown("• Scenarios based on quarterly prediction volatility")

                if len(df[df["Ticker"] == selected_ticker]["EPS"].dropna()) >= 4:
                    recent_quarters = (
                        df[df["Ticker"] == selected_ticker]["EPS"].dropna().tail(4)
                    )
                    st.markdown("**Current TTM Components:**")
                    st.markdown(
                        f"• Last 4 quarters: ${recent_quarters.iloc[0]:.2f} + "
                        f"${recent_quarters.iloc[1]:.2f} + "
                        f"${recent_quarters.iloc[2]:.2f} + "
                        f"${recent_quarters.iloc[3]:.2f} = "
                        f"${recent_quarters.sum():.2f}"
                    )
                    st.markdown("**Predicted TTM Components:**")
                    st.markdown(
                        f"• Next TTM: ${recent_quarters.iloc[1]:.2f} + "
                        f"${recent_quarters.iloc[2]:.2f} + "
                        f"${recent_quarters.iloc[3]:.2f} + "
                        f"${prediction['predicted_eps']:.2f} = "
                        f"${prediction['predicted_eps_ttm']:.2f}"
                    )

        # Price Prediction Section
        if prediction and prediction["predicted_price"] is not None:
            st.subheader("💰 Price Prediction")

            # All price prediction metrics in one compact row
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

            with col1:
                st.markdown(
                    f"<small>🔴 **Worst**<br>${prediction['worst_case_price']:.2f}<br>{prediction['worst_case_price_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"<small>⭐ **Base**<br>${prediction['predicted_price']:.2f}<br>{prediction['predicted_price_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"<small>🟢 **Best**<br>${prediction['best_case_price']:.2f}<br>{prediction['best_case_price_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"<small>**Current**<br>${prediction['current_price']:.2f}</small>",
                    unsafe_allow_html=True,
                )

            with col5:
                st.markdown(
                    f"<small>**P/E**<br>{prediction['current_multiple']:.1f}x</small>",
                    unsafe_allow_html=True,
                )

            with col6:
                price_range_span = (
                    prediction["best_case_price"] - prediction["worst_case_price"]
                )
                price_range_pct = (
                    (price_range_span / prediction["predicted_price"]) * 100
                    if prediction["predicted_price"] != 0
                    else 0
                )
                st.markdown(
                    f"<small>**Range**<br>${price_range_span:.2f}<br>({price_range_pct:.0f}%)</small>",
                    unsafe_allow_html=True,
                )

            with col7:
                st.markdown(
                    f"<small>**Impact**<br>{prediction['predicted_price_growth']:+.1f}%</small>",
                    unsafe_allow_html=True,
                )

            with col8:
                confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                st.markdown(
                    f"<small>**Confidence**<br>{confidence_color.get(prediction['confidence'], '⚪')} "
                    f"{prediction['confidence']}</small>",
                    unsafe_allow_html=True,
                )

            # Enhanced Price chart with all scenarios
            price_pred_chart = create_price_prediction_chart(df, selected_ticker)
            if price_pred_chart:
                st.plotly_chart(price_pred_chart, use_container_width=True)

            with st.expander("💰 Price Prediction Methodology"):
                col_price_method1, col_price_method2 = st.columns(2)

                with col_price_method1:
                    st.write("**Valuation Method:**")
                    st.write("• Price = EPS TTM × P/E Multiple")
                    st.write(f"• Current P/E: {prediction['current_multiple']:.1f}x")
                    st.write("• Assumes multiple remains constant")
                    st.write("• Scenarios based on EPS TTM predictions")

                with col_price_method2:
                    st.write("**Price Calculation:**")
                    st.write(
                        f"• Current: ${prediction['current_eps_ttm']:.2f} × "
                        f"{prediction['current_multiple']:.1f}x = "
                        f"${prediction['current_price']:.2f}"
                    )
                    st.write(
                        f"• Base Case: ${prediction['predicted_eps_ttm']:.2f} × "
                        f"{prediction['current_multiple']:.1f}x = "
                        f"${prediction['predicted_price']:.2f}"
                    )
                    st.write(
                        f"• Best Case: ${prediction['best_case_eps_ttm']:.2f} × "
                        f"{prediction['current_multiple']:.1f}x = "
                        f"${prediction['best_case_price']:.2f}"
                    )
                    st.write(
                        f"• Worst Case: ${prediction['worst_case_eps_ttm']:.2f} × "
                        f"{prediction['current_multiple']:.1f}x = "
                        f"${prediction['worst_case_price']:.2f}"
                    )

                st.markdown("**Important Assumptions:**")
                st.markdown(
                    "• **Constant Multiple**: P/E ratio remains at current levels"
                )
                st.markdown("• **EPS-Driven**: Price moves are purely earnings-driven")
                st.markdown(
                    "• **No Market Factors**: Excludes sentiment, sector rotation, "
                    "macro events"
                )
                st.markdown(
                    "• **Historical Basis**: Multiple reflects recent valuation "
                    "preference"
                )

        # Raw data table for selected ticker
        st.subheader(f"Raw Data for {selected_ticker}")
        ticker_data = df[df["Ticker"] == selected_ticker]
        st.dataframe(ticker_data, use_container_width=True)

    with tab2:
        st.header("Multi-Ticker Comparison")

        # Multi-select for tickers
        default_comparison_tickers = ["AAPL", "GOOGL", "AMZN", "META", "NVDA", "BRK.B"]
        # Only include tickers that actually exist in the data
        default_tickers = [
            ticker
            for ticker in default_comparison_tickers
            if ticker in available_tickers
        ]
        # If none of the defaults exist, fall back to first 4 tickers
        if not default_tickers:
            default_tickers = (
                available_tickers[:4]
                if len(available_tickers) >= 4
                else available_tickers
            )

        comparison_tickers = st.multiselect(
            "Select tickers to compare:", available_tickers, default=default_tickers
        )

        if comparison_tickers:
            # Allow multiple metric comparisons
            st.subheader("📊 Comparison Charts")

            # Available metrics for comparison
            metric_options = [
                "EPS",
                "EPS_TTM",
                "Revenue",
                "Revenue_TTM",
                "Price",
                "Multiple",
                "DivAmt",
                "DivYield",
                "PayoutRatio",
                "PriceVolatility",
                "RevenueConsistency",
                "EPSMomentum",
                "PEGRatio",
                "PEGYRatio",
            ]

            # Available QoQ metrics for comparison
            qoq_metric_options = [
                "EPS_QoQ",
                "EPS_TTM_QoQ",
                "Revenue_QoQ",
                "Revenue_TTM_QoQ",
                "Price_QoQ",
                "Multiple_QoQ",
                "DivAmt_QoQ",
                "DivYield_QoQ",
                "PayoutRatio_QoQ",
                "PriceVolatility_QoQ",
                "RevenueConsistency_QoQ",
                "EPSMomentum_QoQ",
                "PEGRatio_QoQ",
            ]
            num_charts = st.selectbox(
                "How many comparison charts would you like?", [1, 2, 3, 4], index=3
            )

            # Create the specified number of charts
            for i in range(num_charts):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col2:
                    # Chart type selection
                    chart_type = st.selectbox(
                        f"Type for Chart {i+1}:",
                        ["Regular", "QoQ"],
                        key=f"chart_type_{i}",
                        index=0,
                    )

                with col3:
                    # Metric selection based on chart type
                    if chart_type == "QoQ":
                        available_options = qoq_metric_options
                        metric_key = f"qoq_metric_{i}"
                    else:
                        available_options = metric_options
                        metric_key = f"metric_{i}"

                    selected_metric = st.selectbox(
                        f"Metric for Chart {i+1}:",
                        available_options,
                        key=metric_key,
                        index=i if i < len(available_options) else 0,
                    )

                with col1:
                    # Set y-axis range for specific metrics
                    if selected_metric == "Multiple":
                        yaxis_range = [-1, 60]
                    elif chart_type == "QoQ":
                        yaxis_range = None  # Let QoQ charts auto-scale
                    else:
                        yaxis_range = None

                    # Create comparison chart
                    comparison_chart = create_comparison_chart(
                        df, comparison_tickers, selected_metric, yaxis_range=yaxis_range
                    )
                    st.plotly_chart(comparison_chart, use_container_width=True)

                # Add some spacing between charts
                if i < num_charts - 1:
                    st.markdown("---")
        else:
            st.info("Please select at least one ticker for comparison")

    with tab3:
        st.header("Rolling Averages Summary")
        st.info(
            "📊 Latest rolling average values for QoQ growth rates across all "
            "tickers with sector classifications"
        )

        # Calculate rolling averages for all tickers
        rolling_summary_data = []
        missing_classifications = []

        for ticker in available_tickers:
            ticker_data = df[df["Ticker"] == ticker].copy()
            ticker_summary = {"Ticker": ticker}

            # Add stock classification data if available
            if CLASSIFICATIONS_AVAILABLE:
                classification = get_stock_classification(ticker)
                if classification:
                    ticker_summary["Sector"] = classification.sector.value
                    ticker_summary["Industry"] = classification.industry.value
                    ticker_summary["Sub_Industry"] = classification.sub_industry.value
                else:
                    # Track missing classifications for optional display
                    missing_classifications.append(ticker)
                    ticker_summary["Sector"] = "Unclassified"
                    ticker_summary["Industry"] = "Unclassified"
                    ticker_summary["Sub_Industry"] = "Unclassified"
            else:
                ticker_summary["Sector"] = "N/A"
                ticker_summary["Industry"] = "N/A"
                ticker_summary["Sub_Industry"] = "N/A"

            # For each metric, calculate the latest rolling averages
            for metric in [
                "EPS",
                "Revenue",
                "EPS_TTM",
                "Revenue_TTM",
                "Price",
                "DivAmt",
                "DivYield",
                "DivYieldAnnual",
                "PayoutRatio",
                "PEGRatio",
                "EPSMomentum",
                "PriceVolatility",
                "RevenueConsistency",
            ]:
                qoq_col = f"{metric}_QoQ"
                if qoq_col in ticker_data.columns:
                    qoq_values = ticker_data[qoq_col].dropna()
                    if len(qoq_values) > 0:
                        # Calculate rolling means
                        rolling_4q = qoq_values.rolling(window=4, min_periods=1).mean()
                        rolling_8q = qoq_values.rolling(window=8, min_periods=1).mean()
                        rolling_12q = qoq_values.rolling(
                            window=12, min_periods=1
                        ).mean()

                        # Get the latest values
                        ticker_summary[f"{metric}_4Q_Avg"] = (
                            rolling_4q.iloc[-1] if len(rolling_4q) > 0 else np.nan
                        )
                        ticker_summary[f"{metric}_8Q_Avg"] = (
                            rolling_8q.iloc[-1] if len(rolling_8q) > 0 else np.nan
                        )
                        ticker_summary[f"{metric}_12Q_Avg"] = (
                            rolling_12q.iloc[-1] if len(rolling_12q) > 0 else np.nan
                        )
                    else:
                        ticker_summary[f"{metric}_4Q_Avg"] = np.nan
                        ticker_summary[f"{metric}_8Q_Avg"] = np.nan
                        ticker_summary[f"{metric}_12Q_Avg"] = np.nan
                else:
                    ticker_summary[f"{metric}_4Q_Avg"] = np.nan
                    ticker_summary[f"{metric}_8Q_Avg"] = np.nan
                    ticker_summary[f"{metric}_12Q_Avg"] = np.nan

            # Add latest Multiple and Revenue_TTM values
            latest_multiple = (
                ticker_data["Multiple"].dropna().iloc[-1]
                if (
                    "Multiple" in ticker_data.columns
                    and not ticker_data["Multiple"].dropna().empty
                )
                else np.nan
            )
            latest_revenue_ttm = (
                ticker_data["Revenue_TTM"].dropna().iloc[-1]
                if (
                    "Revenue_TTM" in ticker_data.columns
                    and not ticker_data["Revenue_TTM"].dropna().empty
                )
                else np.nan
            )

            ticker_summary["Latest_Multiple"] = latest_multiple
            ticker_summary["Latest_Revenue_TTM"] = latest_revenue_ttm

            rolling_summary_data.append(ticker_summary)

        if rolling_summary_data:
            rolling_df = pd.DataFrame(rolling_summary_data)

            # Create sector-based filtering options
            if CLASSIFICATIONS_AVAILABLE and "Sector" in rolling_df.columns:
                st.subheader("🎯 Filter by Sector")
                sectors = sorted(rolling_df["Sector"].unique())

                # Multi-select for sectors
                selected_sectors = st.multiselect(
                    "Select sectors to display (leave empty for all):",
                    sectors,
                    default=[],
                    help="Choose specific sectors to filter the data",
                )

                # Filter data if sectors are selected
                if selected_sectors:
                    filtered_rolling_df = rolling_df[
                        rolling_df["Sector"].isin(selected_sectors)
                    ]
                    st.info(
                        f"Showing {len(filtered_rolling_df)} tickers from "
                        f"{len(selected_sectors)} selected sector(s)"
                    )
                else:
                    filtered_rolling_df = rolling_df
                    st.info(f"Showing all {len(filtered_rolling_df)} tickers")
            else:
                filtered_rolling_df = rolling_df
                st.info(f"Showing all {len(filtered_rolling_df)} tickers")

            # Create a comprehensive rolling averages table
            st.subheader("📊 Comprehensive QoQ Rolling Averages Summary (%)")

            # Select columns for display - including classification columns
            display_cols = ["Ticker"]

            # Add classification columns if available
            if CLASSIFICATIONS_AVAILABLE:
                display_cols.extend(["Sector", "Industry", "Sub_Industry"])

            # Add columns for each metric with descriptive names
            for metric in [
                "EPS",
                "Revenue",
                "EPS_TTM",
                "Revenue_TTM",
                "Price",
                "DivAmt",
                "DivYield",
                "DivYieldAnnual",
                "PayoutRatio",
                "PEGRatio",
            ]:
                for period in ["4Q", "8Q", "12Q"]:
                    col_name = f"{metric}_{period}_Avg"
                    if col_name in filtered_rolling_df.columns:
                        display_cols.append(col_name)

            # Add latest values columns
            display_cols.extend(["Latest_Multiple", "Latest_Revenue_TTM"])

            # Create display dataframe
            comprehensive_df = filtered_rolling_df[display_cols].copy()

            # Rename columns for better readability
            column_mapping = {
                "Sub_Industry": "Sub-Industry",
                "EPS_4Q_Avg": "EPS 4Q Avg",
                "EPS_8Q_Avg": "EPS 8Q Avg",
                "EPS_12Q_Avg": "EPS 12Q Avg",
                "Revenue_4Q_Avg": "Revenue 4Q Avg",
                "Revenue_8Q_Avg": "Revenue 8Q Avg",
                "Revenue_12Q_Avg": "Revenue 12Q Avg",
                "EPS_TTM_4Q_Avg": "EPS TTM 4Q Avg",
                "EPS_TTM_8Q_Avg": "EPS TTM 8Q Avg",
                "EPS_TTM_12Q_Avg": "EPS TTM 12Q Avg",
                "Revenue_TTM_4Q_Avg": "Revenue TTM 4Q Avg",
                "Revenue_TTM_8Q_Avg": "Revenue TTM 8Q Avg",
                "Revenue_TTM_12Q_Avg": "Revenue TTM 12Q Avg",
                "Price_4Q_Avg": "Price 4Q Avg",
                "Price_8Q_Avg": "Price 8Q Avg",
                "Price_12Q_Avg": "Price 12Q Avg",
                "DivAmt_4Q_Avg": "Div Amount 4Q Avg",
                "DivAmt_8Q_Avg": "Div Amount 8Q Avg",
                "DivAmt_12Q_Avg": "Div Amount 12Q Avg",
                "DivYield_4Q_Avg": "Div Yield Q 4Q Avg",
                "DivYield_8Q_Avg": "Div Yield Q 8Q Avg",
                "DivYield_12Q_Avg": "Div Yield Q 12Q Avg",
                "DivYieldAnnual_4Q_Avg": "Div Yield Annual 4Q Avg",
                "DivYieldAnnual_8Q_Avg": "Div Yield Annual 8Q Avg",
                "DivYieldAnnual_12Q_Avg": "Div Yield Annual 12Q Avg",
                "PayoutRatio_4Q_Avg": "Payout Ratio 4Q Avg",
                "PayoutRatio_8Q_Avg": "Payout Ratio 8Q Avg",
                "PayoutRatio_12Q_Avg": "Payout Ratio 12Q Avg",
                "PEGRatio_4Q_Avg": "PEG Ratio 4Q Avg",
                "PEGRatio_8Q_Avg": "PEG Ratio 8Q Avg",
                "PEGRatio_12Q_Avg": "PEG Ratio 12Q Avg",
                "Latest_Multiple": "Latest P/E Multiple",
                "Latest_Revenue_TTM": "Latest Revenue TTM ($M)",
            }

            comprehensive_df = comprehensive_df.rename(columns=column_mapping)

            # Configure all numeric columns to display appropriately
            column_config = {}
            for col in comprehensive_df.columns:
                if col in ["Ticker", "Sector", "Industry", "Sub-Industry"]:
                    continue
                elif col == "Latest P/E Multiple":
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.1fx",
                        help="Price-to-Earnings Multiple (Price / EPS TTM)",
                    )
                elif col == "Latest Revenue TTM ($M)":
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="$%.0f",
                        help=("Latest TTM Revenue in " "Millions"),
                    )
                else:
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.1f%%",
                        help=f"{col.replace(' Avg', ' Rolling Average')}",
                    )

            # Display the comprehensive table
            st.dataframe(
                comprehensive_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )

            # Show information about missing classifications if any
            if CLASSIFICATIONS_AVAILABLE and missing_classifications:
                with st.expander(
                    f"ℹ️ Data Quality Info - {len(missing_classifications)} "
                    "ticker(s) missing classifications"
                ):
                    st.info(
                        f"The following {len(missing_classifications)} ticker(s) "
                        "are marked as 'Unclassified' because they don't have "
                        "sector/industry classifications in the system:"
                    )
                    st.write(", ".join(sorted(missing_classifications)))

            # Add sector analysis if classifications are available
            if CLASSIFICATIONS_AVAILABLE and "Sector" in filtered_rolling_df.columns:
                st.markdown("---")
                st.subheader("📈 Sector Analysis")

                # Create sector summary statistics
                sector_stats = []
                for sector in filtered_rolling_df["Sector"].unique():
                    if sector not in ["Unknown", "Unclassified", "N/A"]:
                        sector_data = filtered_rolling_df[
                            filtered_rolling_df["Sector"] == sector
                        ]

                        # Calculate average rolling metrics by sector
                        sector_summary = {"Sector": sector, "Count": len(sector_data)}

                        for metric in [
                            "EPS",
                            "Revenue",
                            "EPS_TTM",
                            "Revenue_TTM",
                            "Price",
                        ]:
                            for period in [
                                "4Q",
                                "8Q",
                            ]:  # Just show 4Q and 8Q for sector summary
                                col_name = f"{metric}_{period}_Avg"
                                if col_name in sector_data.columns:
                                    avg_value = sector_data[col_name].mean()
                                    sector_summary[
                                        f"{metric}_{period}_Sector_Avg"
                                    ] = avg_value

                        sector_stats.append(sector_summary)

                if sector_stats:
                    sector_df = pd.DataFrame(sector_stats)

                    # Rename columns for better display
                    sector_column_mapping = {
                        "Count": "Ticker Count",
                        "EPS_4Q_Sector_Avg": "EPS 4Q Sector Avg",
                        "EPS_8Q_Sector_Avg": "EPS 8Q Sector Avg",
                        "Revenue_4Q_Sector_Avg": "Revenue 4Q Sector Avg",
                        "Revenue_8Q_Sector_Avg": "Revenue 8Q Sector Avg",
                        "EPS_TTM_4Q_Sector_Avg": "EPS TTM 4Q Sector Avg",
                        "EPS_TTM_8Q_Sector_Avg": "EPS TTM 8Q Sector Avg",
                        "Revenue_TTM_4Q_Sector_Avg": ("Revenue TTM 4Q Sector Avg"),
                        "Revenue_TTM_8Q_Sector_Avg": ("Revenue TTM 8Q Sector Avg"),
                        "Price_4Q_Sector_Avg": "Price 4Q Sector Avg",
                        "Price_8Q_Sector_Avg": "Price 8Q Sector Avg",
                    }

                    sector_df = sector_df.rename(columns=sector_column_mapping)

                    # Configure sector table columns
                    sector_column_config = {}
                    for col in sector_df.columns:
                        if col in ["Sector", "Ticker Count"]:
                            continue
                        else:
                            sector_column_config[col] = st.column_config.NumberColumn(
                                col,
                                format="%.1f%%",
                                help=f"Average {col} across sector",
                            )

                    st.dataframe(
                        sector_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=sector_column_config,
                    )

            # Add option to view individual metric tables
            st.markdown("---")
            with st.expander("📋 View Individual Metric Tables"):
                # EPS QoQ Rolling Averages Table
                st.subheader("📈 EPS QoQ Rolling Averages (%)")
                eps_cols = ["Ticker"]
                if CLASSIFICATIONS_AVAILABLE:
                    eps_cols.extend(
                        ["Sector", "Industry", "Sub_Industry"]
                    )  # Use original column name
                eps_cols.extend(["EPS_4Q_Avg", "EPS_8Q_Avg", "EPS_12Q_Avg"])

                eps_rolling_df = filtered_rolling_df[eps_cols].copy()
                # Rename columns
                eps_rename = {
                    "Sub_Industry": "Sub-Industry",
                    "EPS_4Q_Avg": "4Q Rolling Avg",
                    "EPS_8Q_Avg": "8Q Rolling Avg",
                    "EPS_12Q_Avg": "12Q Rolling Avg",
                }
                eps_rolling_df = eps_rolling_df.rename(columns=eps_rename)

                eps_column_config = {
                    "4Q Rolling Avg": st.column_config.NumberColumn(
                        "4Q Rolling Avg", format="%.1f%%"
                    ),
                    "8Q Rolling Avg": st.column_config.NumberColumn(
                        "8Q Rolling Avg", format="%.1f%%"
                    ),
                    "12Q Rolling Avg": st.column_config.NumberColumn(
                        "12Q Rolling Avg", format="%.1f%%"
                    ),
                }
                st.dataframe(
                    eps_rolling_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=eps_column_config,
                )

                # Revenue QoQ Rolling Averages Table
                st.subheader("💰 Revenue QoQ Rolling Averages (%)")
                revenue_cols = ["Ticker"]
                if CLASSIFICATIONS_AVAILABLE:
                    revenue_cols.extend(
                        ["Sector", "Industry", "Sub_Industry"]
                    )  # Use original column name
                revenue_cols.extend(
                    ["Revenue_4Q_Avg", "Revenue_8Q_Avg", "Revenue_12Q_Avg"]
                )

                revenue_rolling_df = filtered_rolling_df[revenue_cols].copy()
                revenue_rename = {
                    "Sub_Industry": "Sub-Industry",
                    "Revenue_4Q_Avg": "4Q Rolling Avg",
                    "Revenue_8Q_Avg": "8Q Rolling Avg",
                    "Revenue_12Q_Avg": "12Q Rolling Avg",
                }
                revenue_rolling_df = revenue_rolling_df.rename(columns=revenue_rename)
                st.dataframe(
                    revenue_rolling_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=eps_column_config,
                )

                # EPS TTM QoQ Rolling Averages Table
                st.subheader("📊 EPS TTM QoQ Rolling Averages (%)")
                eps_ttm_cols = ["Ticker"]
                if CLASSIFICATIONS_AVAILABLE:
                    eps_ttm_cols.extend(
                        ["Sector", "Industry", "Sub_Industry"]
                    )  # Use original column name
                eps_ttm_cols.extend(
                    ["EPS_TTM_4Q_Avg", "EPS_TTM_8Q_Avg", "EPS_TTM_12Q_Avg"]
                )
                eps_ttm_rolling_df = filtered_rolling_df[eps_ttm_cols].copy()
                eps_ttm_rename = {
                    "Sub_Industry": "Sub-Industry",
                    "EPS_TTM_4Q_Avg": "4Q Rolling Avg",
                    "EPS_TTM_8Q_Avg": "8Q Rolling Avg",
                    "EPS_TTM_12Q_Avg": "12Q Rolling Avg",
                }
                eps_ttm_rolling_df = eps_ttm_rolling_df.rename(columns=eps_ttm_rename)
                st.dataframe(
                    eps_ttm_rolling_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=eps_column_config,
                )

                # Revenue TTM QoQ Rolling Averages Table
                st.subheader("💼 Revenue TTM QoQ Rolling Averages (%)")
                revenue_ttm_cols = ["Ticker"]
                if CLASSIFICATIONS_AVAILABLE:
                    revenue_ttm_cols.extend(
                        ["Sector", "Industry", "Sub_Industry"]
                    )  # Use original column name
                revenue_ttm_cols.extend(
                    ["Revenue_TTM_4Q_Avg", "Revenue_TTM_8Q_Avg", "Revenue_TTM_12Q_Avg"]
                )
                revenue_ttm_rolling_df = filtered_rolling_df[revenue_ttm_cols].copy()
                revenue_ttm_rename = {
                    "Sub_Industry": "Sub-Industry",
                    "Revenue_TTM_4Q_Avg": "4Q Rolling Avg",
                    "Revenue_TTM_8Q_Avg": "8Q Rolling Avg",
                    "Revenue_TTM_12Q_Avg": "12Q Rolling Avg",
                }
                revenue_ttm_rolling_df = revenue_ttm_rolling_df.rename(
                    columns=revenue_ttm_rename,
                )
                st.dataframe(
                    revenue_ttm_rolling_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=eps_column_config,
                )

                # Price QoQ Rolling Averages Table
                st.subheader("💹 Price QoQ Rolling Averages (%)")
                price_cols = ["Ticker"]
                if CLASSIFICATIONS_AVAILABLE:
                    price_cols.extend(
                        ["Sector", "Industry", "Sub_Industry"]
                    )  # Use original column name
                price_cols.extend(["Price_4Q_Avg", "Price_8Q_Avg", "Price_12Q_Avg"])
                price_rolling_df = filtered_rolling_df[price_cols].copy()
                price_rename = {
                    "Sub_Industry": "Sub-Industry",
                    "Price_4Q_Avg": "4Q Rolling Avg",
                    "Price_8Q_Avg": "8Q Rolling Avg",
                    "Price_12Q_Avg": "12Q Rolling Avg",
                }
                price_rolling_df = price_rolling_df.rename(columns=price_rename)
                st.dataframe(
                    price_rolling_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=eps_column_config,
                )

            # Download button for the complete rolling averages data
            csv = filtered_rolling_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Rolling Averages Data",
                data=csv,
                file_name=(
                    "rolling_averages_summary_"
                    f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                mime="text/csv",
            )

    with tab4:
        st.header("📚 Methodology & Calculations")
        st.markdown("---")

        st.markdown(
            """
        This section explains how each metric is calculated using **AAPL** as our
        example ticker.
        All calculations are performed on a per-ticker basis using quarterly data.
        """
        )

        # Core Metrics Section
        st.subheader("🔢 Core Metrics")

        with st.expander("**EPS TTM**", expanded=False):
            st.markdown(
                """
            **Formula**: `EPS_TTM = Sum of last 4 quarters of EPS`

            **Example (AAPL)**:
            - Q4 2023: $2.18
            - Q1 2024: $2.18
            - Q2 2024: $1.40
            - Q3 2024: $1.64
            - **EPS TTM = $2.18 + $2.18 + $1.40 + $1.64 = $7.40**

            **Purpose**: Shows annualized earnings performance, smoothing out quarterly
            volatility.
            """
            )

        with st.expander("**P/E Multiple**", expanded=False):
            st.markdown(
                """
            **Formula**: `Multiple = Price / EPS_TTM`

            **Example (AAPL)**:
            - Current Price: $150.00
            - EPS TTM: $7.40
            - **P/E Multiple = $150.00 / $7.40 = 20.3x**

            **Purpose**: Shows how much investors pay for each dollar of earnings.
            Lower is generally better value.
            """
            )

        with st.expander("**Dividend Yield (Quarterly & Annual)**", expanded=False):
            st.markdown(
                """
            **Quarterly Formula**: `DivYield = (DivAmt / Price) × 100`
            **Annual Formula**: `DivYieldAnnual = (DivAmt × 4 / Price) × 100`

            **Example (AAPL)**:
            - Quarterly Dividend: $0.24
            - Current Price: $150.00
            - **Quarterly Yield = ($0.24 / $150.00) × 100 = 0.16%**
            - **Annual Yield = ($0.24 × 4 / $150.00) × 100 = 0.64%**

            **Usage**:
            - **Quarterly**: Used for QoQ analysis and trending
            - **Annual**: Used for comparative analysis and PEGY ratio

            **Purpose**: Shows income return as percentage of stock price.
            Annual yield is standard for investment comparison.
            """
            )

        # Advanced Metrics Section
        st.subheader("📊 Advanced Financial Metrics")

        with st.expander("**Payout Ratio**", expanded=False):
            st.markdown(
                """
            **Formula**: `PayoutRatio = (Annual Dividends / EPS_TTM) × 100`

            **Example (AAPL)**:
            - Quarterly Dividend: $0.24
            - Annual Dividends: $0.24 × 4 = $0.96
            - EPS TTM: $7.40
            - **Payout Ratio = ($0.96 / $7.40) × 100 = 13.0%**

            **Purpose**: Shows what percentage of earnings are paid as dividends.
            Higher ratios may indicate less retained earnings for growth.
            """
            )

        with st.expander("**PEG Ratio**", expanded=False):
            st.markdown(
                """
            **Formula**: `PEG = P/E Multiple / Annual EPS Growth Rate`

            **Calculation Steps**:
            1. Calculate 4-quarter rolling average of EPS QoQ growth
            2. Annualize: `(1 + quarterly_rate/100)^4 - 1`
            3. Divide P/E by annualized growth rate

            **Example (AAPL)**:
            - P/E Multiple: 20.3x
            - 4Q Avg EPS Growth: 8% quarterly
            - Annualized Growth: (1.08)^4 - 1 = 36.0%
            - **PEG Ratio = 20.3 / 36.0 = 0.56**

            **Purpose**: PEG < 1.0 suggests stock may be undervalued relative to
            growth. Lower is better.
            """
            )

        with st.expander("**PEGY Ratio**", expanded=False):
            st.markdown(
                """
            **Formula**: `PEGY = PEG Ratio / Annual Dividend Yield`

            **Example (AAPL)**:
            - PEG Ratio: 0.56
            - Annual Dividend Yield: 0.64%
            - **PEGY Ratio = 0.56 / 0.64 = 0.88**

            **Note**: Uses annualized dividend yield for standard comparison across
            different payment frequencies.

            **Purpose**: Incorporates dividend income into growth valuation.
            Lower values suggest better value when considering both growth and income.
            """
            )

        # Momentum & Quality Metrics
        st.subheader("🚀 Momentum & Quality Metrics")

        with st.expander("**EPS Growth Momentum**", expanded=False):
            st.markdown(
                """
            **Formula**: `EPSMomentum = EPS_4Q_Rolling_Avg - EPS_8Q_Rolling_Avg`

            **Example (AAPL)**:
            - Recent 4Q Average EPS Growth: 12%
            - Recent 8Q Average EPS Growth: 8%
            - **EPS Momentum = 12% - 8% = +4.0 percentage points**

            **Interpretation**:
            - Positive: Earnings growth is accelerating
            - Negative: Earnings growth is decelerating

            **Purpose**: Identifies whether earnings growth is accelerating or
            slowing down.
            """
            )

        with st.expander("**Price Volatility**", expanded=False):
            st.markdown(
                """
            **Formula**: `PriceVolatility = Standard Deviation of Price_QoQ over
            8 quarters`

            **Example (AAPL)**:
            - Last 8 quarters Price QoQ: [15%, -8%, 22%, -12%, 18%, -5%, 25%, -10%]
            - **Price Volatility = Standard Deviation = 14.2%**

            **Purpose**: Higher values indicate more volatile stock price movements.
            Risk measure.
            """
            )

        with st.expander("**Revenue Consistency**", expanded=False):
            st.markdown(
                """
            **Formula**: `RevenueConsistency = 100 - (Coefficient of Variation × 100)`

            **Where**: `Coefficient of Variation = Standard Deviation / |Mean|`

            **Example (AAPL)**:
            - 8Q Revenue Growth: Mean = 8%, Std Dev = 3.2%
            - CV = 3.2% / 8% = 0.40
            - **Revenue Consistency = 100 - (40) = 60%**

            **Purpose**: Higher scores indicate more consistent revenue growth.
            Quality measure.
            """
            )

        with st.expander("**Dividend Growth Rate**", expanded=False):
            st.markdown(
                """
            **Enhanced Methodology**: Tracks actual dividend progression over time,
            not just QoQ fluctuations

            **Calculation Steps**:
            1. **Identify Dividend Change Points**: Find when dividend amount actually
               changed
               - Example: $0.24 → $0.25 → $0.26 (actual increases)
               - Ignores periods where dividend stayed constant

            2. **Calculate Growth Between Changes**:
               - From $0.24 to $0.25: +4.17% growth
               - From $0.25 to $0.26: +4.00% growth

            3. **Annualized Growth Rate**:
               - Formula: `((Last_Dividend / First_Dividend)^(1/Years) - 1) * 100`
               - If tracked over 3 years: ((0.26/0.24)^(1/3) - 1) * 100 = 2.7% annually

            **Additional Metrics**:
            - **Dividend Increase Frequency**: How often dividends are raised
              (increases per year)
            - **Average Dividend Increase**: Mean percentage increase when
              dividends are raised

            **Example (AAPL)**:
            - Dividend progression: $0.22 → $0.23 → $0.24 → $0.25 → $0.26
            - **Annual Growth Rate: 4.3%**
            - **Increase Frequency: 1.2 times per year**
            - **Average Increase: 4.2% when raised**

            **Purpose**: More accurate assessment of dividend growth by focusing on
            actual progression rather than quarterly noise.
            """
            )

        # Relative Performance Metrics
        st.subheader("📈 Relative Performance Metrics")

        with st.expander("**Sector Rankings**", expanded=False):
            st.markdown(
                """
            **Methodology**: Each ticker is ranked within its sector for key metrics

            **Ranking Logic**:
            - **Higher is Better**: EPS_TTM, Revenue_TTM, DivYield (Q),
              DivYieldAnnual, Revenue Consistency, EPS Momentum
            - **Lower is Better**: P/E Multiple, Price Volatility, PEG Ratio, PEGY Ratio

            **Example (AAPL in Technology)**:
            - EPS_TTM Sector Rank: 3/25 (3rd highest EPS in Technology)
            - PEG Ratio Sector Rank: 8/25 (8th lowest PEG in Technology)

            **Purpose**: Shows relative performance within peer group.
            Rank 1 = best in sector.
            """
            )

        with st.expander("**Outperformance Ratios**", expanded=False):
            st.markdown(
                """
            **Formula**: `Outperformance = (Ticker Metric / Benchmark Average) × 100`

            **Benchmarks**:
            - **Market Outperformance**: vs. all tickers average
            - **Sector Outperformance**: vs. sector average

            **Example (AAPL)**:
            - AAPL EPS Growth: 15%
            - Technology Sector Avg: 12%
            - Market Avg: 8%
            - **Sector Outperformance = (15% / 12%) × 100 = 125%**
            - **Market Outperformance = (15% / 8%) × 100 = 188%**

            **Purpose**: Values >100% indicate outperformance. Shows relative
            strength vs benchmarks.
            """
            )

        with st.expander("**Downside Capture**", expanded=False):
            st.markdown(
                """
            **Formula**: `DownsideCapture = (Avg Ticker Return in Down Markets /
            Avg Market Return in Down Markets) × 100`

            **Calculation Steps**:
            1. Identify quarters where market (all tickers avg) was negative
            2. Calculate average ticker return during those periods
            3. Calculate average market return during those periods
            4. Compute ratio

            **Example (AAPL)**:
            - Down market quarters: Market avg -8%, -12%, -5%
            - AAPL during same quarters: -6%, -9%, -3%
            - Market avg during down periods: -8.3%
            - AAPL avg during down periods: -6.0%
            - **Downside Capture = (-6.0% / -8.3%) × 100 = 72%**

            **Interpretation**:
            - **<100%**: Stock falls less than market (defensive)
            - **>100%**: Stock falls more than market (aggressive)

            **Purpose**: Risk measure. Lower values indicate better downside protection.
            """
            )

        # QoQ Rolling Averages
        st.subheader("📊 Quarter-over-Quarter (QoQ) Analysis")

        with st.expander("**QoQ Calculations & Rolling Averages**", expanded=False):
            st.markdown(
                """
            **QoQ Formula**: `QoQ_Change = ((Current_Quarter - Previous_Quarter) /
            Previous_Quarter) × 100`

            **Rolling Averages**:
            - **4Q Rolling**: Average of last 4 quarters QoQ changes
            - **8Q Rolling**: Average of last 8 quarters QoQ changes
            - **12Q Rolling**: Average of last 12 quarters QoQ changes

            **Example (AAPL EPS)**:
            - Q1: $2.00, Q2: $2.20 → QoQ = 10%
            - Q2: $2.20, Q3: $2.40 → QoQ = 9.1%
            - Q3: $2.40, Q4: $2.60 → QoQ = 8.3%
            - Q4: $2.60, Q1: $2.80 → QoQ = 7.7%

            - **4Q Rolling Average = (10% + 9.1% + 8.3% + 7.7%) / 4 = 8.8%**

            **Purpose**: Smooths volatility and shows trends. Longer periods provide
            more stable indicators.
            """
            )

        # Data Notes
        st.subheader("📝 Important Notes")

        st.info(
            """
        **Data Requirements**:
        - Minimum 4 quarters needed for TTM calculations
        - Minimum 8 quarters needed for EPS Momentum
        - Minimum 3 down market periods needed for Downside Capture

        **Handling Missing Data**:
        - Calculations skip periods with missing data
        - Results marked as N/A when insufficient data available
        - Rankings only include tickers with valid data for that metric

        **Frequency**:
        - All calculations based on quarterly reporting data
        - Metrics updated each quarter as new data becomes available
        - Rolling averages provide smoothed trend indicators
        """
        )

        st.success(
            "💡 **Pro Tip**: Use multiple metrics together for comprehensive "
            "analysis. No single metric tells the complete story!"
        )


if __name__ == "__main__":
    main()
