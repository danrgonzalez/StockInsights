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
    create_comparison_chart,
    create_eps_prediction_chart,
    create_eps_ttm_prediction_chart,
    create_metric_chart,
    create_price_prediction_chart,
    create_qoq_chart,
)
from data_utils import calculate_qoq_changes, load_data, predict_next_eps
from ui_components import display_summary_stats

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
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
    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Individual Analysis",
            "📊 Multi-Ticker Comparison",
            "📈 Rolling Averages Summary",
        ]
    )

    with tab1:
        st.header(f"Analysis for {selected_ticker}")

        # Summary stats
        display_summary_stats(df, selected_ticker)
        st.markdown("---")

        # Charts for individual ticker
        st.subheader("📊 Absolute Values")
        col1, col2 = st.columns(2)

        with col1:
            eps_chart = create_metric_chart(
                df, selected_ticker, "EPS", "Earnings Per Share"
            )
            if eps_chart:
                st.plotly_chart(eps_chart, use_container_width=True)
            else:
                st.info("No EPS data available for this ticker")

        # EPS Prediction Section
        st.subheader("🔮 EPS Prediction")
        prediction = predict_next_eps(df, selected_ticker)
        if prediction:
            # Three-scenario display
            st.markdown("**Next Quarter EPS Scenarios:**")
            col_worst, col_base, col_best = st.columns(3)

            with col_worst:
                st.metric(
                    "🔴 Worst Case",
                    f"${prediction['worst_case_eps']:.2f}",
                    f"{prediction['worst_case_growth']:+.1f}%",
                    delta_color="inverse",
                )

            with col_base:
                st.metric(
                    "⭐ Base Case",
                    f"${prediction['predicted_eps']:.2f}",
                    f"{prediction['predicted_growth']:+.1f}%",
                )

            with col_best:
                st.metric(
                    "🟢 Best Case",
                    f"${prediction['best_case_eps']:.2f}",
                    f"{prediction['best_case_growth']:+.1f}%",
                    delta_color="normal",
                )

            # Additional metrics row
            col_current, col_confidence, col_volatility = st.columns(3)
            with col_current:
                st.metric("Current EPS", f"${prediction['latest_eps']:.2f}")

            with col_confidence:
                confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                st.metric(
                    "Confidence Level",
                    f"{confidence_color.get(prediction['confidence'], '⚪')} {prediction['confidence']}",
                )

            with col_volatility:
                st.metric("Volatility (±1σ)", f"{prediction['volatility']:.1f}%")

            # Scenario range summary
            range_span = prediction["best_case_eps"] - prediction["worst_case_eps"]
            range_pct = (
                (range_span / prediction["predicted_eps"]) * 100
                if prediction["predicted_eps"] != 0
                else 0
            )

            st.info(
                f"📊 **Scenario Range**: ${prediction['worst_case_eps']:.2f} to ${prediction['best_case_eps']:.2f} "
                f"(${range_span:.2f} spread, {range_pct:.0f}% of base case)"
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
                    st.write(f"• Recent 4Q Growth: {prediction['growth_4q']:.1f}%")
                    if prediction["growth_8q"] is not None:
                        st.write(f"• Recent 8Q Growth: {prediction['growth_8q']:.1f}%")
                        st.write("• Weighting: 70% recent + 30% long-term")

                with col_method2:
                    st.write("**Statistical Analysis:**")
                    st.write(f"• 4Q Std Dev: {prediction['std_4q']:.1f}%")
                    if prediction["std_8q"] is not None:
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
                "Insufficient historical data for EPS prediction (requires at least 4 quarters)"
            )

        # EPS TTM Prediction Section
        if prediction and prediction["predicted_eps_ttm"] is not None:
            st.subheader("📊 EPS TTM Prediction")
            st.markdown("**Next Quarter Impact on Trailing Twelve Months:**")

            col_ttm_worst, col_ttm_base, col_ttm_best = st.columns(3)

            with col_ttm_worst:
                st.metric(
                    "🔴 Worst Case TTM",
                    f"${prediction['worst_case_eps_ttm']:.2f}",
                    f"{prediction['worst_case_eps_ttm_growth']:+.1f}%",
                    delta_color="inverse",
                )

            with col_ttm_base:
                st.metric(
                    "⭐ Base Case TTM",
                    f"${prediction['predicted_eps_ttm']:.2f}",
                    f"{prediction['predicted_eps_ttm_growth']:+.1f}%",
                )

            with col_ttm_best:
                st.metric(
                    "🟢 Best Case TTM",
                    f"${prediction['best_case_eps_ttm']:.2f}",
                    f"{prediction['best_case_eps_ttm_growth']:+.1f}%",
                    delta_color="normal",
                )

            # Current TTM for comparison
            col_current_ttm, col_ttm_range = st.columns(2)
            with col_current_ttm:
                st.metric("Current EPS TTM", f"${prediction['current_eps_ttm']:.2f}")

            with col_ttm_range:
                ttm_range_span = (
                    prediction["best_case_eps_ttm"] - prediction["worst_case_eps_ttm"]
                )
                ttm_range_pct = (
                    (ttm_range_span / prediction["predicted_eps_ttm"]) * 100
                    if prediction["predicted_eps_ttm"] != 0
                    else 0
                )
                st.metric(
                    "TTM Range Spread", f"${ttm_range_span:.2f} ({ttm_range_pct:.0f}%)"
                )

            # TTM Impact explanation
            st.info(
                f"📈 **TTM Impact**: The predicted quarter will replace the oldest quarter in the TTM calculation, "
                f"potentially changing annual earnings by {prediction['predicted_eps_ttm_growth']:+.1f}% in the base case scenario."
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
                    st.markdown(f"**Current TTM Components:**")
                    st.markdown(
                        f"• Last 4 quarters: ${recent_quarters.iloc[0]:.2f} + ${recent_quarters.iloc[1]:.2f} + ${recent_quarters.iloc[2]:.2f} + ${recent_quarters.iloc[3]:.2f} = ${recent_quarters.sum():.2f}"
                    )
                    st.markdown(f"**Predicted TTM Components:**")
                    st.markdown(
                        f"• Next TTM: ${recent_quarters.iloc[1]:.2f} + ${recent_quarters.iloc[2]:.2f} + ${recent_quarters.iloc[3]:.2f} + ${prediction['predicted_eps']:.2f} = ${prediction['predicted_eps_ttm']:.2f}"
                    )

        # Price Prediction Section
        if prediction and prediction["predicted_price"] is not None:
            st.subheader("💰 Price Prediction")
            st.markdown(
                "**Multiple-Based Price Scenarios (Current P/E × Predicted EPS TTM):**"
            )

            col_price_worst, col_price_base, col_price_best = st.columns(3)

            with col_price_worst:
                st.metric(
                    "🔴 Worst Case Price",
                    f"${prediction['worst_case_price']:.2f}",
                    f"{prediction['worst_case_price_growth']:+.1f}%",
                    delta_color="inverse",
                )

            with col_price_base:
                st.metric(
                    "⭐ Base Case Price",
                    f"${prediction['predicted_price']:.2f}",
                    f"{prediction['predicted_price_growth']:+.1f}%",
                )

            with col_price_best:
                st.metric(
                    "🟢 Best Case Price",
                    f"${prediction['best_case_price']:.2f}",
                    f"{prediction['best_case_price_growth']:+.1f}%",
                    delta_color="normal",
                )

            # Additional price metrics
            col_current_price, col_current_pe, col_price_range = st.columns(3)
            with col_current_price:
                st.metric("Current Price", f"${prediction['current_price']:.2f}")

            with col_current_pe:
                st.metric(
                    "Current P/E Multiple", f"{prediction['current_multiple']:.1f}x"
                )

            with col_price_range:
                price_range_span = (
                    prediction["best_case_price"] - prediction["worst_case_price"]
                )
                price_range_pct = (
                    (price_range_span / prediction["predicted_price"]) * 100
                    if prediction["predicted_price"] != 0
                    else 0
                )
                st.metric(
                    "Price Range Spread",
                    f"${price_range_span:.2f} ({price_range_pct:.0f}%)",
                )

            # Price prediction explanation
            st.info(
                f"📈 **Multiple-Based Valuation**: Price predictions assume current P/E multiple ({prediction['current_multiple']:.1f}x) "
                f"remains constant, applied to predicted EPS TTM scenarios. Base case suggests "
                f"{prediction['predicted_price_growth']:+.1f}% price movement."
            )

            # Enhanced Price chart with all scenarios
            price_pred_chart = create_price_prediction_chart(df, selected_ticker)
            if price_pred_chart:
                st.plotly_chart(price_pred_chart, use_container_width=True)

            with st.expander("💰 Price Prediction Methodology"):
                col_price_method1, col_price_method2 = st.columns(2)

                with col_price_method1:
                    st.write("**Valuation Method:**")
                    st.write(f"• Price = EPS TTM × P/E Multiple")
                    st.write(f"• Current P/E: {prediction['current_multiple']:.1f}x")
                    st.write(f"• Assumes multiple remains constant")
                    st.write("• Scenarios based on EPS TTM predictions")

                with col_price_method2:
                    st.write("**Price Calculation:**")
                    st.write(
                        f"• Current: ${prediction['current_eps_ttm']:.2f} × {prediction['current_multiple']:.1f}x = ${prediction['current_price']:.2f}"
                    )
                    st.write(
                        f"• Base Case: ${prediction['predicted_eps_ttm']:.2f} × {prediction['current_multiple']:.1f}x = ${prediction['predicted_price']:.2f}"
                    )
                    st.write(
                        f"• Best Case: ${prediction['best_case_eps_ttm']:.2f} × {prediction['current_multiple']:.1f}x = ${prediction['best_case_price']:.2f}"
                    )
                    st.write(
                        f"• Worst Case: ${prediction['worst_case_eps_ttm']:.2f} × {prediction['current_multiple']:.1f}x = ${prediction['worst_case_price']:.2f}"
                    )

                st.markdown("**Important Assumptions:**")
                st.markdown(
                    "• **Constant Multiple**: P/E ratio remains at current levels"
                )
                st.markdown("• **EPS-Driven**: Price moves are purely earnings-driven")
                st.markdown(
                    "• **No Market Factors**: Excludes sentiment, sector rotation, macro events"
                )
                st.markdown(
                    "• **Historical Basis**: Multiple reflects recent valuation preference"
                )

        st.markdown("---")

        with col1:
            # EPS TTM Chart
            eps_ttm_chart = create_metric_chart(
                df, selected_ticker, "EPS_TTM", "EPS - Trailing Twelve Months"
            )
            if eps_ttm_chart:
                st.plotly_chart(eps_ttm_chart, use_container_width=True)
            else:
                st.info("No EPS TTM data available for this ticker")

            price_chart = create_metric_chart(
                df, selected_ticker, "Price", "Stock Price"
            )
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.info("No Price data available for this ticker")

            # Multiple Chart (P/E TTM)
            multiple_chart = create_metric_chart(
                df, selected_ticker, "Multiple", "P/E Multiple (Price / EPS TTM)"
            )
            if multiple_chart:
                st.plotly_chart(multiple_chart, use_container_width=True)
            else:
                st.info("No Multiple data available for this ticker")

        with col2:
            revenue_chart = create_metric_chart(
                df, selected_ticker, "Revenue", "Revenue"
            )
            if revenue_chart:
                st.plotly_chart(revenue_chart, use_container_width=True)
            else:
                st.info("No Revenue data available for this ticker")

            # Revenue TTM Chart
            revenue_ttm_chart = create_metric_chart(
                df, selected_ticker, "Revenue_TTM", "Revenue - Trailing Twelve Months"
            )
            if revenue_ttm_chart:
                st.plotly_chart(revenue_ttm_chart, use_container_width=True)
            else:
                st.info("No Revenue TTM data available for this ticker")

            div_chart = create_metric_chart(
                df, selected_ticker, "DivAmt", "Dividend Amount"
            )
            if div_chart:
                st.plotly_chart(div_chart, use_container_width=True)
            else:
                st.info("No Dividend data available for this ticker")

        # QoQ Change Charts
        st.subheader("📈 Quarter-over-Quarter % Changes")
        col3, col4 = st.columns(2)

        with col3:
            eps_qoq_chart = create_qoq_chart(df, selected_ticker, "EPS", "EPS")
            if eps_qoq_chart:
                st.plotly_chart(eps_qoq_chart, use_container_width=True)
            else:
                st.info("No EPS QoQ data available for this ticker")

            # EPS TTM QoQ Chart
            eps_ttm_qoq_chart = create_qoq_chart(
                df, selected_ticker, "EPS_TTM", "EPS TTM"
            )
            if eps_ttm_qoq_chart:
                st.plotly_chart(eps_ttm_qoq_chart, use_container_width=True)
            else:
                st.info("No EPS TTM QoQ data available for this ticker")

            price_qoq_chart = create_qoq_chart(df, selected_ticker, "Price", "Price")
            if price_qoq_chart:
                st.plotly_chart(price_qoq_chart, use_container_width=True)
            else:
                st.info("No Price QoQ data available for this ticker")

            # Multiple QoQ Chart
            multiple_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Multiple", "P/E Multiple"
            )
            if multiple_qoq_chart:
                st.plotly_chart(multiple_qoq_chart, use_container_width=True)
            else:
                st.info("No Multiple QoQ data available for this ticker")

        with col4:
            revenue_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Revenue", "Revenue"
            )
            if revenue_qoq_chart:
                st.plotly_chart(revenue_qoq_chart, use_container_width=True)
            else:
                st.info("No Revenue QoQ data available for this ticker")

            # Revenue TTM QoQ Chart
            revenue_ttm_qoq_chart = create_qoq_chart(
                df, selected_ticker, "Revenue_TTM", "Revenue TTM"
            )
            if revenue_ttm_qoq_chart:
                st.plotly_chart(revenue_ttm_qoq_chart, use_container_width=True)
            else:
                st.info("No Revenue TTM QoQ data available for this ticker")

            # Add QoQ summary stats (including TTM)
            st.subheader("📊 QoQ Change Summary")
            ticker_data = df[df["Ticker"] == selected_ticker]

            qoq_metrics = []
            for metric in [
                "EPS",
                "EPS_TTM",
                "Revenue",
                "Revenue_TTM",
                "Price",
                "Multiple",
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
                                    f"{avg_qoq:.1f}%"
                                    if not np.isnan(avg_qoq)
                                    else "N/A"
                                ),
                                "Periods with Growth": (
                                    f"{(qoq_data > 0).sum()}/{len(qoq_data)}"
                                ),
                            }
                        )

            if qoq_metrics:
                qoq_df = pd.DataFrame(qoq_metrics)
                st.dataframe(qoq_df, use_container_width=True, hide_index=True)

        # Raw data table for selected ticker
        st.subheader(f"Raw Data for {selected_ticker}")
        ticker_data = df[df["Ticker"] == selected_ticker]
        st.dataframe(ticker_data, use_container_width=True)

    with tab2:
        st.header("Multi-Ticker Comparison")

        # Multi-select for tickers
        default_comparison_tickers = ["AAPL", "GOOGL", "AMZN", "META"]
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
            ]
            num_charts = st.selectbox(
                "How many comparison charts would you like?", [1, 2, 3, 4], index=1
            )

            # Create the specified number of charts
            for i in range(num_charts):
                col1, col2 = st.columns([3, 1])

                with col2:
                    selected_metric = st.selectbox(
                        f"Metric for Chart {i+1}:",
                        metric_options,
                        key=f"metric_{i}",
                        index=i if i < len(metric_options) else 0,
                    )

                with col1:
                    # Set y-axis range for "Multiple" metric
                    yaxis_range = [-1, 60] if selected_metric == "Multiple" else None
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
            for metric in ["EPS", "Revenue", "EPS_TTM", "Revenue_TTM", "Price"]:
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
            for metric in ["EPS", "Revenue", "EPS_TTM", "Revenue_TTM", "Price"]:
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
                        help=("Latest Trailing Twelve Months Revenue in " "Millions"),
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
                    f"rolling_averages_summary_"
                    f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
