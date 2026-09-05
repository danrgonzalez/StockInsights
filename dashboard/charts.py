import re

import plotly.express as px
import plotly.graph_objects as go

from core.enums import ChartDefaults, DerivedMetric, RollingWindow
from dashboard.data_utils import predict_next_eps


def _report_ticks(data, extra_index=None, extra_label=None):
    """Map each row's Index to its fiscal-quarter Report label (e.g. "Q1'26")
    so charts can keep Index as the plotted x-axis while showing quarter/year
    tick labels instead of raw integers.
    """
    ticks = (
        data[["Index", "Report"]]
        .dropna()
        .drop_duplicates(subset=["Index"])
        .sort_values("Index")
    )
    tickvals = ticks["Index"].tolist()
    ticktext = ticks["Report"].tolist()
    if (
        extra_index is not None
        and extra_label is not None
        and extra_index not in tickvals
    ):
        tickvals.append(extra_index)
        ticktext.append(extra_label)
    return tickvals, ticktext


def _next_quarter_label(report):
    """Given a fiscal-quarter label like "Q4'25", return the next quarter's
    label ("Q1'26"). Returns None if the label doesn't parse.
    """
    match = re.match(r"Q(\d)'(\d+)", str(report).strip())
    if not match:
        return None
    quarter, year = int(match.group(1)), int(match.group(2))
    if quarter == 4:
        return f"Q1'{(year + 1) % 100:02d}"
    return f"Q{quarter + 1}'{year:02d}"


def create_metric_chart(df, ticker, metric, title, height=400):
    ticker_data = df[df["Ticker"] == ticker].copy()
    if metric not in ticker_data.columns or ticker_data[metric].isna().all():
        return None
    clean_data = ticker_data.dropna(subset=[metric])
    if len(clean_data) == 0:
        return None

    # Include all data in the chart
    fig = px.line(
        clean_data,
        x="Index",
        y=metric,
        title=f"{title}",
        hover_data=["Report"],
        markers=True,
    )

    # Add horizontal lines for P/E Multiple chart
    if metric == DerivedMetric.MULTIPLE.value:
        fig.add_hline(
            y=ChartDefaults.PE_LOW, line_dash="solid", line_color="yellow", line_width=2
        )
        fig.add_hline(
            y=ChartDefaults.PE_MEDIUM,
            line_dash="solid",
            line_color="yellow",
            line_width=2,
        )
        fig.add_hline(
            y=ChartDefaults.PE_HIGH,
            line_dash="solid",
            line_color="yellow",
            line_width=2,
        )

        # Ensure y-axis range includes all reference lines
        data_max = clean_data[metric].max()
        data_min = clean_data[metric].min()
        y_max = max(ChartDefaults.MULTIPLE_Y_MAX, data_max + 2)
        y_min = max(0, min(ChartDefaults.MULTIPLE_Y_MIN, data_min - 2))
        fig.update_yaxes(range=[y_min, y_max])

    # Set x-axis range to show last N quarters by default
    display_quarters = RollingWindow.DISPLAY_QUARTERS
    if len(clean_data) > 1:
        x_max = clean_data["Index"].max() + 0.5
        if len(clean_data) > display_quarters:
            x_min = clean_data["Index"].iloc[-display_quarters] - 0.5
        else:
            x_min = clean_data["Index"].min() - 0.5
        fig.update_xaxes(range=[x_min, x_max])

    # Set y-axis range based on visible data
    if metric != DerivedMetric.MULTIPLE.value:
        visible_data = (
            clean_data.tail(display_quarters)
            if len(clean_data) > display_quarters
            else clean_data
        )
        data_range = visible_data[metric].max() - visible_data[metric].min()
        padding = data_range * ChartDefaults.Y_AXIS_PADDING if data_range > 0 else 1
        y_min = visible_data[metric].min() - padding
        y_max = visible_data[metric].max() + padding
        fig.update_yaxes(range=[y_min, y_max])

    tickvals, ticktext = _report_ticks(clean_data)
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        xaxis_title="",
        yaxis_title=title,
        hovermode="x unified",
        height=height,
        margin=ChartDefaults.standard_margin(),
    )
    return fig


def create_qoq_chart(df, ticker, metric, title, height=240):
    ticker_data = df[df["Ticker"] == ticker].copy()
    qoq_column = f"{metric}_QoQ"
    if qoq_column not in ticker_data.columns or ticker_data[qoq_column].isna().all():
        return create_blank_placeholder(
            f"{title} - QoQ % Change - No Data Available", height
        )
    full_clean_data = ticker_data.dropna(subset=[qoq_column])
    if len(full_clean_data) == 0:
        return create_blank_placeholder(
            f"{title} - QoQ % Change - No Data Available", height
        )

    # Use full dataset for both rolling averages and display
    clean_data = full_clean_data  # Include all data in the chart

    fig = px.line(
        clean_data,
        x="Index",
        y=qoq_column,
        title=f"<span style='font-size:12px'>{title} - QoQ % Change</span>",
        hover_data=["Report", metric],
        markers=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="yellow", opacity=0.8)
    if (
        metric
        in [
            "EPS",
            "Revenue",
            "EPS_TTM",
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
        ]
        and len(full_clean_data) > 0
    ):
        # Calculate rolling averages from full dataset
        qoq_values = clean_data[qoq_column]
        rolling_4q = qoq_values.rolling(
            window=RollingWindow.SHORT, min_periods=1
        ).mean()
        rolling_8q = qoq_values.rolling(window=RollingWindow.LONG, min_periods=1).mean()
        rolling_12q = qoq_values.rolling(
            window=RollingWindow.EXTENDED, min_periods=1
        ).mean()

        fig.add_trace(
            go.Scatter(
                x=clean_data["Index"],
                y=rolling_4q,
                mode="lines",
                name="4Q Rolling Avg",
                line=dict(color="cyan", width=1),
                hovertemplate="4Q Rolling Avg: %{y:.1f}%<extra></extra>",
                visible="legendonly",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=clean_data["Index"],
                y=rolling_8q,
                mode="lines",
                name="8Q Rolling Avg",
                line=dict(color="magenta", width=2),
                hovertemplate="8Q Rolling Avg: %{y:.1f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=clean_data["Index"],
                y=rolling_12q,
                mode="lines",
                name="12Q Rolling Avg",
                line=dict(color="purple", width=3),
                hovertemplate="12Q Rolling Avg: %{y:.1f}%<extra></extra>",
                visible="legendonly",
            )
        )
    # Set x-axis range to show last N quarters by default
    display_quarters = RollingWindow.DISPLAY_QUARTERS
    if len(clean_data) > 1:
        x_max = clean_data["Index"].max() + 0.5
        if len(clean_data) > display_quarters:
            x_min = clean_data["Index"].iloc[-display_quarters] - 0.5
        else:
            x_min = clean_data["Index"].min() - 0.5
        fig.update_xaxes(range=[x_min, x_max])

    # Set y-axis range based on visible data with padding
    visible_data = (
        clean_data.tail(display_quarters)
        if len(clean_data) > display_quarters
        else clean_data
    )
    data_range = visible_data[qoq_column].max() - visible_data[qoq_column].min()
    padding = data_range * ChartDefaults.Y_AXIS_PADDING if data_range > 0 else 5
    y_min = visible_data[qoq_column].min() - padding
    y_max = visible_data[qoq_column].max() + padding
    fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        xaxis_title="",
        yaxis_title="QoQ % Change",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=ChartDefaults.compact_margin(),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_traces(
        line=dict(color="green"),
        marker=dict(
            color=clean_data[qoq_column],
            colorscale=["red", "green"],
            cmin=(
                -abs(clean_data[qoq_column]).max()
                if not clean_data[qoq_column].empty
                else 0
            ),
            cmax=(
                abs(clean_data[qoq_column]).max()
                if not clean_data[qoq_column].empty
                else 0
            ),
        ),
        selector=dict(mode="lines+markers"),
    )
    return fig


def create_blank_placeholder(title="No Data Available", height=400):
    """Create a blank placeholder chart to maintain layout symmetry"""
    fig = go.Figure()

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Value",
        height=height,
        margin=ChartDefaults.standard_margin(),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Add a centered text annotation
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="No data available",
        showarrow=False,
        font=dict(size=14, color="gray"),
        align="center",
    )

    # Remove axes
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    return fig


def create_combined_peg_pegy_chart(df, ticker, height=400):
    """Create a combined chart showing both PEG and PEGY ratios"""
    ticker_data = df[df["Ticker"] == ticker].copy()

    # Check if we have data for both metrics
    peg_available = (
        "PEGRatio" in ticker_data.columns and not ticker_data["PEGRatio"].isna().all()
    )
    pegy_available = (
        "PEGYRatio" in ticker_data.columns and not ticker_data["PEGYRatio"].isna().all()
    )

    if not peg_available and not pegy_available:
        return None

    fig = go.Figure()

    # Add PEG Ratio line if available
    if peg_available:
        peg_data = ticker_data.dropna(subset=["PEGRatio"])
        if len(peg_data) > 0:
            # Show last 20 quarters by default
            if len(peg_data) > RollingWindow.DISPLAY_QUARTERS:
                display_peg = peg_data.tail(RollingWindow.DISPLAY_QUARTERS)
            else:
                display_peg = peg_data

            fig.add_trace(
                go.Scatter(
                    x=display_peg["Index"],
                    y=display_peg["PEGRatio"],
                    mode="lines+markers",
                    name="PEG Ratio",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>PEG Ratio</b><br>Index: %{x}<br>"
                        "Value: %{y:.2f}<extra></extra>"
                    ),
                )
            )

    # Add PEGY Ratio line if available
    if pegy_available:
        pegy_data = ticker_data.dropna(subset=["PEGYRatio"])
        if len(pegy_data) > 0:
            # Show last 20 quarters by default
            if len(pegy_data) > RollingWindow.DISPLAY_QUARTERS:
                display_pegy = pegy_data.tail(RollingWindow.DISPLAY_QUARTERS)
            else:
                display_pegy = pegy_data

            fig.add_trace(
                go.Scatter(
                    x=display_pegy["Index"],
                    y=display_pegy["PEGYRatio"],
                    mode="lines+markers",
                    name="PEGY Ratio",
                    line=dict(color="red", width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>PEGY Ratio</b><br>Index: %{x}<br>"
                        "Value: %{y:.2f}<extra></extra>"
                    ),
                )
            )

    # Add reference line at good value threshold
    fig.add_hline(
        y=ChartDefaults.PEG_GOOD_VALUE,
        line_dash="dash",
        line_color="green",
        opacity=0.7,
        annotation_text=f"Good Value Threshold ({ChartDefaults.PEG_GOOD_VALUE})",
    )

    # Set axis ranges based on visible data
    all_data = []
    if peg_available:
        peg_visible = (
            ticker_data.dropna(subset=["PEGRatio"]).tail(RollingWindow.DISPLAY_QUARTERS)
            if len(ticker_data.dropna(subset=["PEGRatio"]))
            > RollingWindow.DISPLAY_QUARTERS
            else ticker_data.dropna(subset=["PEGRatio"])
        )
        all_data.extend(peg_visible["PEGRatio"].tolist())
    if pegy_available:
        pegy_visible = (
            ticker_data.dropna(subset=["PEGYRatio"]).tail(
                RollingWindow.DISPLAY_QUARTERS
            )
            if len(ticker_data.dropna(subset=["PEGYRatio"]))
            > RollingWindow.DISPLAY_QUARTERS
            else ticker_data.dropna(subset=["PEGYRatio"])
        )
        all_data.extend(pegy_visible["PEGYRatio"].tolist())

    if all_data:
        # Set x-axis range
        all_indices = []
        if peg_available:
            peg_visible = (
                ticker_data.dropna(subset=["PEGRatio"]).tail(
                    RollingWindow.DISPLAY_QUARTERS
                )
                if len(ticker_data.dropna(subset=["PEGRatio"]))
                > RollingWindow.DISPLAY_QUARTERS
                else ticker_data.dropna(subset=["PEGRatio"])
            )
            all_indices.extend(peg_visible["Index"].tolist())
        if pegy_available:
            pegy_visible = (
                ticker_data.dropna(subset=["PEGYRatio"]).tail(
                    RollingWindow.DISPLAY_QUARTERS
                )
                if len(ticker_data.dropna(subset=["PEGYRatio"]))
                > RollingWindow.DISPLAY_QUARTERS
                else ticker_data.dropna(subset=["PEGYRatio"])
            )
            all_indices.extend(pegy_visible["Index"].tolist())

        if all_indices:
            x_min = min(all_indices) - 0.5
            x_max = max(all_indices) + 0.5
            fig.update_xaxes(range=[x_min, x_max])

        # Set y-axis range
        data_range = max(all_data) - min(all_data)
        padding = data_range * ChartDefaults.Y_AXIS_PADDING if data_range > 0 else 0.5
        y_min = max(0, min(all_data) - padding)  # Don't go below 0
        y_max = max(all_data) + padding
        # Ensure we show at least up to 1.5 to include the reference line area
        y_max = max(y_max, 1.5)
        fig.update_yaxes(range=[y_min, y_max])

    tickvals, ticktext = _report_ticks(ticker_data)
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        title="PEG & PEGY Ratios Comparison",
        xaxis_title="",
        yaxis_title="Ratio Value",
        hovermode="x unified",
        height=height,
        margin=ChartDefaults.standard_margin(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_comparison_chart(df, tickers, metric, yaxis_range=None):
    # Filter data for selected tickers
    data = df[df["Ticker"].isin(tickers)]

    # Check if this is a QoQ metric
    is_qoq = metric.endswith("_QoQ")

    fig = px.line(
        data,
        x="Index",  # <-- Use Index for alignment
        y=metric,
        color="Ticker",
        markers=True,
        title=f"{metric} Comparison (Aligned by Index)",
    )

    # Add horizontal line at y=0 for QoQ charts
    if is_qoq:
        fig.add_hline(y=0, line_dash="dash", line_color="yellow", opacity=0.8)

    # Set x-axis range to show last N quarters by default
    display_quarters = RollingWindow.DISPLAY_QUARTERS
    if len(data) > 0:
        max_index = data["Index"].max()
        if max_index >= display_quarters:
            x_min = max_index - (display_quarters - 0.5)
            x_max = max_index + 0.5
            fig.update_xaxes(range=[x_min, x_max])

    # Set y-axis range if provided
    if yaxis_range is not None:
        fig.update_yaxes(range=yaxis_range)
    fig.update_layout(
        legend_title_text="Ticker",
        xaxis_title="",
        margin=ChartDefaults.standard_margin(),
    )
    return fig


def create_eps_prediction_chart(df, ticker, prediction=None):
    """
    Create EPS chart with prediction point for next quarter.

    Args:
        df (pandas.DataFrame): Stock data with QoQ calculations
        ticker (str): Stock ticker symbol
        prediction (dict, optional): Pre-computed predict_next_eps result.
            Computed here when not supplied, so callers rendering several
            prediction charts for one ticker can compute it once.

    Returns:
        plotly.graph_objects.Figure or None: Chart with prediction
    """
    # Start with regular EPS chart
    fig = create_metric_chart(df, ticker, "EPS", "Earnings Per Share")

    if fig is None:
        return None

    # Get prediction data
    if prediction is None:
        prediction = predict_next_eps(df, ticker)

    if prediction is None:
        # Return regular chart if no prediction possible
        return fig

    # Add prediction scenarios as points
    next_index = prediction["next_index"]

    # Best case scenario
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["best_case_eps"]],
            mode="markers",
            marker=dict(
                size=12,
                color="green",
                symbol="triangle-up",
                line=dict(width=2, color="darkgreen"),
            ),
            name="Best Case",
            hovertemplate=(
                "<b>Best Case EPS</b><br>"
                f"Value: ${prediction['best_case_eps']:.2f}<br>"
                f"Growth: {prediction['best_case_growth']:+.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Base case prediction (most likely)
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["predicted_eps"]],
            mode="markers",
            marker=dict(
                size=15, color="red", symbol="star", line=dict(width=2, color="darkred")
            ),
            name="Base Case",
            hovertemplate=(
                "<b>Base Case EPS</b><br>"
                f"Value: ${prediction['predicted_eps']:.2f}<br>"
                f"Growth: {prediction['predicted_growth']:+.1f}%<br>"
                f"Confidence: {prediction['confidence']}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Worst case scenario
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["worst_case_eps"]],
            mode="markers",
            marker=dict(
                size=12,
                color="orange",
                symbol="triangle-down",
                line=dict(width=2, color="darkorange"),
            ),
            name="Worst Case",
            hovertemplate=(
                "<b>Worst Case EPS</b><br>"
                f"Value: ${prediction['worst_case_eps']:.2f}<br>"
                f"Growth: {prediction['worst_case_growth']:+.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add prediction uncertainty band (shaded area)
    fig.add_trace(
        go.Scatter(
            x=[next_index, next_index, next_index, next_index],
            y=[
                prediction["worst_case_eps"],
                prediction["best_case_eps"],
                prediction["best_case_eps"],
                prediction["worst_case_eps"],
            ],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.1)",  # Light red
            line=dict(color="rgba(255,255,255,0)"),  # Transparent line
            name="Uncertainty Band",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add dashed lines connecting last actual to each scenario
    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")
    eps_data = ticker_data["EPS"].dropna()

    if len(eps_data) > 0:
        last_index = ticker_data.loc[eps_data.index[-1], "Index"]
        latest_eps = prediction["latest_eps"]

        # Line to base case
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[latest_eps, prediction["predicted_eps"]],
                mode="lines",
                line=dict(dash="dash", width=2, color="red"),
                name="Base Case Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line to best case
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[latest_eps, prediction["best_case_eps"]],
                mode="lines",
                line=dict(dash="dot", width=1, color="green"),
                name="Best Case Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line to worst case
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[latest_eps, prediction["worst_case_eps"]],
                mode="lines",
                line=dict(dash="dot", width=1, color="orange"),
                name="Worst Case Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Update title to indicate prediction
    prediction_title = "Earnings Per Share with Next Quarter Prediction"

    # Extend x-axis to include prediction points
    ticker_data = df[df["Ticker"] == ticker].copy()
    clean_data = ticker_data.dropna(subset=["EPS"])
    if len(clean_data) > 0:
        x_max = max(clean_data["Index"].max(), next_index) + 0.5
        if len(clean_data) > RollingWindow.DISPLAY_QUARTERS:
            x_min = clean_data["Index"].iloc[-RollingWindow.DISPLAY_QUARTERS] - 0.5
        else:
            x_min = clean_data["Index"].min() - 0.5
        fig.update_xaxes(range=[x_min, x_max])

        last_report = clean_data.loc[clean_data["Index"].idxmax(), "Report"]
        tickvals, ticktext = _report_ticks(
            clean_data,
            extra_index=next_index,
            extra_label=_next_quarter_label(last_report),
        )
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

        # Extend y-axis to include all prediction values
        all_y_values = []
        # Add historical data (visible range)
        visible_data = (
            clean_data.tail(RollingWindow.DISPLAY_QUARTERS)
            if len(clean_data) > RollingWindow.DISPLAY_QUARTERS
            else clean_data
        )
        all_y_values.extend(visible_data["EPS"].dropna().tolist())
        # Add prediction values
        all_y_values.extend(
            [
                prediction["predicted_eps"],
                prediction["best_case_eps"],
                prediction["worst_case_eps"],
            ]
        )

        if all_y_values:
            data_range = max(all_y_values) - min(all_y_values)
            padding = data_range * ChartDefaults.Y_AXIS_PADDING if data_range > 0 else 1
            y_min = min(all_y_values) - padding
            y_max = max(all_y_values) + padding
            fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        title=prediction_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_eps_ttm_prediction_chart(df, ticker, prediction=None):
    """
    Create EPS TTM chart with prediction scenarios for next quarter impact.

    Args:
        df (pandas.DataFrame): Stock data with QoQ calculations
        ticker (str): Stock ticker symbol
        prediction (dict, optional): Pre-computed predict_next_eps result.
            Computed here when not supplied, so callers rendering several
            prediction charts for one ticker can compute it once.

    Returns:
        plotly.graph_objects.Figure or None: Chart with TTM predictions
    """
    # Start with regular EPS TTM chart
    fig = create_metric_chart(df, ticker, "EPS_TTM", "EPS - TTM")

    if fig is None:
        return None

    # Get prediction data
    if prediction is None:
        prediction = predict_next_eps(df, ticker)

    if prediction is None or prediction["predicted_eps_ttm"] is None:
        # Return regular chart if no TTM prediction possible
        return fig

    # Add TTM prediction scenarios as points
    next_index = prediction["next_index"]

    # Best case TTM scenario
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["best_case_eps_ttm"]],
            mode="markers",
            marker=dict(
                size=12,
                color="green",
                symbol="triangle-up",
                line=dict(width=2, color="darkgreen"),
            ),
            name="Best Case TTM",
            hovertemplate=(
                "<b>Best Case EPS TTM</b><br>"
                f"Value: ${prediction['best_case_eps_ttm']:.2f}<br>"
                f"Growth: {prediction['best_case_eps_ttm_growth']:+.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Base case TTM prediction (most likely)
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["predicted_eps_ttm"]],
            mode="markers",
            marker=dict(
                size=15, color="red", symbol="star", line=dict(width=2, color="darkred")
            ),
            name="Base Case TTM",
            hovertemplate=(
                "<b>Base Case EPS TTM</b><br>"
                f"Value: ${prediction['predicted_eps_ttm']:.2f}<br>"
                f"Growth: {prediction['predicted_eps_ttm_growth']:+.1f}%<br>"
                f"Confidence: {prediction['confidence']}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Worst case TTM scenario
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["worst_case_eps_ttm"]],
            mode="markers",
            marker=dict(
                size=12,
                color="orange",
                symbol="triangle-down",
                line=dict(width=2, color="darkorange"),
            ),
            name="Worst Case TTM",
            hovertemplate=(
                "<b>Worst Case EPS TTM</b><br>"
                f"Value: ${prediction['worst_case_eps_ttm']:.2f}<br>"
                f"Growth: {prediction['worst_case_eps_ttm_growth']:+.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add TTM prediction uncertainty band (shaded area)
    fig.add_trace(
        go.Scatter(
            x=[next_index, next_index, next_index, next_index],
            y=[
                prediction["worst_case_eps_ttm"],
                prediction["best_case_eps_ttm"],
                prediction["best_case_eps_ttm"],
                prediction["worst_case_eps_ttm"],
            ],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.1)",  # Light red
            line=dict(color="rgba(255,255,255,0)"),  # Transparent line
            name="TTM Uncertainty Band",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add dashed lines connecting last actual TTM to each scenario
    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")
    eps_ttm_data = ticker_data["EPS_TTM"].dropna()

    if len(eps_ttm_data) > 0:
        last_index = ticker_data.loc[eps_ttm_data.index[-1], "Index"]
        current_eps_ttm = prediction["current_eps_ttm"]

        # Line to base case TTM
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[current_eps_ttm, prediction["predicted_eps_ttm"]],
                mode="lines",
                line=dict(dash="dash", width=2, color="red"),
                name="Base Case TTM Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line to best case TTM
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[current_eps_ttm, prediction["best_case_eps_ttm"]],
                mode="lines",
                line=dict(dash="dot", width=1, color="green"),
                name="Best Case TTM Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line to worst case TTM
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[current_eps_ttm, prediction["worst_case_eps_ttm"]],
                mode="lines",
                line=dict(dash="dot", width=1, color="orange"),
                name="Worst Case TTM Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Update title to indicate TTM prediction
    ttm_prediction_title = "EPS TTM with Next Quarter Impact Prediction"

    # Extend x-axis to include prediction points
    ticker_data = df[df["Ticker"] == ticker].copy()
    clean_data = ticker_data.dropna(subset=["EPS_TTM"])
    if len(clean_data) > 0:
        x_max = max(clean_data["Index"].max(), next_index) + 0.5
        if len(clean_data) > RollingWindow.DISPLAY_QUARTERS:
            x_min = clean_data["Index"].iloc[-RollingWindow.DISPLAY_QUARTERS] - 0.5
        else:
            x_min = clean_data["Index"].min() - 0.5
        fig.update_xaxes(range=[x_min, x_max])

        last_report = clean_data.loc[clean_data["Index"].idxmax(), "Report"]
        tickvals, ticktext = _report_ticks(
            clean_data,
            extra_index=next_index,
            extra_label=_next_quarter_label(last_report),
        )
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

        # Extend y-axis to include all TTM prediction values
        all_y_values = []
        # Add historical data (visible range)
        visible_data = (
            clean_data.tail(RollingWindow.DISPLAY_QUARTERS)
            if len(clean_data) > RollingWindow.DISPLAY_QUARTERS
            else clean_data
        )
        all_y_values.extend(visible_data["EPS_TTM"].dropna().tolist())
        # Add TTM prediction values
        all_y_values.extend(
            [
                prediction["predicted_eps_ttm"],
                prediction["best_case_eps_ttm"],
                prediction["worst_case_eps_ttm"],
            ]
        )

        if all_y_values:
            data_range = max(all_y_values) - min(all_y_values)
            padding = data_range * ChartDefaults.Y_AXIS_PADDING if data_range > 0 else 1
            y_min = min(all_y_values) - padding
            y_max = max(all_y_values) + padding
            fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        title=ttm_prediction_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_price_prediction_chart(df, ticker, prediction=None):
    """
    Create Price chart with prediction scenarios based on EPS_TTM predictions and
    current Multiple.

    Args:
        df (pandas.DataFrame): Stock data with QoQ calculations
        ticker (str): Stock ticker symbol
        prediction (dict, optional): Pre-computed predict_next_eps result.
            Computed here when not supplied, so callers rendering several
            prediction charts for one ticker can compute it once.

    Returns:
        plotly.graph_objects.Figure or None: Chart with price predictions
    """
    # Start with regular Price chart
    fig = create_metric_chart(df, ticker, "Price", "Stock Price")

    if fig is None:
        return None

    # Get prediction data
    if prediction is None:
        prediction = predict_next_eps(df, ticker)

    if prediction is None or prediction["predicted_price"] is None:
        # Return regular chart if no price prediction possible
        return fig

    # Add price prediction scenarios as points
    next_index = prediction["next_index"]

    # Best case price scenario
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["best_case_price"]],
            mode="markers",
            marker=dict(
                size=12,
                color="green",
                symbol="triangle-up",
                line=dict(width=2, color="darkgreen"),
            ),
            name="Best Case Price",
            hovertemplate=(
                "<b>Best Case Price</b><br>"
                f"Value: ${prediction['best_case_price']:.2f}<br>"
                f"Growth: {prediction['best_case_price_growth']:+.1f}%<br>"
                f"P/E: {prediction['current_multiple']:.1f}x<br>"
                "<extra></extra>"
            ),
        )
    )

    # Base case price prediction (most likely)
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["predicted_price"]],
            mode="markers",
            marker=dict(
                size=15, color="red", symbol="star", line=dict(width=2, color="darkred")
            ),
            name="Base Case Price",
            hovertemplate=(
                "<b>Base Case Price</b><br>"
                f"Value: ${prediction['predicted_price']:.2f}<br>"
                f"Growth: {prediction['predicted_price_growth']:+.1f}%<br>"
                f"P/E: {prediction['current_multiple']:.1f}x<br>"
                f"Confidence: {prediction['confidence']}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Worst case price scenario
    fig.add_trace(
        go.Scatter(
            x=[next_index],
            y=[prediction["worst_case_price"]],
            mode="markers",
            marker=dict(
                size=12,
                color="orange",
                symbol="triangle-down",
                line=dict(width=2, color="darkorange"),
            ),
            name="Worst Case Price",
            hovertemplate=(
                "<b>Worst Case Price</b><br>"
                f"Value: ${prediction['worst_case_price']:.2f}<br>"
                f"Growth: {prediction['worst_case_price_growth']:+.1f}%<br>"
                f"P/E: {prediction['current_multiple']:.1f}x<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add price prediction uncertainty band (shaded area)
    fig.add_trace(
        go.Scatter(
            x=[next_index, next_index, next_index, next_index],
            y=[
                prediction["worst_case_price"],
                prediction["best_case_price"],
                prediction["best_case_price"],
                prediction["worst_case_price"],
            ],
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.1)",  # Light red
            line=dict(color="rgba(255,255,255,0)"),  # Transparent line
            name="Price Uncertainty Band",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add dashed lines connecting last actual price to each scenario
    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")
    price_data = ticker_data["Price"].dropna()

    if len(price_data) > 0:
        last_index = ticker_data.loc[price_data.index[-1], "Index"]
        current_price = prediction["current_price"]

        # Line to base case price
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[current_price, prediction["predicted_price"]],
                mode="lines",
                line=dict(dash="dash", width=2, color="red"),
                name="Base Case Price Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line to best case price
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[current_price, prediction["best_case_price"]],
                mode="lines",
                line=dict(dash="dot", width=1, color="green"),
                name="Best Case Price Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line to worst case price
        fig.add_trace(
            go.Scatter(
                x=[last_index, next_index],
                y=[current_price, prediction["worst_case_price"]],
                mode="lines",
                line=dict(dash="dot", width=1, color="orange"),
                name="Worst Case Price Line",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Update title to indicate price prediction
    price_prediction_title = "Stock Price with Multiple-Based Prediction"

    # Extend x-axis to include prediction points
    ticker_data = df[df["Ticker"] == ticker].copy()
    clean_data = ticker_data.dropna(subset=["Price"])
    if len(clean_data) > 0:
        x_max = max(clean_data["Index"].max(), next_index) + 0.5
        if len(clean_data) > RollingWindow.DISPLAY_QUARTERS:
            x_min = clean_data["Index"].iloc[-RollingWindow.DISPLAY_QUARTERS] - 0.5
        else:
            x_min = clean_data["Index"].min() - 0.5
        fig.update_xaxes(range=[x_min, x_max])

        last_report = clean_data.loc[clean_data["Index"].idxmax(), "Report"]
        tickvals, ticktext = _report_ticks(
            clean_data,
            extra_index=next_index,
            extra_label=_next_quarter_label(last_report),
        )
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

        # Extend y-axis to include all price prediction values
        all_y_values = []
        # Add historical data (visible range)
        visible_data = (
            clean_data.tail(RollingWindow.DISPLAY_QUARTERS)
            if len(clean_data) > RollingWindow.DISPLAY_QUARTERS
            else clean_data
        )
        all_y_values.extend(visible_data["Price"].dropna().tolist())
        # Add price prediction values
        all_y_values.extend(
            [
                prediction["predicted_price"],
                prediction["best_case_price"],
                prediction["worst_case_price"],
            ]
        )

        if all_y_values:
            data_range = max(all_y_values) - min(all_y_values)
            padding = data_range * ChartDefaults.Y_AXIS_PADDING if data_range > 0 else 1
            y_min = min(all_y_values) - padding
            y_max = max(all_y_values) + padding
            fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        title=price_prediction_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
