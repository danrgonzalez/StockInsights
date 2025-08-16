import plotly.express as px
import plotly.graph_objects as go

def create_metric_chart(df, ticker, metric, title):
    ticker_data = df[df['Ticker'] == ticker].copy()
    if metric not in ticker_data.columns or ticker_data[metric].isna().all():
        return None
    clean_data = ticker_data.dropna(subset=[metric])
    if len(clean_data) == 0:
        return None
    fig = px.line(
        clean_data,
        x='Index',
        y=metric,
        title=f"{title} Over Time (Most Recent = Higher Index)",
        hover_data=['Report'],
        markers=True
    )
    fig.update_layout(
        xaxis_title="Index (Higher = More Recent)",
        yaxis_title=title,
        hovermode='x unified'
    )
    return fig

def create_qoq_chart(df, ticker, metric, title):
    ticker_data = df[df['Ticker'] == ticker].copy()
    qoq_column = f'{metric}_QoQ'
    if (
        qoq_column not in ticker_data.columns or
        ticker_data[qoq_column].isna().all()
    ):
        return None
    clean_data = ticker_data.dropna(subset=[qoq_column])
    if len(clean_data) == 0:
        return None
    fig = px.line(
        clean_data,
        x='Index',
        y=qoq_column,
        title=f"{title} - Quarter over Quarter % Change",
        hover_data=['Report', metric],
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    if (
        metric in ['EPS', 'Revenue', 'EPS_TTM', 'Revenue_TTM', 'Price'] and
        len(clean_data) > 0
    ):
        qoq_values = clean_data[qoq_column]
        rolling_4q = qoq_values.rolling(window=4, min_periods=1).mean()
        rolling_8q = qoq_values.rolling(window=8, min_periods=1).mean()
        rolling_12q = qoq_values.rolling(window=12, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=clean_data['Index'],
            y=rolling_4q,
            mode='lines',
            name='4Q Rolling Avg',
            line=dict(color='cyan', width=1),
            hovertemplate='4Q Rolling Avg: %{y:.1f}%<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=clean_data['Index'],
            y=rolling_8q,
            mode='lines',
            name='8Q Rolling Avg',
            line=dict(color='magenta', width=2),
            hovertemplate='8Q Rolling Avg: %{y:.1f}%<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=clean_data['Index'],
            y=rolling_12q,
            mode='lines',
            name='12Q Rolling Avg',
            line=dict(color='purple', width=3),
            hovertemplate='12Q Rolling Avg: %{y:.1f}%<extra></extra>'
        ))
    fig.update_layout(
        xaxis_title="Index (Higher = More Recent)",
        yaxis_title="QoQ % Change",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_traces(
        line=dict(color='green'),
        marker=dict(
            color=clean_data[qoq_column],
            colorscale=['red', 'green'],
            cmin=-abs(clean_data[qoq_column]).max()
            if not clean_data[qoq_column].empty else 0,
            cmax=abs(clean_data[qoq_column]).max()
            if not clean_data[qoq_column].empty else 0
        ),
        selector=dict(mode='lines+markers')
    )
    return fig

def create_comparison_chart(df, tickers, metric, yaxis_range=None):
    # Filter data for selected tickers
    data = df[df['Ticker'].isin(tickers)]
    fig = px.line(
        data,
        x='Index',  # <-- Use Index for alignment
        y=metric,
        color='Ticker',
        markers=True,
        title=f"{metric} Comparison (Aligned by Index)"
    )
    # Set y-axis range if provided
    if yaxis_range is not None:
        fig.update_yaxes(range=yaxis_range)
    fig.update_layout(
        legend_title_text='Ticker',
        xaxis_title="Index (Higher = More Recent)"
    )
    return fig