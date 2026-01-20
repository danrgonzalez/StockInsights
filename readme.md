# Stock Analysis Workflow

## Quick Start

```bash
python scripts/run_analysis.py
```

Or run the dashboard directly:

```bash
streamlit run dashboard/app.py
```

---

A comprehensive Python-based stock data analysis tool that processes quarterly financial data and provides an interactive dashboard for visualization and comparison.

## Data Pipeline Overview

The following diagram shows how raw stock data flows through the system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐
│ data/StockData   │     │ data/quotes/*.csv│
│     .xlsx        │     │                  │
│  (Raw Input)     │     │ (Optional)       │
│                  │     │                  │
│  - Ticker        │     │  - Symbol        │
│  - Report        │     │  - SimpleMoving  │
│  - EPS           │     │    Avg           │
│  - Revenue       │     │                  │
│  - Price         │     │                  │
│  - DivAmt        │     │                  │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └───────────┬────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: scripts/indexer.py                             │
│                     (Run once when data changes)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Load raw Excel data                                                     │
│  2. Add sequential Index column per ticker (1=oldest, N=newest)             │
│  3. Align all tickers to end at same max index (enables comparison)         │
│  4. Optionally update latest Price with SimpleMovingAvg from CSV            │
│  5. Validate data integrity (no duplicates, correct sequences)              │
│  6. Remove unnamed/empty columns                                            │
│                                                                             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ data/StockData_Indexed │
                    │     .xlsx              │
                    │                        │
                    │  + Index column        │
                    │  + Cleaned data        │
                    │  + Updated prices      │
                    └────────────┬───────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: calculate_qoq_changes()                        │
│                     (Run at dashboard load / library use)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each ticker, calculate:                                                │
│                                                                             │
│  TTM (Trailing Twelve Months):          Derived Metrics:                    │
│  ├─ EPS_TTM = sum(last 4 EPS)           ├─ Multiple = Price / EPS_TTM       │
│  └─ Revenue_TTM = sum(last 4 Revenue)   ├─ DivYield = DivAmt / Price        │
│                                         ├─ DivYieldAnnual = DivAmt*4/Price  │
│  QoQ Changes (% change from prior Q):   └─ PayoutRatio = DivAmt*4 / EPS_TTM │
│  ├─ EPS_QoQ                                                                 │
│  ├─ Revenue_QoQ                         Advanced Analytics:                 │
│  ├─ Price_QoQ                           ├─ EPSMomentum (4Q avg - 8Q avg)    │
│  ├─ EPS_TTM_QoQ                         ├─ PriceVolatility (8Q std dev)     │
│  ├─ Revenue_TTM_QoQ                     ├─ RevenueConsistency (CV score)    │
│  ├─ Multiple_QoQ                        ├─ PEGRatio (P/E / growth rate)     │
│  ├─ DivAmt_QoQ                          ├─ PEGYRatio (PEG / div yield)      │
│  └─ DivYield_QoQ                        └─ DivGrowthRate (annualized)       │
│                                                                             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: Additional Analysis (Optional)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  calculate_sector_rankings()     - Rank tickers within their sector         │
│  calculate_outperformance_ratios() - Compare vs sector/market averages      │
│  calculate_downside_capture()    - Measure downside risk vs market          │
│  predict_next_eps()              - Forecast next quarter EPS                │
│                                                                             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OUTPUT: Dashboard or Core Library                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dashboard (dashboard/app.py):      Core Library (from core import ...):    │
│  ├─ Interactive charts              ├─ Use in other Streamlit apps          │
│  ├─ Multi-ticker comparison         ├─ Build APIs with Flask/FastAPI        │
│  ├─ Rolling averages table          ├─ Create CLI tools                     │
│  └─ EPS predictions                 └─ Jupyter notebook analysis            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why the Index Column Matters

The `scripts/indexer.py` adds an `Index` column that enables time-aligned comparison across tickers with different history lengths:

```
Before Indexing:                    After Indexing:

AAPL has 40 quarters                AAPL: Index 1-40  (starts at 1)
NVDA has 32 quarters                NVDA: Index 9-40  (starts at 9)
                                           ↑
                                    All tickers END at same index (40)
                                    This aligns their most recent data
```

This alignment is critical for the multi-ticker comparison feature in the dashboard.

---

## Overview

This project consists of two main parts:

1. **Core Library** (`core/`) - Reusable data processing and analysis logic with no UI dependencies
2. **Streamlit Dashboard** - Interactive web interface for visualization

### Using the Core Library

The `core/` module contains pure Python data processing functions that can be imported into any application (Streamlit, Flask, CLI, Jupyter notebooks, etc.):

```python
from core import (
    load_stock_data,
    calculate_qoq_changes,
    calculate_sector_rankings,
    calculate_outperformance_ratios,
    calculate_downside_capture,
    predict_next_eps,
)

# Load and process data
df = load_stock_data("data/StockData_Indexed.xlsx")
df = calculate_qoq_changes(df)
df = calculate_sector_rankings(df)

# Get EPS predictions for any ticker
prediction = predict_next_eps(df, "AAPL")
print(f"Predicted EPS: ${prediction['predicted_eps']:.2f}")
print(f"Confidence: {prediction['confidence']}")
```

### Dashboard Components

- **Stock Indexer** (`scripts/indexer.py`) - Processes raw stock data and adds time-based indexing
- **Stock Dashboard** (`dashboard/app.py`) - Interactive Streamlit dashboard for data visualization
- **Workflow Runner** (`scripts/run_analysis.py`) - Automated script to run both components sequentially

## Running the Application

### Environment Setup (Conda Users)
```bash
source setup_env.sh
```

### Option 1: Automated Workflow (Recommended)
```bash
python scripts/run_analysis.py
```
This will automatically:
1. Run the stock indexer to process your data
2. Launch the interactive dashboard in your browser

### Option 2: Manual Step-by-Step
```bash
# Step 1: Process the raw data
python scripts/indexer.py

# Step 2: Launch the dashboard
streamlit run dashboard/app.py
```

### Option 3: Using Core Library Only (No Dashboard)
```python
from core import load_stock_data, calculate_qoq_changes, predict_next_eps

# Load and process
df = load_stock_data("data/StockData_Indexed.xlsx")
df = calculate_qoq_changes(df)

# Analyze any ticker
for ticker in ["AAPL", "MSFT", "GOOGL"]:
    prediction = predict_next_eps(df, ticker)
    if prediction:
        print(f"{ticker}: ${prediction['predicted_eps']:.2f} ({prediction['confidence']})")
```

## Project Structure

```
StockInsights/
├── core/                          # Core library (no UI dependencies)
│   ├── __init__.py               # Public API exports
│   ├── data_processing.py        # Data loading and calculations
│   ├── predictions.py            # EPS prediction logic
│   ├── strategies.py             # EPS prediction strategies
│   ├── backtesting.py            # Strategy backtesting
│   └── classifications.py        # Sector/industry classifications
│
├── dashboard/                     # Streamlit dashboard components
│   ├── __init__.py               # Package exports
│   ├── app.py                    # Main Streamlit dashboard
│   ├── data_utils.py             # Dashboard-specific data utilities
│   ├── charts.py                 # Plotly chart generation
│   └── ui_components.py          # Reusable UI components
│
├── scripts/                       # CLI tools
│   ├── indexer.py                # Data preprocessing script
│   └── run_analysis.py           # Workflow automation
│
├── data/                          # Data files
│   ├── StockData.xlsx            # Input data file (user-provided)
│   ├── StockData_Indexed.xlsx    # Generated indexed data
│   └── quotes/                   # Optional price data directory
│
├── config/                        # Configuration files
│   └── ticker_strategy_mapping.json  # Optimal strategies per ticker
│
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Tool configuration
└── readme.md                    # This file
```

## Requirements

### Required Files
- `data/StockData.xlsx` - Your raw quarterly stock data file

### Python Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas streamlit plotly numpy pydantic openpyxl
```

### Development Dependencies (Optional)
```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Input Data Format

Your `StockData.xlsx` file should contain the following columns:

| Column   | Description                        | Example     |
|----------|------------------------------------|-------------|
| Ticker   | Stock symbol                       | AAPL        |
| Report   | Reporting period/date              | 2023-Q1     |
| EPS      | Earnings Per Share                 | 2.18        |
| Revenue  | Quarterly revenue (in millions)    | 117154      |
| Price    | Stock price                        | 157.96      |
| DivAmt   | Dividend amount (optional)         | 0.24        |

## Core Library API

### Data Loading
```python
from core import load_stock_data

df = load_stock_data("data/StockData_Indexed.xlsx")
```

### Calculate Metrics
```python
from core import calculate_qoq_changes, calculate_sector_rankings

# Calculate QoQ changes, TTM values, P/E ratios, dividend yields, etc.
df = calculate_qoq_changes(df)

# Add sector rankings (requires Sector column from classifications)
df = calculate_sector_rankings(df)
```

### EPS Predictions
```python
from core import predict_next_eps

prediction = predict_next_eps(df, "AAPL")
# Returns dict with:
# - predicted_eps, best_case_eps, worst_case_eps
# - predicted_eps_ttm, predicted_price
# - confidence level, methodology used
```

## Dashboard Features

### Tab 1: Individual Analysis
- Absolute value charts (EPS, Revenue, Price, etc.)
- Quarter-over-Quarter (QoQ) percentage change analysis
- Trailing Twelve Months (TTM) calculations
- P/E multiple analysis
- Rolling average trend lines
- EPS predictions with confidence intervals

### Tab 2: Multi-Ticker Comparison
- Side-by-side metric comparisons
- Customizable chart selection
- Time-aligned analysis using Index

### Tab 3: Rolling Averages Summary
- Comprehensive QoQ growth summaries
- Sector-based filtering
- 4Q, 8Q, and 12Q rolling averages
- Downloadable CSV reports

### Tab 4: Methodology
- Documentation of calculations
- Prediction strategy explanations

## Key Metrics

| Metric            | Description                                           |
|-------------------|-------------------------------------------------------|
| QoQ               | Quarter-over-Quarter percentage change                |
| TTM               | Trailing Twelve Months (sum of last 4 quarters)       |
| P/E Multiple      | Price divided by EPS TTM                              |
| PEG Ratio         | P/E divided by earnings growth rate                   |
| PEGY Ratio        | PEG divided by dividend yield                         |
| EPS Momentum      | Difference between 4Q and 8Q average growth           |
| Downside Capture  | Stock's decline relative to market during downturns   |

## Prediction Strategies

The system uses backtested optimal strategies per ticker:

| Strategy        | Description                                        |
|-----------------|---------------------------------------------------|
| weighted_growth | 70% recent + 30% long-term weighted average       |
| simple_average  | Simple mean of last 4 quarters                    |
| seasonal        | Year-over-year same-quarter patterns              |
| trend_analysis  | Linear regression extrapolation                   |
| momentum        | Exponentially weighted recent performance         |

Run `python multi_ticker_backtest.py` to optimize strategies for your data.

## Troubleshooting

### Common Issues

**File not found errors:**
```
ERROR: data/StockData.xlsx not found
```
Solution: Ensure `StockData.xlsx` is in the `data/` folder.

**Missing dependencies:**
```
ERROR: Missing required packages
```
Solution: Run `pip install -r requirements.txt`

**Port already in use:**
```
Error: Port 8501 is already in use
```
Solution: Run `streamlit run dashboard/app.py --server.port 8502`

## Updating Data

1. Update `data/StockData.xlsx` with new records
2. Re-run the indexer: `python scripts/indexer.py`
3. Refresh the dashboard (or restart it)

## Development

### Code Quality
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Prediction Strategies

1. Add strategy function to `core/strategies.py`
2. Register in `STRATEGIES` dict
3. Run backtesting to evaluate performance

### Extending the Core Library

The `core/` module is designed for reuse in other applications. To add new calculations:

1. Add function to `core/data_processing.py`
2. Export in `core/__init__.py`
3. Document in this README
