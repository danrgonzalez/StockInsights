# Stock Analysis Workflow

## Quick Start

> **Always activate the `stockinsights` environment first.** The dashboard needs
> `streamlit>=1.49` and will crash on an older one. See
> [Environment Setup](#environment-setup-conda-users).

```bash
source setup_env.sh            # activate stockinsights (required, every session)
python scripts/run_analysis.py # Stage 1 (indexer) + launch dashboard
```

Or, if the data is already indexed, launch the dashboard directly:

```bash
source setup_env.sh
streamlit run dashboard/app.py
```

**Before running:** close `data/StockData.xlsx` in Excel. An open workbook leaves a
`~$StockData.xlsx` lock file, and Excel will overwrite pipeline changes when it saves.

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

First-time setup — create the `stockinsights` environment:
```bash
conda create -n stockinsights -c conda-forge python=3.11
conda activate stockinsights
pip install -r requirements.txt
```

For every session after that, activate the environment with:
```bash
source setup_env.sh
```

> **Apple silicon (M1/M2/M3/M4) note:** Use a native **arm64** conda
> (e.g. [Miniconda for Apple silicon](https://docs.conda.io/projects/miniconda/)).
> An Intel build of Anaconda runs under Rosetta, which Apple is deprecating
> (support ends after macOS 27). To confirm your environment is native, run
> `python -c "import platform; print(platform.machine())"` — it should print
> `arm64`, not `x86_64`.

### Pre-flight checks

```bash
source setup_env.sh

# 1. Confirm you are in the right environment - this MUST print the conda path
python -c "import sys, streamlit; print(sys.executable); print('streamlit', streamlit.__version__)"
#    expected: /Users/<you>/miniconda3/envs/stockinsights/bin/python
#              streamlit 1.58.0

# 2. Confirm Excel does not hold the workbook open
ls data/ | grep '~\$' && echo "CLOSE EXCEL FIRST" || echo "ok"
```

If `sys.executable` points at a system Python (e.g. `/Library/Frameworks/...` or
`/usr/bin/python3`), stop — see [Wrong Python environment](#wrong-python-environment).

### Option 1: Automated Workflow (Recommended)
```bash
source setup_env.sh
python scripts/run_analysis.py
```
This will automatically:
1. Run Stage 1 (`scripts/indexer.py`) to rebuild `data/StockData_Indexed.xlsx`
2. Launch the interactive dashboard in your browser

### Option 2: Manual Step-by-Step
```bash
source setup_env.sh

# Stage 1: rebuild the indexed workbook from the raw data
python scripts/indexer.py

# Stages 2-3 + UI: the dashboard runs these on load
streamlit run dashboard/app.py
```

Port 8501 is the Streamlit default and may already be taken by another app.
Pick a free port explicitly if so:

```bash
streamlit run dashboard/app.py --server.port 8502
```

Check what is holding a port with `lsof -nP -iTCP:8501 -sTCP:LISTEN`.

### Option 3: Run Stages 1-3 headlessly (no dashboard)

Useful for verifying the pipeline end to end without opening the UI:

```bash
source setup_env.sh
python scripts/indexer.py          # Stage 1

python - <<'EOF'
from core import (load_stock_data, calculate_qoq_changes, calculate_sector_rankings,
                  calculate_outperformance_ratios, calculate_downside_capture,
                  predict_next_eps)

df = load_stock_data("data/StockData_Indexed.xlsx")   # applies the exclusion list
print("loaded          ", df.shape, df["Ticker"].nunique(), "tickers")

df = calculate_qoq_changes(df);           print("stage 2         ", df.shape)
df = calculate_sector_rankings(df);       print("stage 3a rank   ", df.shape)
df = calculate_outperformance_ratios(df); print("stage 3b outperf", df.shape)
df = calculate_downside_capture(df);      print("stage 3c downside", df.shape)

ok = sum(predict_next_eps(df, t) is not None for t in df["Ticker"].unique())
print("stage 3d predict", ok, "/", df["Ticker"].nunique())
EOF
```

Expected shape progression (as of 2026-09-02): `(8456, 8)` -> `(8456, 32)` ->
`(8456, 32)` -> `(8456, 37)` -> `(8456, 38)`, and 136/137 predictions.

> `calculate_sector_rankings()` currently adds **0 columns**: it needs a `Sector`
> column that the source data does not carry. The classifications exist in
> `core/classifications.py` but are never joined onto the DataFrame, so sector
> ranking is presently inert.

### Option 4: Using Core Library Only

```python
from core import load_stock_data, calculate_qoq_changes, predict_next_eps

df = load_stock_data("data/StockData_Indexed.xlsx")
df = calculate_qoq_changes(df)

for ticker in ["AAPL", "MSFT", "GOOGL"]:
    prediction = predict_next_eps(df, ticker)
    if prediction:
        print(f"{ticker}: ${prediction['predicted_eps']:.2f} ({prediction['confidence']})")
```

> **Note:** `predict_next_eps` reads `config/ticker_strategy_mapping.json` by a
> path relative to the **current working directory**. Run from the repo root, or
> the per-ticker strategy silently falls back to `weighted_growth` while still
> reporting "backtested optimal".

### Option 5: Export every computed metric to a file

Runs the same pipeline the dashboard runs and writes the results to disk, so the
numbers can be read - or handed to another agent - without launching Streamlit:

```bash
source setup_env.sh
python scripts/export_dashboard_data.py
```

Writes to `data/exports/` (about 5 seconds):

| File | Contents |
| --- | --- |
| `stock_analysis_export.json` | Self-describing bundle: metadata, a full data dictionary, per-ticker snapshot and sector aggregates. ~1.4 MB. |
| `quarterly_metrics.csv` | Full panel, one row per ticker per quarter, 56 columns. |
| `ticker_snapshot.csv` | The per-ticker snapshot flattened for spreadsheets. |
| `sector_summary.csv` | Per-sector averages and medians. |
| `README.md` | Guide to the four files above. |

Each ticker in the JSON carries its latest values, QoQ summaries, 4Q/8Q/12Q
rolling averages, peer ranks within its sector and the whole panel, a
next-quarter EPS/TTM/price forecast, and `data_flags.warnings` naming anything
that makes its own metrics unreliable.

Useful flags:

```bash
python scripts/export_dashboard_data.py --output-dir /tmp/export
python scripts/export_dashboard_data.py --include-history   # embeds every
                                                            # quarterly row in
                                                            # the JSON (~17 MB)
```

Two deliberate differences from the dashboard, both of them fixes for the
limitations noted above:

- The exporter joins `core/classifications.py` onto the DataFrame, so sector
  comparisons are populated rather than inert.
- It `chdir`s to the repo root before predicting, so the per-ticker strategy
  mapping always loads and forecasts do not depend on where you ran it from.

Its `peer_comparison` block also ranks each ticker's latest quarter against
other tickers' latest quarters, which is what a sector rank is normally read as.
The pipeline's own `*_SectorRank` columns rank a row against every
ticker-quarter row in the sector across all history; they are still exported in
`quarterly_metrics.csv`, labelled as such in the data dictionary.

## Excluded Tickers

`config/excluded_tickers.json` is the single source of truth for tickers with known
data problems. Entries with `"exclude": true` are filtered out at data load, so the
dashboard, predictions and backtests all ignore the same set.

| Ticker | Status | Reason |
|--------|--------|--------|
| `S` | excluded | Delisted (Sprint). Last report 2019-10-25. No real earnings dates. |
| `JWN` | excluded | Taken private. 585 days stale. No real earnings dates. |
| `SKX` | excluded | Acquired. 403 days stale. No real earnings dates. |
| `BRK/B` | excluded | 10 quarters missing (Q2'17-Q3'19); TTM/QoQ and Index alignment are wrong. |
| `BABA` | watch | One suspect earnings date; financials sound, so still included. |
| `EA` | watch | "Buyout" note in the sheet; still reporting. |

To bring a ticker back, set its `"exclude"` to `false` — no re-index needed, since
`StockData_Indexed.xlsx` still contains all 141 tickers.

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
│   ├── run_analysis.py           # Workflow automation
│   └── export_dashboard_data.py  # Export all computed metrics to JSON/CSV
│
├── data/                          # Data files
│   ├── StockData.xlsx            # Input data file (user-provided)
│   ├── StockData_Indexed.xlsx    # Generated indexed data
│   ├── exports/                  # Generated metric exports (JSON + CSV)
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

## Getting the Data

**The data files are deliberately not in this repository.** `.gitignore` excludes
`*.xlsx` and `*.csv`, so a fresh clone has no `data/` directory and nothing will
run until you supply one. This keeps the repo small and avoids committing a
binary that changes every quarter.

You need to create `data/` and place the raw workbook in it:

```
data/
├── StockData.xlsx          # you supply this — the raw quarterly panel
├── StockData_Indexed.xlsx  # generated: python scripts/indexer.py
└── exports/                # generated: python scripts/export_dashboard_data.py
```

`StockData.xlsx` is maintained by hand from quarterly filings, one row per
ticker per fiscal quarter, in the format below. Once it is in place:

```bash
python scripts/indexer.py          # builds StockData_Indexed.xlsx
streamlit run dashboard/app.py     # reads the indexed workbook
```

Everything under `data/` is either hand-maintained or regenerable from
`StockData.xlsx`, so nothing there needs backing up beyond that one file —
**but it is not version-controlled either, so keep your own copy.** Corrections
made to the workbook (label fixes, backfilled earnings dates) live only in your
copy; see `TODO.md` for the running record of what has been changed and why.

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

### Wrong Python environment

**By far the most common failure.** The app requires `streamlit>=1.49` (it uses
`width="stretch"`). A system Python with an older Streamlit crashes mid-render:

```
TypeError: 'str' object cannot be interpreted as an integer
  ... in dataframe: proto.width = width
```

The Streamlit server still starts and answers on its port, so the app *looks* fine
until you actually open it. Diagnose:

```bash
python -c "import sys, streamlit; print(sys.executable, streamlit.__version__)"
```

If that is not `.../miniconda3/envs/stockinsights/bin/python` with `1.58.0`:

```bash
source setup_env.sh
```

To confirm the whole app script runs before opening a browser:

```bash
python -c "import runpy; runpy.run_path('dashboard/app.py', run_name='__main__')" 2>&1 | tail -5
```
Streamlit prints "bare mode" / ScriptRunContext warnings here — those are expected.
A traceback is not.

### Excel holds the workbook open

Writes to `data/StockData.xlsx` are silently lost, or clobbered when Excel next saves.

```bash
ls data/ | grep '~\$'    # a ~$StockData.xlsx lock file means it is open
```
Solution: close the workbook in Excel before running anything that writes it.

### Port already in use
```
Error: Port 8501 is already in use
```
Solution: `streamlit run dashboard/app.py --server.port 8502`.
Check the occupant first with `lsof -nP -iTCP:8501 -sTCP:LISTEN` — it may be a
different project you do not want to kill.

### A ticker shows no EPS prediction

Expected for companies whose EPS crosses zero repeatedly (e.g. `INTC`): every
quarter-over-quarter growth exceeds the +/-200% outlier filter, leaving the strategy
with nothing to fit. The dashboard shows an info message rather than failing.

### A ticker is missing entirely

Check `config/excluded_tickers.json` — see [Excluded Tickers](#excluded-tickers).
The dashboard also prints an `st.info` naming everything it dropped at load.

### File not found errors
```
ERROR: data/StockData.xlsx not found
```
Solution: ensure `StockData.xlsx` is in the `data/` folder, and run from the repo root.

### Missing dependencies
```
ERROR: Missing required packages
```
Solution: `source setup_env.sh` first; if still missing, `pip install -r requirements.txt`.

## Updating Data

1. Update `data/StockData.xlsx` with new records
2. Re-run the indexer: `python scripts/indexer.py`
3. Refresh the dashboard (or restart it)

Because `data/` is untracked (see [Getting the Data](#getting-the-data)), step 1
is not recoverable from git. Keep a backup of `StockData.xlsx` before editing it.

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
