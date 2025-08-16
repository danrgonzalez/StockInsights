# Stock Analysis Workflow

A comprehensive Python-based stock data analysis tool that processes quarterly financial data and provides an interactive dashboard for visualization and comparison.

## 📋 Overview

This workflow consists of three main components:
1. **Stock Indexer** (`stock_indexer.py`) - Processes raw stock data and adds time-based indexing
2. **Stock Dashboard** (`stock_dashboard.py`) - Interactive Streamlit dashboard for data visualization
3. **Workflow Runner** (`run_stock_analysis.py`) - Automated script to run both components sequentially

## 🚀 Quick Start

### Environment Setup (Conda Users)
If you're using conda and have the `stockinsights` environment set up:
```bash
source setup_env.sh
```

### Option 1: Automated Workflow (Recommended)
```bash
python run_stock_analysis.py
```
This will automatically:
1. Run the stock indexer to process your data
2. Launch the interactive dashboard in your browser

### Option 2: Manual Step-by-Step
```bash
# Step 1: Process the raw data
python stock_indexer.py

# Step 2: Launch the dashboard
streamlit run stock_dashboard.py
```

## 📦 Requirements

### Required Files
- `StockData.xlsx` - Your raw quarterly stock data file
- `stock_indexer.py` - Data processing script
- `stock_dashboard.py` - Dashboard application
- `stock_classifications.py` - Stock sector/industry classifications (optional)
- `run_stock_analysis.py` - Workflow automation script

### Python Dependencies
Install the required packages:
```bash
pip install pandas streamlit plotly numpy pydantic
```

Or install from a requirements file:
```bash
pip install -r requirements.txt
```

#### Development Dependencies (Optional)
For code formatting and quality tools:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

**requirements.txt:**
```
pandas>=1.5.0
streamlit>=1.28.0
plotly>=5.15.0
numpy>=1.24.0
pydantic>=2.0.0
openpyxl>=3.1.0
```

## 📊 Input Data Format

Your `StockData.xlsx` file should contain the following columns:
- **Ticker** - Stock symbol (e.g., AAPL, GOOGL)
- **Report** - Reporting period/date
- **EPS** - Earnings Per Share
- **Revenue** - Quarterly revenue (in millions)
- **Price** - Stock price
- **DivAmt** - Dividend amount (optional)

### Example Data Structure:
| Ticker | Report    | EPS  | Revenue | Price | DivAmt |
|--------|-----------|------|---------|-------|--------|
| AAPL   | 2023-Q1   | 2.18 | 117154  | 157.96| 0.24   |
| AAPL   | 2023-Q2   | 1.26 | 81797   | 193.97| 0.24   |
| GOOGL  | 2023-Q1   | 1.17 | 69787   | 108.22| 0.00   |

## 🔧 Component Details

### 1. Stock Indexer (`stock_indexer.py`)

**Purpose:** Processes raw stock data and adds sequential indexing for time-series analysis.

**What it does:**
- Reads `StockData.xlsx`
- Adds an `Index` column for each ticker (1 = oldest, N = newest)
- Validates data integrity
- Outputs `StockData_Indexed.xlsx`

**Key Features:**
- Independent indexing per ticker
- Data integrity verification
- Handles missing values gracefully
- Removes unnamed/empty columns

**Output:** `StockData_Indexed.xlsx` with added Index column

### 2. Stock Dashboard (`stock_dashboard.py`)

**Purpose:** Interactive web-based dashboard for stock data analysis and visualization.

**Features:**
- **Individual Analysis Tab:**
  - Absolute value charts (EPS, Revenue, Price, etc.)
  - Quarter-over-Quarter (QoQ) percentage change analysis
  - Trailing Twelve Months (TTM) calculations
  - P/E multiple analysis
  - Rolling average trend lines

- **Multi-Ticker Comparison Tab:**
  - Side-by-side metric comparisons
  - Customizable chart selection
  - Time-aligned analysis using Index

- **Rolling Averages Summary Tab:**
  - Comprehensive QoQ growth summaries
  - Sector-based filtering (if classifications available)
  - 4Q, 8Q, and 12Q rolling averages
  - Downloadable CSV reports

**Access:** Opens automatically in your default web browser at `http://localhost:8501`

### 3. Workflow Runner (`run_stock_analysis.py`)

**Purpose:** Automates the entire workflow with dependency checking and error handling.

**Features:**
- Dependency validation
- File existence checks
- Automated sequential execution
- User prompts for dashboard launch
- Comprehensive error reporting

## 📈 Dashboard Usage

### Navigation
The dashboard has three main tabs:

1. **📋 Individual Analysis**
   - Select any ticker from the sidebar
   - View absolute values and QoQ changes
   - Analyze TTM metrics and P/E multiples

2. **📊 Multi-Ticker Comparison**
   - Select multiple tickers for comparison
   - Choose metrics to compare
   - View time-aligned charts

3. **📈 Rolling Averages Summary**
   - View comprehensive growth summaries
   - Filter by sector (if classifications available)
   - Download summary data

### Key Metrics Explained

- **QoQ (Quarter-over-Quarter):** Percentage change from previous quarter
- **TTM (Trailing Twelve Months):** Sum of last 4 quarters
- **P/E Multiple:** Price divided by EPS TTM
- **Rolling Averages:** 4Q, 8Q, and 12Q moving averages of QoQ changes

## 🛠️ Troubleshooting

### Common Issues

**1. "File not found" errors:**
```
ERROR: StockData.xlsx not found in current directory!
```
**Solution:** Ensure `StockData.xlsx` is in the same folder as the scripts.

**2. Missing dependencies:**
```
ERROR: Missing required packages: streamlit, plotly
```
**Solution:** Install missing packages with `pip install streamlit plotly`

**3. Data format issues:**
```
ERROR: Core data verification failed!
```
**Solution:** Check that your Excel file has the required columns (Ticker, Report, EPS, Revenue, Price).

**4. Port already in use:**
```
Error: Port 8501 is already in use
```
**Solution:**
- Close other Streamlit instances
- Or specify a different port: `streamlit run stock_dashboard.py --server.port 8502`

### Validation Checks

The indexer performs several validation checks:
- ✅ Data integrity verification
- ✅ Index sequence validation
- ✅ No duplicate indices per ticker
- ✅ Proper time alignment

## 📁 File Structure

```
your_project_folder/
├── StockData.xlsx                 # Input data file
├── StockData_Indexed.xlsx         # Generated by indexer
├── stock_indexer.py              # Data processing script
├── stock_dashboard.py            # Dashboard application
├── stock_classifications.py      # Sector classifications (optional)
├── run_stock_analysis.py         # Workflow automation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎯 Best Practices

1. **Data Preparation:**
   - Ensure consistent ticker symbols
   - Use consistent date formats
   - Clean data before processing

2. **Performance:**
   - For large datasets (>10,000 rows), consider data filtering
   - Close unused browser tabs when running dashboard

3. **Analysis:**
   - Use TTM metrics for annual comparisons
   - Focus on rolling averages for trend analysis
   - Compare similar sector companies

## 💡 Tips

- **First-time users:** Start with the automated workflow (`run_stock_analysis.py`)
- **Power users:** Run components individually for more control
- **Data updates:** Re-run the indexer whenever you update `StockData.xlsx`
- **Sharing:** Export data from the dashboard's download buttons
- **Customization:** Modify sector classifications in `stock_classifications.py`

## 🔄 Updating Data

When you have new quarterly data:

1. Update `StockData.xlsx` with new records
2. Re-run the indexer: `python stock_indexer.py`
3. Refresh the dashboard (it will automatically load the new data)

## 📞 Support

If you encounter issues:

1. Check that all files are in the same directory
2. Verify your Python environment has all dependencies
3. Ensure your data follows the expected format
4. Check the console output for specific error messages

## 🚀 Getting Started Checklist

- [ ] Install Python dependencies (`pip install pandas streamlit plotly numpy pydantic openpyxl`)
- [ ] Place `StockData.xlsx` in the project folder
- [ ] Ensure all Python scripts are in the same folder
- [ ] Run `python run_stock_analysis.py`
- [ ] Access dashboard at `http://localhost:8501`
- [ ] Explore the three analysis tabs

---

**Happy analyzing! 📈✨**
