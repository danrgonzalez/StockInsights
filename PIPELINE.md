# Pipeline Map

What goes in, what comes out, and which numbers are facts rather than inferences.

Counts below were read from the live workbook on 2026-09-06, not copied from
documentation. Re-check them before trusting them: `readme.md` and `TODO.md`
have both drifted from the data before.

| | |
|---|---|
| Quarterly rows | 8,667 |
| Tickers held | 141 |
| Active universe | 137 |
| Derived columns | ~56, computed in memory |
| Earnings dates from filings | 99.4% (8,617 of 8,667) |

---

## Inputs

### `data/StockData.xlsx` — hand-maintained, irreplaceable

8,667 rows across 141 tickers. Eight columns:

| Column | Notes |
|--------|-------|
| `Earnings Report Date` | renamed to `EarningsDate` by the indexer |
| `Ticker` | `BRK/B` here, normalised to `BRK.B` at load |
| `Report` | fiscal quarter label, e.g. `Q3'26` |
| `EPS`, `Revenue`, `Price`, `DivAmt` | see the price caveat below |
| *(unnamed 8th column)* | one cell only: EA's `Buyout` note |

**Untracked by git.** Everything else in this document can be rebuilt from it;
this file cannot be rebuilt from anything. Keep your own backups — see
[Getting the Data](readme.md#getting-the-data).

### `data/quotes/*.csv` — thinkorswim exports, a *live* input

Seven dated `Quotes.tos` files. `scripts/indexer.py` auto-selects the newest by
filename and reads its `SimpleMovingAvg` column, which **overwrites each
ticker's most recent `Price`**. This is not a static input: re-running the
indexer with a newer quotes file changes the latest row's price, and every
valuation ratio derived from it.

These files also carry `Sector` / `Industry` / `Sub-Industry`, which duplicate
`core/classifications.py`. The pipeline uses the hardcoded Python, not the CSV.

### `config/` — three files, three different roles

| File | Role |
|------|------|
| `ticker_status.json` | lifecycle and exclusions — hand-edited |
| `earnings_dates.json` | 8,617 SEC-sourced dates, ~248 KB — generated |
| `ticker_strategy_mapping.json` | **research artifact; predictions do not read it** |

`ticker_status.json` keeps two independent ideas apart, and the distinction
matters: `status` (`active` / `acquired`) is a fact about the company, while
`exclude_from_active` is a decision about whether it counts toward benchmarks. A
data-quality hold is still an active company.

`ticker_strategy_mapping.json` is still written by the multi-ticker backtest, but
per-ticker strategy selection was retired after losing to a single global default
out-of-sample on two independent windows. See `StrategyPolicy` in `core/enums.py`.

### `data.sec.gov` — network, on demand

Read only when you run `scripts/fetch_earnings_dates.py`. Supplies earnings
dates, and supplied the quarterly revenue for the BRK/B backfill.

---

## The flow

```
data/StockData.xlsx ──┐
                      ├──► scripts/indexer.py ──► data/StockData_Indexed.xlsx
data/quotes/*.csv ────┘                                      │
                                                             ▼
                                     core/ computes ~56 derived columns
                                     (TTM, QoQ, Multiple, PEG, dividend
                                      yields, sector ranks, peer benchmarks,
                                      next-quarter EPS forecasts)
                                                             │
                        ┌────────────────────────────────────┼────────────────────────────────┐
                        ▼                                    ▼                                ▼
                 dashboard/app.py                     data/exports/              scripts/acquisition_profile.py
```

`scripts/indexer.py` assigns each ticker a sequential `Index` (1..N, aligned so
all 141 share the same maximum), renames the date column, and applies the price
override described above.

**The derived columns exist only in memory.** They are recomputed on every
dashboard load and never persisted — they reach disk only through the exporter.

---

## Outputs

### `data/StockData_Indexed.xlsx`

The working file everything downstream reads. Same 8,667 rows, plus `Index` and
the renamed `EarningsDate`. Regenerable — do not hand-edit it; edit the raw
workbook and re-run the indexer.

### `data/exports/` — the headless bundle

| File | Contents |
|------|----------|
| `stock_analysis_export.json` | ~1.4 MB, self-describing, includes a data dictionary |
| `quarterly_metrics.csv` | the full panel, all derived columns |
| `ticker_snapshot.csv` | one row per ticker |
| `sector_summary.csv` | sector aggregates |
| `README.md` | generated, describes the bundle |

### The dashboard

`streamlit run dashboard/app.py` — four tabs over the 137-ticker active
universe. Reads only; writes nothing.

### Acquisition profiles

`scripts/acquisition_profile.py` lines the four acquired tickers up on their
announcement dates, where quarter `0` is the last report before the market knew.

---

## Entry points

| Command | Reads | Writes |
|---------|-------|--------|
| `python scripts/run_analysis.py` | raw workbook + quotes | indexed workbook, then launches the dashboard |
| `python scripts/indexer.py` | raw workbook + quotes | `data/StockData_Indexed.xlsx` |
| `streamlit run dashboard/app.py` | indexed workbook + config | nothing |
| `python scripts/export_dashboard_data.py` | indexed workbook + config | `data/exports/` |
| `python scripts/acquisition_profile.py` | indexed workbook + config | stdout, optional CSV |
| `python scripts/fetch_earnings_dates.py` | **SEC EDGAR** + raw workbook | `config/earnings_dates.json` |
| `python scripts/fit_earnings_dates.py` | registry + workbook | workbook dates, with `--apply` |

All of these need the `stockinsights` conda env — see
[Troubleshooting](readme.md#wrong-python-environment).

---

## Two things that will bite

Neither is visible in the data itself.

### 1. One price column, two meanings

Historical rows hold the **mean daily close from that report to the next** — a
forward-looking quarterly average. Verified at 0.35% mean error across 84
quarters and 6 tickers.

But each ticker's **latest** row is whatever `SimpleMovingAvg` the newest quotes
file carried, written in by the indexer:

| Latest row | In the workbook | After indexing |
|------------|-----------------|----------------|
| `AAPL` | 303.00 | **323.02** |
| `MSFT` | 447.00 | **502.99** |

So the most recent quarter's `Multiple`, `PEGRatio` and `DivYield` rest on a
different price basis than every row before it.

Checking this column against a report-date closing price will suggest it is
wrong. It is not — the window is forward-looking, and it was mis-diagnosed as
"inconsistent" once already for exactly that reason.

### 2. 141 tickers held, 137 analysed

Four acquired companies are loaded but held out of the active universe:

| Ticker | Company | Completed |
|--------|---------|-----------|
| `S` | Sprint, merged into T-Mobile | 2020-04-01 |
| `JWN` | Nordstrom, taken private | 2025-05-20 |
| `SKX` | Skechers, acquired by 3G Capital | 2025-09-12 |
| `EA` | Electronic Arts, taken private | 2026-08-04 |

A company frozen in 2019 has no business in a 2026 sector benchmark. They are
excluded, not deleted — their final quarters are what the pre-acquisition
profiling reads.

```python
from core import load_stock_data
from core.enums import FilePaths

active = load_stock_data(FilePaths.DATA_FILE)                             # 137
everything = load_stock_data(FilePaths.DATA_FILE, include_excluded=True)  # 141
```

---

## Related

- [`readme.md`](readme.md) — setup, data sourcing, metric definitions
- [`TODO.md`](TODO.md) — the backlog, and the record of what each fix actually changed
- `config/ticker_status.json` — per-ticker lifecycle and the reason for each hold
