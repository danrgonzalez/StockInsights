# StockInsights

Quarterly fundamentals for 141 US-listed companies, hand-maintained in one Excel
workbook, turned into derived metrics, peer comparisons and next-quarter EPS
forecasts by a pure-Python `core/` library. One command regenerates every
output to disk; a Streamlit dashboard reads the same numbers interactively.

This file is the whole documentation. Facts below were checked against the
code and data on **2026-09-06**; re-check the counts after any data change.

| | |
|---|---|
| Quarterly rows | 8,667 across 141 tickers, Q1'10 to Q2'27 |
| Active universe | 137 tickers, 8,454 rows (4 acquired companies held out) |
| Derived columns | 56 in the exported panel |
| Earnings dates from SEC filings | 8,617 of 8,667 (99.4%); the other 50 are modelled |
| Forecasts | 137 of 137 tickers |
| Full pipeline run | about 15 seconds |
| Tests | 74, `pytest tests` |

---

## Quick start

```bash
source setup_env.sh              # activate the stockinsights conda env - every session
python scripts/run_analysis.py   # regenerate every output, then launch the dashboard
```

Flags:

```bash
python scripts/run_analysis.py --no-dashboard          # outputs only
python scripts/run_analysis.py --skip-backtest         # skip the slowest step (~4 s)
python scripts/run_analysis.py --fetch-earnings-dates  # refresh the SEC registry first (network)
streamlit run dashboard/app.py                         # dashboard only, data already indexed
```

**Before running:** close `data/StockData.xlsx` in Excel. An open workbook
leaves a `~$StockData.xlsx` lock file and Excel overwrites pipeline edits when it
saves. Check with `ls data/ | grep '~\$'`.

**The environment is not optional.** The dashboard needs `streamlit>=1.49`
(it passes `width="stretch"`); the machine's system Python has 1.45.1 and the
app dies mid-render with `TypeError: 'str' object cannot be interpreted as an
integer`. The server still answers HTTP 200, so the failure is only visible in
the browser. See [Troubleshooting](#troubleshooting).

First-time setup:

```bash
conda create -n stockinsights -c conda-forge python=3.11
conda activate stockinsights
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

On Apple silicon use a native arm64 conda; `python -c "import platform;
print(platform.machine())"` should print `arm64`.

---

## What one run does

`scripts/run_analysis.py` is the single entry point. It runs these steps in
order, saves each step's console output to `data/exports/logs/<step>.log`, and
writes a manifest. A failed step is reported in the summary and the dashboard is
not launched.

| Step | Runs | Writes |
|---|---|---|
| `index` | `scripts/indexer.py` | `data/StockData_Indexed.xlsx` |
| `price_overrides` | raw vs indexed comparison | `data/exports/price_overrides.csv` |
| `load_stats` | the loader | universe counts and exclusions, into the manifest |
| `export` | `scripts/export_dashboard_data.py` | `data/exports/stock_analysis_export.json`, `quarterly_metrics.csv`, `ticker_snapshot.csv`, `sector_summary.csv`, `README.md` |
| `acquisitions` | `scripts/acquisition_profile.py` | `data/exports/acquisition_profiles.csv` |
| `earnings_dates_fit` | `scripts/fit_earnings_dates.py` | report only: how accurate the modelled dates are |
| `backtest` | `core.backtesting` | `data/exports/backtest_results.csv`, `backtest_detail.csv`, and `config/ticker_strategy_mapping.json` |
| manifest | | `data/exports/manifest.json`: git commit, input SHA-256s, quotes file used, universe counts, per-step status and timing |

Nothing under `data/exports/` is hand-edited. Delete the directory and re-run to
get it all back. The backtest step rewrites the tracked
`config/ticker_strategy_mapping.json` whenever the data has changed, so expect
it in `git status` after a run; predictions do not read it (see
[Forecasts](#forecasts)).

### Flow

```
data/StockData.xlsx ──┐
                      ├──► scripts/indexer.py ──► data/StockData_Indexed.xlsx
data/quotes/*.csv ────┘                                      │
                                                             ▼
                                     core/ computes the derived columns
                                     (TTM, QoQ, Multiple, PEG, dividend
                                      yields, sector ranks, peer benchmarks,
                                      downside capture, EPS forecasts)
                                                             │
               ┌─────────────────┬───────────────────────────┼──────────────────┐
               ▼                 ▼                           ▼                  ▼
        dashboard/app.py   data/exports/        acquisition_profiles.csv   backtest_*.csv
        (recomputes on     (persisted by
         every load)        run_analysis.py)
```

---

## Repository layout

```
StockInsights/
├── core/                       # pure Python, no UI dependencies
│   ├── data_processing.py      # loader, calculate_qoq_changes, rankings, outperformance, downside capture
│   ├── predictions.py          # predict_next_eps: EPS -> TTM -> price scenarios
│   ├── strategies.py           # the five growth strategies
│   ├── backtesting.py          # per-ticker strategy backtest
│   ├── acquisitions.py         # pre-acquisition profiles
│   ├── ticker_status.py        # active / acquired / held-out, from config
│   ├── classifications.py      # sector / industry / sub-industry per ticker, hardcoded
│   └── enums.py                # every column name, threshold, window and path
├── dashboard/                  # Streamlit UI: app.py, charts.py, ui_components.py, data_utils.py
├── scripts/
│   ├── run_analysis.py         # ONE-STOP PIPELINE, see above
│   ├── indexer.py              # Stage 1: Index column + price override
│   ├── export_dashboard_data.py
│   ├── acquisition_profile.py
│   ├── fetch_earnings_dates.py # SEC EDGAR -> config/earnings_dates.json
│   └── fit_earnings_dates.py   # model the dates EDGAR cannot supply
├── config/
│   ├── ticker_status.json      # hand-edited: lifecycle and holds
│   ├── earnings_dates.json     # generated: 8,617 SEC-sourced dates (~250 KB)
│   └── ticker_strategy_mapping.json  # generated by the backtest; research artifact
├── data/                       # UNTRACKED, see Data
├── tests/test_data_processing.py
├── setup_env.sh                # locates conda, activates stockinsights
├── requirements.txt            # pinned: streamlit 1.58.0, pandas 2.2.3, numpy 2.0.2, plotly 6.1.2, pydantic 2.11.7, openpyxl 3.1.5
└── requirements-dev.txt        # pre-commit, black, isort, flake8, autoflake, pytest
```

---

## Data

### `data/` is not in git

`.gitignore` excludes `*.xlsx` and `*.csv`. A fresh clone has no `data/` and
nothing runs until you supply `data/StockData.xlsx`. Everything else in `data/`
is regenerable from it. **That one workbook is irreplaceable and not
version-controlled: keep your own backup**, especially before editing it, since
label fixes and backfills live only in your copy.

```
data/
├── StockData.xlsx           # you supply this
├── StockData_Indexed.xlsx   # generated
├── quotes/                  # thinkorswim exports, YYYY-MM-DD-Quote.csv
└── exports/                 # generated
```

### `data/StockData.xlsx`, the raw input

One row per ticker per fiscal quarter, rows contiguous and chronological per
ticker. Columns:

| Column | Notes |
|---|---|
| `Earnings Report Date` | renamed to `EarningsDate` by the indexer |
| `Ticker` | `BRK/B` here, normalised to `BRK.B` at load; `BK` aliases to `BNY` |
| `Report` | fiscal quarter label, `Q3'26` |
| `EPS` | USD per share |
| `Revenue` | USD millions |
| `Price` | see [the price convention](#one-price-column-two-meanings) |
| `DivAmt` | USD per share, blank when none |
| unnamed 8th column | one cell only, EA's buyout note; dropped |

**Label rule.** For each ticker, anchor on the latest row's label and decrement
one quarter per row backwards. Run this as a validator after any data entry; it
catches year-rollover slips that produce no duplicate. It holds for all 141
tickers. When a ticker fails it, first establish whether the labels are wrong or
rows are missing: reconciling 4-quarter revenue sums against reported annual
revenue is the sharpest test. BRK/B failed the rule because 10 real quarters were
missing, not because the labels were wrong.

### `data/quotes/*.csv`, a live input

Seven dated thinkorswim exports; the newest by filename (`2026-09-04`) is used.
The indexer reads its `SimpleMovingAvg` column and **overwrites each ticker's
latest `Price`**. Re-running with a newer file changes the latest row's price and
every ratio built on it. `data/exports/price_overrides.csv` records exactly
which tickers were overridden and by how much; on 2026-09-04 that was 139 of 141.
The two kept their workbook price: `JWN` (acquired, not in the file) and `BNY`,
whose row is listed under the old symbol `BK` with `loading` in the moving-average
cell, so it was skipped. A fresh export with the cell populated fixes BNY.

The quotes files also carry sector columns; the pipeline ignores them and uses
`core/classifications.py`.

### `config/`

| File | Role |
|---|---|
| `ticker_status.json` | hand-edited. `status` is a fact about the company (`active` / `acquired`); `exclude_from_active` is a decision about whether it counts toward benchmarks. A data-quality hold is still an active company. |
| `earnings_dates.json` | generated by `fetch_earnings_dates.py`. Each entry is `date|source`, sources `8K`, `10Q`, `10K`, `6K~`. |
| `ticker_strategy_mapping.json` | generated by the backtest. Predictions do not read it. |

### Outputs

`data/StockData_Indexed.xlsx` is the working file everything downstream reads:
the same 8,667 rows plus `Index` and the renamed date column. Never hand-edit it.

`data/exports/`:

| File | Contents |
|---|---|
| `stock_analysis_export.json` | Start here. ~1.4 MB, self-describing: metadata, a full data dictionary, a per-ticker snapshot (latest values with staleness, QoQ summary, 4Q/8Q/12Q rolling averages, peer ranks in sector and market, forecast, data-quality warnings), sector aggregates. `--include-history` embeds every row (~17 MB). |
| `quarterly_metrics.csv` | the full panel, 8,454 rows x 56 columns |
| `ticker_snapshot.csv` | the JSON snapshot flattened, one row per ticker |
| `sector_summary.csv` | per-sector averages and medians |
| `price_overrides.csv` | per ticker: workbook price, indexed price, overridden, change % |
| `acquisition_profiles.csv` | one row per acquired ticker: deal terms and run-up |
| `backtest_results.csv` | every strategy scored on every ticker, `IsBest` marks the winner |
| `backtest_detail.csv` | the per-period predictions behind those scores |
| `manifest.json` | provenance of the run |
| `logs/` | console output of each step |
| `README.md` | generated guide to the bundle |

Units: USD; Revenue in millions; ratios and `*_QoQ` in percent. Missing values
are `null` in JSON and empty in CSV.

### Updating data

1. Close Excel. Back up `data/StockData.xlsx`. Add the new rows.
2. Drop a new `data/quotes/YYYY-MM-DD-Quote.csv` if you want current prices.
3. `python scripts/run_analysis.py`.
4. Optionally `--fetch-earnings-dates` to pull the new quarters' dates from EDGAR.

---

## Calculations

All in `core/data_processing.py`, applied per ticker in `Index` order. Column
names, thresholds and windows are constants in `core/enums.py`.

### The Index column

`scripts/indexer.py` numbers each ticker's rows so that every ticker's latest
row lands on the same maximum (`Index = cumcount + max_records - records + 1`).
AAPL with 40 quarters gets 1..40; a ticker with 32 gets 9..40. All cross-ticker
comparison and the multi-ticker charts align on it. The indexer also verifies
the core columns were not altered (except the price override) and refuses to
write if any ticker has duplicate or misaligned indices.

### Derived columns

| Column | Definition |
|---|---|
| `EPS_TTM`, `Revenue_TTM` | rolling 4-quarter sum, needs all 4 |
| `Multiple` | `Price / EPS_TTM`; **NaN when `EPS_TTM <= 0`** |
| `DivYield`, `DivYieldAnnual` | `DivAmt / Price * 100`, and `* 4` |
| `PayoutRatio` | `DivAmt * 4 / EPS_TTM * 100`; NaN when `EPS_TTM <= 0` |
| `<metric>_QoQ` | percent change vs the ticker's previous row, for EPS, Revenue, Price, EPS_TTM, Revenue_TTM, Multiple, DivAmt, DivYield, PayoutRatio; `inf` off a zero base is set to NaN |
| `EPSMomentum` | 4Q mean of `EPS_QoQ` minus 8Q mean, percentage points |
| `PriceVolatility` | rolling 8Q std of `Price_QoQ` (min 4) |
| `RevenueConsistency` | `100 / (1 + std8Q(Revenue_QoQ) / 10)`, bounded (0, 100]; 100 is a perfectly steady line, 50 means growth typically swings 10 pp |
| `PEGRatio` | `Multiple / annualised 4Q mean EPS growth`; **NaN when growth is not positive** |
| `PEGYRatio` | `PEGRatio / DivYieldAnnual` where the yield is > 0 |
| `DivGrowthRate` | annualised CAGR between the first and last detected dividend change; NaN with fewer than two changes |
| `DivIncreaseFreq`, `AvgDivIncrease` | increases per year, mean size of an increase |
| `DownsideCapture` | ticker's mean `Price_QoQ` in quarters when the market mean was negative, as % of the market's mean; needs 3 such quarters |
| `Status` | `active` / `acquired` from `ticker_status.json` |
| `Sector`, `Industry`, `Sub_Industry` | from `core/classifications.py`, 10 sectors, every ticker classified |

Why the masks: a P/E of -50 is not cheaper than 12, and with `.abs()` on growth
a shrinking company looked cheap. Before the masks `Multiple` reached -33,282 and
a money-losing company sorted to rank 1 wherever lower is better.

### Peer comparisons

Peer comparisons are between companies, so they use **one row per ticker, its
latest quarter** (`latest_row_per_ticker`), never every historical row. Ranking
rows instead let a 60-quarter ticker outvote a 20-quarter one and produced
"ranks" of 2,115; the largest sector has 36 tickers and that is now the maximum.

| Column | Meaning |
|---|---|
| `<metric>_SectorRank` | rank among sector peers, 1 = best; higher is better for EPS_TTM, Revenue_TTM, DivYield, DivYieldAnnual, RevenueConsistency, EPSMomentum; lower is better for Multiple, PriceVolatility, PEGRatio, PEGYRatio. Repeated on every row of the ticker. |
| `<metric>_MarketOutperf`, `_SectorOutperf` | level metrics (EPS_TTM, Revenue_TTM): value as % of the mean across tickers' latest quarters, 100 = average |
| `<metric>_MarketGapPP`, `_SectorGapPP` | percent metrics (Price_QoQ, EPS_QoQ, Revenue_QoQ): difference in percentage points, because the mean of a percent series sits near zero and a ratio explodes |

The export's `peer_comparison` block adds sector and market rank, peer counts,
percentiles and the gap to the average for 15 metrics, withholding stale values
and non-positive valuations.

### Latest values and staleness

"Latest" is read as the last non-null value, which reaches back through blank
quarters: DAL's dividend card once showed the dividend it last paid in Q4'19.
`core.latest_with_age` returns the value, its quarter and how many quarters
stale it is. The dashboard cards flag stale values with a warning icon, the
cross-ticker table shows a value only when it belongs to the latest quarter, and
the export lists them in `latest_stale_quarters` and keeps them out of peer ranks.

### Forecasts

`core.predict_next_eps` forecasts next-quarter EPS, then TTM (last 3 actual
quarters plus the forecast) and price (predicted TTM at the current multiple,
held constant), each as base, best and worst case (base plus or minus one
volatility band, floor 5%). Growth observations outside plus or minus 200% are
dropped as outliers; predicted growth is floored at -50%, worst case at -75%,
best case capped at +200%. Confidence is High / Medium / Low from how many clean
observations were available.

**Every ticker uses one global strategy, `weighted_growth`** (70% of the 4Q mean
QoQ growth plus 30% of the 8Q mean). The other four (`simple_average`,
`momentum`, `trend_analysis`, `seasonal`) are tried in that order only when the
chosen one cannot fit; today that is `BA` alone, which falls back to momentum.
17 tickers have EPS that crosses zero in the last 12 quarters; they still get a
forecast, but it is flagged (`eps_crosses_zero`, a warning in the dashboard, a
caution in the methodology string) because percent growth across a sign change
is not meaningful.

Why not a strategy per ticker: it was tested out of sample on 2026-09-05, the
strategy chosen on one 8-quarter window and scored on the next.

| Window | Per-ticker | Always `weighted_growth` | Head to head |
|---|---|---|---|
| latest 8 quarters | 53.27 | **50.59** | 22 better, 54 worse (p = 0.0002) |
| quarters -16..-9 | 71.13 | **67.82** | 21 better, 54 worse (p = 0.0001) |

Lower is better. The selection-window winner repeated out of sample only 39% of
the time (chance 20%), the median margin between best and second-best was 2.0%,
and even a perfect-foresight oracle beat the global default by only 11%. The
mapping file's apparent edge was in-sample. `StrategyPolicy` in `core/enums.py`
holds this; `USE_TICKER_MAPPING = True` restores the old behaviour. Switching
moved AAPL's forecast from $2.92 (seasonal) to $2.22.

Forecasts extrapolate a ticker's own EPS history at a constant P/E. They know
nothing about guidance, macro conditions or news, and are not investment advice.

### Backtest

`core.backtesting.run_multi_ticker_backtest` scores each strategy on each ticker
over the last 8 quarters (needs 12 EPS points). Score = 0.6 x mean absolute
percent error + 0.4 x mean growth error, lower is better. All scores and the
per-period predictions are persisted by the pipeline; the winner per ticker goes
to `config/ticker_strategy_mapping.json`.

---

## Ticker universe

141 tickers are loaded; 137 are analysed. `load_stock_data(path)` returns the
active universe; `load_stock_data(path, include_excluded=True)` returns all 141.

| Ticker | Status | Held out | Why |
|---|---|---|---|
| `S` | acquired | yes | Sprint, merged into T-Mobile; announced 2018-04-29, completed 2020-04-01 |
| `JWN` | acquired | yes | Nordstrom, taken private by the family and El Puerto de Liverpool at $24.25; announced 2024-12-23, completed 2025-05-20 |
| `SKX` | acquired | yes | Skechers, 3G Capital at $63, $9.42B; announced 2025-05-05, completed 2025-09-12 |
| `EA` | acquired | yes | Electronic Arts, PIF / Silver Lake / Affinity at $210, $55B; announced 2025-09-29, completed 2026-08-04 |
| `BRK/B` | active | no | 10 missing quarters (Q2'17-Q3'19) backfilled 2026-09-06 from SEC XBRL |
| `BNY` | active | no | renamed from `BK` 2026-05-21; missing Q2'26 row added 2026-09-06 |
| `BABA` | active | no | 55-day gap between 2026-03-19 and 2026-05-13 verified real |
| `HAIN` | active | no | watch: sub-$1 Nasdaq bid-price warning, reverse split planned |

Acquired tickers are excluded, not deleted: a company frozen in 2019 has no
business in a 2026 sector benchmark, but its final quarters are what
pre-acquisition profiling reads. To bring a ticker back, set
`exclude_from_active` to `false`; no re-index needed.

### Pre-acquisition profiles

`core/acquisitions.py` aligns each acquired ticker's quarters on its
**announcement** date: quarter 0 is the last report before the market knew,
positive offsets are reports filed while the deal was pending.

```bash
python scripts/acquisition_profile.py                   # summary, run-up, P/E aligned
python scripts/acquisition_profile.py --metric EPS_TTM
python scripts/acquisition_profile.py --ticker EA
```

P/E multiple by quarters to announcement: EA re-rated up into its buyout (18.2 at
-8, 27.7 at 0, 36.2 at +1), Nordstrom was cheap throughout (8.6 to 11.9),
Skechers was compressing (19.7 to 14.9). Sprint's near-zero earnings make its
multiple meaningless. `deal_price_vs_last_report_pct` is deliberately not called
a premium: the last report can be a quarter stale, so Skechers computes to +0.5%
against an announced move of about 24%.

---

## Things that will bite

### One price column, two meanings

Historical rows hold the **mean daily close from that report date to the next**,
a forward-looking quarterly average, verified at 0.35% mean error across 84
quarters and 6 tickers. Each ticker's **latest** row is whatever
`SimpleMovingAvg` the newest quotes file carried. So the most recent quarter's
`Multiple`, `PEGRatio` and `DivYield` rest on a different price basis than every
row before it. Checking this column against a report-date closing price will
suggest it is wrong; it is not, and it was mis-diagnosed that way once already.

### Earnings dates are real, with a documented remainder

`scripts/fetch_earnings_dates.py` pulls dates from SEC EDGAR, preferring the 8-K
with item 2.02 (the earnings release itself) and falling back to the 10-Q/10-K
filing date. Sources today: 6,690 from 8-Ks, 1,592 from 10-Qs, 226 from 10-Ks,
109 proximity-matched 6-Ks for the foreign filers BABA and BIDU (weaker). Three
traps it handles: a retired ticker can be reissued (`S` now resolves to
SentinelOne, so Sprint's CIK is pinned), a reorganisation splits filings across
CIKs (XOM, GOOGL, DIS, AVGO, BLK, MDT merge a predecessor), and 6-Ks carry no
item codes.

The 50 modelled dates are quarters that cannot exist on EDGAR: pre-IPO for META,
WDAY, GPRO and BABA, and either side of the ETN and AVGO reorganisations.
`fit_earnings_dates.py` fits a per-ticker line with per-fiscal-quarter offsets
and reports leave-one-out accuracy every run: median 2.0 days, p90 5.6, 99.2%
within 14 days.

### The dashboard recomputes; the exports persist

Every dashboard load recomputes all derived columns from the indexed workbook.
The exports only reflect the last pipeline run. Before this was made one
command, the export went a day stale while five commits changed the data and
the forecasting policy, and nothing detected it. `manifest.json` now records the
inputs' hashes and the git commit, so staleness is checkable.

### Report labels and Excel

Labels sort wrongly as strings (`Q1'26` below `Q4'10`); use `report_sort_key`.
Seven label typos and a 36-row VFC fiscal-convention relabel were fixed in the
workbook on 2026-09-02 and exist only in the user's copy.

---

## Dashboard

`streamlit run dashboard/app.py`, four tabs over the active universe. Reads
only; writes nothing. Data flow: `load_data` -> `calculate_qoq_changes` ->
`attach_classifications` -> rankings -> outperformance -> downside capture, the
same functions the exporter runs.

- **Individual Analysis**: six summary cards with stale-value flags; last 20
  quarters of each metric paired with its QoQ chart; QoQ summary table; EPS,
  TTM and price forecast scenarios with methodology; raw data.
- **Multi-Ticker Comparison**: up to four charts, any metric or its QoQ, aligned
  on `Index`.
- **Rolling Averages Summary**: 4Q/8Q/12Q rolling QoQ means for every ticker,
  sector filter, sector averages, per-metric tables, CSV download.
- **Methodology**: the formulas, with AAPL as the worked example.

---

## Core library

```python
from core import (load_stock_data, calculate_qoq_changes, calculate_sector_rankings,
                  calculate_outperformance_ratios, calculate_downside_capture,
                  predict_next_eps)
from core.data_processing import attach_classifications
from core.enums import FilePaths

df = load_stock_data(FilePaths.DATA_FILE)          # 137 tickers, 8,454 rows
df = attach_classifications(df)                    # Sector column, needed by the next two
df = calculate_qoq_changes(df)
df = calculate_sector_rankings(df)
df = calculate_outperformance_ratios(df)
df = calculate_downside_capture(df)
prediction = predict_next_eps(df, "AAPL")          # dict, see the export's data dictionary
```

`FilePaths` are absolute, resolved from the repo root, so results do not depend
on the working directory (forecasts once differed 18% between the repo root and
`/tmp`). `load_stock_data_with_stats` additionally reports duplicates dropped
and tickers excluded. To add a calculation: implement it in `core/`, export it
from `core/__init__.py`, add it to the exporter so it is persisted, and document
it here.

---

## Development

```bash
pytest tests                     # 74 tests over calculate_qoq_changes and the loader
pre-commit run --all-files       # black, isort, flake8 (88 cols), autoflake, hygiene hooks
```

The pre-commit hook must run under the `stockinsights` env (python 3.11); black
26 needs 3.10+, and a hook wired to a homebrew 3.9 fails with "Package 'black'
requires a different Python".

Local tooling notes for this machine: the repo is **public** at
`github.com/danrgonzalez/StockInsights`, so confirm before pushing. A 2015 git
2.6.4 that shadowed `/usr/bin/git` was disabled on 2026-09-05; the empty
`helper =` line in `~/.gitconfig`'s credential block is the correct reset idiom
and must not be "fixed". `gh auth setup-git` is configured and an ed25519 key is
registered with GitHub, so plain `git push origin main` works.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `TypeError: 'str' object cannot be interpreted as an integer` in the browser | wrong Python; `source setup_env.sh`. Verify with `python -c "import sys, streamlit; print(sys.executable, streamlit.__version__)"`: expect the conda path and 1.58.0 |
| Workbook edits vanish or get clobbered | Excel has it open; look for `~$StockData.xlsx` |
| `Port 8501 is already in use` | `streamlit run dashboard/app.py --server.port 8502`; check the occupant with `lsof -nP -iTCP:8501 -sTCP:LISTEN` |
| A ticker is missing | held out in `config/ticker_status.json`; the dashboard names what it dropped at load |
| `data/StockData.xlsx not found` | supply the workbook; every script resolves paths from the repo root |
| A run's numbers differ from the dashboard | the export is from an older run; re-run `scripts/run_analysis.py` and compare `manifest.json` |
| Backtest changed `config/ticker_strategy_mapping.json` | expected whenever data changed; commit or discard, forecasts are unaffected |

---

## Known open questions

- BRK/B's Q2'16 (2.03) and Q3'16 (2.92) EPS look high against Berkshire's
  reported operating earnings, which imply about 1.87 and 1.97.
- BNY's latest price is the workbook value, not a Sep 4 moving average, because
  the quotes row was still `loading`.
- `RevenueConsistency`'s scale of 10 pp is a judgement call that sets how harshly
  volatility is punished. Face validity looks right (NFLX 89, WDAY 87 and CL 85 at the
  top; DECK, FSLR and M at 20 at the bottom) but it deserves a second opinion.
- The per-strategy backtest scores are now persisted but nothing consumes them
  yet.

## History

- **2026-09-02** Full-repo review found ~18 defects. Fixed 7 `Report` typos and
  relabelled 36 VFC rows; established the label rule; content-based earnings-date
  column detection; exclusion list; dedupe keeps the newest row.
- **2026-09-05** Absolute `FilePaths`; valuation masks on non-positive earnings;
  PEG sign fix; bounded `RevenueConsistency`; QoQ `inf` -> NaN; single loader;
  tests added; sector feature made live (it had added 0 columns because `Sector`
  was never joined) and peer maths moved to one row per ticker; stale-value
  flags; per-ticker strategy selection retired for one global default; fallback
  chain and zero-crossing flag; headless exporter; git tooling repaired.
- **2026-09-06** Acquired companies separated from the active universe with
  profiling; `BK` -> `BNY` with alias and the missing Q2'26 row; BRK/B's 10
  missing quarters backfilled from XBRL (revenue reconciles exactly to 2017-2019
  annual totals); earnings dates sourced from SEC EDGAR (99.4% real, LOO median
  2.0 days); `run_analysis.py` made the one-stop pipeline; documentation
  consolidated into this file.
