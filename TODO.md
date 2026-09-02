# StockInsights — Open Issues

Tracked backlog from the full-repo review of 2026-09-02. Line numbers are as of that
date; verify before acting.

**Priority key:** P1 produces wrong numbers under a confident label · P2 correctness
bug with visible impact · P3 quality/maintainability · P4 data sourcing.

---

## P1 — Silently wrong results

### 1. `predict_next_eps` returns different numbers depending on working directory
`FilePaths.STRATEGY_MAPPING_FILE` is the relative path `config/ticker_strategy_mapping.json`.
`load_ticker_strategy_mapping` swallows `FileNotFoundError` and returns `None`
(`core/backtesting.py:512`), so `get_ticker_strategy` falls back to `weighted_growth`
while `core/predictions.py:180` still stamps the result "backtested optimal".

```
from repo root:  $2.58  "Ticker-specific Seasonal (backtested optimal)"
from /tmp:       $2.19  "Ticker-specific Weighted Growth (backtested optimal)"
```

An 18% different AAPL forecast under an identical label. Matters because the readme
advertises `core/` for notebooks and services, where CWD is not the repo root.

- [ ] Resolve `FilePaths` entries against `Path(__file__).parents[1]`, as
      `core/exclusions.py` already does.
- [ ] Make the fallback visible in the methodology string instead of claiming
      "backtested optimal".

### 2. Negative earnings invert the sector rankings
318 rows have `EPS_TTM < 0`, making `Multiple` negative (min **-33,282**).
`DerivedMetric.negative_ranking_metrics()` ranks Multiple/PEG/PEGY *ascending*
("lower is better"), so a money-losing company sorts to **rank 1**. `PayoutRatio`
shares the cause (range -8,600 to 5,800).

- [ ] Mask `Multiple`, `PEGRatio`, `PEGYRatio`, `PayoutRatio` to `NaN` where
      `EPS_TTM <= 0` in `core/data_processing.py`.

### 3. `PEGRatio` hides earnings decline
`core/data_processing.py:247` divides by `eps_growth_annual.abs()`, so -30% and +30%
annual EPS growth produce an identical PEG. Shrinking companies look cheap.

- [ ] Drop the `.abs()`; return `NaN` for non-positive growth.

### 4. `RevenueConsistency` is unusable for most rows
`100 - (rolling_std / |rolling_mean|) * 100` explodes when mean `Revenue_QoQ` nears
zero, which is normal. Dataset-wide: median **-318**, p05 **-3,348**, min
**-1,860,112**. It is charted as a real metric and sits in `positive_ranking_metrics()`.

- [ ] Replace with a bounded definition (not a clamp on the current one).

---

## P2 — Correctness bugs

### 5. Backtest ranking prints wrong ranks
`core/backtesting.py:257` — `rank = idx + 1` uses the pre-sort DataFrame index. For
AAPL the five sorted strategies print as ranks `1, 2, 5, 3, 4`.

- [ ] Use `enumerate` over the sorted frame.

### 6. Division by zero on an empty ticker set
`core/backtesting.py:406` — `successful_tests / total * 100`.

- [ ] Guard `total == 0`.

### 7. Indexer assigns the Index column positionally
`scripts/indexer.py:167` — `df.groupby("Ticker", sort=False).apply(...).values` is
correct only because the source keeps each ticker's rows contiguous (141 runs /
141 tickers). Reproduced garbage on a non-contiguous frame (`A -> 1,3,3`, `B -> 2,2`).
Also raises a pandas deprecation warning.

- [ ] Rewrite with `groupby(...).cumcount()`.

### 8. Sector ranking is dead code
`calculate_sector_rankings` returns immediately without a `Sector` column, and the
data has none — confirmed live: it adds **0 columns**. The sector half of
`calculate_outperformance_ratios` never runs either, while `dashboard/app.py:1617`
documents Sector Rankings in detail. `get_stock_classification` already works; the
classifications are simply never joined onto the DataFrame.

- [ ] Join sector/industry onto `df` in `dashboard/app.py:main()` (one line), or
      remove the feature and its Methodology entry.

---

## P3 — Quality and maintainability

### 9. No tests
~7,450 lines, nearly all financial arithmetic, zero test files. Root cause of items
2-4 going unnoticed.

- [ ] Add tests over `calculate_qoq_changes` covering negative EPS_TTM, zero-crossing
      revenue growth, and TTM boundaries — write these *with* the P1 fixes.

### 10. `dashboard.data_utils.load_data` duplicates `core.load_stock_data`
It documents itself as "a Streamlit-specific wrapper" but copies the whole body. Every
load-time fix must currently land in both — the exclusion filter and the
`keep="last"` change both had to be applied twice.

- [ ] Make it delegate to `core.load_stock_data` and keep only caching + `st` messaging.

### 11. `CLASSIFICATIONS_AVAILABLE` is hardcoded `True`
`dashboard/app.py:22`, beside an unguarded import — roughly 7 `else:` branches are unreachable.

- [ ] Remove the flag and the dead branches.

### 12. `predict_next_eps` runs 4x per ticker render
Once in `app.py`, once in each of three prediction charts, each re-reading the JSON
mapping. Measured cost is negligible; the duplication is the problem.

- [ ] Compute once in `app.py` and pass into the chart functions.

### 13. Deprecated Pydantic v1 API
`core/classifications.py:1150` uses `stock_info.dict()` under the pinned pydantic 2.11.7.

- [ ] Switch to `.model_dump()`.

### 14. Pre-commit pins ~2 years behind `requirements-dev.txt`
black 23.7 vs `>=25.1`, flake8 6.0 vs `>=7.3`, isort 5.12 vs `>=6.0` — hooks and a
local run disagree.

- [ ] Align `.pre-commit-config.yaml` with `requirements-dev.txt`.

### 15. Repo hygiene
- [ ] `.claude/settings.local.json` is tracked but is a local file.
- [ ] `data/` is entirely untracked (`.gitignore` excludes `*.xlsx`/`*.csv`), so the
      repo cannot run from a fresh clone **and the spreadsheet fixes made on
      2026-09-02 are not under version control**. Decide on a data strategy.
- [ ] `setup_env.sh` hardcodes `/Users/dgonzalez/miniconda3`.

---

## P4 — Data sourcing

### 16. BRK/B is missing 10 real quarters
`Q1'17` jumps straight to `Q4'19` (Q2'17-Q3'19 absent). The labels are **correct** —
4-quarter revenue sums reconcile to Berkshire's reported annual revenue within
~$1,000M (FY13 exact, FY16 exact). Relabeling to close the gap makes those sums
$29-43B/yr short, so **do not "fix" it by relabeling.**

Currently excluded via `config/excluded_tickers.json`. While unfixed, its TTM and QoQ
bridge a 2.5-year hole and its `Index` alignment is off by 10 quarters.

- [ ] Backfill Q2'17-Q3'19, then set `"exclude": false`.

### 17. Three tickers have no real earnings dates
`JWN`, `S`, `SKX` (149 rows) — all dates are estimates from the modal fiscal
convention, p90 error ~214 days, potentially a full quarter off. All three are also
delisted/acquired and currently excluded.

- [ ] Supply one real earnings date each (pulls them to ~4-day accuracy), or drop
      the tickers permanently.

### 18. BABA has one suspect earnings date
Rows 783-784 are 55 days apart (`2026-03-19` -> `2026-05-13`); every other spacing in
the file is 84-112. Financials are sound, so BABA is **not** excluded.

- [ ] Confirm whether `2026-03-19` should be ~`2026-02-19`.

### 19. EA is being acquired
The sheet carries a `Buyout` note at Q2'26. Data is current; it will stop reporting.

- [ ] Exclude once it stops reporting.

### 20. 8,268 of 8,653 earnings dates are estimates
Generated 2026-09-02 at 91.3125 days/quarter with per-ticker, per-fiscal-quarter
offsets. Leave-one-out accuracy on the 383 real dates: median 4 days, p90 13, 91%
within two weeks. Long-range extrapolation is structurally sound but unverifiable.

- [ ] Treat as approximations, never as reportable facts. Re-run the fit (don't patch)
      whenever real dates are added.

---

## Strategy / method

### 21. The per-ticker strategy mapping is stale and likely fits noise
Re-backtesting the first 25 tickers: the stored choice is no longer the winner for
**10 of 25**, and the median margin between best and second-best is **2.2%** over 8
observations (AAPL: 0.1%). The same 8 quarters both select the strategy and back the
"backtested optimal" claim, so the reported accuracy is in-sample.

- [ ] Decide: hold out the selection window, or require a minimum margin before
      deviating from a single global default. This is a method decision, not a bug fix.

### 22. `INTC` produces no prediction
Its EPS crosses zero repeatedly, so every YoY growth exceeds the ±200% outlier filter
and the seasonal strategy has nothing to fit. Pre-existing behaviour; the app shows an
info message.

- [ ] Decide whether zero-crossing EPS deserves a dedicated strategy or an explicit
      "not predictable" state.

---

## Done — 2026-09-02

- [x] Fixed 7 `Report` label typos (AMZN `!2'26`, F/KO/MRK year rollovers, MMM
      `Q4'26`->`Q4'25`, STZ `Q1'26`->`Q1'27`).
- [x] Relabeled 36 VFC rows for its fiscal-year convention change.
- [x] Established the canonical rule: anchor on each ticker's latest quarter and
      decrement one per row backwards. Holds for 140/141 (BRK/B is the exception).
- [x] Backfilled all 8,268 missing earnings dates; fixed MGM `4/29/2026MGM` and
      DECK `2029-01-26`.
- [x] Added `config/excluded_tickers.json` + `core/exclusions.py`, applied at load in
      both loaders.
- [x] `keep="first"` -> `keep="last"` in both loaders, so a corrected duplicate wins.
- [x] Made `indexer.py` earnings-date detection content-based, so a named
      "Earnings Report Date" column survives as `EarningsDate`.
- [x] Documented the `stockinsights` conda env requirement in readme.md — system
      python has streamlit 1.45.1, the app needs >=1.49, and the server returns
      HTTP 200 even when the app would crash on open.
