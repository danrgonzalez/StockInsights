# StockInsights — Open Issues

Tracked backlog from the full-repo review of 2026-09-02. Line numbers are as of that
date; verify before acting.

**Priority key:** P1 produces wrong numbers under a confident label · P2 correctness
bug with visible impact · P3 quality/maintainability · P4 data sourcing.

---

## P3 — Quality and maintainability

### 15. Repo hygiene
- [ ] `.claude/settings.local.json` is tracked but is a local file.
- [ ] `data/` is entirely untracked (`.gitignore` excludes `*.xlsx`/`*.csv`), so the
      repo cannot run from a fresh clone **and the spreadsheet fixes made on
      2026-09-02 are not under version control**. Decide on a data strategy.
- [ ] `setup_env.sh` hardcodes `/Users/dgonzalez/miniconda3`.

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

## Local environment — git tooling

Machine-level issues, not repo issues. They blocked the 2026-09-02 push to origin and
will keep causing trouble until fixed.

### 23. A 2015 git shadows the modern one on PATH  *(root cause of 24)*
`/etc/paths` lists `/usr/local/bin` before `/usr/bin`, so:

| Path | Version |
|------|---------|
| `/usr/local/bin/git` | **2.6.4 (2015)** — wins |
| `/usr/bin/git` | 2.50.1 (Apple Git-155) |
| `/usr/local/git/bin/git` | 2.6.4 — old standalone Git-for-Mac installer |
| `~/opt/homebrew/bin/git` | 2.43.2 |

2.6.4 also lacks `git remote get-url` and `git config --show-origin`, and Streamlit
warns about it at startup ("requires Git 2.7.0 or later").

- [ ] Reorder `/etc/paths` so `/usr/bin` precedes `/usr/local/bin`, or remove the
      stale 2.6.4 installs (`/usr/local/bin/git`, `/usr/local/git/`).

### 24. `credential.helper` appears broken but is not
`~/.gitconfig` contains:

```
[credential]
	helper = osxkeychain
	helper =
	helper = /usr/local/share/gcm-core/git-credential-manager-core
```

The empty middle value is the **correct, documented idiom** for resetting the helper
list (git >= 2.9) so only gcm-core applies. Git 2.6.4 predates that behaviour and tries
to execute a helper named `""`, producing:

```
git: 'credential-' is not a git command
```

So the config is right and the git is wrong. **Do not "fix" the config** — fixing
item 23 makes this disappear. gcm-core itself works (v2.0.632).

- [ ] No action beyond item 23. Re-verify once PATH is fixed.

### 25. SSH key is not registered with GitHub
`~/.ssh/id_rsa.pub` exists (RSA 4096, `SHA256:7WdzGyVzN/jvq5PRPv92+NcpvVEijeTiW41e56hm5ss`)
but `ssh -T git@github.com` returns `Permission denied (publickey)`. Auth currently
works only via `gh` over https.

- [ ] Add the key at https://github.com/settings/keys, or drop the idea and stay on
      https via `gh`.

### 26. `gh` is not registered as git's credential helper
`gh` is authenticated (as `danrgonzalez`), but git cannot reach that token on its own —
the 2026-09-02 push needed an explicit one-off override:

```bash
/usr/bin/git -c credential.helper= -c credential.helper='!gh auth git-credential' \
    push origin main
```

- [ ] Run `gh auth setup-git` for a permanent fix (after item 23), so a plain
      `git push` works.

### 27. `gh` CLI is three years old
Version 2.5.1 (2022-02-15). Its token had silently expired, which is what surfaced all
of the above.

- [ ] Upgrade (`brew upgrade gh`).

---

## Done — 2026-09-05

- [x] **Items 8 + 29** — the sector feature is live and its arithmetic is a peer
      comparison. `core.attach_classifications` joins Sector/Industry/Sub_Industry
      onto the frame (the exporter's private copy deleted in favour of it) and
      `dashboard/app.py:main()` calls it before the sector analytics, so the feature
      adds **23 columns where it previously added 0**.
      `calculate_sector_rankings` and `calculate_outperformance_ratios` now work off
      `latest_row_per_ticker` — one row per company, its latest quarter. Max
      `_SectorRank` fell from **2,115 to 36**, which is exactly the largest sector's
      ticker count; AAPL's `Multiple_SectorRank` is **16 of 35** IT peers, not 957.
      The market benchmark for `Revenue_TTM` moved from the row-weighted **51,927**
      to **79,613** per ticker, the 1.53x gap the backlog predicted, and is no longer
      pulled by long-history tickers. Percent-unit metrics (`Price_QoQ`, `EPS_QoQ`,
      `Revenue_QoQ`) are reported as a percentage-point gap in new `_MarketGapPP` /
      `_SectorGapPP` columns; level metrics keep the ratio form. Non-positive
      valuations are already excluded via item 2. The Methodology expanders for both
      Sector Rankings and Outperformance were rewritten — they had documented the old
      behaviour to the user — and the exporter's data dictionary with them.

- [x] **Item 28** — added `core.latest_with_age(ticker_data, column)`, returning the
      value, its source quarter and how many quarters stale it is. The six summary
      cards now flag a stale value with ⚠️ and explain it in the tooltip (DAL:
      "Dividend amount is from Q4'19, 26 quarters before the latest reported quarter,
      which has no value for it"). The two `Latest_*` columns in the cross-ticker
      comparison table show a value only when it belongs to the latest quarter —
      under a "Latest" header, a 2019 figure compared against everyone else's current
      one is simply wrong. Flagging confirmed for DAL, LUV, EXPE, AAL, INTC, GPRO and
      PVH; AAPL and other current tickers are unflagged.

- [x] **Item 2** — `Multiple`, `PayoutRatio`, `PEGRatio` and `PEGYRatio` are `NaN`
      wherever `EPS_TTM <= 0`, masked at the two sources (`Multiple`, `PayoutRatio`)
      so the two derived ratios inherit it. All 289 negative-earnings rows now carry
      `NaN` on all four. `Multiple` min went from **-33,282 to 1.59**, `PayoutRatio`
      from -8,600 to 0.00 — a money-losing company can no longer sort to rank 1 where
      "lower is better".
- [x] **Item 3** — dropped the `.abs()` in `PEGRatio`; non-positive growth returns
      `NaN`. A shrinking company no longer looks cheap.
- [x] **Item 4** — `RevenueConsistency` is now `100 / (1 + std / SCALE)` over the 8Q
      rolling std of `Revenue_QoQ`, with the near-zero `|mean|` denominator gone
      entirely. Bounded (0, 100] on every row; dataset median moved from **-318 to
      53.3**. `SCALE` (`Thresholds.CONSISTENCY_VOLATILITY_SCALE`) is set to 10
      percentage points — the dataset's median rolling std, which centres the score
      and gives the widest spread. **That constant is a judgement call**: it sets how
      harshly volatility is punished and is worth a second opinion. Face validity
      looks right (NFLX 89, CL 85 at the top; EA 19, DECK 20, M 20 at the bottom).
- [x] **Item 9** — added `tests/test_data_processing.py`, 27 tests over
      `calculate_qoq_changes` covering negative and zero `EPS_TTM`, PEG against a
      declining company, revenue growth crossing zero, TTM window boundaries, the
      dividend and `Report`-ordering fixes. `pytest>=9.1.1` added to
      `requirements-dev.txt`.
- [x] **Item 10** — `core.load_stock_data_with_stats` is now the single loader
      implementation, reporting `duplicates_dropped` / `excluded` / `error`;
      `core.load_stock_data` and `dashboard.data_utils.load_data` are both thin
      wrappers over it. The dashboard keeps its caching and `st` messaging and lost
      the copied body — 110 lines to 71 — and both loaders were verified to return
      identical frames.
- [x] **New, found by the tests** — `*_QoQ` percent changes off a zero base produced
      `inf`, which was never cleaned the way the level ratios were, so it propagated
      into every rolling mean built on the series. Present in the real data: 16
      infinities across `EPS_QoQ`, `DivAmt_QoQ`, `DivYield_QoQ`, `PayoutRatio_QoQ`.
      Now replaced with `NaN`; the panel has none left.

- [x] **Item 1** — `FilePaths` entries are now absolute, resolved from
      `core/enums.py` via `REPO_ROOT`, so `core/` behaves identically from a notebook
      or a service. AAPL predicts $2.92 from both the repo root and `/tmp`; it used to
      differ by 18%. Added `get_ticker_strategy_with_source`, so the methodology string
      only claims "backtested optimal" when the strategy really came from the mapping —
      the two fallbacks now read "(default; backtested strategy mapping unavailable)"
      and "(default; no backtested strategy for X)".
- [x] **Item 11** — removed `CLASSIFICATIONS_AVAILABLE` and its 11 guard sites
      (8 unwrapped, 2 dead `else:` branches deleted, 3 `and` guards simplified);
      app.py is 18 lines shorter.
- [x] **Item 12** — the three prediction charts take an optional pre-computed
      `prediction`; `app.py` computes it once. 4 calls per ticker render -> 1, with the
      three chart figures verified byte-identical. Calling a chart standalone still
      computes its own.
- [x] **Item 30** — the four metrics with no `_QoQ` series (`PEGRatio`, `EPSMomentum`,
      `PriceVolatility`, `RevenueConsistency`) are gone from the rolling-average and
      tab-1 lists, and `Multiple` was added. Verified: 12 permanently blank columns ->
      0, the 27 columns that carried data are unchanged. All four hand-maintained
      metric lists (and the rename map) now derive from `DerivedMetric.qoq_metrics()`,
      which is why they had drifted apart in the first place.
- [x] **Item 31** — all three:
      `report_sort_key`/`report_range` added to `core/data_processing.py` and used by
      the sidebar, which now reads **Q1'10 to Q2'27** instead of Q1'10 to Q4'26 (the
      exporter's private copy of that helper was deleted in favour of the shared one);
      `DivYieldAnnual` dropped from `qoq_metrics()` since its QoQ was identical to
      `DivYield_QoQ` on all 4,813 rows; `DivGrowthRate` returns `NaN` rather than `0.0`
      when growth cannot be measured, which corrects exactly the three tickers the
      backlog named — `AAL`, `EA`, `PVH` — the other 35 unmeasurable tickers were
      already absent from the export.

- [x] **Item 5** — backtest ranking printed the pre-sort DataFrame index as the rank
      (`rank = idx + 1`). Now `enumerate` over the sorted frame. AAPL reproduced the
      documented `1,2,5,3,4` (its sorted frame carries index order `[0,1,4,2,3]`) and
      now prints `1,2,3,4,5`. Scope was wider than recorded: **94 of 137 tickers**
      sort non-monotonically and so printed wrong ranks.
- [x] **Item 6** — guarded `successful_tests / total` in `run_multi_ticker_backtest`;
      an empty ticker set now prints "Success rate: n/a (no tickers tested)".
- [x] **Item 7** — `scripts/indexer.py` builds `Index` from
      `groupby.cumcount() + max_records - transform("size") + 1`. Reproduces the
      existing 8,656-row `Index` column exactly, is correct on a non-contiguous frame
      (old: `1,2,3,2,3`; new: `1,2,2,3,3`), and no longer raises the pandas
      `DataFrameGroupBy.apply` deprecation warning.
- [x] **Item 13** — `stock_info.dict()` -> `.model_dump()` in `core/classifications.py`.
- [x] **Item 14** — `.pre-commit-config.yaml` pinned to the versions actually installed
      in the `stockinsights` env: black 26.5.1, isort 8.0.1, flake8 7.3.0,
      autoflake v2.3.3, pre-commit-hooks v6.0.0. black 26 then reformatted 4 files
      (148 lines, conditional-expression parenthesization only — ASTs verified
      identical before/after). The bump also exposed the reason the hooks had gone
      stale: `.git/hooks/pre-commit` was wired to a homebrew python 3.9, and black 26
      needs >= 3.10, so the first commit attempt died with "Package 'black' requires a
      different Python". Added `default_language_version: python: python3.11` to the
      config and reinstalled the hook from the `stockinsights` env. All 10 hooks now
      pass on `pre-commit run --all-files`.

- [x] Added `scripts/export_dashboard_data.py`: runs the dashboard pipeline headlessly
      and writes `data/exports/` — a self-describing JSON bundle (metadata, data
      dictionary, per-ticker snapshot, sector aggregates) plus three CSVs. Documented
      as readme Option 5.
- [x] The exporter works around items 1, 2 and 8 for its own output only — it `chdir`s
      to the repo root so the strategy mapping always loads, joins the classifications
      onto the frame, excludes non-positive valuations from peer ranks, and flags
      stale values per ticker. **None of these are fixed in `core/`**; the items stay
      open.
- [x] Surfaced items 28-31 while validating the exported numbers against the raw
      workbook.

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
- [x] Pushed 12 commits to origin, taking the public repo from `d84ef38`
      (2025-09-17) to current. Needed a one-off credential override, see item 26.
- [x] Documented the `stockinsights` conda env requirement in readme.md — system
      python has streamlit 1.45.1, the app needs >=1.49, and the server returns
      HTTP 200 even when the app would crash on open.
