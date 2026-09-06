#!/usr/bin/env python3
"""
Build the registry of real earnings dates from SEC EDGAR.

The workbook's ``EarningsDate`` column is almost entirely modelled: the
2026-09-02 backfill fitted every date and the real ones it was seeded from are
no longer distinguishable in the file. This fetches actual 10-Q/10-K filing
dates from EDGAR so there is a verifiable ground truth to anchor on, written to
``config/earnings_dates.json``.

A filing date is not the same as the earnings announcement -- a company usually
issues its release a few days before the 10-Q lands -- so these are close but
not exact. They are still far better than a pure model, and unlike the model
they are checkable against a primary source.

Usage:
    python scripts/fetch_earnings_dates.py
    python scripts/fetch_earnings_dates.py --ticker AAPL --ticker BNY
    python scripts/fetch_earnings_dates.py --tolerance 30
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from core.enums import Column, FilePaths  # noqa: E402

REGISTRY = REPO_ROOT / "config" / "earnings_dates.json"
TICKER_FILE = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC's ticker file lists only current registrants, so delisted companies are
# absent -- and worse, a retired ticker can be reissued to someone else. "S" now
# belongs to SentinelOne, not Sprint, so resolving it automatically would fetch
# the wrong company's filings entirely.
CIK_OVERRIDES = {
    "S": [101830],  # Sprint Corp (ticker since reissued to SentinelOne)
    "JWN": [72333],  # Nordstrom, taken private 2025
    "SKX": [1065837],  # Skechers, taken private 2025
    "EA": [712515],  # Electronic Arts, taken private 2026
}

# A reorganisation leaves the company's older filings under its predecessor's
# CIK, so the current one covers only part of the panel. Each entry lists every
# CIK a company has filed under, newest first; their filings are merged.
CIK_PREDECESSORS = {
    "XOM": [34088],  # Exxon Mobil Corp, before the 2025 holdco
    "GOOGL": [1288776],  # Google Inc, before the 2015 Alphabet reorg
    "DIS": [1001039],  # TWDC Enterprises 18 Corp, before the 2019 holdco
    "AVGO": [1441634],  # Avago Technologies, before the 2018 redomicile
    "BLK": [1364742],  # BlackRock Finance, before the 2024 reorg
    "MDT": [64670],  # Medtronic Inc, before the 2015 Irish redomicile
}

# Foreign private issuers file 6-K/20-F instead of 8-K/10-Q, and a 6-K carries
# no item codes, so an earnings 6-K cannot be told from any other announcement.
# These are matched on proximity alone and are therefore weaker evidence -- the
# stored "source" says so.
FOREIGN_FORMS = ["6-K", "20-F", "40-F"]

# Registry values are stored as "YYYY-MM-DD|code" -- one line per quarter keeps
# the file small enough to live in git alongside the code that generates it.
SOURCE_CODES = {
    "8-K item 2.02": "8K",
    "10-Q filing": "10Q",
    "10-K filing": "10K",
    "6-K (proximity match, weaker)": "6K~",
    "20-F (proximity match, weaker)": "6K~",
    "40-F (proximity match, weaker)": "6K~",
}

# SEC asks for no more than 10 requests/second and a descriptive User-Agent.
USER_AGENT = "StockInsights research (danrgonzalez@gmail.com)"
REQUEST_DELAY = 0.15


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def sec_symbol(ticker: str) -> str:
    """The workbook writes BRK/B; SEC writes BRK-B."""
    return ticker.replace("/", "-").replace(".", "-").upper()


def build_cik_map(tickers: list[str]) -> dict[str, list[int]]:
    """Every CIK a ticker has filed under, newest first."""
    raw = get(TICKER_FILE)
    lookup = {v["ticker"]: v["cik_str"] for v in raw.values()}
    out = {}
    for ticker in tickers:
        ciks = list(CIK_OVERRIDES.get(ticker) or [])
        if not ciks:
            current = lookup.get(sec_symbol(ticker))
            if current:
                ciks = [current]
        ciks += CIK_PREDECESSORS.get(ticker, [])
        if ciks:
            out[ticker] = ciks
    return out


def earnings_filings(ciks: list[int]) -> pd.DataFrame:
    """Dated earnings events for a company, best source first.

    An 8-K carrying item 2.02 ("Results of Operations and Financial Condition")
    *is* the earnings release, so its event date is the announcement date. A
    10-Q/10-K filing date is only a proxy: some companies file the same day they
    announce, others weeks later -- BNY's filings trail its announcements by
    around three weeks -- so the 8-K is used wherever one exists and the
    periodic filing is the fallback for quarters that predate 8-K coverage.
    """
    frames = []
    for one in ciks:
        time.sleep(REQUEST_DELAY)
        data = get(SUBMISSIONS.format(cik=one))
        frames.append(pd.DataFrame(data["filings"]["recent"]))
        for extra in data["filings"].get("files", []):
            time.sleep(REQUEST_DELAY)
            frames.append(
                pd.DataFrame(get(f"https://data.sec.gov/submissions/{extra['name']}"))
            )
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True)
    for col in ("filingDate", "reportDate"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    items = df.get("items", pd.Series("", index=df.index)).astype(str)
    announcements = df[(df["form"] == "8-K") & items.str.contains(r"\b2\.02\b")].copy()
    # The 8-K's reportDate is the event date; fall back to when it was filed.
    announcements["date"] = announcements["reportDate"].fillna(
        announcements["filingDate"]
    )
    announcements["source"] = "8-K item 2.02"

    periodic = df[df["form"].isin(["10-Q", "10-K"])].copy()
    periodic["date"] = periodic["filingDate"]
    periodic["source"] = periodic["form"] + " filing"

    foreign = df[df["form"].isin(FOREIGN_FORMS)].copy()
    foreign["date"] = foreign["filingDate"]
    foreign["source"] = foreign["form"] + " (proximity match, weaker)"

    out = pd.concat([announcements, periodic, foreign], ignore_index=True)
    out = out.dropna(subset=["date"])
    # Best evidence first: the earnings 8-K, then a periodic filing, then a
    # foreign filing that only happens to fall near the expected date.
    rank = {"8-K item 2.02": 0}
    out["rank"] = [rank.get(s, 2 if "proximity" in s else 1) for s in out["source"]]
    return out.sort_values(["date", "rank"])


def match_to_panel(panel: pd.DataFrame, filings: pd.DataFrame, tolerance: int) -> dict:
    """Attach each workbook row to the filing that reported that quarter.

    Matched on the existing (modelled) date only to decide *which* quarter a
    filing belongs to; the value stored is the filing's own date. Quarters are
    ~91 days apart, so a tolerance well under that is unambiguous.
    """
    out = {}
    used = set()
    for _, row in panel.iterrows():
        current = row[Column.EARNINGS_DATE.value]
        if pd.isna(current):
            continue
        gaps = (filings["date"] - current).abs()
        near = filings[gaps <= pd.Timedelta(days=tolerance)]
        # closest first, and an 8-K ahead of a periodic filing at equal distance
        near = near.assign(gap=gaps[near.index]).sort_values(["gap", "rank"])
        for pos in near.index:
            if pos in used:
                continue
            used.add(pos)
            date = filings.loc[pos, "date"].strftime("%Y-%m-%d")
            out[str(row[Column.REPORT.value])] = (
                f"{date}|{SOURCE_CODES[filings.loc[pos, 'source']]}"
            )
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=FilePaths.DATA_FILE)
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=45,
        help="max days between the modelled date and a filing to call it a match",
    )
    args = parser.parse_args()

    panel = pd.read_excel(args.source)
    panel[Column.EARNINGS_DATE.value] = pd.to_datetime(
        panel[Column.EARNINGS_DATE.value], errors="coerce"
    )
    tickers = args.tickers or sorted(panel[Column.TICKER.value].unique())

    print(f"resolving {len(tickers)} tickers to CIKs ...")
    ciks = build_cik_map(tickers)
    unresolved = [t for t in tickers if t not in ciks]
    if unresolved:
        print(f"  no CIK for: {', '.join(unresolved)}")

    registry = {}
    if REGISTRY.exists():
        registry = json.loads(REGISTRY.read_text()).get("tickers", {})

    total_matched = 0
    for i, ticker in enumerate(tickers, 1):
        if ticker not in ciks:
            continue
        try:
            time.sleep(REQUEST_DELAY)
            filings = earnings_filings(ciks[ticker])
        except Exception as exc:  # network or unexpected shape
            print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} FAILED: {exc}")
            continue

        rows = panel[panel[Column.TICKER.value] == ticker]
        matched = match_to_panel(rows, filings, args.tolerance)
        registry[ticker] = matched
        total_matched += len(matched)
        print(
            f"  [{i:3d}/{len(tickers)}] {ticker:6s} "
            f"{len(matched):3d}/{len(rows):3d} quarters matched "
            f"({len(filings)} filings on EDGAR)"
        )

    REGISTRY.write_text(
        json.dumps(
            {
                "_comment": (
                    "Real earnings dates from SEC EDGAR, keyed by ticker then "
                    "Report label. Each value is 'YYYY-MM-DD|source'. Sources: "
                    "8K = the 8-K carrying item 2.02 (Results of Operations), "
                    "which is the earnings release itself and the best evidence; "
                    "10Q/10K = the periodic filing date, which trails the "
                    "announcement by a few days; 6K~ = a foreign issuer's 6-K "
                    "matched on proximity only, because 6-Ks carry no item codes "
                    "- treat those as weaker. Regenerate with "
                    "scripts/fetch_earnings_dates.py."
                ),
                "_generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "tickers": {
                    t: dict(sorted(q.items())) for t, q in sorted(registry.items())
                },
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {REGISTRY}")
    print(f"  {total_matched:,} real dates across {len(registry)} tickers")


if __name__ == "__main__":
    main()
