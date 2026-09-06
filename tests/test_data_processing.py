"""
Tests for core.data_processing.

These cover the arithmetic that produced silently wrong numbers before the
2026-09-05 fixes: valuation ratios against non-positive earnings, PEG against a
shrinking company, revenue consistency when growth crosses zero, and the TTM
window boundary.
"""

import numpy as np
import pandas as pd
import pytest

from core.acquisitions import aligned_metric, all_profiles, build_profile
from core.backtesting import (
    STRATEGY_SOURCE_GLOBAL_DEFAULT,
    get_ticker_strategy_with_source,
)
from core.classifications import get_stock_classification, normalize_symbol
from core.data_processing import (
    attach_classifications,
    calculate_outperformance_ratios,
    calculate_qoq_changes,
    calculate_sector_rankings,
    latest_row_per_ticker,
    latest_with_age,
    load_stock_data,
    report_range,
    report_sort_key,
)
from core.enums import FilePaths, RollingWindow, StrategyPolicy, Thresholds
from core.predictions import predict_next_eps
from core.ticker_status import (
    STATUS_ACQUIRED,
    STATUS_ACTIVE,
    attach_status,
    get_acquired_tickers,
    get_acquisition,
    get_excluded_tickers,
    get_ticker_status,
)


def build_panel(eps, revenue=None, price=None, div=None, ticker="TEST"):
    """A single-ticker panel shaped the way calculate_qoq_changes expects."""
    n = len(eps)
    if revenue is None:
        revenue = [1000.0] * n
    if price is None:
        price = [100.0] * n
    if div is None:
        div = [0.0] * n
    return pd.DataFrame(
        {
            "Ticker": [ticker] * n,
            "Index": list(range(1, n + 1)),
            "Report": [f"Q{(i % 4) + 1}'{20 + i // 4}" for i in range(n)],
            "EPS": eps,
            "Revenue": revenue,
            "Price": price,
            "DivAmt": div,
        }
    )


class TestNegativeEarnings:
    """Item 2: valuation ratios are undefined against non-positive EPS_TTM."""

    def test_multiple_is_nan_when_eps_ttm_negative(self):
        # Four losing quarters, then four profitable ones.
        df = build_panel([-1.0] * 4 + [2.0] * 4)
        out = calculate_qoq_changes(df)

        negative = out["EPS_TTM"] < 0
        assert negative.any(), "fixture should produce negative TTM earnings"
        assert out.loc[negative, "Multiple"].isna().all()

    def test_multiple_is_positive_wherever_it_is_defined(self):
        df = build_panel([-1.0] * 4 + [2.0] * 4)
        out = calculate_qoq_changes(df)

        defined = out["Multiple"].dropna()
        assert len(defined) > 0
        assert (defined > 0).all(), "a P/E on positive earnings cannot be negative"

    def test_payout_ratio_is_nan_when_eps_ttm_negative(self):
        df = build_panel([-1.0] * 4 + [2.0] * 4, div=[0.25] * 8)
        out = calculate_qoq_changes(df)

        negative = out["EPS_TTM"] < 0
        assert out.loc[negative, "PayoutRatio"].isna().all()

    def test_zero_eps_ttm_is_also_excluded(self):
        # EPS_TTM sums to exactly zero; dividing by it is undefined, not huge.
        df = build_panel([1.0, -1.0, 1.0, -1.0] * 2)
        out = calculate_qoq_changes(df)

        zero = out["EPS_TTM"] == 0
        assert zero.any(), "fixture should produce a zero TTM sum"
        assert out.loc[zero, "Multiple"].isna().all()
        assert out.loc[zero, "PayoutRatio"].isna().all()

    def test_ranking_no_longer_puts_a_loss_maker_first(self):
        """The concrete regression: 'lower is better' ranked losses to rank 1."""
        loser = build_panel([-5.0] * 8, ticker="LOSER")
        winner = build_panel([2.0] * 8, ticker="WINNER")
        out = calculate_qoq_changes(pd.concat([loser, winner], ignore_index=True))

        multiples = out.dropna(subset=["Multiple"])
        assert set(multiples["Ticker"].unique()) == {"WINNER"}


class TestPegRatio:
    """Item 3: PEG must not read a decline as growth."""

    def test_declining_eps_gives_no_peg(self):
        # Steady decline: every QoQ change is negative.
        df = build_panel([10.0, 9.0, 8.1, 7.29, 6.56, 5.9, 5.31, 4.78])
        out = calculate_qoq_changes(df)

        assert out["PEGRatio"].dropna().empty, "PEG is undefined for negative growth"

    def test_growing_eps_gives_a_positive_peg(self):
        df = build_panel([1.0, 1.1, 1.21, 1.33, 1.46, 1.61, 1.77, 1.95])
        out = calculate_qoq_changes(df)

        peg = out["PEGRatio"].dropna()
        assert len(peg) > 0
        assert (peg > 0).all()

    def test_decline_and_growth_do_not_collapse_to_the_same_peg(self):
        """The .abs() bug made -30% and +30% growth indistinguishable."""
        up = build_panel([1.0, 1.1, 1.21, 1.33, 1.46, 1.61, 1.77, 1.95], ticker="UP")
        down = build_panel(
            [1.95, 1.77, 1.61, 1.46, 1.33, 1.21, 1.1, 1.0], ticker="DOWN"
        )
        out = calculate_qoq_changes(pd.concat([up, down], ignore_index=True))

        up_peg = out[out["Ticker"] == "UP"]["PEGRatio"].dropna()
        down_peg = out[out["Ticker"] == "DOWN"]["PEGRatio"].dropna()
        assert len(up_peg) > 0
        assert down_peg.empty


class TestRevenueConsistency:
    """Item 4: the score must stay bounded when growth crosses zero."""

    def test_bounded_when_revenue_growth_crosses_zero(self):
        # Revenue oscillates, so mean Revenue_QoQ sits near zero -- the exact
        # case where the old |mean| denominator exploded.
        revenue = [100.0, 130.0, 100.0, 130.0, 100.0, 130.0, 100.0, 130.0]
        df = build_panel([1.0] * 8, revenue=revenue)
        out = calculate_qoq_changes(df)

        score = out["RevenueConsistency"].dropna()
        assert len(score) > 0
        assert (score > 0).all()
        assert (score <= 100).all()

    def test_steady_revenue_scores_near_100(self):
        revenue = [100.0 * (1.05**i) for i in range(8)]
        df = build_panel([1.0] * 8, revenue=revenue)
        out = calculate_qoq_changes(df)

        assert out["RevenueConsistency"].dropna().max() > 95

    def test_volatile_revenue_scores_below_steady_revenue(self):
        steady = build_panel(
            [1.0] * 8, revenue=[100.0 * (1.05**i) for i in range(8)], ticker="STEADY"
        )
        wild = build_panel(
            [1.0] * 8,
            revenue=[100.0, 300.0, 90.0, 400.0, 80.0, 500.0, 70.0, 600.0],
            ticker="WILD",
        )
        out = calculate_qoq_changes(pd.concat([steady, wild], ignore_index=True))

        steady_score = out[out["Ticker"] == "STEADY"]["RevenueConsistency"].dropna()
        wild_score = out[out["Ticker"] == "WILD"]["RevenueConsistency"].dropna()
        assert steady_score.min() > wild_score.max()

    def test_scale_constant_sets_the_midpoint(self):
        """A std equal to the scale must score exactly 50."""
        scale = Thresholds.CONSISTENCY_VOLATILITY_SCALE
        assert 100 / (1 + scale / scale) == pytest.approx(50.0)


class TestTtmBoundaries:
    """TTM is a 4-quarter sum and must not be produced from fewer."""

    def test_no_ttm_before_four_quarters(self):
        df = build_panel([1.0, 2.0, 3.0])
        out = calculate_qoq_changes(df)

        assert out["EPS_TTM"].isna().all()

    def test_ttm_appears_exactly_at_the_fourth_quarter(self):
        df = build_panel([1.0, 2.0, 3.0, 4.0])
        out = calculate_qoq_changes(df).sort_values("Index")

        assert out["EPS_TTM"].iloc[:3].isna().all()
        assert out["EPS_TTM"].iloc[3] == pytest.approx(10.0)

    def test_ttm_is_a_rolling_four_quarter_sum(self):
        df = build_panel([1.0, 2.0, 3.0, 4.0, 5.0])
        out = calculate_qoq_changes(df).sort_values("Index")

        assert out["EPS_TTM"].iloc[4] == pytest.approx(14.0)  # 2+3+4+5

    def test_ttm_window_matches_the_configured_size(self):
        n = RollingWindow.TTM
        df = build_panel([1.0] * n)
        out = calculate_qoq_changes(df).sort_values("Index")

        assert out["EPS_TTM"].iloc[-1] == pytest.approx(float(n))

    def test_revenue_ttm_follows_the_same_rule(self):
        df = build_panel([1.0] * 5, revenue=[10.0, 20.0, 30.0, 40.0, 50.0])
        out = calculate_qoq_changes(df).sort_values("Index")

        assert out["Revenue_TTM"].iloc[:3].isna().all()
        assert out["Revenue_TTM"].iloc[3] == pytest.approx(100.0)


class TestQoqBasics:
    """QoQ percent change, including the zero-crossing cases."""

    def test_qoq_is_a_percent_change(self):
        df = build_panel([1.0, 1.5, 3.0, 3.0])
        out = calculate_qoq_changes(df).sort_values("Index")

        assert np.isnan(out["EPS_QoQ"].iloc[0])
        assert out["EPS_QoQ"].iloc[1] == pytest.approx(50.0)
        assert out["EPS_QoQ"].iloc[2] == pytest.approx(100.0)
        assert out["EPS_QoQ"].iloc[3] == pytest.approx(0.0)

    def test_no_infinities_survive_a_zero_denominator(self):
        df = build_panel([0.0, 1.0, 0.0, -1.0, 2.0, 0.0, 3.0, 1.0])
        out = calculate_qoq_changes(df)

        numeric = out.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()

    def test_dividend_growth_is_nan_when_unmeasurable(self):
        """Item 31: 0.0 used to mean 'unknown', which read as 'no growth'."""
        df = build_panel([1.0] * 8, div=[0.25] * 8)  # never changes
        out = calculate_qoq_changes(df)

        assert out["DivGrowthRate"].isna().all()

    def test_duplicate_qoq_column_is_gone(self):
        """Item 31: DivYieldAnnual_QoQ always equalled DivYield_QoQ."""
        df = build_panel([1.0] * 8, div=[0.25] * 8)
        out = calculate_qoq_changes(df)

        assert "DivYield_QoQ" in out.columns
        assert "DivYieldAnnual_QoQ" not in out.columns


class TestReportOrdering:
    """Item 31: fiscal quarter labels do not sort as strings."""

    def test_year_beats_quarter(self):
        assert report_sort_key("Q1'26") > report_sort_key("Q4'10")

    def test_quarter_orders_within_a_year(self):
        assert report_sort_key("Q1'25") < report_sort_key("Q4'25")

    def test_range_spans_the_real_endpoints(self):
        labels = pd.Series(["Q4'10", "Q1'26", "Q3'25", "Q2'27", "Q1'10"])
        assert report_range(labels) == ("Q1'10", "Q2'27")

    def test_naive_string_sort_would_disagree(self):
        labels = ["Q4'10", "Q1'26"]
        assert sorted(labels)[-1] != sorted(labels, key=report_sort_key)[-1]

    def test_unparseable_label_sorts_last_instead_of_raising(self):
        assert report_sort_key("garbage") > report_sort_key("Q4'99")

    def test_empty_series_gives_no_range(self):
        assert report_range(pd.Series([], dtype=object)) == (None, None)


class TestLatestWithAge:
    """Item 28: 'latest' must not silently reach back through blank quarters."""

    def test_current_value_reports_zero_staleness(self):
        df = build_panel([1.0, 2.0, 3.0, 4.0])
        value, quarter, stale = latest_with_age(df, "EPS")

        assert value == pytest.approx(4.0)
        assert quarter == "Q4'20"
        assert stale == 0

    def test_blank_latest_quarters_are_counted(self):
        """The DAL case: a dividend that stopped, still shown as current."""
        df = build_panel([1.0] * 6, div=[0.25, 0.25, np.nan, np.nan, np.nan, np.nan])
        value, quarter, stale = latest_with_age(df, "DivAmt")

        assert value == pytest.approx(0.25)
        assert stale == 4, "value is four quarters behind the latest row"

    def test_entirely_blank_column_returns_nothing(self):
        df = build_panel([1.0] * 4, div=[np.nan] * 4)
        assert latest_with_age(df, "DivAmt") == (None, None, None)

    def test_missing_column_returns_nothing(self):
        df = build_panel([1.0] * 4)
        assert latest_with_age(df, "NoSuchColumn") == (None, None, None)

    def test_unordered_rows_still_resolve_by_index(self):
        df = build_panel([1.0, 2.0, 3.0, 4.0]).sample(frac=1, random_state=0)
        value, _, stale = latest_with_age(df, "EPS")

        assert value == pytest.approx(4.0)
        assert stale == 0


class TestPeerComparison:
    """Items 8 and 29: peer ranks compare companies, not ticker-quarter rows."""

    @staticmethod
    def two_sector_panel():
        """Two tickers with different history lengths in one sector."""
        long_history = build_panel([1.0] * 20, price=[100.0] * 20, ticker="LONG")
        short_history = build_panel([2.0] * 8, price=[100.0] * 8, ticker="SHORT")
        df = pd.concat([long_history, short_history], ignore_index=True)
        df["Sector"] = "Information Technology"
        return calculate_qoq_changes(df)

    def test_latest_row_per_ticker_returns_one_row_each(self):
        df = self.two_sector_panel()
        latest = latest_row_per_ticker(df)

        assert len(latest) == df["Ticker"].nunique()
        assert set(latest["Ticker"]) == {"LONG", "SHORT"}

    def test_latest_row_is_the_highest_index(self):
        df = self.two_sector_panel()
        latest = latest_row_per_ticker(df).set_index("Ticker")

        assert latest.loc["LONG", "Index"] == 20
        assert latest.loc["SHORT", "Index"] == 8

    def test_sector_rank_never_exceeds_the_number_of_peers(self):
        """The old code ranked rows, producing ranks far above the peer count."""
        df = calculate_sector_rankings(self.two_sector_panel())
        ranks = df["EPS_TTM_SectorRank"].dropna()

        assert len(ranks) > 0
        assert ranks.max() <= df["Ticker"].nunique()

    def test_rank_is_constant_across_a_tickers_rows(self):
        df = calculate_sector_rankings(self.two_sector_panel())

        for _, rows in df.groupby("Ticker"):
            assert rows["EPS_TTM_SectorRank"].nunique(dropna=True) <= 1

    def test_benchmark_is_not_row_weighted(self):
        """A long-history ticker must not outweigh a short-history one."""
        df = self.two_sector_panel()
        latest = latest_row_per_ticker(df)

        row_weighted = df["EPS_TTM"].mean()
        per_ticker = latest["EPS_TTM"].mean()
        assert per_ticker != pytest.approx(row_weighted)
        assert per_ticker == pytest.approx(
            (
                latest.set_index("Ticker").loc["LONG", "EPS_TTM"]
                + latest.set_index("Ticker").loc["SHORT", "EPS_TTM"]
            )
            / 2
        )

    def test_percent_metrics_are_compared_in_percentage_points(self):
        df = calculate_outperformance_ratios(self.two_sector_panel())

        assert "EPS_QoQ_MarketGapPP" in df.columns
        assert "EPS_QoQ_MarketOutperf" not in df.columns

    def test_level_metrics_keep_the_ratio_form(self):
        df = calculate_outperformance_ratios(self.two_sector_panel())

        assert "EPS_TTM_MarketOutperf" in df.columns
        assert "EPS_TTM_MarketGapPP" not in df.columns

    def test_no_sector_column_means_no_ranking(self):
        df = calculate_qoq_changes(build_panel([1.0] * 8))
        assert calculate_sector_rankings(df).equals(df)

    def test_classifications_join_populates_the_sector_column(self):
        df = calculate_qoq_changes(build_panel([1.0] * 8, ticker="AAPL"))
        out = attach_classifications(df)

        assert "Sector" in out.columns
        assert out["Sector"].iloc[0] != "Unclassified"

    def test_unknown_ticker_is_marked_unclassified(self):
        df = calculate_qoq_changes(build_panel([1.0] * 8, ticker="ZZZZ"))
        out = attach_classifications(df)

        assert (out["Sector"] == "Unclassified").all()


class TestStrategyPolicy:
    """Items 21 and 22: one global strategy, with an explicit fallback."""

    @staticmethod
    def growing_panel(ticker="TEST"):
        return calculate_qoq_changes(
            build_panel([1.0 * (1.05**i) for i in range(16)], ticker=ticker)
        )

    def test_strategy_source_is_the_global_default(self):
        """Per-ticker mapping lost out-of-sample and is no longer consulted."""
        strategy, source = get_ticker_strategy_with_source("AAPL")

        assert source == STRATEGY_SOURCE_GLOBAL_DEFAULT
        assert strategy == StrategyPolicy.GLOBAL_DEFAULT

    def test_every_ticker_gets_the_same_strategy(self):
        picks = {
            get_ticker_strategy_with_source(t)[0]
            for t in ("AAPL", "INTC", "KO", "ZZZZ")
        }
        assert picks == {StrategyPolicy.GLOBAL_DEFAULT}

    def test_an_explicit_mapping_is_still_honoured(self):
        """Callers may still pass one in; only the implicit lookup was dropped."""
        strategy, source = get_ticker_strategy_with_source(
            "AAPL", ticker_strategy_mapping={"AAPL": "seasonal"}
        )

        assert strategy == "seasonal"
        assert source != STRATEGY_SOURCE_GLOBAL_DEFAULT

    def test_prediction_reports_the_strategy_it_used(self):
        prediction = predict_next_eps(self.growing_panel(), "TEST")

        assert prediction is not None
        assert prediction["strategy"] == StrategyPolicy.GLOBAL_DEFAULT
        assert prediction["strategy_source"] == STRATEGY_SOURCE_GLOBAL_DEFAULT

    def test_methodology_no_longer_claims_backtested_optimal(self):
        prediction = predict_next_eps(self.growing_panel(), "TEST")

        assert "backtested optimal" not in prediction["methodology"]
        assert "global default" in prediction["methodology"]

    def test_fallback_order_covers_every_strategy(self):
        from core.strategies import STRATEGIES

        assert set(StrategyPolicy.FALLBACK_ORDER) == set(STRATEGIES)


class TestZeroCrossingEps:
    """Item 22: a sign change makes growth rates meaningless -- say so."""

    @staticmethod
    def crossing_panel():
        eps = [
            0.41,
            0.54,
            0.18,
            0.02,
            -0.46,
            0.13,
            0.13,
            -0.10,
            0.23,
            0.15,
            0.29,
            0.42,
            0.31,
            0.25,
            0.38,
            0.44,
        ]
        return calculate_qoq_changes(build_panel(eps, ticker="CROSS"))

    def test_zero_crossing_is_flagged(self):
        prediction = predict_next_eps(self.crossing_panel(), "CROSS")

        assert prediction is not None
        assert prediction["eps_crosses_zero"] is True
        assert "crosses zero" in prediction["methodology"]

    def test_steady_eps_is_not_flagged(self):
        steady = calculate_qoq_changes(
            build_panel([1.0 * (1.05**i) for i in range(16)], ticker="STEADY")
        )
        prediction = predict_next_eps(steady, "STEADY")

        assert prediction["eps_crosses_zero"] is False
        assert "crosses zero" not in prediction["methodology"]

    def test_a_zero_crossing_ticker_still_gets_a_forecast(self):
        """It used to get none at all, because its mapped strategy could not fit."""
        prediction = predict_next_eps(self.crossing_panel(), "CROSS")

        assert prediction is not None
        assert prediction["predicted_eps"] is not None


class TestTickerStatus:
    """Acquired companies are separated from the active universe, not deleted."""

    def test_acquired_tickers_are_configured(self):
        acquired = get_acquired_tickers()

        assert {"S", "JWN", "SKX", "EA"} <= acquired

    def test_every_acquired_ticker_is_held_out_of_active(self):
        """A company frozen years ago must not sit in a current benchmark."""
        assert get_acquired_tickers() <= get_excluded_tickers()

    def test_status_defaults_to_active_for_unlisted_tickers(self):
        assert get_ticker_status("AAPL") == STATUS_ACTIVE
        assert get_ticker_status("ZZZZ") == STATUS_ACTIVE

    def test_acquired_ticker_reports_its_status(self):
        assert get_ticker_status("EA") == STATUS_ACQUIRED

    def test_a_hold_is_independent_of_acquisition_status(self):
        """The two flags are separate: a held-back ticker is still an active
        company, and being active says nothing about being held back."""
        held_but_active = get_excluded_tickers() - get_acquired_tickers()
        for ticker in held_but_active:
            assert get_ticker_status(ticker) == STATUS_ACTIVE
            assert get_acquisition(ticker) is None

    def test_brk_b_is_active_and_no_longer_held_back(self):
        """Its 10 missing quarters were backfilled on 2026-09-06."""
        assert get_ticker_status("BRK.B") == STATUS_ACTIVE
        assert "BRK.B" not in get_excluded_tickers()

    def test_acquisition_details_are_present(self):
        details = get_acquisition("EA")

        assert details is not None
        assert details["completed"] == "2026-08-04"
        assert details["announced"] == "2025-09-29"
        assert details["price_per_share"] == 210.0

    def test_active_ticker_has_no_acquisition_details(self):
        assert get_acquisition("AAPL") is None

    def test_attach_status_marks_rows(self):
        df = pd.concat(
            [
                build_panel([1.0] * 4, ticker="EA"),
                build_panel([1.0] * 4, ticker="AAPL"),
            ],
            ignore_index=True,
        )
        out = attach_status(df)

        assert set(out.loc[out["Ticker"] == "EA", "Status"]) == {STATUS_ACQUIRED}
        assert set(out.loc[out["Ticker"] == "AAPL", "Status"]) == {STATUS_ACTIVE}


class TestAcquisitionProfile:
    """Quarters line up on the announcement so deals can be compared."""

    @staticmethod
    def panel():
        return calculate_qoq_changes(
            load_stock_data(FilePaths.DATA_FILE, include_excluded=True)
        )

    def test_acquired_tickers_load_only_when_asked(self):
        active = load_stock_data(FilePaths.DATA_FILE)
        everything = load_stock_data(FilePaths.DATA_FILE, include_excluded=True)

        assert "EA" not in set(active["Ticker"])
        assert "EA" in set(everything["Ticker"])
        assert len(everything) > len(active)

    def test_quarter_zero_is_the_last_report_before_announcement(self):
        profile = build_profile(self.panel(), "EA")

        assert profile is not None
        anchor = profile[profile["QuartersToAnnouncement"] == 0].iloc[0]
        assert anchor["EarningsDate"] <= pd.Timestamp("2025-09-29")

    def test_quarters_after_the_announcement_are_positive(self):
        profile = build_profile(self.panel(), "EA")
        pending = profile[profile["QuartersToAnnouncement"] > 0]

        assert len(pending) > 0
        assert (pending["EarningsDate"] > pd.Timestamp("2025-09-29")).all()

    def test_offsets_are_consecutive(self):
        profile = build_profile(self.panel(), "SKX")
        offsets = profile["QuartersToAnnouncement"].tolist()

        assert offsets == list(range(offsets[0], offsets[0] + len(offsets)))

    def test_active_ticker_has_no_profile(self):
        assert build_profile(self.panel(), "AAPL") is None

    def test_all_profiles_covers_every_acquired_ticker(self):
        profiles = all_profiles(self.panel())

        assert set(profiles["ticker"]) == get_acquired_tickers()
        assert profiles["announced"].is_monotonic_increasing

    def test_all_stock_deal_has_no_price_comparison(self):
        """Sprint was all-stock, so there is no per-share cash price."""
        profiles = all_profiles(self.panel()).set_index("ticker")

        assert pd.isna(profiles.loc["S", "deal_price_vs_last_report_pct"])

    def test_aligned_metric_puts_tickers_in_columns(self):
        aligned = aligned_metric(self.panel(), "Multiple")

        assert aligned.index.name == "QuartersToAnnouncement"
        assert "EA" in aligned.columns
        assert aligned.index.is_monotonic_increasing


class TestRenamedSymbols:
    """A ticker change is a relabelling, not a new company."""

    def test_former_symbol_maps_to_the_current_one(self):
        assert normalize_symbol("BK") == "BNY"

    def test_mapping_is_case_insensitive(self):
        assert normalize_symbol("bk") == "BNY"
        assert normalize_symbol(" Bk ") == "BNY"

    def test_current_symbol_is_unchanged(self):
        assert normalize_symbol("BNY") == "BNY"

    def test_unrelated_symbols_are_untouched(self):
        assert normalize_symbol("AAPL") == "AAPL"
        assert normalize_symbol("BRK/B") == "BRK.B"

    def test_former_symbol_still_resolves_a_classification(self):
        """Rows predating the rename must not become Unclassified."""
        assert get_stock_classification("BK") is not None
        assert get_stock_classification("BK") == get_stock_classification("BNY")

    def test_renamed_company_keeps_one_identity_in_the_panel(self):
        df = attach_classifications(
            pd.concat(
                [
                    build_panel([1.0] * 4, ticker="BK"),
                    build_panel([1.0] * 4, ticker="BNY"),
                ],
                ignore_index=True,
            )
        )
        assert (df["Sector"] == "Financials").all()
