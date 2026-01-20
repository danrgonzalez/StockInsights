"""
Core Enumerations for StockInsights

This module provides consistent vocabulary across the entire codebase through
well-defined enums for strategies, metrics, confidence levels, and other
domain concepts.

Usage:
    from core.enums import Strategy, Confidence, Metric, Scenario

    # Use enum values consistently
    strategy = Strategy.WEIGHTED_GROWTH
    if prediction["confidence"] == Confidence.HIGH.value:
        ...
"""

from enum import Enum


class Strategy(str, Enum):
    """EPS prediction strategy algorithms."""

    WEIGHTED_GROWTH = "weighted_growth"
    SIMPLE_AVERAGE = "simple_average"
    MOMENTUM = "momentum"
    TREND_ANALYSIS = "trend_analysis"
    SEASONAL = "seasonal"

    @classmethod
    def default(cls) -> "Strategy":
        """Return the default strategy."""
        return cls.WEIGHTED_GROWTH

    @classmethod
    def list_all(cls) -> list[str]:
        """Return all strategy names as strings."""
        return [s.value for s in cls]


class Confidence(str, Enum):
    """Prediction confidence levels based on data availability."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

    @property
    def emoji(self) -> str:
        """Return the emoji indicator for this confidence level."""
        return {
            Confidence.HIGH: "🟢",
            Confidence.MEDIUM: "🟡",
            Confidence.LOW: "🔴",
        }[self]


class Scenario(str, Enum):
    """Prediction scenario types for EPS forecasting."""

    BASE = "predicted"
    BEST = "best_case"
    WORST = "worst_case"

    @classmethod
    def eps_keys(cls) -> dict[str, str]:
        """Return mapping of scenario to EPS key names."""
        return {
            cls.BASE.value: "predicted_eps",
            cls.BEST.value: "best_case_eps",
            cls.WORST.value: "worst_case_eps",
        }

    @classmethod
    def growth_keys(cls) -> dict[str, str]:
        """Return mapping of scenario to growth key names."""
        return {
            cls.BASE.value: "predicted_growth",
            cls.BEST.value: "best_case_growth",
            cls.WORST.value: "worst_case_growth",
        }


class Metric(str, Enum):
    """Base financial metrics from source data."""

    EPS = "EPS"
    REVENUE = "Revenue"
    PRICE = "Price"
    DIVIDEND_AMOUNT = "DivAmt"
    INDEX = "Index"

    @classmethod
    def numeric_columns(cls) -> list[str]:
        """Return list of numeric column names for data loading."""
        return [
            cls.EPS.value,
            cls.REVENUE.value,
            cls.PRICE.value,
            cls.DIVIDEND_AMOUNT.value,
            cls.INDEX.value,
        ]


class DerivedMetric(str, Enum):
    """Calculated metrics derived from base metrics."""

    # Trailing Twelve Months
    EPS_TTM = "EPS_TTM"
    REVENUE_TTM = "Revenue_TTM"

    # Quarter-over-Quarter changes
    EPS_QOQ = "EPS_QoQ"
    REVENUE_QOQ = "Revenue_QoQ"
    PRICE_QOQ = "Price_QoQ"
    EPS_TTM_QOQ = "EPS_TTM_QoQ"
    REVENUE_TTM_QOQ = "Revenue_TTM_QoQ"
    DIVIDEND_QOQ = "DivAmt_QoQ"
    DIVIDEND_YIELD_QOQ = "DivYield_QoQ"
    DIVIDEND_YIELD_ANNUAL_QOQ = "DivYieldAnnual_QoQ"
    PAYOUT_RATIO_QOQ = "PayoutRatio_QoQ"
    MULTIPLE_QOQ = "Multiple_QoQ"

    # Valuation ratios
    MULTIPLE = "Multiple"
    PEG_RATIO = "PEGRatio"
    PEGY_RATIO = "PEGYRatio"

    # Dividend metrics
    DIVIDEND_YIELD = "DivYield"
    DIVIDEND_YIELD_ANNUAL = "DivYieldAnnual"
    PAYOUT_RATIO = "PayoutRatio"
    DIVIDEND_GROWTH_RATE = "DivGrowthRate"
    DIVIDEND_INCREASE_FREQ = "DivIncreaseFreq"
    AVG_DIVIDEND_INCREASE = "AvgDivIncrease"

    # Advanced analytics
    EPS_MOMENTUM = "EPSMomentum"
    PRICE_VOLATILITY = "PriceVolatility"
    REVENUE_CONSISTENCY = "RevenueConsistency"
    DOWNSIDE_CAPTURE = "DownsideCapture"

    @classmethod
    def ttm_metrics(cls) -> list[str]:
        """Return TTM metric names."""
        return [cls.EPS_TTM.value, cls.REVENUE_TTM.value]

    @classmethod
    def qoq_metrics(cls) -> list[str]:
        """Return base metrics that get QoQ calculations."""
        return [
            Metric.EPS.value,
            Metric.REVENUE.value,
            Metric.PRICE.value,
            cls.EPS_TTM.value,
            cls.REVENUE_TTM.value,
            cls.MULTIPLE.value,
            Metric.DIVIDEND_AMOUNT.value,
            cls.DIVIDEND_YIELD.value,
            cls.DIVIDEND_YIELD_ANNUAL.value,
            cls.PAYOUT_RATIO.value,
        ]

    @classmethod
    def positive_ranking_metrics(cls) -> list[str]:
        """Return metrics where higher is better for sector ranking."""
        return [
            cls.EPS_TTM.value,
            cls.REVENUE_TTM.value,
            cls.DIVIDEND_YIELD.value,
            cls.DIVIDEND_YIELD_ANNUAL.value,
            cls.REVENUE_CONSISTENCY.value,
            cls.EPS_MOMENTUM.value,
        ]

    @classmethod
    def negative_ranking_metrics(cls) -> list[str]:
        """Return metrics where lower is better for sector ranking."""
        return [
            cls.MULTIPLE.value,
            cls.PRICE_VOLATILITY.value,
            cls.PEG_RATIO.value,
            cls.PEGY_RATIO.value,
        ]


class Column(str, Enum):
    """Standard DataFrame column names."""

    TICKER = "Ticker"
    REPORT = "Report"
    INDEX = "Index"
    SECTOR = "Sector"
    INDUSTRY = "Industry"
    SUB_INDUSTRY = "Sub_Industry"

    @classmethod
    def required_columns(cls) -> list[str]:
        """Return columns required for basic data loading."""
        return [cls.TICKER.value, cls.REPORT.value, cls.INDEX.value]


class TimeWindow(str, Enum):
    """Rolling time window periods for calculations."""

    FOUR_QUARTERS = "4Q"
    EIGHT_QUARTERS = "8Q"
    TWELVE_QUARTERS = "12Q"

    @property
    def quarters(self) -> int:
        """Return the number of quarters in this window."""
        return {
            TimeWindow.FOUR_QUARTERS: 4,
            TimeWindow.EIGHT_QUARTERS: 8,
            TimeWindow.TWELVE_QUARTERS: 12,
        }[self]

    @property
    def avg_suffix(self) -> str:
        """Return the column suffix for rolling averages."""
        return f"{self.value}_Avg"


class DataQuality(str, Enum):
    """Data quality and classification status values."""

    UNKNOWN = "Unknown"
    UNCLASSIFIED = "Unclassified"
    NOT_AVAILABLE = "N/A"

    @classmethod
    def invalid_values(cls) -> list[str | None]:
        """Return list of values that indicate invalid/missing classification."""
        return [
            cls.UNKNOWN.value,
            cls.UNCLASSIFIED.value,
            cls.NOT_AVAILABLE.value,
            None,
        ]


class RankingSuffix(str, Enum):
    """Suffixes for ranking and outperformance columns."""

    SECTOR_RANK = "_SectorRank"
    MARKET_OUTPERF = "_MarketOutperf"
    SECTOR_OUTPERF = "_SectorOutperf"

    def column_name(self, metric: str) -> str:
        """Generate full column name for a metric with this suffix."""
        return f"{metric}{self.value}"


# Thresholds and constants as a namespace class
class Thresholds:
    """Constants for prediction and calculation thresholds."""

    # Outlier detection
    GROWTH_OUTLIER_MAX = 200  # ±200% growth considered outlier
    GROWTH_OUTLIER_MIN = -200

    # Prediction caps
    PREDICTED_GROWTH_FLOOR = -50  # Cap at -50% for positive EPS companies
    WORST_CASE_FLOOR = -75  # Cap worst case at -75%
    BEST_CASE_CEILING = 200  # Cap best case at +200%

    # Volatility
    MIN_VOLATILITY = 5.0  # Minimum 5% volatility floor

    # Minimum data requirements
    MIN_QUARTERS_FOR_PREDICTION = 4
    MIN_QOQ_DATA_POINTS = 3
    MIN_QUARTERS_FOR_SEASONAL = 8
    MIN_DOWNSIDE_PERIODS = 3


# Prediction result keys as constants
class PredictionKey:
    """Standard keys for prediction result dictionaries."""

    # EPS predictions
    PREDICTED_EPS = "predicted_eps"
    BEST_CASE_EPS = "best_case_eps"
    WORST_CASE_EPS = "worst_case_eps"
    LATEST_EPS = "latest_eps"

    # Growth rates
    PREDICTED_GROWTH = "predicted_growth"
    BEST_CASE_GROWTH = "best_case_growth"
    WORST_CASE_GROWTH = "worst_case_growth"

    # TTM predictions
    CURRENT_EPS_TTM = "current_eps_ttm"
    PREDICTED_EPS_TTM = "predicted_eps_ttm"
    BEST_CASE_EPS_TTM = "best_case_eps_ttm"
    WORST_CASE_EPS_TTM = "worst_case_eps_ttm"

    # TTM growth rates
    PREDICTED_EPS_TTM_GROWTH = "predicted_eps_ttm_growth"
    BEST_CASE_EPS_TTM_GROWTH = "best_case_eps_ttm_growth"
    WORST_CASE_EPS_TTM_GROWTH = "worst_case_eps_ttm_growth"

    # Price predictions
    CURRENT_PRICE = "current_price"
    CURRENT_MULTIPLE = "current_multiple"
    PREDICTED_PRICE = "predicted_price"
    BEST_CASE_PRICE = "best_case_price"
    WORST_CASE_PRICE = "worst_case_price"

    # Price growth rates
    PREDICTED_PRICE_GROWTH = "predicted_price_growth"
    BEST_CASE_PRICE_GROWTH = "best_case_price_growth"
    WORST_CASE_PRICE_GROWTH = "worst_case_price_growth"

    # Metadata
    VOLATILITY = "volatility"
    CONFIDENCE = "confidence"
    DATA_POINTS = "data_points"
    METHODOLOGY = "methodology"
    NEXT_INDEX = "next_index"

    # Strategy-specific
    GROWTH_4Q = "growth_4q"
    GROWTH_8Q = "growth_8q"
    STD_4Q = "std_4q"
    STD_8Q = "std_8q"
