"""
Backtesting module for EPS prediction strategies.

This module provides functionality for:
- Backtesting individual strategies on historical data
- Running multi-ticker backtests to find optimal strategies
- Analyzing and comparing strategy performance
- Saving/loading ticker-to-strategy mappings
"""

import json

import numpy as np
import pandas as pd

from core.enums import Column, Metric, PredictionKey, Strategy
from core.strategies import get_all_strategies


def backtest_strategy(df, ticker, strategy_func, n_periods=8):
    """
    Backtest a single strategy on historical data.

    Args:
        df: Full stock data
        ticker: Stock ticker to test
        strategy_func: Strategy function to test
        n_periods: Number of historical periods to test

    Returns:
        dict: Backtest results with predictions and actual values
    """
    ticker_col = Column.TICKER.value
    index_col = Column.INDEX.value
    eps_col = Metric.EPS.value

    ticker_data = df[df[ticker_col] == ticker].copy()
    ticker_data = ticker_data.sort_values(index_col)

    if len(ticker_data) < n_periods + 8:  # Need enough data for backtesting
        return None

    eps_data = ticker_data[eps_col].dropna()
    if len(eps_data) < n_periods + 4:  # Need enough EPS data
        return None

    results = []

    # For each of the last n_periods quarters
    for i in range(n_periods):
        # Create training data (everything up to this point)
        cutoff_idx = len(ticker_data) - n_periods + i
        train_data = ticker_data.iloc[:cutoff_idx].copy()

        # Get the actual next quarter value
        if cutoff_idx < len(ticker_data):
            actual_eps = ticker_data.iloc[cutoff_idx][eps_col]
            actual_index = ticker_data.iloc[cutoff_idx][index_col]

            if pd.isna(actual_eps):
                continue

            # Make prediction using strategy
            prediction = strategy_func(train_data)

            if prediction is not None:
                predicted_eps = prediction[PredictionKey.PREDICTED_EPS]
                latest_eps = prediction[PredictionKey.LATEST_EPS]

                # Calculate prediction error
                abs_error = abs(predicted_eps - actual_eps)
                pct_error = (
                    (abs_error / abs(actual_eps)) * 100
                    if actual_eps != 0
                    else float("inf")
                )

                # Calculate actual growth rate
                actual_growth = (
                    ((actual_eps - latest_eps) / abs(latest_eps)) * 100
                    if latest_eps != 0
                    else 0
                )
                predicted_growth = prediction[PredictionKey.PREDICTED_GROWTH]
                growth_error = abs(predicted_growth - actual_growth)

                results.append(
                    {
                        "period": i,
                        "actual_eps": actual_eps,
                        PredictionKey.PREDICTED_EPS: predicted_eps,
                        PredictionKey.LATEST_EPS: latest_eps,
                        "actual_growth": actual_growth,
                        PredictionKey.PREDICTED_GROWTH: predicted_growth,
                        "abs_error": abs_error,
                        "pct_error": pct_error,
                        "growth_error": growth_error,
                        "actual_index": actual_index,
                        PredictionKey.CONFIDENCE: prediction.get(
                            PredictionKey.CONFIDENCE, "Unknown"
                        ),
                    }
                )

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    pred_eps_key = PredictionKey.PREDICTED_EPS
    summary = {
        "n_predictions": len(results_df),
        "mean_abs_error": results_df["abs_error"].mean(),
        "mean_pct_error": results_df["pct_error"].mean(),
        "median_pct_error": results_df["pct_error"].median(),
        "mean_growth_error": results_df["growth_error"].mean(),
        "predictions_within_10pct": (results_df["pct_error"] <= 10).sum(),
        "predictions_within_20pct": (results_df["pct_error"] <= 20).sum(),
        "rmse": np.sqrt(
            ((results_df[pred_eps_key] - results_df["actual_eps"]) ** 2).mean()
        ),
        "accuracy_score": None,  # Will calculate below
        "results_detail": results_df,
    }

    # Calculate accuracy score (lower is better)
    # Weighted combination of percentage error and growth prediction error
    accuracy_score = (
        0.6 * summary["mean_pct_error"] + 0.4 * summary["mean_growth_error"]
    )
    summary["accuracy_score"] = accuracy_score

    return summary


def run_backtest_comparison(df, ticker="AAPL", n_periods=8):
    """
    Run backtesting comparison across all strategies.

    Args:
        df: Stock data
        ticker: Stock ticker to test on
        n_periods: Number of historical periods to test

    Returns:
        dict: Results for each strategy
    """
    print(f"\n Running backtest comparison for {ticker}")
    print(f" Testing last {n_periods} quarters")
    print("=" * 60)

    strategies = get_all_strategies()
    results = {}

    for strategy_name, strategy_func in strategies.items():
        print(f"\n Testing strategy: {strategy_name}")

        try:
            result = backtest_strategy(df, ticker, strategy_func, n_periods)

            if result is not None:
                results[strategy_name] = result

                # Print summary for this strategy
                print(f"   Predictions made: {result['n_predictions']}")
                print(f"   Mean % Error: {result['mean_pct_error']:.1f}%")
                print(f"   Median % Error: {result['median_pct_error']:.1f}%")
                print(f"   Growth Error: {result['mean_growth_error']:.1f}%")
                print(f"   Accuracy Score: {result['accuracy_score']:.1f}")
                within_10 = result["predictions_within_10pct"]
                within_20 = result["predictions_within_20pct"]
                n_pred = result["n_predictions"]
                print(f"   Within 10%: {within_10}/{n_pred}")
                print(f"   Within 20%: {within_20}/{n_pred}")

            else:
                print("   Strategy failed - insufficient data")
                results[strategy_name] = None

        except Exception as e:
            print(f"   Strategy failed with error: {str(e)}")
            results[strategy_name] = None

    return results


def analyze_backtest_results(results):
    """
    Analyze and rank the backtest results.

    Args:
        results: Results from run_backtest_comparison

    Returns:
        pandas.DataFrame: Ranked strategy performance
    """
    if not results:
        return None

    # Filter out failed strategies
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        return None

    # Create comparison dataframe
    comparison_data = []

    for strategy_name, result in valid_results.items():
        n_pred = result["n_predictions"]
        comparison_data.append(
            {
                "Strategy": strategy_name,
                "Accuracy Score": result["accuracy_score"],
                "Mean % Error": result["mean_pct_error"],
                "Median % Error": result["median_pct_error"],
                "Growth Error": result["mean_growth_error"],
                "RMSE": result["rmse"],
                "Within 10%": result["predictions_within_10pct"],
                "Within 20%": result["predictions_within_20pct"],
                "Total Predictions": n_pred,
                "Success Rate 10%": (result["predictions_within_10pct"] / n_pred * 100)
                if n_pred > 0
                else 0,
                "Success Rate 20%": (result["predictions_within_20pct"] / n_pred * 100)
                if n_pred > 0
                else 0,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by accuracy score (lower is better)
    comparison_df = comparison_df.sort_values("Accuracy Score")

    return comparison_df


def get_best_strategy(results):
    """
    Get the name of the best performing strategy.

    Args:
        results: Results from run_backtest_comparison

    Returns:
        str: Name of best strategy
    """
    comparison_df = analyze_backtest_results(results)

    if comparison_df is None or len(comparison_df) == 0:
        return Strategy.WEIGHTED_GROWTH.value  # Fallback to original

    return comparison_df.iloc[0]["Strategy"]


def print_detailed_comparison(results):
    """
    Print detailed comparison of all strategies.

    Args:
        results: Results from run_backtest_comparison
    """
    comparison_df = analyze_backtest_results(results)

    if comparison_df is None:
        print("No valid results to compare")
        return

    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE RANKING")
    print("=" * 80)

    for idx, row in comparison_df.iterrows():
        rank = idx + 1

        print(f"\n{rank}. {row['Strategy'].upper()}")
        print(f"   Accuracy Score: {row['Accuracy Score']:.1f} (lower is better)")
        mean_err = row["Mean % Error"]
        median_err = row["Median % Error"]
        print(f"   Mean Error: {mean_err:.1f}% | Median: {median_err:.1f}%")
        print(f"   Growth Prediction Error: {row['Growth Error']:.1f}%")
        rate_10 = row["Success Rate 10%"]
        rate_20 = row["Success Rate 20%"]
        print(f"   Success Rates: {rate_10:.0f}% (10%) | {rate_20:.0f}% (20%)")
        print(f"   RMSE: {row['RMSE']:.4f}")

    print("\n" + "=" * 80)
    print(f"WINNER: {comparison_df.iloc[0]['Strategy'].upper()}")
    print(f"Best accuracy score: {comparison_df.iloc[0]['Accuracy Score']:.1f}")
    print("=" * 80)


# Multi-ticker backtesting functions


def run_multi_ticker_backtest(df, n_periods=8, min_data_points=12):
    """
    Run backtesting on all tickers to find the best strategy for each.

    Args:
        df: Stock data with all tickers
        n_periods: Number of historical periods to test
        min_data_points: Minimum data points required for backtesting

    Returns:
        tuple: (ticker_results dict, ticker_best_strategies dict)
    """
    print("\n" + "=" * 80)
    print("MULTI-TICKER STRATEGY OPTIMIZATION")
    print("=" * 80)

    ticker_col = Column.TICKER.value
    eps_col = Metric.EPS.value

    available_tickers = sorted(df[ticker_col].unique())
    print(f"Found {len(available_tickers)} tickers in dataset")
    print(f"Testing {n_periods} historical quarters per ticker")
    print(f"Minimum {min_data_points} data points required for testing")

    strategies = get_all_strategies()
    ticker_results = {}
    ticker_best_strategies = {}

    successful_tests = 0
    failed_tests = 0

    for i, ticker in enumerate(available_tickers, 1):
        print(f"\n{'='*60}")
        print(f"[{i:2d}/{len(available_tickers)}] Testing {ticker}")
        print(f"{'='*60}")

        # Check if ticker has enough data
        ticker_data = df[df[ticker_col] == ticker]
        eps_data = ticker_data[eps_col].dropna()

        if len(eps_data) < min_data_points:
            print(
                f"   Insufficient data: {len(eps_data)} quarters "
                f"(need {min_data_points})"
            )
            failed_tests += 1
            continue

        # Run backtest for each strategy on this ticker
        ticker_strategy_results = {}

        for strategy_name, strategy_func in strategies.items():
            try:
                result = backtest_strategy(df, ticker, strategy_func, n_periods)
                if result is not None:
                    ticker_strategy_results[strategy_name] = result
                    print(
                        f"   {strategy_name:15} -> "
                        f"Accuracy: {result['accuracy_score']:.1f}"
                    )
                else:
                    msg = "Failed (insufficient data)"
                    print(f"   {strategy_name:15} -> {msg}")
            except Exception as e:
                print(f"   {strategy_name:15} -> Error: {str(e)}")

        if ticker_strategy_results:
            # Find best strategy for this ticker
            best_strategy = get_best_strategy(ticker_strategy_results)
            best_score = ticker_strategy_results[best_strategy]["accuracy_score"]

            ticker_results[ticker] = ticker_strategy_results
            ticker_best_strategies[ticker] = best_strategy

            print(f"   WINNER: {best_strategy.upper()} (Score: {best_score:.1f})")
            successful_tests += 1
        else:
            print(f"   All strategies failed for {ticker}")
            failed_tests += 1

    print(f"\n{'='*80}")
    print("MULTI-TICKER BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    total = successful_tests + failed_tests
    print(f"Success rate: {(successful_tests / total * 100):.1f}%")

    return ticker_results, ticker_best_strategies


def analyze_ticker_strategies(ticker_best_strategies):
    """
    Analyze the distribution of best strategies across tickers.

    Args:
        ticker_best_strategies: Mapping of ticker to best strategy

    Returns:
        pandas.DataFrame: Strategy distribution analysis
    """
    if not ticker_best_strategies:
        print("No strategy data to analyze")
        return None

    print(f"\n{'='*80}")
    print("STRATEGY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")

    # Count strategy usage
    strategy_counts = pd.Series(ticker_best_strategies).value_counts()
    total_tickers = len(ticker_best_strategies)

    print(f"Strategy performance across {total_tickers} tickers:")
    print()

    for i, (strategy, count) in enumerate(strategy_counts.items(), 1):
        percentage = (count / total_tickers) * 100

        print(f"{i}. {strategy.upper()}")
        print(f"   Tickers: {count}")
        print(f"   Percentage: {percentage:.1f}%")

        # Show some example tickers
        example_tickers = [
            ticker
            for ticker, strat in ticker_best_strategies.items()
            if strat == strategy
        ]
        if len(example_tickers) <= 5:
            print(f"   Examples: {', '.join(example_tickers)}")
        else:
            extra = len(example_tickers) - 5
            print(f"   Examples: {', '.join(example_tickers[:5])} (+{extra} more)")
        print()

    # Create summary DataFrame
    strategy_df = pd.DataFrame(
        {
            "Strategy": strategy_counts.index,
            "Ticker_Count": strategy_counts.values,
            "Percentage": (strategy_counts.values / total_tickers * 100).round(1),
        }
    )

    return strategy_df


def save_ticker_strategy_mapping(
    ticker_best_strategies, filename="config/ticker_strategy_mapping.json"
):
    """
    Save the ticker-to-strategy mapping to a JSON file.

    Args:
        ticker_best_strategies: Mapping of ticker to best strategy
        filename: Output filename
    """
    if not ticker_best_strategies:
        print("No mapping data to save")
        return

    try:
        with open(filename, "w") as f:
            json.dump(ticker_best_strategies, f, indent=2, sort_keys=True)

        print(f"\nTicker strategy mapping saved to: {filename}")
        print(f"Mapped {len(ticker_best_strategies)} tickers")

    except Exception as e:
        print(f"Failed to save mapping: {str(e)}")


def load_ticker_strategy_mapping(
    filename="config/ticker_strategy_mapping.json", verbose=False
):
    """
    Load the ticker-to-strategy mapping from a JSON file.

    Args:
        filename: Input filename
        verbose: Whether to print status messages (default: False)

    Returns:
        dict: Mapping of ticker to best strategy, or None if loading fails
    """
    try:
        with open(filename, "r") as f:
            mapping = json.load(f)

        if verbose:
            print(f"Loaded ticker strategy mapping from: {filename}")
            print(f"Found mappings for {len(mapping)} tickers")
        return mapping

    except FileNotFoundError:
        if verbose:
            print(f"Mapping file not found: {filename}")
        return None
    except Exception as e:
        if verbose:
            print(f"Failed to load mapping: {str(e)}")
        return None


def get_ticker_strategy(
    ticker, ticker_strategy_mapping=None, default_strategy=None, verbose=False
):
    """
    Get the best strategy for a specific ticker.

    Args:
        ticker: Stock ticker symbol
        ticker_strategy_mapping: Pre-loaded mapping
        default_strategy: Fallback strategy if ticker not found
        verbose: Whether to print status messages (default: False)

    Returns:
        str: Best strategy name for the ticker
    """
    if default_strategy is None:
        default_strategy = Strategy.SEASONAL.value

    if ticker_strategy_mapping is None:
        ticker_strategy_mapping = load_ticker_strategy_mapping(verbose=verbose)

    if ticker_strategy_mapping and ticker in ticker_strategy_mapping:
        return ticker_strategy_mapping[ticker]

    # Fallback to default strategy
    if verbose:
        print(f"No specific strategy for {ticker}, using {default_strategy}")
    return default_strategy


def display_detailed_ticker_results(ticker_results, top_n=10):
    """
    Display detailed results for top performing tickers.

    Args:
        ticker_results: Full backtest results for all tickers
        top_n: Number of top tickers to show details for
    """
    if not ticker_results:
        print("No detailed results to display")
        return

    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS FOR TOP {top_n} TICKERS")
    print(f"{'='*80}")

    # Calculate best scores for each ticker
    ticker_best_scores = {}
    for ticker, strategies in ticker_results.items():
        if strategies:
            best_score = min(result["accuracy_score"] for result in strategies.values())
            ticker_best_scores[ticker] = best_score

    # Sort by best score (lower is better)
    sorted_tickers = sorted(ticker_best_scores.items(), key=lambda x: x[1])[:top_n]

    for i, (ticker, best_score) in enumerate(sorted_tickers, 1):
        print(f"\n{i:2d}. {ticker} (Best Score: {best_score:.1f})")
        print("-" * 40)

        # Show all strategy results for this ticker
        ticker_strategies = ticker_results[ticker]
        sorted_strategies = sorted(
            ticker_strategies.items(), key=lambda x: x[1]["accuracy_score"]
        )

        for j, (strategy, result) in enumerate(sorted_strategies, 1):
            marker = "* " if j == 1 else "  "
            score = result["accuracy_score"]
            err = result["mean_pct_error"]
            w20 = result["predictions_within_20pct"]
            n = result["n_predictions"]
            print(
                f"   {marker}{strategy:15} | Score: {score:5.1f} | "
                f"Mean Error: {err:5.1f}% | 20%: {w20}/{n}"
            )


if __name__ == "__main__":
    # Import here to avoid circular imports when used as library
    from dashboard.data_utils import calculate_qoq_changes, load_data

    print("Starting Multi-Ticker Strategy Optimization...")

    # Load data
    df = load_data("data/StockData_Indexed.xlsx")
    if df is None:
        print("Failed to load data")
        exit(1)

    # Calculate QoQ changes
    print("Calculating QoQ changes...")
    df = calculate_qoq_changes(df)

    # Run multi-ticker backtest
    ticker_results, ticker_best_strategies = run_multi_ticker_backtest(
        df, n_periods=8, min_data_points=12
    )

    if ticker_best_strategies:
        # Analyze strategy distribution
        strategy_distribution = analyze_ticker_strategies(ticker_best_strategies)

        # Display detailed results
        display_detailed_ticker_results(ticker_results, top_n=10)

        # Save the mapping
        save_ticker_strategy_mapping(ticker_best_strategies)

        print("\nMulti-ticker optimization complete!")
        print(f"Generated strategies for {len(ticker_best_strategies)} tickers")

    else:
        print("No successful strategy mappings generated")
