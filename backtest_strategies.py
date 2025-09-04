import numpy as np
import pandas as pd

from data_utils import calculate_qoq_changes, load_data
from strategies import get_all_strategies


def backtest_strategy(df, ticker, strategy_func, n_periods=8):
    """
    Backtest a single strategy on historical data.

    Args:
        df (pandas.DataFrame): Full stock data
        ticker (str): Stock ticker to test
        strategy_func (function): Strategy function to test
        n_periods (int): Number of historical periods to test

    Returns:
        dict: Backtest results with predictions and actual values
    """
    ticker_data = df[df["Ticker"] == ticker].copy()
    ticker_data = ticker_data.sort_values("Index")

    if len(ticker_data) < n_periods + 8:  # Need enough data for backtesting
        return None

    eps_data = ticker_data["EPS"].dropna()
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
            actual_eps = ticker_data.iloc[cutoff_idx]["EPS"]
            actual_index = ticker_data.iloc[cutoff_idx]["Index"]

            if pd.isna(actual_eps):
                continue

            # Make prediction using strategy
            prediction = strategy_func(train_data)

            if prediction is not None:
                predicted_eps = prediction["predicted_eps"]
                latest_eps = prediction["latest_eps"]

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
                predicted_growth = prediction["predicted_growth"]
                growth_error = abs(predicted_growth - actual_growth)

                results.append(
                    {
                        "period": i,
                        "actual_eps": actual_eps,
                        "predicted_eps": predicted_eps,
                        "latest_eps": latest_eps,
                        "actual_growth": actual_growth,
                        "predicted_growth": predicted_growth,
                        "abs_error": abs_error,
                        "pct_error": pct_error,
                        "growth_error": growth_error,
                        "actual_index": actual_index,
                        "confidence": prediction.get("confidence", "Unknown"),
                    }
                )

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    summary = {
        "n_predictions": len(results_df),
        "mean_abs_error": results_df["abs_error"].mean(),
        "mean_pct_error": results_df["pct_error"].mean(),
        "median_pct_error": results_df["pct_error"].median(),
        "mean_growth_error": results_df["growth_error"].mean(),
        "predictions_within_10pct": (results_df["pct_error"] <= 10).sum(),
        "predictions_within_20pct": (results_df["pct_error"] <= 20).sum(),
        "rmse": np.sqrt(
            ((results_df["predicted_eps"] - results_df["actual_eps"]) ** 2).mean()
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
        df (pandas.DataFrame): Stock data
        ticker (str): Stock ticker to test on
        n_periods (int): Number of historical periods to test

    Returns:
        dict: Results for each strategy
    """
    print(f"\n🚀 Running backtest comparison for {ticker}")
    print(f"📊 Testing last {n_periods} quarters")
    print("=" * 60)

    strategies = get_all_strategies()
    results = {}

    for strategy_name, strategy_func in strategies.items():
        print(f"\n📈 Testing strategy: {strategy_name}")

        try:
            backtest_result = backtest_strategy(df, ticker, strategy_func, n_periods)

            if backtest_result is not None:
                results[strategy_name] = backtest_result

                # Print summary for this strategy
                print(f"   ✅ Predictions made: {backtest_result['n_predictions']}")
                print(f"   📍 Mean % Error: {backtest_result['mean_pct_error']:.1f}%")
                print(
                    f"   🎯 Median % Error: {backtest_result['median_pct_error']:.1f}%"
                )
                print(f"   📈 Growth Error: {backtest_result['mean_growth_error']:.1f}%")
                print(f"   🎲 Accuracy Score: {backtest_result['accuracy_score']:.1f}")
                print(
                    f"   ✨ Within 10%: {backtest_result['predictions_within_10pct']}/{backtest_result['n_predictions']}"
                )
                print(
                    f"   ⭐ Within 20%: {backtest_result['predictions_within_20pct']}/{backtest_result['n_predictions']}"
                )

            else:
                print(f"   ❌ Strategy failed - insufficient data")
                results[strategy_name] = None

        except Exception as e:
            print(f"   ❌ Strategy failed with error: {str(e)}")
            results[strategy_name] = None

    return results


def analyze_backtest_results(results):
    """
    Analyze and rank the backtest results.

    Args:
        results (dict): Results from run_backtest_comparison

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
                "Total Predictions": result["n_predictions"],
                "Success Rate 10%": (
                    result["predictions_within_10pct"] / result["n_predictions"] * 100
                )
                if result["n_predictions"] > 0
                else 0,
                "Success Rate 20%": (
                    result["predictions_within_20pct"] / result["n_predictions"] * 100
                )
                if result["n_predictions"] > 0
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
        results (dict): Results from run_backtest_comparison

    Returns:
        str: Name of best strategy
    """
    comparison_df = analyze_backtest_results(results)

    if comparison_df is None or len(comparison_df) == 0:
        return "weighted_growth"  # Fallback to original

    return comparison_df.iloc[0]["Strategy"]


def print_detailed_comparison(results):
    """
    Print detailed comparison of all strategies.

    Args:
        results (dict): Results from run_backtest_comparison
    """
    comparison_df = analyze_backtest_results(results)

    if comparison_df is None:
        print("❌ No valid results to compare")
        return

    print("\n" + "=" * 80)
    print("🏆 STRATEGY PERFORMANCE RANKING")
    print("=" * 80)

    for idx, row in comparison_df.iterrows():
        rank = idx + 1
        emoji = (
            "🥇"
            if rank == 1
            else "🥈"
            if rank == 2
            else "🥉"
            if rank == 3
            else f"{rank}️⃣"
        )

        print(f"\n{emoji} {row['Strategy'].upper()}")
        print(f"   🎯 Accuracy Score: {row['Accuracy Score']:.1f} (lower is better)")
        print(
            f"   📊 Mean Error: {row['Mean % Error']:.1f}% | Median: {row['Median % Error']:.1f}%"
        )
        print(f"   📈 Growth Prediction Error: {row['Growth Error']:.1f}%")
        print(
            f"   ✅ Success Rates: {row['Success Rate 10%']:.0f}% (±10%) | {row['Success Rate 20%']:.0f}% (±20%)"
        )
        print(f"   📐 RMSE: {row['RMSE']:.4f}")

    print("\n" + "=" * 80)
    print(f"🏆 WINNER: {comparison_df.iloc[0]['Strategy'].upper()}")
    print(f"💡 Best accuracy score: {comparison_df.iloc[0]['Accuracy Score']:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    # Load data and run backtest
    df = load_data("StockData_Indexed.xlsx")

    if df is not None:
        # Calculate QoQ changes
        df = calculate_qoq_changes(df)

        # Run backtest comparison
        results = run_backtest_comparison(df, ticker="AAPL", n_periods=8)

        # Print detailed results
        print_detailed_comparison(results)

        # Show best strategy
        best_strategy = get_best_strategy(results)
        print(f"\n🎯 Recommended strategy: {best_strategy}")

    else:
        print("❌ Failed to load data for backtesting")
