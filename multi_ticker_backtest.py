import json

import pandas as pd

from backtest_strategies import backtest_strategy, get_best_strategy
from data_utils import calculate_qoq_changes, load_data
from strategies import get_all_strategies


def run_multi_ticker_backtest(df, n_periods=8, min_data_points=12):
    """
    Run backtesting on all available tickers to find the best strategy for each.

    Args:
        df (pandas.DataFrame): Stock data with all tickers
        n_periods (int): Number of historical periods to test
        min_data_points (int): Minimum data points required for backtesting

    Returns:
        dict: Results for each ticker with their best strategy
    """
    print("\n" + "=" * 80)
    print("🔬 MULTI-TICKER STRATEGY OPTIMIZATION")
    print("=" * 80)

    available_tickers = sorted(df["Ticker"].unique())
    print(f"📊 Found {len(available_tickers)} tickers in dataset")
    print(f"🎯 Testing {n_periods} historical quarters per ticker")
    print(f"📋 Minimum {min_data_points} data points required for testing")

    strategies = get_all_strategies()
    ticker_results = {}
    ticker_best_strategies = {}

    successful_tests = 0
    failed_tests = 0

    for i, ticker in enumerate(available_tickers, 1):
        print(f"\n{'='*60}")
        print(f"📈 [{i:2d}/{len(available_tickers)}] Testing {ticker}")
        print(f"{'='*60}")

        # Check if ticker has enough data
        ticker_data = df[df["Ticker"] == ticker]
        eps_data = ticker_data["EPS"].dropna()

        if len(eps_data) < min_data_points:
            print(
                f"   ⚠️  Insufficient data: {len(eps_data)} quarters "
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
                        f"   ✅ {strategy_name:15} -> "
                        f"Accuracy: {result['accuracy_score']:.1f}"
                    )
                else:
                    print(f"   ❌ {strategy_name:15} -> Failed (insufficient data)")
            except Exception as e:
                print(f"   ❌ {strategy_name:15} -> Error: {str(e)}")

        if ticker_strategy_results:
            # Find best strategy for this ticker
            best_strategy = get_best_strategy(ticker_strategy_results)
            best_score = ticker_strategy_results[best_strategy]["accuracy_score"]

            ticker_results[ticker] = ticker_strategy_results
            ticker_best_strategies[ticker] = best_strategy

            print(f"   🏆 WINNER: {best_strategy.upper()} (Score: {best_score:.1f})")
            successful_tests += 1
        else:
            print(f"   ❌ All strategies failed for {ticker}")
            failed_tests += 1

    print(f"\n{'='*80}")
    print("📊 MULTI-TICKER BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful tests: {successful_tests}")
    print(f"❌ Failed tests: {failed_tests}")
    print(
        f"📈 Success rate: {(successful_tests/(successful_tests+failed_tests)*100):.1f}%"
    )

    return ticker_results, ticker_best_strategies


def analyze_ticker_strategies(ticker_best_strategies):
    """
    Analyze the distribution of best strategies across tickers.

    Args:
        ticker_best_strategies (dict): Mapping of ticker to best strategy

    Returns:
        pandas.DataFrame: Strategy distribution analysis
    """
    if not ticker_best_strategies:
        print("❌ No strategy data to analyze")
        return None

    print(f"\n{'='*80}")
    print("📈 STRATEGY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")

    # Count strategy usage
    strategy_counts = pd.Series(ticker_best_strategies).value_counts()
    total_tickers = len(ticker_best_strategies)

    print(f"📊 Strategy performance across {total_tickers} tickers:")
    print()

    for i, (strategy, count) in enumerate(strategy_counts.items(), 1):
        percentage = (count / total_tickers) * 100
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"

        print(f"{emoji} {strategy.upper()}")
        print(f"   💼 Tickers: {count}")
        print(f"   📊 Percentage: {percentage:.1f}%")

        # Show some example tickers
        example_tickers = [
            ticker
            for ticker, strat in ticker_best_strategies.items()
            if strat == strategy
        ]
        if len(example_tickers) <= 5:
            print(f"   🎯 Examples: {', '.join(example_tickers)}")
        else:
            print(
                f"   🎯 Examples: {', '.join(example_tickers[:5])} "
                f"(+{len(example_tickers)-5} more)"
            )
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
    ticker_best_strategies, filename="ticker_strategy_mapping.json"
):
    """
    Save the ticker-to-strategy mapping to a JSON file.

    Args:
        ticker_best_strategies (dict): Mapping of ticker to best strategy
        filename (str): Output filename
    """
    if not ticker_best_strategies:
        print("❌ No mapping data to save")
        return

    try:
        with open(filename, "w") as f:
            json.dump(ticker_best_strategies, f, indent=2, sort_keys=True)

        print(f"\n💾 Ticker strategy mapping saved to: {filename}")
        print(f"📊 Mapped {len(ticker_best_strategies)} tickers")

    except Exception as e:
        print(f"❌ Failed to save mapping: {str(e)}")


def load_ticker_strategy_mapping(filename="ticker_strategy_mapping.json"):
    """
    Load the ticker-to-strategy mapping from a JSON file.

    Args:
        filename (str): Input filename

    Returns:
        dict: Mapping of ticker to best strategy, or None if loading fails
    """
    try:
        with open(filename, "r") as f:
            mapping = json.load(f)

        print(f"📂 Loaded ticker strategy mapping from: {filename}")
        print(f"📊 Found mappings for {len(mapping)} tickers")
        return mapping

    except FileNotFoundError:
        print(f"⚠️  Mapping file not found: {filename}")
        return None
    except Exception as e:
        print(f"❌ Failed to load mapping: {str(e)}")
        return None


def get_ticker_strategy(
    ticker, ticker_strategy_mapping=None, default_strategy="seasonal"
):
    """
    Get the best strategy for a specific ticker.

    Args:
        ticker (str): Stock ticker symbol
        ticker_strategy_mapping (dict, optional): Pre-loaded mapping
        default_strategy (str): Fallback strategy if ticker not found

    Returns:
        str: Best strategy name for the ticker
    """
    if ticker_strategy_mapping is None:
        ticker_strategy_mapping = load_ticker_strategy_mapping()

    if ticker_strategy_mapping and ticker in ticker_strategy_mapping:
        return ticker_strategy_mapping[ticker]

    # Fallback to default strategy
    print(f"⚠️  No specific strategy found for {ticker}, using {default_strategy}")
    return default_strategy


def display_detailed_ticker_results(ticker_results, top_n=10):
    """
    Display detailed results for top performing tickers.

    Args:
        ticker_results (dict): Full backtest results for all tickers
        top_n (int): Number of top tickers to show details for
    """
    if not ticker_results:
        print("❌ No detailed results to display")
        return

    print(f"\n{'='*80}")
    print(f"🔍 DETAILED RESULTS FOR TOP {top_n} TICKERS")
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
            emoji = "🏆" if j == 1 else "🥈" if j == 2 else "🥉" if j == 3 else "  "
            print(
                f"   {emoji} {strategy:15} | Score: {result['accuracy_score']:5.1f} | "
                f"Mean Error: {result['mean_pct_error']:5.1f}% | "
                f"±20%: {result['predictions_within_20pct']}/{result['n_predictions']}"
            )


if __name__ == "__main__":
    print("🚀 Starting Multi-Ticker Strategy Optimization...")

    # Load data
    df = load_data("StockData_Indexed.xlsx")
    if df is None:
        print("❌ Failed to load data")
        exit(1)

    # Calculate QoQ changes
    print("📊 Calculating QoQ changes...")
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

        print("\n🎯 Multi-ticker optimization complete!")
        print(f"📊 Generated strategies for {len(ticker_best_strategies)} tickers")

    else:
        print("❌ No successful strategy mappings generated")
