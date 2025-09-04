#!/usr/bin/env python3
"""
Test script for ticker-specific prediction system.
"""

from data_utils import calculate_qoq_changes, load_data, predict_next_eps


def test_ticker_specific_predictions():
    """Test the new ticker-specific prediction system."""
    print("🧪 Testing Ticker-Specific Prediction System")
    print("=" * 60)

    # Load data
    print("📊 Loading data...")
    df = load_data("StockData_Indexed.xlsx")
    if df is None:
        print("❌ Failed to load data")
        return False

    # Calculate QoQ changes
    print("📈 Calculating QoQ changes...")
    df = calculate_qoq_changes(df)

    # Test predictions for a few sample tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    print(f"\n🎯 Testing predictions for {len(test_tickers)} tickers:")
    print("-" * 60)

    successful_predictions = 0
    total_tests = len(test_tickers)

    for ticker in test_tickers:
        print(f"\n📈 Testing {ticker}:")

        try:
            prediction = predict_next_eps(df, ticker)

            if prediction is not None:
                print(f"   ✅ Strategy: {prediction['methodology']}")
                print(f"   📊 Predicted EPS: ${prediction['predicted_eps']:.2f}")
                print(f"   📈 Growth: {prediction['predicted_growth']:+.1f}%")
                print(f"   🎯 Confidence: {prediction['confidence']}")
                print(f"   📉 Volatility: {prediction['volatility']:.1f}%")

                # Check if EPS_TTM prediction is available
                if prediction.get("predicted_eps_ttm") is not None:
                    print(
                        f"   💰 Predicted EPS TTM: "
                        f"${prediction['predicted_eps_ttm']:.2f}"
                    )

                # Check if price prediction is available
                if prediction.get("predicted_price") is not None:
                    print(f"   💵 Predicted Price: ${prediction['predicted_price']:.2f}")
                    print(f"   📊 Current P/E: {prediction['current_multiple']:.1f}x")

                successful_predictions += 1

            else:
                print("   ❌ No prediction possible (insufficient data)")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful predictions: {successful_predictions}/{total_tests}")
    print(f"📈 Success rate: {(successful_predictions/total_tests*100):.1f}%")

    if successful_predictions == total_tests:
        print("🎉 All tests passed! Ticker-specific system is working.")
        return True
    elif successful_predictions > 0:
        print("⚠️  Some tests passed, system partially working.")
        return True
    else:
        print("❌ All tests failed, system needs debugging.")
        return False


def show_strategy_mapping_sample():
    """Show a sample of the strategy mapping."""
    print(f"\n{'='*60}")
    print("📋 STRATEGY MAPPING SAMPLE")
    print(f"{'='*60}")

    try:
        from multi_ticker_backtest import load_ticker_strategy_mapping

        mapping = load_ticker_strategy_mapping()

        if mapping:
            # Show first 10 mappings
            sample_tickers = list(mapping.keys())[:10]

            for ticker in sample_tickers:
                strategy = mapping[ticker].replace("_", " ").title()
                print(f"   {ticker:6} -> {strategy}")

            print(f"\n   ... and {len(mapping) - 10} more tickers")

            # Show strategy distribution
            from collections import Counter

            strategy_counts = Counter(mapping.values())

            print("\n📊 Strategy Distribution:")
            for strategy, count in strategy_counts.most_common():
                percentage = (count / len(mapping)) * 100
                strategy_name = strategy.replace("_", " ").title()
                print(f"   {strategy_name:20} {count:3d} tickers ({percentage:4.1f}%)")

        else:
            print("❌ No strategy mapping found")

    except Exception as e:
        print(f"❌ Error loading strategy mapping: {str(e)}")


if __name__ == "__main__":
    # Test the ticker-specific prediction system
    success = test_ticker_specific_predictions()

    # Show strategy mapping sample
    show_strategy_mapping_sample()

    if success:
        print("\n🎉 Ticker-specific prediction system is ready!")
    else:
        print("\n❌ System needs debugging before deployment")
