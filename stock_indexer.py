import pandas as pd


def process_stock_data(file_path):
    """
    Read Excel file and add reverse index to each ticker's data.
    Each ticker gets its own independent index from 1 to N
    (where N = number of records for that ticker).
    The most recent record for each ticker gets the highest index.

    Args:
        file_path (str): Path to the Excel file

    Returns:
        pandas.DataFrame: Processed dataframe with reverse index column named 'Index'
    """
    print("Reading Excel file...")
    df = pd.read_excel(file_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("Adding reverse index to each ticker...")

    max_records_per_ticker = df.groupby("Ticker").size().max()

    def add_ticker_index(group):
        n_records = len(group)
        start_index = max_records_per_ticker - n_records + 1
        end_index = max_records_per_ticker + 1
        return pd.Series(range(start_index, end_index), index=group.index)

    df["Index"] = df.groupby("Ticker", sort=False).apply(add_ticker_index).values

    remaining_cols = [
        col
        for col in df.columns
        if col not in ["Ticker", "Index"] and "Unnamed:" not in str(col)
    ]
    cols = ["Ticker", "Index"] + remaining_cols
    df_indexed = df[cols]

    final_unnamed = [col for col in df_indexed.columns if "Unnamed:" in str(col)]
    if final_unnamed:
        print(f"ERROR: Unnamed columns still present after processing: {final_unnamed}")
        df_indexed = df_indexed.drop(columns=final_unnamed)
        print(f"Force dropped remaining unnamed columns: {final_unnamed}")

    print(f"Final processed columns after cleanup: {list(df_indexed.columns)}")

    ticker_counts = df_indexed["Ticker"].value_counts()
    print(f"\nUnique tickers: {len(ticker_counts)}")
    print(f"Average records per ticker: {ticker_counts.mean():.1f}")
    print(f"Global maximum index: {max_records_per_ticker} (all tickers end here)")

    print("\n=== DUPLICATE INDEX CHECK ===")
    duplicate_check_passed = True
    for ticker in df_indexed["Ticker"].unique():
        ticker_data = df_indexed[df_indexed["Ticker"] == ticker]
        index_counts = ticker_data["Index"].value_counts()
        duplicates = index_counts[index_counts > 1]
        if len(duplicates) > 0:
            print(f"ERROR: {ticker} has duplicate indices: {duplicates.index.tolist()}")
            duplicate_check_passed = False
        else:
            ticker_max = ticker_data["Index"].max()
            if ticker_max != max_records_per_ticker:
                print(
                    f"ERROR: {ticker} should end at {max_records_per_ticker} "
                    f"but ends at {ticker_max}"
                )
                duplicate_check_passed = False

    if duplicate_check_passed:
        print("✓ No duplicate indices found within any ticker!")
        print("✓ All tickers correctly end at the global maximum index!")
    else:
        print("✗ INDEX ISSUES DETECTED!")
        return None

    print(f"\n=== INDEX RANGES BY TICKER (All end at {max_records_per_ticker}) ===")
    for ticker in sorted(df_indexed["Ticker"].unique()):
        ticker_data = df_indexed[df_indexed["Ticker"] == ticker]
        min_idx = ticker_data["Index"].min()
        max_idx = ticker_data["Index"].max()
        count = len(ticker_data)
        expected_start = max_records_per_ticker - count + 1
        print(
            f"{ticker}: {count} records, Index range: {min_idx} to {max_idx} "
            f"(expected: {expected_start} to {max_records_per_ticker})"
        )

    print("\nSample of processed data:")
    print(df_indexed.head(15))

    return df_indexed


def verify_core_data_integrity(original_df, processed_df):
    """
    Verify that core data columns were not changed.
    Only checks: Ticker, Report, EPS, Revenue, Price, and DivAmt

    Args:
        original_df (pandas.DataFrame): Original dataframe
        processed_df (pandas.DataFrame): Processed dataframe with index column

    Returns:
        bool: True if core data integrity is maintained
    """
    print("\n=== CORE DATA VERIFICATION ===")
    core_columns = ["Ticker", "Report", "EPS", "Revenue", "Price", "DivAmt"]

    if len(original_df) != len(processed_df):
        print(
            f"ERROR: Row count changed! Original: {len(original_df)}, "
            f"Processed: {len(processed_df)}"
        )
        return False

    for col in core_columns:
        if col not in original_df.columns:
            print(f"INFO: Column '{col}' not found in original data")
            continue
        if col not in processed_df.columns:
            print(f"ERROR: Column '{col}' missing from processed data!")
            return False
        orig_series = original_df[col].fillna("__MISSING__")
        proc_series = processed_df[col].fillna("__MISSING__")
        if not orig_series.equals(proc_series):
            print(f"ERROR: Data in column '{col}' was modified!")
            return False
        else:
            print(f"✓ Column '{col}' unchanged")

    print("✓ All core data columns verified - no changes detected!")
    return True


def verify_index_integrity(df):
    """
    Additional verification specifically for the Index column.
    Ensures each ticker ends at the global maximum and has no duplicates.

    Args:
        df (pandas.DataFrame): Processed dataframe with Index column

    Returns:
        bool: True if index integrity is maintained
    """
    print("\n=== INDEX INTEGRITY VERIFICATION ===")
    global_max = df["Index"].max()
    issues_found = False

    for ticker in df["Ticker"].unique():
        ticker_data = df[df["Ticker"] == ticker].copy()
        duplicate_indices = ticker_data["Index"].duplicated()
        if duplicate_indices.any():
            dup_values = ticker_data[duplicate_indices]["Index"].values
            print(f"ERROR: {ticker} has duplicate indices: {dup_values}")
            issues_found = True
            continue
        ticker_max = ticker_data["Index"].max()
        if ticker_max != global_max:
            print(
                f"ERROR: {ticker} should end at {global_max} but ends at {ticker_max}"
            )
            issues_found = True
            continue
        n_records = len(ticker_data)
        expected_start = global_max - n_records + 1
        expected_indices = list(range(expected_start, global_max + 1))
        actual_indices = sorted(ticker_data["Index"].values)
        if expected_indices != actual_indices:
            print(f"ERROR: {ticker} index sequence incorrect!")
            print(f"  Expected: {expected_indices}")
            print(f"  Actual: {actual_indices}")
            issues_found = True
            continue
        print(
            f"✓ {ticker}: {n_records} records, indices {expected_start} to {global_max}"
        )

    if not issues_found:
        print(
            "✓ All tickers end at the global maximum with correct sequential indices!"
        )
        return True
    else:
        print("✗ INDEX INTEGRITY ISSUES FOUND!")
        return False


def save_processed_data(df, output_path):
    """
    Save the processed dataframe to Excel file.

    Args:
        df (pandas.DataFrame): Processed dataframe
        output_path (str): Path for output file
    """
    print(f"\nSaving processed data to {output_path}...")
    df.to_excel(output_path, index=False)
    print("File saved successfully!")


def main():
    input_file = "StockData.xlsx"
    output_file = "StockData_Indexed.xlsx"

    try:
        print("Reading original Excel file...")
        original_df_raw = pd.read_excel(input_file)
        print(f"Original file columns: {list(original_df_raw.columns)}")

        unnamed_cols = [
            col for col in original_df_raw.columns if "Unnamed:" in str(col)
        ]
        if unnamed_cols:
            print(f"Found unnamed columns in original: {unnamed_cols}")
            original_df = original_df_raw.drop(columns=unnamed_cols)
            print(f"Cleaned original columns: {list(original_df.columns)}")
        else:
            print("No unnamed columns found in original")
            original_df = original_df_raw

        processed_df = process_stock_data(input_file)
        if processed_df is None:
            print("\nERROR: Processing failed due to index issues! Stopping execution.")
            return

        print(f"Final processed columns: {list(processed_df.columns)}")

        if not verify_core_data_integrity(original_df, processed_df):
            print("\nERROR: Core data verification failed! Stopping execution.")
            return

        if not verify_index_integrity(processed_df):
            print("\nERROR: Index integrity verification failed! Stopping execution.")
            return

        save_processed_data(processed_df, output_file)

        print("\n=== FINAL OUTPUT VERIFICATION ===")
        print(f"Output file columns: {list(processed_df.columns)}")

        final_unnamed = [col for col in processed_df.columns if "Unnamed:" in str(col)]
        if final_unnamed:
            print(f"WARNING: Unnamed columns still present: {final_unnamed}")
        else:
            print("✓ Confirmed: No unnamed columns in final output")

        print("\nExample of indexed data for specific tickers:")
        problem_tickers = ["LULU", "LUV"]
        example_tickers = []

        for ticker in problem_tickers:
            if ticker in processed_df["Ticker"].unique():
                example_tickers.append(ticker)

        if len(example_tickers) < 3:
            for ticker in processed_df["Ticker"].unique():
                if ticker not in example_tickers:
                    example_tickers.append(ticker)
                    if len(example_tickers) >= 3:
                        break

        for ticker in example_tickers:
            ticker_data = processed_df[processed_df["Ticker"] == ticker]
            print(f"\n{ticker} ({len(ticker_data)} records):")
            print(ticker_data[["Ticker", "Index", "Report", "EPS", "Revenue"]].head(8))
            if len(ticker_data) > 8:
                print("...")
                print(
                    ticker_data[["Ticker", "Index", "Report", "EPS", "Revenue"]].tail(3)
                )

        ticker_counts = processed_df["Ticker"].value_counts()
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total unique tickers: {len(ticker_counts)}")
        print(f"Total records: {len(processed_df)}")
        print(f"Average records per ticker: {ticker_counts.mean():.1f}")

        print("\nTop 5 tickers by number of records:")
        for i, (ticker, count) in enumerate(ticker_counts.head().items(), 1):
            print(f"  {i}. {ticker}: {count} records (indices 1 to {count})")

        print("\n✓ Processing completed successfully!")
        print("✓ Each ticker now has independent sequential indices from 1 to N")
        print("✓ No duplicate indices within any ticker")
        print(
            "✓ Most recent record for each ticker has the highest index for that ticker"
        )

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
