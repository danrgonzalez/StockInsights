#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis Runner
Runs the stock indexer script first, then launches the Streamlit dashboard.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def check_file_exists(filepath):
    """Check if a file exists and return True/False"""
    return Path(filepath).exists()


def run_stock_indexer():
    """Run the stock indexer script"""
    print("=" * 60)
    print("STEP 1: Running Stock Indexer")
    print("=" * 60)

    # Check if the indexer script exists
    if not check_file_exists("stock_indexer.py"):
        print("ERROR: stock_indexer.py not found in current directory!")
        print("Please ensure stock_indexer.py is in the same folder as this script.")
        return False

    # Check if the input file exists
    if not check_file_exists("StockData.xlsx"):
        print("ERROR: StockData.xlsx not found in current directory!")
        print("Please ensure StockData.xlsx is in the same folder as this script.")
        return False

    try:
        # Run the stock indexer
        print("Running stock_indexer.py...")
        result = subprocess.run(
            [sys.executable, "stock_indexer.py"],
            capture_output=True, text=True, check=True
        )

        # Print the output from the indexer
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors from indexer:")
            print(result.stderr)

        # Check if the output file was created
        if check_file_exists("StockData_Indexed.xlsx"):
            print("StockData_Indexed.xlsx created successfully!")
            return True
        else:
            print("ERROR: StockData_Indexed.xlsx was not created!")
            return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR running stock indexer: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running stock indexer: {e}")
        return False


def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n" + "=" * 60)
    print("STEP 2: Launching Streamlit Dashboard")
    print("=" * 60)

    # Check if the dashboard script exists
    if not check_file_exists("stock_dashboard.py"):
        print("ERROR: stock_dashboard.py not found in current directory!")
        print("Please ensure stock_dashboard.py is in the same folder as this script.")
        return False

    try:
        print("Starting Streamlit dashboard...")
        print("The dashboard will open in your default web browser.")
        print("Press Ctrl+C to stop the dashboard when you're done.")
        print("\nDashboard URL will typically be: http://localhost:8501")
        print("-" * 60)

        # Give user a moment to read the message
        time.sleep(2)

        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "stock_dashboard.py"])

    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
        return True
    except FileNotFoundError:
        print("ERROR: Streamlit not installed!")
        print("Please install Streamlit with: pip install streamlit")
        return False
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    required_packages = [
        'pandas',
        'streamlit',
        'plotly',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")

    if missing_packages:
        print(f"\nERROR: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with one of the following commands:")
        print(f"conda install {' '.join(missing_packages)}")
        print("or, if you prefer pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✓ All dependencies are installed!")
    return True


def main():
    """Main function to run the complete workflow"""
    print("Stock Analysis Workflow Runner")
    print("=" * 60)

    # Check current directory
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")

    # List relevant files in current directory
    print("\nFiles in current directory:")
    for file in [
        "stock_indexer.py",
        "stock_dashboard.py",
        "StockData.xlsx",
        "StockData_Indexed.xlsx"
    ]:
        status = "✓" if check_file_exists(file) else "✗"
        print(f"  {status} {file}")

    print("\n" + "=" * 60)

    # Check dependencies first
    if not check_dependencies():
        print("\nWorkflow aborted due to missing dependencies.")
        return

    print("\n" + "=" * 60)

    # Step 1: Run the stock indexer
    indexer_success = run_stock_indexer()

    if not indexer_success:
        print("\nWorkflow aborted due to indexer failure.")
        return

    # Ask user if they want to proceed to dashboard
    print("\n" + "=" * 60)
    response = input("Stock indexing completed! Launch dashboard? (y/n): ").lower().strip()

    if response in ['y', 'yes', '']:
        # Step 2: Launch the dashboard
        launch_dashboard()
    else:
        print("Dashboard launch skipped. You can run it manually with:")
        print("streamlit run stock_dashboard.py")

    print("\nWorkflow completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check that all required files are in the current directory.")
