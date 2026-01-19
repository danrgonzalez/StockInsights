#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Analysis Runner
Runs the stock indexer script first, then launches the Streamlit dashboard.
"""

import os
import subprocess
import sys
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
    if not check_file_exists("scripts/indexer.py"):
        print("ERROR: scripts/indexer.py not found!")
        print("Please ensure you're running from the project root directory.")
        return False

    # Check if the input file exists
    if not check_file_exists("data/StockData.xlsx"):
        print("ERROR: data/StockData.xlsx not found!")
        print("Please ensure data/StockData.xlsx exists.")
        return False

    try:
        # Run the stock indexer
        print("Running scripts/indexer.py...")
        result = subprocess.run(
            [sys.executable, "scripts/indexer.py"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Print the output from the indexer
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors from indexer:")
            print(result.stderr)

        # Check if the output file was created
        if check_file_exists("data/StockData_Indexed.xlsx"):
            print("data/StockData_Indexed.xlsx created successfully!")
            return True
        else:
            print("ERROR: data/StockData_Indexed.xlsx was not created!")
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
    if not check_file_exists("dashboard/app.py"):
        print("ERROR: dashboard/app.py not found!")
        print("Please ensure you're running from the project root directory.")
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
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])

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

    required_packages = ["pandas", "streamlit", "plotly", "numpy"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  {package} - MISSING")

    if missing_packages:
        print(f"\nERROR: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with one of the following commands:")
        print(f"conda install {' '.join(missing_packages)}")
        print("or, if you prefer pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("All dependencies are installed!")
    return True


def main():
    """Main function to run the complete workflow"""
    print("Stock Analysis Workflow Runner")
    print("=" * 60)

    # Check current directory
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")

    # List relevant files in current directory
    print("\nFiles check:")
    for file in [
        "scripts/indexer.py",
        "dashboard/app.py",
        "data/StockData.xlsx",
        "data/StockData_Indexed.xlsx",
    ]:
        status = "OK" if check_file_exists(file) else "MISSING"
        print(f"  [{status}] {file}")

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

    # Step 2: Launch the dashboard immediately
    print("\nData preprocessing completed successfully!")
    launch_dashboard()

    print("\nWorkflow completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check that all required files are in the current directory.")
