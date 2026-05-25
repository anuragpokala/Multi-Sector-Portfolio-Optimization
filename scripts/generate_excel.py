#!/usr/bin/env python3
"""
Convenience script to generate Excel file with backtest results.

This script:
1. Runs comprehensive backtests for all 6 strategies
2. Generates Excel file with 3 tabs for easy graphing
3. Displays instructions for creating Virginia Tech themed graphs

Usage:
    python scripts/generate_excel.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.backtest_all_strategies import backtest_all_strategies
from src.excel_exporter import create_formatted_excel, print_graphing_instructions


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("EXCEL GENERATION WORKFLOW")
    print("="*70)
    print("\nThis script will:")
    print("  1. Run backtests for all 6 portfolio strategies")
    print("  2. Generate Excel file with 3 comparison tabs")
    print("  3. Display graphing instructions")
    print("\n" + "="*70 + "\n")
    
    # Step 1: Run backtests
    print("STEP 1: Running comprehensive backtests...")
    print("-" * 70)
    backtest_results, dates = backtest_all_strategies()
    
    # Step 2: Create Excel file
    print("\n" + "="*70)
    print("STEP 2: Generating Excel file...")
    print("-" * 70)
    
    output_path = 'outputs/consumers_optimization.xlsx'
    excel_path = create_formatted_excel(backtest_results, dates, output_path)
    
    # Step 3: Display instructions
    print_graphing_instructions()
    
    # Summary
    print("\n" + "="*70)
    print("SUCCESS! Workflow Complete")
    print("="*70)
    print(f"\n✓ Excel file created: {excel_path}")
    print("✓ Ready for graphing in Excel")
    print("\nNext steps:")
    print("  1. Open the Excel file")
    print("  2. Follow the instructions above to create graphs")
    print("  3. Apply Virginia Tech colors as specified")
    print("\nFor detailed instructions, see: GRAPHING_GUIDE.md")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
