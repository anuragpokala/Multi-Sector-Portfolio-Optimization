"""
Excel Export Module

Generates Excel file with backtest results formatted for easy graph creation.
Creates 3 tabs for different strategy comparisons.
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime


def create_excel_output(excel_data, output_path='outputs/consumers_optimization.xlsx'):
    """
    Create Excel file with 3 tabs for backtest comparisons.
    
    Parameters
    ----------
    excel_data : dict
        Dictionary with 3 dataframes:
        - 'BL_MV_Comparison': Before, Mean-Variance, Black-Litterman
        - 'RP_HRP_Comparison': Before, Risk Parity, HRP
        - 'Hybrid_Comparison': Before, Hybrid BL-HRP
    output_path : str
        Output file path
        
    Returns
    -------
    str
        Path to created Excel file
    """
    print(f"\nCreating Excel file: {output_path}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write each dataframe to a separate sheet
        for sheet_name, df in excel_data.items():
            # Format Date column properly
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.date
            
            # Write to Excel
            df_copy.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get the worksheet
            worksheet = writer.sheets[sheet_name]
            
            # Format header row
            for cell in worksheet[1]:
                cell.font = Font(bold=True, size=11)
                cell.alignment = Alignment(horizontal='center')
                cell.fill = PatternFill(start_color='D9D9D9', end_color='D9D9D9', fill_type='solid')
            
            # Set column widths
            worksheet.column_dimensions['A'].width = 12  # Date
            for col in ['B', 'C', 'D', 'E']:
                if col in [cell.column_letter for row in worksheet.iter_rows() for cell in row]:
                    worksheet.column_dimensions[col].width = 16
            
            # Format percentage columns (all except Date)
            for row in worksheet.iter_rows(min_row=2, min_col=2, max_col=worksheet.max_column):
                for cell in row:
                    cell.number_format = '0.00'  # Two decimal places
                    cell.alignment = Alignment(horizontal='right')
            
            print(f"  ✓ Created sheet: {sheet_name}")
    
    print(f"\n✓ Excel file created successfully: {output_path}")
    return output_path


def create_formatted_excel(backtest_results, dates, output_path='outputs/consumers_optimization.xlsx'):
    """
    Create formatted Excel file directly from backtest results.
    
    Parameters
    ----------
    backtest_results : dict
        Dictionary of backtest results from backtester
    dates : pd.DatetimeIndex
        Date index for the data
    output_path : str
        Output file path
        
    Returns
    -------
    str
        Path to created Excel file
    """
    # Calculate cumulative returns as percentages
    cum_returns = {}
    portfolio_dates = None
    
    for strategy_name, result in backtest_results.items():
        portfolio_value = result['portfolio_value']
        initial_value = portfolio_value.iloc[0]
        cum_return_pct = ((portfolio_value / initial_value) - 1) * 100
        cum_returns[strategy_name] = cum_return_pct.values
        
        # Use portfolio dates from the backtest (which has correct length)
        if portfolio_dates is None:
            portfolio_dates = portfolio_value.index
    
    # Create three comparison dataframes using portfolio dates
    excel_data = {
        'BL_MV_Comparison': pd.DataFrame({
            'Date': portfolio_dates,
            'Before': cum_returns['Before'],
            'Mean-Variance': cum_returns['Mean-Variance'],
            'Black-Litterman': cum_returns['Black-Litterman']
        }),
        'RP_HRP_Comparison': pd.DataFrame({
            'Date': portfolio_dates,
            'Before': cum_returns['Before'],
            'Risk Parity': cum_returns['Risk Parity'],
            'HRP': cum_returns['HRP']
        }),
        'Hybrid_Comparison': pd.DataFrame({
            'Date': portfolio_dates,
            'Before': cum_returns['Before'],
            'Hybrid': cum_returns['Hybrid']
        })
    }
    
    return create_excel_output(excel_data, output_path)


def print_graphing_instructions():
    """Print instructions for creating graphs in Excel."""
    print("\n" + "="*70)
    print("GRAPHING INSTRUCTIONS")
    print("="*70)
    
    instructions = """
To create graphs in Excel:

1. Open consumers_optimization.xlsx

2. For BL/MV Comparison Graph:
   - Go to "BL_MV_Comparison" tab
   - Select columns A through D (Date, Before, Mean-Variance, Black-Litterman)
   - Select all data rows (including header)
   - Insert → Line Chart → Line with Markers (or Line)
   
3. For RP/HRP Comparison Graph:
   - Go to "RP_HRP_Comparison" tab
   - Select columns A through D (Date, Before, Risk Parity, HRP)
   - Select all data rows (including header)
   - Insert → Line Chart → Line with Markers (or Line)
   
4. For Hybrid Comparison Graph:
   - Go to "Hybrid_Comparison" tab
   - Select columns A through C (Date, Before, Hybrid)
   - Select all data rows (including header)
   - Insert → Line Chart → Line with Markers (or Line)

5. Color the lines with Virginia Tech colors:
   - Click on a line in the chart
   - Right-click → Format Data Series → Line Color
   - Use these colors:
     * Brown: RGB(99, 0, 49) or #630031
     * Maroon: RGB(134, 31, 65) or #861F41
     * Orange: RGB(232, 119, 34) or #E87722
     * Red: RGB(207, 69, 32) or #CF4520
     * Black: RGB(0, 0, 0) or #000000

Suggested color assignments:
  - Before portfolio: Maroon (#861F41)
  - Mean-Variance: Orange (#E87722)
  - Black-Litterman: Red (#CF4520)
  - Risk Parity: Brown (#630031)
  - HRP: Black (#000000)
  - Hybrid: Red (#E87722)

For more details, see GRAPHING_GUIDE.md
"""
    print(instructions)


if __name__ == '__main__':
    # This can be run standalone if backtest results are pickled
    import pickle
    import os
    
    if os.path.exists('outputs/backtest_results.pkl'):
        print("Loading backtest results...")
        with open('outputs/backtest_results.pkl', 'rb') as f:
            data = pickle.load(f)
        
        results = data['results']
        dates = data['dates']
        
        # Create Excel file
        excel_path = create_formatted_excel(results, dates)
        
        # Print instructions
        print_graphing_instructions()
        
        print(f"\n✓ Done! Excel file ready: {excel_path}")
    else:
        print("Error: Run scripts/backtest_all_strategies.py first to generate backtest results.")
