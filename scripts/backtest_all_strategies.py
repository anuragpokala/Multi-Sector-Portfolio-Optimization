"""
Comprehensive Backtesting Script for All Portfolio Strategies

This script backtests all 6 portfolio strategies using exact allocations
from the screenshots/config and generates results for Excel export.

Strategies:
1. Before (baseline)
2. Mean-Variance
3. Black-Litterman  
4. Risk Parity
5. HRP (Hierarchical Risk Parity)
6. Hybrid BL-HRP

Output: Daily cumulative returns for each strategy, ready for Excel export.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import yaml
from src import data_loader, backtester


def load_exact_weights_from_config(config_path='config.yaml'):
    """
    Load exact allocations from config.yaml.
    
    Returns
    -------
    dict
        Dictionary with strategy names as keys and weight dictionaries as values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    strategies = {
        'Before': config['before_weights_exact'],
        'Mean-Variance': config['mean_variance_exact'],
        'Black-Litterman': config['black_litterman_exact'],
        'Risk Parity': config['risk_parity_exact'],
        'HRP': config['hrp_exact'],
        'Hybrid': config['hybrid_exact']
    }
    
    tickers = config['tickers']
    rf = config['risk_free_rate']
    
    return strategies, tickers, rf, config


def weights_dict_to_array(weights_dict, tickers):
    """Convert weight dictionary to numpy array matching ticker order."""
    return np.array([weights_dict[ticker] for ticker in tickers])


def backtest_all_strategies():
    """
    Run comprehensive backtest for all 6 strategies using exact weights.
    
    Returns
    -------
    dict
        Backtest results for each strategy
    """
    print("="*70)
    print("COMPREHENSIVE PORTFOLIO BACKTESTING")
    print("="*70)
    print("\nLoading configuration and data...")
    
    # Load config and exact weights
    strategies, tickers, rf, config = load_exact_weights_from_config()
    
    # Load historical data
    prices = data_loader.load_prices(
        tickers, 
        config['data']['start_date'],
        config['data']['end_date']
    )
    
    returns = data_loader.compute_returns(prices)
    
    print(f"Data loaded: {len(prices)} days, {len(tickers)} stocks")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Convert weight dictionaries to arrays
    weights_arrays = {}
    for strategy_name, weights_dict in strategies.items():
        weights_arrays[strategy_name] = weights_dict_to_array(weights_dict, tickers)
        
        # Verify weights
        total = weights_arrays[strategy_name].sum()
        print(f"\n{strategy_name} weights sum: {total:.4f}")
        for ticker, weight in weights_dict.items():
            print(f"  {ticker}: {weight:.2%}")
    
    # Run backtests
    print("\n" + "="*70)
    print("RUNNING BACKTESTS")
    print("="*70)
    
    backtest_results = {}
    
    for strategy_name, weights in weights_arrays.items():
        print(f"\nBacktesting {strategy_name}...")
        
        result = backtester.backtest_portfolio(
            weights=weights,
            tickers=tickers,
            prices=prices,
            returns=returns,
            risk_free_rate=rf,
            initial_capital=10000
        )
        
        backtest_results[strategy_name] = result
        
        # Print summary
        final_value = result['portfolio_value'].iloc[-1]
        total_return = (final_value / result['initial_capital'] - 1) * 100
        annual_return = result['metrics']['Annual Return'] * 100
        annual_vol = result['metrics']['Annual Volatility'] * 100
        
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Annual Return: {annual_return:.2f}%")
        print(f"  Annual Volatility: {annual_vol:.2f}%")
    
    return backtest_results, prices.index


def prepare_excel_data(backtest_results, dates):
    """
    Prepare data for Excel export with 3 separate tabs.
    
    Returns
    -------
    dict
        Three dataframes: 'BL_MV_Comparison', 'RP_HRP_Comparison', 'Hybrid_Comparison'
    """
    # Calculate cumulative returns (as percentages) for all strategies
    cum_returns = {}
    
    for strategy_name, result in backtest_results.items():
        portfolio_value = result['portfolio_value']
        initial_value = portfolio_value.iloc[0]
        cum_return_pct = ((portfolio_value / initial_value) - 1) * 100
        cum_returns[strategy_name] = cum_return_pct
    
    # Create three comparison dataframes
    
    # Tab 1: Before, Mean-Variance, Black-Litterman
    bl_mv_df = pd.DataFrame({
        'Date': dates,
        'Before': cum_returns['Before'].values,
        'Mean-Variance': cum_returns['Mean-Variance'].values,
        'Black-Litterman': cum_returns['Black-Litterman'].values
    })
    
    # Tab 2: Before, Risk Parity, HRP
    rp_hrp_df = pd.DataFrame({
        'Date': dates,
        'Before': cum_returns['Before'].values,
        'Risk Parity': cum_returns['Risk Parity'].values,
        'HRP': cum_returns['HRP'].values
    })
    
    # Tab 3: Before, Hybrid
    hybrid_df = pd.DataFrame({
        'Date': dates,
        'Before': cum_returns['Before'].values,
        'Hybrid': cum_returns['Hybrid'].values
    })
    
    return {
        'BL_MV_Comparison': bl_mv_df,
        'RP_HRP_Comparison': rp_hrp_df,
        'Hybrid_Comparison': hybrid_df
    }


if __name__ == '__main__':
    print("Starting comprehensive backtest...\n")
    
    # Run backtests
    results, dates = backtest_all_strategies()
    
    # Prepare Excel data
    print("\n" + "="*70)
    print("PREPARING EXCEL DATA")
    print("="*70)
    
    excel_data = prepare_excel_data(results, dates)
    
    print(f"\nCreated {len(excel_data)} comparison tables:")
    for tab_name, df in excel_data.items():
        print(f"  - {tab_name}: {len(df)} rows, {len(df.columns)} columns")
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print("\nTo generate Excel file, run:")
    print("  python scripts/generate_excel.py")
    print("\nOr use the data directly from excel_data variable")
    
    # Save results for Excel export script
    import pickle
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/backtest_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'dates': dates, 'excel_data': excel_data}, f)
    
    print("\nResults saved to outputs/backtest_results.pkl")
