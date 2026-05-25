"""
Backtest Runner

Run historical backtests on Before, Mean-Variance, and Black-Litterman portfolios.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml

# Import modules
from src import data_loader, estimators, mv_optimizer, bl_model, backtester

print('\n' + '='*70)
print('PORTFOLIO BACKTEST ANALYSIS')
print('='*70)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

tickers = config['tickers']
before_weights = np.array([config['before_weights'][t] for t in tickers])
rf = config['risk_free_rate']

# Load historical data
print('\n📊 Loading historical price data...')
prices = data_loader.load_prices(tickers, config['data']['start_date'], config['data']['end_date'])
returns = data_loader.compute_returns(prices, method='log')

print(f'\n✓ Loaded {len(returns)} trading days of data')
print(f'  Date range: {returns.index[0].date()} to {returns.index[-1].date()}')

# Estimate parameters for optimization
print('\n⚙️  Estimating covariance and returns...')
Sigma, _ = estimators.estimate_covariance_shrinkage(returns)
mu_hist = estimators.estimate_expected_returns(returns, shrinkage=0.0)

# Run Mean-Variance optimization
print('\n🔵 Running Mean-Variance optimization...')
mv_results = mv_optimizer.optimize_mean_variance(
    mu=mu_hist.values,
    Sigma=Sigma.values,
    constraints_config=config['constraints'],
    rf=rf,
    lambda_grid=np.logspace(-1, 1, 20),
    w_before=before_weights
)
w_mv = mv_results['weights']

# Run Black-Litterman optimization
print('\n🟢 Running Black-Litterman optimization...')
pi = bl_model.compute_equilibrium_returns(
    Sigma.values, before_weights, config['black_litterman']['delta']
)
views_config = config['black_litterman']['views']
P, q, Omega = bl_model.encode_views(tickers, views_config, len(tickers))

bl_results = bl_model.optimize_black_litterman(
    pi=pi,
    Sigma=Sigma.values,
    P=P, q=q, Omega=Omega,
    tau=config['black_litterman']['tau'],
    constraints_config=config['constraints'],
    rf=rf,
    w_before=before_weights
)
w_bl = bl_results['weights']

# Run backtests
print('\n📈 Running backtests...')
print('='*70)

before_backtest = backtester.backtest_portfolio(
    weights=before_weights,
    tickers=tickers,
    prices=prices,
    returns=returns,
    risk_free_rate=rf,
    initial_capital=10000
)

mv_backtest = backtester.backtest_portfolio(
    weights=w_mv,
    tickers=tickers,
    prices=prices,
    returns=returns,
    risk_free_rate=rf,
    initial_capital=10000
)

bl_backtest = backtester.backtest_portfolio(
    weights=w_bl,
    tickers=tickers,
    prices=prices,
    returns=returns,
    risk_free_rate=rf,
    initial_capital=10000
)

# Create results dictionary
results_dict = {
    "Before (Baseline)": before_backtest,
    "Mean-Variance": mv_backtest,
    "Black-Litterman": bl_backtest
}

# Generate summary and plot
print('\n📊 Generating performance summary and visualization...')
summary_df = backtester.summarize_and_plot_strategies(
    results_dict,
    risk_free_rate=rf,
    trading_days=252,
    mar_annual=0.0,
    output_path='outputs/backtest_comparison.png'
)

print('\n' + '='*70)
print('BACKTEST SUMMARY (Raw Numbers)')
print('='*70)
print(summary_df.to_string())
print('='*70)

# Save summary to CSV
summary_df.to_csv('outputs/backtest_summary.csv')
print('\n✓ Results saved to outputs/backtest_summary.csv')

# Show final portfolio values
print('\n💰 Final Portfolio Values (started with $10,000):')
print('='*70)
for name, res in results_dict.items():
    final_value = res['portfolio_value'].iloc[-1]
    total_return = (final_value / res['initial_capital'] - 1) * 100
    print(f'  {name:20s}: ${final_value:,.2f}  ({total_return:+.2f}%)')
print('='*70)

# Show weight allocations
print('\n📋 Portfolio Weight Allocations:')
print('='*70)
weight_df = pd.DataFrame({
    'Before': before_weights,
    'Mean-Variance': w_mv,
    'Black-Litterman': w_bl
}, index=tickers)
print(weight_df.applymap(lambda x: f'{x:.2%}').to_string())
print('='*70)

print('\n✅ Backtest complete!')
print('📁 Check outputs/ directory for:')
print('   - backtest_comparison.png')
print('   - backtest_summary.csv')
