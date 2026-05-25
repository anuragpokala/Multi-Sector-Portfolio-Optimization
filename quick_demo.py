#!/usr/bin/env python3
"""
Quick demonstration of the portfolio optimization pipeline.
This uses sample data - run the full notebook for real market data analysis.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from src import estimators, mv_optimizer, bl_model, metrics, constraints

def main():
    print('\n' + '#'*70)
    print('# Consumer Portfolio Optimization - Quick Demo')
    print('#'*70)

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tickers = config['tickers']
    before_weights = np.array([config['before_weights'][t] for t in tickers])

    print('\n📊 Portfolio Setup:')
    print(f'   Tickers: {" ".join(tickers)}')
    print(f'   Risk-free rate: {config["risk_free_rate"]:.1%}')
    print(f'   Max weight constraint: {config["constraints"]["max_weight"]:.0%}')

    # Generate sample data (normally would use historical prices)
    print('\n⚙️  Generating sample data for demo...')
    np.random.seed(42)
    n_days = 252 * 3
    # Add some realistic expected returns
    daily_returns = np.array([0.0002, 0.0001, 0.0003, 0.0002, 0.0002, 0.0001, 0.0000, 0.0003])
    returns_data = np.random.randn(n_days, 8) * 0.01 + daily_returns
    returns_df = pd.DataFrame(returns_data, columns=tickers)

    # Estimate parameters
    Sigma, shrinkage = estimators.estimate_covariance_shrinkage(returns_df)
    mu_hist = estimators.estimate_expected_returns(returns_df)

    print(f'   ✓ Covariance shrinkage: {shrinkage:.2%}')
    print(f'   ✓ Expected returns range: {mu_hist.min():.1%} to {mu_hist.max():.1%}')

    # Mean-Variance Optimization
    print('\n🔵 Mean-Variance Optimization:')
    mv_results = mv_optimizer.optimize_mean_variance(
        mu=mu_hist.values,
        Sigma=Sigma.values,
        constraints_config=config['constraints'],
        rf=config['risk_free_rate'],
        lambda_grid=np.logspace(-1, 1, 20),
        w_before=before_weights
    )

    w_mv = mv_results['weights']
    print(f'   Return: {mv_results["return"]:.2%}')
    print(f'   Volatility: {mv_results["volatility"]:.2%}')
    print(f'   Sharpe: {mv_results["sharpe"]:.3f}')

    # Black-Litterman Optimization  
    print('\n🟢 Black-Litterman Optimization:')
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
        rf=config['risk_free_rate'],
        w_before=before_weights
    )

    w_bl = bl_results['weights']
    print(f'   Return: {bl_results["return"]:.2%}')
    print(f'   Volatility: {bl_results["volatility"]:.2%}')
    print(f'   Sharpe: {bl_results["sharpe"]:.3f}')

    # Comparison
    print('\n📈 Portfolio Weights Comparison:')
    print('\n' + '-'*70)
    print(f"{'Ticker':<10} {'Before':<12} {'Mean-Variance':<15} {'Black-Litterman':<15}")
    print('-'*70)
    for i, ticker in enumerate(tickers):
        print(f'{ticker:<10} {before_weights[i]:>10.1%}  {w_mv[i]:>13.1%}  {w_bl[i]:>17.1%}')
    print('-'*70)

    # Calculate turnover
    mv_turnover = np.sum(np.abs(w_mv - before_weights))
    bl_turnover = np.sum(np.abs(w_bl - before_weights))

    print(f'\n📊 Key Metrics:')
    print(f'   MV Turnover: {mv_turnover:.1%}')
    print(f'   BL Turnover: {bl_turnover:.1%}')

    # Validate constraints
    is_valid_mv, _ = constraints.validate_weights(w_mv, config['constraints'], before_weights)
    is_valid_bl, _ = constraints.validate_weights(w_bl, config['constraints'], before_weights)
    print(f'   MV Constraints: {"✓ PASS" if is_valid_mv else "✗ FAIL"}')
    print(f'   BL Constraints: {"✓ PASS" if is_valid_bl else "✗ FAIL"}')

    print('\n' + '#'*70)
    print('✨ Demo complete! For full analysis with real data, run the notebook.')
    print('#'*70)
    print('\n📓 Next step: jupyter notebook notebooks/portfolio_optimization.ipynb\n')

if __name__ == '__main__':
    main()
