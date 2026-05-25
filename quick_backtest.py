"""
Quick Backtest - Uses cached data from notebook run
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import backtester
from src import backtester

print('\n' + '='*70)
print('QUICK BACKTEST ANALYSIS')
print('='*70)

# Load the weights from CSV
weights_df = pd.read_csv('outputs/portfolio_weights.csv', index_col=0)
tickers = weights_df.index.tolist()

w_before = weights_df['Before'].values
w_mv = weights_df['Mean-Variance'].values  
w_bl = weights_df['Black-Litterman'].values

# Clean near-zero weights
w_mv = np.where(np.abs(w_mv) < 1e-10, 0.0, w_mv)
w_bl = np.where(np.abs(w_bl) < 1e-10, 0.0, w_bl)

print(f'\n✓ Loaded weights for {len(tickers)} tickers')
print(f'  Tickers: {", ".join(tickers)}')

# Load price/returns data (this should be cached from earlier notebook run)
print('\n📊 Loading historical data...')
from src import data_loader

try:
    prices = data_loader.load_prices(tickers, '2023-01-23', '2026-01-23')
    returns = data_loader.compute_returns(prices, method='log')
    print(f'✓ Loaded {len(returns)} trading days')
except Exception as e:
    print(f'Error loading data: {e}')
    sys.exit(1)

# Run backtests
print('\n📈 Running backtests...')
rf = 0.045

before_backtest = backtester.backtest_portfolio(
    weights=w_before, tickers=tickers, prices=prices, returns=returns,
    risk_free_rate=rf, initial_capital=10000
)
print('  ✓ Before portfolio')

mv_backtest = backtester.backtest_portfolio(
    weights=w_mv, tickers=tickers, prices=prices, returns=returns,
    risk_free_rate=rf, initial_capital=10000
)
print('  ✓ Mean-Variance portfolio')

bl_backtest = backtester.backtest_portfolio(
    weights=w_bl, tickers=tickers, prices=prices, returns=returns,
    risk_free_rate=rf, initial_capital=10000
)
print('  ✓ Black-Litterman portfolio')

# Calculate metrics
results_dict = {
    "Before": before_backtest,
    "Mean-Variance": mv_backtest,
    "Black-Litterman": bl_backtest
}

# Manual summary (avoid plotting issues)
print('\n' + '='*70)
print('PERFORMANCE SUMMARY')
print('='*70)

rows = []
for name, res in results_dict.items():
    m = res['metrics']
    ann_return = m['Annual Return']
    ann_vol = m['Annual Volatility']
    sharpe = m['Sharpe Ratio']
    
    # Sortino
    sortino = backtester.sortino_ratio(res['portfolio_returns'], risk_free_rate=rf)
    
    # Max drawdown
    max_dd, _, _ = backtester.max_drawdown(res['portfolio_value'])
    
    final_value = res['portfolio_value'].iloc[-1]
    total_return = (final_value / 10000 - 1)
    
    print(f'\n{name}:')
    print(f'  Annual Return:    {ann_return:>8.2%}')
    print(f'  Annual Volatility: {ann_vol:>8.2%}')
    print(f'  Sharpe Ratio:     {sharpe:>8.3f}')
    print(f'  Sortino Ratio:    {sortino:>8.3f}')
    print(f'  Max Drawdown:     {max_dd:>8.2%}')
    print(f'  Final Value:      ${final_value:>8,.2f}')
    print(f'  Total Return:     {total_return:>8.2%}')
    
    rows.append({
        'Strategy': name,
        'Annual Return': ann_return,
        'Annual Vol': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD': max_dd,
        'Final Value': final_value,
        'Total Return': total_return
    })

summary_df = pd.DataFrame(rows).set_index('Strategy')

# Save to CSV
summary_df.to_csv('outputs/backtest_summary.csv')
print(f'\n✓ Saved results to outputs/backtest_summary.csv')

# Create plot
print('\n📊 Creating visualization...')

# Get cumulative returns
common_dates = returns.index
cum_returns = {}
for name, res in results_dict.items():
    pv = res['portfolio_value'].reindex(common_dates)
    cum_returns[name] = (pv / 10000 - 1) * 100

cum_df = pd.DataFrame(cum_returns)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Cumulative returns
cum_df.plot(ax=ax1, linewidth=2.5, alpha=0.8)
ax1.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
ax1.grid(False)  # Remove gridlines
ax1.legend(loc='best')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

# Metrics table (with Risk Adj, without Sharpe and Max DD)
ax2.axis('off')
# Calculate risk-adjusted return for display
risk_adj_col = (summary_df['Annual Return'] - rf) / summary_df['Annual Vol']
summary_df['Risk Adj'] = risk_adj_col

display_data = summary_df[['Annual Return', 'Annual Vol', 'Risk Adj', 'Sortino']].copy()
display_data['Annual Return'] = display_data['Annual Return'].map(lambda x: f'{x:.2%}')
display_data['Annual Vol'] = display_data['Annual Vol'].map(lambda x: f'{x:.2%}')
display_data['Risk Adj'] = display_data['Risk Adj'].map(lambda x: f'{x:.3f}')
display_data['Sortino'] = display_data['Sortino'].map(lambda x: f'{x:.3f}')

table = ax2.table(
    cellText=display_data.values,
    rowLabels=display_data.index,
    colLabels=display_data.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.8)
ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/backtest_comparison.png', dpi=300, bbox_inches='tight')
print('✓ Saved plot to outputs/backtest_comparison.png')

print('\n' + '='*70)
print('✅ BACKTEST COMPLETE!')
print('='*70)
