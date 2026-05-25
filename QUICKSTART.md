# Quick Start Guide

## Installation

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

Copy example configs before your first run (real configs stay local):

```bash
cp config.example.yaml config.yaml
```

Or if you prefer using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Test the implementation:**

```bash
python test_implementation.py
```

This will validate that all modules are working correctly.

## Running the Analysis

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/portfolio_optimization.ipynb
```

Then run all cells in order.

### Option 2: Jupyter Lab

```bash
jupyter lab
```

Navigate to `notebooks/portfolio_optimization.ipynb` and run.

## Expected Outputs

After running the notebook, you'll find the following in the `outputs/` directory:

- `portfolio_weights.csv` - Weight allocations for all three portfolios
- `portfolio_metrics.csv` - Performance metrics comparison
- `weights_comparison.png` - Bar chart comparing weights
- `efficient_frontier.png` - Mean-Variance efficient frontier
- `correlation_heatmap.png` - Asset correlation matrix
- `bl_view_impact.png` - Black-Litterman view impact visualization
- `cumulative_returns.png` - Historical cumulative returns
- `summary_report.txt` - Text summary of results

## Key Configuration

Edit `config.yaml` to customize:

- **Tickers and before weights** - Portfolio composition
- **Date range** - Historical data period
- **Constraints** - Max weight, turnover limits
- **Black-Litterman parameters** - Risk aversion (delta), tau, analyst views
- **Risk-free rate** - For Sharpe ratio calculation

## Modifying Analyst Views

To change the Black-Litterman views, edit the `black_litterman.views` section in `config.yaml`:

```yaml
views:
  TICKER:
    type: "absolute"  # or "relative"
    return: 0.04      # Expected excess return (e.g., +4%)
    confidence: 0.0004  # Lower = higher confidence
```

**Confidence guidelines:**
- High confidence: 0.0004 (2% std dev)
- Medium confidence: 0.0009 (3% std dev)
- Low confidence: 0.0016 (4% std dev)

## Troubleshooting

### Import errors
Make sure you've installed all requirements: `pip install -r requirements.txt`

### Data download issues
If yfinance fails, you can load data from CSV by modifying the notebook to use:
```python
prices = data_loader.load_from_csv('path/to/prices.csv')
```

### Optimization fails
Check that:
- Constraints are not too restrictive
- Data quality is good (no missing values)
- Lambda grid is appropriate for your data

## Next Steps

1. Run the notebook and review results
2. Experiment with different constraints and views
3. Perform sensitivity analysis on key parameters
4. Compare optimization methods via backtests and Excel export
5. Document findings for your team or report

## Questions?

Refer to the inline documentation in:
- Module docstrings in `src/`
- Markdown cells in the notebook
- Comments in `config.yaml`
