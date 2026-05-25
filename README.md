# Multi-Sector Portfolio Optimization

A Python framework for multi-sector equity portfolio optimization, backtesting, and reporting. Built for sector-based sleeves with configurable constraints, multiple optimization objectives, and optional news-sentiment or macro-driven views.

## Features

### Optimization methods

| Method | Description |
|--------|-------------|
| **Mean-Variance (Markowitz)** | Efficient frontier via risk–return tradeoff with Ledoit–Wolf covariance shrinkage |
| **Black-Litterman (analyst views)** | Bayesian blend of equilibrium returns and subjective views |
| **Black-Litterman + sentiment** | FinBERT + financial news APIs (Polygon, Finnhub) for data-driven views |
| **Black-Litterman + macro** | Interest-rate scenario views with sector-specific rate betas (financials) |
| **Risk Parity (RP)** | Equal risk contribution across assets |
| **Hierarchical Risk Parity (HRP)** | Clustering-based weights without matrix inversion |
| **Hybrid BL–HRP** | Combines BL views with HRP diversification |
| **CVaR** | Tail-risk minimization (Rockafellar–Uryasev LP) |

### Sector workflows

Each sector has a dedicated Jupyter notebook and YAML config template:

- **Consumer** — MV, BL (analyst + sentiment blend), RP, HRP, hybrid
- **Industrial** — MV, sentiment-only BL
- **Healthcare** — MV, sentiment BL, CVaR
- **Energy** — MV, sentiment BL, CVaR
- **Financials** — MV, macro BL (rate forecasts), CVaR

### Tooling

- Historical backtests with performance metrics (return, volatility, Sharpe, drawdown)
- Excel export for strategy comparison charts (`scripts/generate_excel.py`)
- Publication-style plots and CSV summaries (written locally under `outputs/`)

## Project structure

```
.
├── config.example.yaml              # Consumer sector template (copy → config.yaml)
├── config_*.example.yaml            # Per-sector templates
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── estimators.py
│   ├── constraints.py
│   ├── mv_optimizer.py
│   ├── bl_model.py
│   ├── bl_macro.py
│   ├── sentiment_analyzer.py
│   ├── cvar_optimizer.py
│   ├── rp_optimizer.py
│   ├── hrp_optimizer.py
│   ├── hybrid_optimizer.py
│   ├── backtester.py
│   ├── excel_exporter.py
│   └── reporting.py
├── scripts/
│   ├── backtest_all_strategies.py
│   └── generate_excel.py
├── notebooks/                     # Sector analysis notebooks
├── API_KEYS_SETUP.md
├── GRAPHING_GUIDE.md
└── QUICKSTART.md
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your portfolio (local only)

Real configs are **not** committed to git. Copy the example templates and add your tickers and baseline weights:

```bash
cp config.example.yaml config.yaml
cp config_industrial.example.yaml config_industrial.yaml
cp config_healthcare.example.yaml config_healthcare.yaml
cp config_financials.example.yaml config_financials.yaml
cp config_energy.example.yaml config_energy.yaml
```

Edit each file with your universe, `before_weights`, constraints, and optional Black-Litterman views.

### 3. API keys (sentiment pipelines)

```bash
cp .env.example .env
```

Add free-tier keys for [Polygon](https://polygon.io/) and [Finnhub](https://finnhub.io/). See [API_KEYS_SETUP.md](API_KEYS_SETUP.md).

### 4. Run analysis

```bash
# Validate installation
python test_implementation.py

# Consumer sector notebook
jupyter notebook notebooks/portfolio_optimization.ipynb

# Or run a full backtest + Excel export
python scripts/generate_excel.py
```

See [QUICKSTART.md](QUICKSTART.md) for more detail.

## Configuration

YAML configs drive each run:

- **Universe** — ticker list and baseline weights (prior for BL / backtest “before” portfolio)
- **Data** — historical start/end dates, return frequency
- **Constraints** — long-only, fully invested, max weight per name, optional turnover cap
- **Optimizers** — risk aversion, BL `tau`/`delta`, sentiment or macro view parameters, CVaR confidence

Example constraints used in this project:

- Long-only, fully invested (weights sum to 1)
- Maximum weight per position (e.g. 20%)
- Optional cap on turnover vs. baseline

## Sentiment-based Black-Litterman

1. Fetch recent news per ticker (Polygon)
2. Score articles with FinBERT (`ProsusAI/finbert`)
3. Cross-check with Finnhub sentiment
4. Rank names into quartiles and map to BL views
5. Optimize with PyPortfolioOpt / custom BL stack

Requires API keys and ~1–2 GB model download on first run.

## Macro-based Black-Litterman (financials)

Maps an interest-rate scenario (e.g. expected cut in bps) to stock-level **rate betas**, then generates relative BL views. Useful when sector returns are driven more by rates than headline sentiment.

## Tech stack

- Python 3.10+
- `numpy`, `pandas`, `scipy`, `cvxpy`
- `yfinance` for price history
- `PyPortfolioOpt` for BL and efficient frontier helpers
- `transformers` / PyTorch for FinBERT
- `jupyter` for interactive workflows

## Privacy & local data

This repository is intended for public sharing. The following stay **local** (see `.gitignore`):

- `config.yaml` and sector configs with real weights and portfolio values
- CSV / Excel exports and `outputs/` directories
- `.env` API keys
- Holdings spreadsheets (`stocks.xlsx`, etc.)

Do not commit files that contain live club or fund positions, share counts, or dollar allocations.
