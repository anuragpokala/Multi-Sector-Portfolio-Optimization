"""
Backtesting Module

Functions for backtesting portfolio strategies and computing performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Dict, Tuple


def calculate_portfolio_performance(
    weights: np.ndarray,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.045,
    trading_days: int = 252
) -> Dict:
    """
    Calculate portfolio performance metrics from weights and returns.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    returns : pd.DataFrame
        Daily returns (dates as index, tickers as columns)
    risk_free_rate : float
        Annual risk-free rate
    trading_days : int
        Trading days per year for annualization
        
    Returns
    -------
    Dict
        Performance metrics including returns, volatility, Sharpe, etc.
    """
    # Calculate portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Annualized metrics
    annual_return = portfolio_returns.mean() * trading_days
    annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else np.nan
    
    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    metrics = {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Portfolio Returns": portfolio_returns,
        "Cumulative Returns": cumulative_returns
    }
    
    return metrics


def backtest_portfolio(
    weights: np.ndarray,
    tickers: list,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.045,
    initial_capital: float = 10000
) -> Dict:
    """
    Backtest a portfolio strategy.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    tickers : list
        Asset tickers
    prices : pd.DataFrame
        Historical prices
    returns : pd.DataFrame
        Historical returns
    risk_free_rate : float
        Annual risk-free rate
    initial_capital : float
        Starting capital
        
    Returns
    -------
    Dict
        Backtest results including metrics and portfolio value over time
    """
    metrics = calculate_portfolio_performance(weights, returns, risk_free_rate)
    portfolio_value = initial_capital * metrics["Cumulative Returns"]
    
    results = {
        "weights": weights,
        "tickers": tickers,
        "metrics": metrics,
        "portfolio_value": portfolio_value,
        "portfolio_returns": metrics["Portfolio Returns"],
        "initial_capital": initial_capital
    }
    
    return results


def downside_deviation(
    r: pd.Series,
    mar_annual: float = 0.0,
    trading_days: int = 252
) -> float:
    """
    Calculate downside deviation (annualized).
    
    Downside deviation only penalizes returns below the MAR (minimum acceptable return),
    unlike standard deviation which penalizes both upside and downside volatility.
    
    Parameters
    ----------
    r : pd.Series
        Daily returns
    mar_annual : float
        Minimum acceptable return (annual)
    trading_days : int
        Trading days per year
        
    Returns
    -------
    float
        Annualized downside deviation
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    
    mar_daily = mar_annual / trading_days
    downside = np.minimum(r - mar_daily, 0.0)
    
    return np.sqrt(np.mean(downside**2)) * np.sqrt(trading_days)


def sortino_ratio(
    r: pd.Series,
    rf_annual: float = 0.045,
    mar_annual: float = 0.0,
    trading_days: int = 252
) -> float:
    """
    Calculate Sortino ratio.
    
    Unlike Sharpe ratio which uses total volatility, Sortino only penalizes
    downside volatility, making it more appropriate for asymmetric return distributions.
    
    Parameters
    ----------
    r : pd.Series
        Daily returns
    rf_annual : float
        Annual risk-free rate
    mar_annual : float
        Minimum acceptable return (annual)
    trading_days : int
        Trading days per year
        
    Returns
    -------
    float
        Sortino ratio
    """
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    
    ann_return = r.mean() * trading_days
    dd = downside_deviation(r, mar_annual=mar_annual, trading_days=trading_days)
    
    if not np.isfinite(dd) or dd <= 0:
        return np.nan
    
    return (ann_return - rf_annual) / dd


def max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and dates.
    
    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative returns over time
        
    Returns
    -------
    Tuple[float, pd.Timestamp, pd.Timestamp]
        (max_drawdown, peak_date, trough_date)
    """
    cum_ret = cumulative_returns.dropna()
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    
    # Find peak date (last max before trough)
    peak_date = running_max[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def summarize_and_plot(
    results_dict: dict,
    rf_annual: float = 0.045,
    trading_days: int = 252,
    mar_annual: float = 0.0
):
    """
    Summarize and plot multiple portfolio strategies with VT color scheme.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping strategy names to backtest results
    rf_annual : float
        Annual risk-free rate
    trading_days : int
        Trading days per year
    mar_annual : float
        Minimum acceptable return for Sortino
        
    Returns
    -------
    pd.DataFrame
        Summary metrics DataFrame
    """
    # Common date index
    common = None
    for res in results_dict.values():
        idx = res["portfolio_value"].dropna().index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise ValueError("No overlapping dates across strategies.")

    # Cumulative % returns (0..100 scale)
    cum_pct = pd.DataFrame({
        name: (res["portfolio_value"].reindex(common) / res["initial_capital"] - 1.0) * 100
        for name, res in results_dict.items()
    })

    # Summary metrics
    rows = []
    for name, res in results_dict.items():
        m = res["metrics"]
        ann_ret = float(m["Annual Return"])
        ann_vol = float(m["Annual Volatility"])
        risk_adj = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan

        port_rets = m["Portfolio Returns"]
        sortino = sortino_ratio(
            port_rets,
            rf_annual=rf_annual,
            mar_annual=mar_annual,
            trading_days=trading_days
        )

        rows.append({
            "Strategy": name,
            "Annual Return": ann_ret,
            "Annual Vol": ann_vol,
            "Risk Adj (Excess/Vol)": risk_adj,
            "Sortino": sortino
        })

    summary_df = pd.DataFrame(rows).set_index("Strategy")

    # Format for display table
    disp = summary_df.copy()
    disp["Annual Return"] = disp["Annual Return"].map(lambda x: f"{x:.2%}" if np.isfinite(x) else "—")
    disp["Annual Vol"] = disp["Annual Vol"].map(lambda x: f"{x:.2%}" if np.isfinite(x) else "—")
    disp["Risk Adj (Excess/Vol)"] = disp["Risk Adj (Excess/Vol)"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "—")
    disp["Sortino"] = disp["Sortino"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "—")

    # VT color scheme
    colors = {
        'Before': '#7f212f', 
        'Mean-Variance': '#c2253c', 
        'BL (Sentiment)': '#300f19',
        'BL (Macro)': '#630031',  # Dark maroon for macro BL
        'CVaR': '#e87722'  # VT orange for CVaR
    }

    fig = plt.figure(figsize=(16, 5))

    # Plot 1: Cumulative Returns
    ax1 = fig.add_subplot(1, 2, 1)
    cum_pct.plot(
        ax=ax1,
        linewidth=2.2,
        color=[colors.get(c, "#1f77b4") for c in cum_pct.columns]
    )

    ax1.set_title("Cumulative Returns (All Strategies)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(frameon=False)
    ax1.grid(False)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    # Plot 2: Performance Table
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")

    tbl = ax2.table(
        cellText=disp.values,
        rowLabels=disp.index,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.1, 1.6)

    # VT maroon table styling
    for (row, col), cell in tbl.get_celld().items():
        # Header row
        if row == 0:
            cell.set_facecolor("#7f212f")
            cell.set_text_props(color="#FFFFFF")

        # First column (strategy names)
        if col == -1:
            cell.set_facecolor("#7f212f")
            cell.set_text_props(weight="bold", color="#FFFFFF")

        # Borders
        cell.set_edgecolor("#7f212f")

    ax2.set_title("Performance Summary", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()

    return summary_df


# Backwards compatibility alias
def summarize_and_plot_strategies(
    results_dict: Dict,
    risk_free_rate: float = 0.045,
    trading_days: int = 252,
    mar_annual: float = 0.0,
    output_path: str = None
):
    """Legacy function name for backwards compatibility."""
    return summarize_and_plot(results_dict, rf_annual=risk_free_rate, trading_days=trading_days, mar_annual=mar_annual)


if __name__ == "__main__":
    print("Backtesting module loaded successfully")
