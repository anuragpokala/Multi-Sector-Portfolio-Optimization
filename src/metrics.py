"""
Metrics Module

Functions for calculating portfolio performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.0,
    w_before: Optional[np.ndarray] = None,
    tickers: Optional[list] = None
) -> Dict:
    """
    Compute comprehensive portfolio performance metrics.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    mu : np.ndarray
        Expected returns (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    rf : float, default=0.0
        Risk-free rate
    w_before : np.ndarray, optional
        Baseline portfolio weights for turnover calculation
    tickers : list, optional
        Asset tickers for detailed output
        
    Returns
    -------
    Dict
        Dictionary of portfolio metrics
    """
    # Basic return and risk
    expected_return = float(mu @ weights)
    variance = float(weights @ Sigma @ weights)
    volatility = float(np.sqrt(variance))
    
    # Risk-adjusted metrics
    excess_return = expected_return - rf
    sharpe_ratio = excess_return / volatility if volatility > 1e-8 else 0.0
    
    # Diversification metrics
    individual_vols = np.sqrt(np.diag(Sigma))
    weighted_avg_vol = float(weights @ individual_vols)
    diversification_ratio = weighted_avg_vol / volatility if volatility > 1e-8 else 1.0
    
    # Concentration metrics
    herfindahl = float(np.sum(weights ** 2))
    effective_n_assets = 1.0 / herfindahl if herfindahl > 0 else 0.0
    
    # Risk contribution analysis
    if volatility > 1e-8:
        marginal_risk = Sigma @ weights / volatility
        risk_contribution = weights * marginal_risk
        pct_risk_contribution = risk_contribution / variance if variance > 0 else np.zeros_like(weights)
    else:
        marginal_risk = np.zeros_like(weights)
        risk_contribution = np.zeros_like(weights)
        pct_risk_contribution = np.zeros_like(weights)
    
    # Turnover (if baseline provided)
    turnover = None
    if w_before is not None:
        turnover = float(np.sum(np.abs(weights - w_before)))
    
    # Package metrics
    metrics = {
        'expected_annual_return': expected_return,
        'expected_annual_volatility': volatility,
        'expected_annual_variance': variance,
        'sharpe_ratio': sharpe_ratio,
        'risk_adjusted_return': sharpe_ratio,  # Alias
        'excess_return': excess_return,
        'diversification_ratio': diversification_ratio,
        'herfindahl_index': herfindahl,
        'effective_n_assets': effective_n_assets,
        'max_weight': float(np.max(weights)),
        'min_weight': float(np.min(weights)),
        'turnover': turnover,
        'marginal_risk': marginal_risk,
        'risk_contribution': risk_contribution,
        'pct_risk_contribution': pct_risk_contribution
    }
    
    return metrics


def format_metrics_table(metrics: Dict, name: str = "Portfolio") -> pd.DataFrame:
    """
    Format metrics dictionary as a readable DataFrame.
    
    Parameters
    ----------
    metrics : Dict
        Portfolio metrics dictionary
    name : str, default="Portfolio"
        Portfolio name for the table
        
    Returns
    -------
    pd.DataFrame
        Formatted metrics table
    """
    # Select key metrics for display
    display_metrics = {
        'Expected Annual Return': f"{metrics['expected_annual_return']:.2%}",
        'Expected Annual Volatility': f"{metrics['expected_annual_volatility']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
        'Risk-Adjusted Return': f"{metrics['risk_adjusted_return']:.4f}",
        'Diversification Ratio': f"{metrics['diversification_ratio']:.4f}",
        'Effective # Assets': f"{metrics['effective_n_assets']:.2f}",
        'Max Weight': f"{metrics['max_weight']:.2%}",
        'Min Weight': f"{metrics['min_weight']:.2%}",
    }
    
    if metrics.get('turnover') is not None:
        display_metrics['Turnover'] = f"{metrics['turnover']:.2%}"
    
    df = pd.DataFrame.from_dict(display_metrics, orient='index', columns=[name])
    return df


def compare_portfolios(
    portfolios: Dict[str, Dict],
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.0,
    w_before: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compare metrics across multiple portfolios.
    
    Parameters
    ----------
    portfolios : Dict[str, Dict]
        Dictionary mapping portfolio names to results dictionaries
        Each result dict should have 'weights' key
    mu : np.ndarray
        Expected returns
    Sigma : np.ndarray
        Covariance matrix
    rf : float, default=0.0
        Risk-free rate
    w_before : np.ndarray, optional
        Baseline weights
        
    Returns
    -------
    pd.DataFrame
        Comparison table with portfolios as columns
    """
    comparison = {}
    
    for name, portfolio in portfolios.items():
        weights = portfolio['weights']
        metrics = compute_portfolio_metrics(weights, mu, Sigma, rf, w_before)
        comparison[name] = metrics
    
    # Create DataFrame with key metrics
    rows = [
        ('Expected Annual Return', 'expected_annual_return', '{:.2%}'),
        ('Expected Annual Volatility', 'expected_annual_volatility', '{:.2%}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.4f}'),
        ('Diversification Ratio', 'diversification_ratio', '{:.4f}'),
        ('Effective # Assets', 'effective_n_assets', '{:.2f}'),
        ('Max Weight', 'max_weight', '{:.2%}'),
        ('Turnover', 'turnover', '{:.2%}'),
    ]
    
    data = {}
    for name, portfolio_metrics in comparison.items():
        data[name] = []
        for label, key, fmt in rows:
            value = portfolio_metrics.get(key)
            if value is not None:
                data[name].append(value)
            else:
                data[name].append(np.nan)
    
    df = pd.DataFrame(data, index=[r[0] for r in rows])
    return df


def compute_risk_decomposition(
    weights: np.ndarray,
    Sigma: np.ndarray,
    tickers: list
) -> pd.DataFrame:
    """
    Decompose portfolio risk by asset.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    tickers : list
        Asset tickers
        
    Returns
    -------
    pd.DataFrame
        Risk decomposition table
    """
    portfolio_vol = np.sqrt(weights @ Sigma @ weights)
    
    # Marginal contribution to risk (MCTR)
    if portfolio_vol > 1e-8:
        mctr = (Sigma @ weights) / portfolio_vol
    else:
        mctr = np.zeros_like(weights)
    
    # Component contribution to risk (CCTR)
    cctr = weights * mctr
    
    # Percentage contribution to risk
    if portfolio_vol > 1e-8:
        pct_contribution = cctr / portfolio_vol
    else:
        pct_contribution = np.zeros_like(weights)
    
    # Individual asset volatilities
    individual_vols = np.sqrt(np.diag(Sigma))
    
    df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights,
        'Individual Vol': individual_vols,
        'MCTR': mctr,
        'CCTR': cctr,
        '% Contribution': pct_contribution
    })
    
    return df


def compute_return_decomposition(
    weights: np.ndarray,
    mu: np.ndarray,
    tickers: list
) -> pd.DataFrame:
    """
    Decompose portfolio return by asset.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    mu : np.ndarray
        Expected returns (N x 1)
    tickers : list
        Asset tickers
        
    Returns
    -------
    pd.DataFrame
        Return decomposition table
    """
    # Contribution to return
    contribution = weights * mu
    
    portfolio_return = np.sum(contribution)
    
    # Percentage contribution
    if abs(portfolio_return) > 1e-8:
        pct_contribution = contribution / portfolio_return
    else:
        pct_contribution = np.zeros_like(weights)
    
    df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights,
        'Expected Return': mu,
        'Contribution': contribution,
        '% Contribution': pct_contribution
    })
    
    return df


def compute_information_ratio(
    w_active: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray
) -> float:
    """
    Calculate information ratio of active portfolio.
    
    Parameters
    ----------
    w_active : np.ndarray
        Active weights (w_portfolio - w_benchmark)
    mu : np.ndarray
        Expected returns
    Sigma : np.ndarray
        Covariance matrix
        
    Returns
    -------
    float
        Information ratio = active_return / tracking_error
    """
    active_return = mu @ w_active
    tracking_error = np.sqrt(w_active @ Sigma @ w_active)
    
    if tracking_error > 1e-8:
        ir = active_return / tracking_error
    else:
        ir = 0.0
    
    return ir


def compute_downside_deviation(
    returns: np.ndarray,
    threshold: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate downside deviation (semi-standard deviation).
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    threshold : float, default=0.0
        Minimum acceptable return (MAR)
    annualization_factor : int, default=252
        Annualization factor
        
    Returns
    -------
    float
        Annualized downside deviation
    """
    downside_returns = returns[returns < threshold]
    
    if len(downside_returns) > 0:
        downside_var = np.mean((downside_returns - threshold) ** 2)
        downside_dev = np.sqrt(downside_var) * np.sqrt(annualization_factor)
    else:
        downside_dev = 0.0
    
    return downside_dev


def compute_sortino_ratio(
    returns: np.ndarray,
    rf: float = 0.0,
    threshold: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Sortino ratio (reward-to-downside-risk).
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    rf : float, default=0.0
        Risk-free rate (annualized)
    threshold : float, default=0.0
        Minimum acceptable return (MAR, annualized)
    annualization_factor : int, default=252
        Annualization factor
        
    Returns
    -------
    float
        Sortino ratio
    """
    annual_return = np.mean(returns) * annualization_factor
    downside_dev = compute_downside_deviation(returns, threshold / annualization_factor, annualization_factor)
    
    if downside_dev > 1e-8:
        sortino = (annual_return - rf) / downside_dev
    else:
        sortino = 0.0
    
    return sortino


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative returns over time
        
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_dd = np.min(drawdown)
    
    return max_dd


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Metrics Module Example")
    print("="*60)
    
    # Sample portfolio
    np.random.seed(42)
    n = 5
    weights = np.array([0.15, 0.20, 0.25, 0.30, 0.10])
    mu = np.array([0.08, 0.12, 0.15, 0.10, 0.09])
    L = np.random.randn(n, n) * 0.05
    Sigma = L @ L.T + np.eye(n) * 0.01
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    # Compute metrics
    metrics = compute_portfolio_metrics(weights, mu, Sigma, rf=0.03)
    
    print("\nPortfolio Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(f"  {k}: {v:.4f}")
    
    # Risk decomposition
    risk_decomp = compute_risk_decomposition(weights, Sigma, tickers)
    print("\nRisk Decomposition:")
    print(risk_decomp.round(4))
