"""
Risk Parity Portfolio Optimization

Implements Risk Parity (Equal Risk Contribution) optimization approach.
Risk Parity allocates capital such that each asset contributes equally to 
total portfolio risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def optimize_risk_parity(
    returns: pd.DataFrame,
    Sigma: np.ndarray,
    constraints_config: Optional[Dict] = None,
    before_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Optimize portfolio using Risk Parity approach.
    
    Risk Parity allocates weights inversely proportional to volatility,
    so each asset contributes equally to portfolio risk.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns (dates x assets)
    Sigma : np.ndarray
        Covariance matrix
    constraints_config : dict, optional
        Constraints configuration (max_weight, etc.)
    before_weights : np.ndarray, optional
        Previous weights (not used in basic RP but kept for interface consistency)
        
    Returns
    -------
    dict
        Dictionary with 'weights' (np.ndarray) and 'metrics'
    """
    n_assets = len(returns.columns)
    tickers = returns.columns.tolist()
    
    # Calculate volatilities (standard deviations)
    volatilities = np.sqrt(np.diag(Sigma))
    
    # Risk Parity: weights inversely proportional to volatility
    # w_i ∝ 1/σ_i
    inverse_vol = 1.0 / volatilities
    weights = inverse_vol / inverse_vol.sum()
    
    # Apply constraints if provided
    if constraints_config:
        max_weight = constraints_config.get('max_weight', 1.0)
        
        # Clip weights to max constraint
        weights = np.clip(weights, 0, max_weight)
        
        # Renormalize to sum to 1
        weights = weights / weights.sum()
        
        # Iteratively enforce max_weight constraint
        # (in case normalization pushed some weights over limit)
        max_iterations = 10
        for _ in range(max_iterations):
            if weights.max() <= max_weight:
                break
            
            # Find assets exceeding max_weight
            over_limit = weights > max_weight
            weights[over_limit] = max_weight
            
            # Redistribute excess to other assets
            remaining_weight = 1.0 - weights[over_limit].sum()
            if remaining_weight > 0:
                under_limit = ~over_limit
                if under_limit.any():
                    weights[under_limit] *= remaining_weight / weights[under_limit].sum()
    
    # Ensure fully invested (sum to 1)
    weights = weights / weights.sum()
    
    # Calculate portfolio metrics
    portfolio_return = np.dot(weights, returns.mean() * 252)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
    
    # Risk contribution for each asset
    marginal_contrib = np.dot(Sigma, weights)
    risk_contrib = weights * marginal_contrib / portfolio_vol
    risk_contrib_pct = risk_contrib / risk_contrib.sum()
    
    metrics = {
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_vol,
        'risk_contributions': dict(zip(tickers, risk_contrib_pct))
    }
    
    return {
        'weights': weights,
        'metrics': metrics,
        'method': 'Risk Parity'
    }


def calculate_risk_contributions(weights: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Calculate risk contribution of each asset.
    
    Risk contribution = w_i * (Σw)_i / σ_p
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    Sigma : np.ndarray
        Covariance matrix
        
    Returns
    -------
    np.ndarray
        Risk contribution for each asset (sums to 1)
    """
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
    marginal_contrib = np.dot(Sigma, weights)
    risk_contrib = weights * marginal_contrib / portfolio_vol
    return risk_contrib / risk_contrib.sum()
