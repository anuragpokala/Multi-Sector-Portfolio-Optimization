"""
Hybrid BL-HRP Portfolio Optimization

Combines Black-Litterman (incorporates views) with HRP (diversification)
to create a balanced approach that considers both views and risk structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def optimize_hybrid_bl_hrp(
    bl_weights: np.ndarray,
    hrp_weights: np.ndarray,
    blend_ratio: float = 0.5,
    tickers: Optional[list] = None,
    returns: Optional[pd.DataFrame] = None,
    Sigma: Optional[np.ndarray] = None
) -> Dict:
    """
    Create hybrid portfolio by blending Black-Litterman and HRP weights.
    
    The hybrid approach combines:
    - BL: incorporates analyst views and forward-looking information
    - HRP: provides diversification through hierarchical clustering
    
    Parameters
    ----------
    bl_weights : np.ndarray
        Black-Litterman optimized weights
    hrp_weights : np.ndarray
        HRP optimized weights
    blend_ratio : float
        Blending ratio (0.5 = 50% BL, 50% HRP)
    tickers : list, optional
        List of ticker symbols
    returns : pd.DataFrame, optional
        Historical returns (for metrics calculation)
    Sigma : np.ndarray, optional
        Covariance matrix (for metrics calculation)
        
    Returns
    -------
    dict
        Dictionary with 'weights' and 'metrics'
    """
    # Blend weights
    hybrid_weights = blend_ratio * bl_weights + (1 - blend_ratio) * hrp_weights
    
    # Normalize to ensure sum to 1 (should already be close)
    hybrid_weights = hybrid_weights / hybrid_weights.sum()
    
    # Calculate metrics if data provided
    metrics = {}
    if returns is not None and Sigma is not None:
        portfolio_return = np.dot(hybrid_weights, returns.mean() * 252)
        portfolio_vol = np.sqrt(np.dot(hybrid_weights, np.dot(Sigma, hybrid_weights)))
        
        metrics = {
            'expected_annual_return': portfolio_return,
            'expected_annual_volatility': portfolio_vol,
            'blend_ratio': blend_ratio
        }
        
        if tickers is not None:
            metrics['weight_breakdown'] = {
                'bl_contribution': {ticker: bl_weights[i] * blend_ratio 
                                   for i, ticker in enumerate(tickers)},
                'hrp_contribution': {ticker: hrp_weights[i] * (1 - blend_ratio) 
                                    for i, ticker in enumerate(tickers)}
            }
    
    return {
        'weights': hybrid_weights,
        'metrics': metrics,
        'method': f'Hybrid BL-HRP ({blend_ratio:.0%} BL / {(1-blend_ratio):.0%} HRP)'
    }


def compare_allocations(
    bl_weights: np.ndarray,
    hrp_weights: np.ndarray,
    hybrid_weights: np.ndarray,
    tickers: list
) -> pd.DataFrame:
    """
    Create comparison table of allocations across methods.
    
    Parameters
    ----------
    bl_weights : np.ndarray
        Black-Litterman weights
    hrp_weights : np.ndarray
        HRP weights
    hybrid_weights : np.ndarray
        Hybrid weights
    tickers : list
        List of ticker symbols
        
    Returns
    -------
    pd.DataFrame
        Comparison table with weights for each method
    """
    comparison = pd.DataFrame({
        'Ticker': tickers,
        'BL': bl_weights,
        'HRP': hrp_weights,
        'Hybrid': hybrid_weights,
        'BL_vs_HRP_Diff': bl_weights - hrp_weights
    })
    
    comparison = comparison.set_index('Ticker')
    
    # Add summary row
    comparison.loc['Total'] = comparison.sum()
    
    return comparison


def analyze_diversification_benefits(
    bl_weights: np.ndarray,
    hrp_weights: np.ndarray,
    hybrid_weights: np.ndarray,
    Sigma: np.ndarray
) -> Dict:
    """
    Analyze diversification benefits of hybrid approach.
    
    Parameters
    ----------
    bl_weights : np.ndarray
        Black-Litterman weights
    hrp_weights : np.ndarray
        HRP weights  
    hybrid_weights : np.ndarray
        Hybrid weights
    Sigma : np.ndarray
        Covariance matrix
        
    Returns
    -------
    dict
        Diversification metrics for each approach
    """
    def calc_metrics(w):
        vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
        effective_n = 1.0 / np.sum(w ** 2)  # Effective number of assets
        hhi = np.sum(w ** 2)  # Herfindahl index (lower is more diversified)
        return {
            'volatility': vol,
            'effective_n_assets': effective_n,
            'herfindahl_index': hhi
        }
    
    return {
        'Black-Litterman': calc_metrics(bl_weights),
        'HRP': calc_metrics(hrp_weights),
        'Hybrid': calc_metrics(hybrid_weights)
    }
