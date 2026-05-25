"""
Estimators Module

Functions for estimating expected returns and covariance matrices.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Tuple
import warnings


def estimate_covariance_shrinkage(
    returns: pd.DataFrame,
    annualization_factor: int = 252,
    method: str = "ledoit_wolf"
) -> Tuple[pd.DataFrame, float]:
    """
    Estimate covariance matrix using shrinkage methods.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns (dates as index, tickers as columns)
    annualization_factor : int, default=252
        Number of trading days per year for annualization
    method : str, default='ledoit_wolf'
        Shrinkage method: 'ledoit_wolf' or 'sample'
        
    Returns
    -------
    Tuple[pd.DataFrame, float]
        (annualized covariance matrix, shrinkage intensity)
        
    Notes
    -----
    - Ledoit-Wolf shrinkage reduces estimation error for small samples
    - Shrinkage intensity ∈ [0, 1]: 0 = sample cov, 1 = fully shrunk
    - Covariance is annualized by multiplying by annualization_factor
    """
    if method == "ledoit_wolf":
        # Fit Ledoit-Wolf estimator
        lw = LedoitWolf()
        lw.fit(returns)
        
        # Get covariance and shrinkage intensity
        cov_daily = lw.covariance_
        shrinkage = lw.shrinkage_
        
        print(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}")
        
    elif method == "sample":
        # Standard sample covariance
        cov_daily = returns.cov().values
        shrinkage = 0.0
        print("Using sample covariance (no shrinkage)")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Annualize covariance matrix
    cov_annual = cov_daily * annualization_factor
    
    # Convert to DataFrame for easier handling
    cov_df = pd.DataFrame(
        cov_annual,
        index=returns.columns,
        columns=returns.columns
    )
    
    # Verify positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov_annual)
    if np.min(eigenvalues) < -1e-8:
        warnings.warn(f"Covariance matrix has negative eigenvalue: {np.min(eigenvalues):.2e}")
    
    print(f"Annualized covariance matrix estimated (shape: {cov_df.shape})")
    print(f"Volatility range: {np.sqrt(np.diag(cov_annual)).min():.2%} to {np.sqrt(np.diag(cov_annual)).max():.2%}")
    
    return cov_df, shrinkage


def estimate_expected_returns(
    returns: pd.DataFrame,
    annualization_factor: int = 252,
    shrinkage: float = 0.0,
    prior_type: str = "equal_weight"
) -> pd.Series:
    """
    Estimate expected returns with optional shrinkage.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns (dates as index, tickers as columns)
    annualization_factor : int, default=252
        Number of trading days per year for annualization
    shrinkage : float, default=0.0
        Shrinkage intensity ∈ [0, 1]
        0 = pure historical mean, 1 = full shrinkage to prior
    prior_type : str, default='equal_weight'
        Type of prior: 'equal_weight' or 'zero'
        
    Returns
    -------
    pd.Series
        Annualized expected returns (shrunk if shrinkage > 0)
        
    Notes
    -----
    - Historical mean can be noisy with small samples
    - Shrinkage toward equal-weight prior reduces estimation error
    - Formula: μ = α * μ_hist + (1 - α) * μ_prior
    """
    # Calculate historical mean returns (annualized)
    mu_hist = returns.mean() * annualization_factor
    
    # Define prior
    if prior_type == "equal_weight":
        # Equal-weight portfolio implies all assets have same expected return
        mu_prior = pd.Series(mu_hist.mean(), index=returns.columns)
    elif prior_type == "zero":
        mu_prior = pd.Series(0.0, index=returns.columns)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")
    
    # Apply shrinkage
    mu_shrunk = (1 - shrinkage) * mu_hist + shrinkage * mu_prior
    
    if shrinkage > 0:
        print(f"Applied {shrinkage:.1%} shrinkage toward {prior_type} prior")
    else:
        print("Using pure historical mean returns (no shrinkage)")
    
    print(f"Expected returns range: {mu_shrunk.min():.2%} to {mu_shrunk.max():.2%}")
    
    return mu_shrunk


def get_correlation_matrix(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Convert covariance matrix to correlation matrix.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    # Extract standard deviations (volatilities)
    std = np.sqrt(np.diag(cov))
    
    # Compute correlation: corr[i,j] = cov[i,j] / (std[i] * std[j])
    corr = cov.values / np.outer(std, std)
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(
        corr,
        index=cov.index,
        columns=cov.columns
    )
    
    return corr_df


def compute_diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Calculate portfolio diversification ratio.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    cov : np.ndarray
        Covariance matrix (N x N)
        
    Returns
    -------
    float
        Diversification ratio = (weighted avg vol) / (portfolio vol)
        
    Notes
    -----
    - Higher ratio indicates better diversification
    - Ratio of 1 means no diversification (single asset)
    - Ratio > 1 indicates diversification benefit
    """
    # Individual volatilities
    vol_individual = np.sqrt(np.diag(cov))
    
    # Weighted average of individual volatilities
    weighted_avg_vol = np.dot(weights, vol_individual)
    
    # Portfolio volatility
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    
    # Diversification ratio
    div_ratio = weighted_avg_vol / portfolio_vol
    
    return div_ratio


def compute_effective_number_assets(weights: np.ndarray) -> float:
    """
    Calculate effective number of assets (ENB, or Herfindahl inverse).
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
        
    Returns
    -------
    float
        Effective number of assets = 1 / Σ(w_i^2)
        
    Notes
    -----
    - Equal-weight portfolio: ENB = N
    - Concentrated portfolio: ENB < N
    - Single asset: ENB = 1
    """
    enb = 1.0 / np.sum(weights ** 2)
    return enb


def validate_covariance_matrix(cov: np.ndarray, tolerance: float = 1e-8) -> Tuple[bool, dict]:
    """
    Validate covariance matrix properties.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix (N x N)
    tolerance : float, default=1e-8
        Numerical tolerance for checks
        
    Returns
    -------
    Tuple[bool, dict]
        (is_valid, diagnostics)
    """
    diagnostics = {}
    is_valid = True
    
    # Check symmetry
    is_symmetric = np.allclose(cov, cov.T, atol=tolerance)
    diagnostics['symmetric'] = is_symmetric
    if not is_symmetric:
        is_valid = False
        warnings.warn("Covariance matrix is not symmetric")
    
    # Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov)
    min_eigenvalue = np.min(eigenvalues)
    diagnostics['min_eigenvalue'] = min_eigenvalue
    diagnostics['condition_number'] = np.max(eigenvalues) / max(min_eigenvalue, 1e-10)
    
    if min_eigenvalue < -tolerance:
        is_valid = False
        warnings.warn(f"Covariance matrix is not PSD (min eigenvalue: {min_eigenvalue:.2e})")
    
    # Check diagonal elements are positive
    diag_positive = np.all(np.diag(cov) > 0)
    diagnostics['diag_positive'] = diag_positive
    if not diag_positive:
        is_valid = False
        warnings.warn("Covariance matrix has non-positive diagonal elements")
    
    return is_valid, diagnostics


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Generate sample returns data
    np.random.seed(42)
    n_days = 252 * 3  # 3 years
    n_assets = 5
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    print("="*60)
    print("Estimator Module Example")
    print("="*60)
    
    # Estimate covariance
    cov, shrinkage = estimate_covariance_shrinkage(returns)
    print(f"\nCovariance matrix:\n{cov.round(4)}")
    
    # Estimate expected returns
    mu = estimate_expected_returns(returns, shrinkage=0.3)
    print(f"\nExpected returns:\n{mu.round(4)}")
    
    # Validate covariance
    is_valid, diagnostics = validate_covariance_matrix(cov.values)
    print(f"\nCovariance validation: {is_valid}")
    print(f"Diagnostics: {diagnostics}")
