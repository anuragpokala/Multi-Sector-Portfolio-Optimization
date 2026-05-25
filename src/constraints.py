"""
Constraints Module

Functions for building portfolio constraints for optimization.
"""

import cvxpy as cp
import numpy as np
from typing import List, Optional, Dict


def build_constraints(
    w: cp.Variable,
    n_assets: int,
    long_only: bool = True,
    fully_invested: bool = True,
    max_weight: Optional[float] = None,
    min_weight: Optional[float] = None,
    turnover_cap: Optional[float] = None,
    w_before: Optional[np.ndarray] = None,
    custom_bounds: Optional[Dict[int, tuple]] = None
) -> List:
    """
    Build list of cvxpy constraints for portfolio optimization.
    
    Parameters
    ----------
    w : cp.Variable
        Portfolio weight variable (N x 1)
    n_assets : int
        Number of assets
    long_only : bool, default=True
        If True, enforce w >= 0 (no short positions)
    fully_invested : bool, default=True
        If True, enforce sum(w) == 1
    max_weight : float, optional
        Maximum weight per asset (e.g., 0.20 for 20% cap)
    min_weight : float, optional
        Minimum weight per asset (e.g., 0.05 for 5% minimum)
    turnover_cap : float, optional
        Maximum turnover vs w_before: sum(|w - w_before|) <= turnover_cap
    w_before : np.ndarray, optional
        Baseline weights for turnover constraint (N x 1)
    custom_bounds : Dict[int, tuple], optional
        Custom bounds for specific assets: {idx: (min, max)}
        
    Returns
    -------
    List
        List of cvxpy constraints
        
    Examples
    --------
    >>> w = cp.Variable(5)
    >>> constraints = build_constraints(
    ...     w, n_assets=5, long_only=True, fully_invested=True, max_weight=0.25
    ... )
    """
    constraints = []
    
    # Long-only constraint: w >= 0
    if long_only:
        constraints.append(w >= 0)
    
    # Fully invested constraint: sum(w) == 1
    if fully_invested:
        constraints.append(cp.sum(w) == 1)
    
    # Maximum weight constraint: w <= max_weight
    if max_weight is not None:
        constraints.append(w <= max_weight)
    
    # Minimum weight constraint: w >= min_weight
    if min_weight is not None:
        constraints.append(w >= min_weight)
    
    # Custom bounds for specific assets
    if custom_bounds is not None:
        for idx, (lb, ub) in custom_bounds.items():
            if lb is not None:
                constraints.append(w[idx] >= lb)
            if ub is not None:
                constraints.append(w[idx] <= ub)
    
    # Turnover constraint: sum(|w - w_before|) <= turnover_cap
    if turnover_cap is not None:
        if w_before is None:
            raise ValueError("w_before must be provided if turnover_cap is specified")
        
        # Turnover is measured as L1 norm of weight changes
        # sum(|w_i - w_before_i|) <= turnover_cap
        constraints.append(cp.norm(w - w_before, 1) <= turnover_cap)
    
    return constraints


def validate_weights(
    weights: np.ndarray,
    constraints_config: dict,
    w_before: Optional[np.ndarray] = None,
    tolerance: float = 1e-4
) -> tuple:
    """
    Validate that optimized weights satisfy constraints.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    constraints_config : dict
        Dictionary with constraint parameters
    w_before : np.ndarray, optional
        Baseline weights for turnover check
    tolerance : float, default=1e-4
        Numerical tolerance for constraint violations
        
    Returns
    -------
    tuple
        (is_valid, violations_dict)
    """
    violations = {}
    is_valid = True
    
    # Check sum to 1
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > tolerance:
        violations['sum'] = f"Weights sum to {weight_sum:.6f}, not 1.0"
        is_valid = False
    
    # Check long-only
    if constraints_config.get('long_only', False):
        if np.any(weights < -tolerance):
            violations['long_only'] = f"Negative weights found: min={np.min(weights):.6f}"
            is_valid = False
    
    # Check max weight
    max_weight = constraints_config.get('max_weight')
    if max_weight is not None:
        if np.any(weights > max_weight + tolerance):
            violations['max_weight'] = f"Weights exceed max: max={np.max(weights):.6f} > {max_weight}"
            is_valid = False
    
    # Check min weight
    min_weight = constraints_config.get('min_weight')
    if min_weight is not None:
        if np.any(weights < min_weight - tolerance):
            violations['min_weight'] = f"Weights below min: min={np.min(weights):.6f} < {min_weight}"
            is_valid = False
    
    # Check turnover
    turnover_cap = constraints_config.get('turnover_cap')
    if turnover_cap is not None and w_before is not None:
        turnover = np.sum(np.abs(weights - w_before))
        if turnover > turnover_cap + tolerance:
            violations['turnover'] = f"Turnover {turnover:.4f} exceeds cap {turnover_cap}"
            is_valid = False
    
    return is_valid, violations


def compute_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """
    Calculate portfolio turnover.
    
    Parameters
    ----------
    w_new : np.ndarray
        New portfolio weights
    w_old : np.ndarray
        Old portfolio weights
        
    Returns
    -------
    float
        Turnover = sum(|w_new - w_old|)
        
    Notes
    -----
    - Turnover represents fraction of portfolio that needs to be traded
    - Turnover of 0 = no changes
    - Turnover of 2 = complete portfolio reversal
    """
    return np.sum(np.abs(w_new - w_old))


def compute_tracking_error(
    w_new: np.ndarray,
    w_benchmark: np.ndarray,
    cov: np.ndarray
) -> float:
    """
    Calculate ex-ante tracking error vs benchmark.
    
    Parameters
    ----------
    w_new : np.ndarray
        New portfolio weights (N x 1)
    w_benchmark : np.ndarray
        Benchmark weights (N x 1)
    cov : np.ndarray
        Covariance matrix (N x N)
        
    Returns
    -------
    float
        Tracking error (annualized volatility of active returns)
        
    Notes
    -----
    - Active weights: w_active = w_new - w_benchmark
    - Tracking error: sqrt(w_active' * Cov * w_active)
    """
    w_active = w_new - w_benchmark
    tracking_error = np.sqrt(w_active @ cov @ w_active)
    return tracking_error


def check_concentration(
    weights: np.ndarray,
    top_n: int = 3
) -> dict:
    """
    Analyze portfolio concentration.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    top_n : int, default=3
        Number of top holdings to analyze
        
    Returns
    -------
    dict
        Concentration metrics
    """
    metrics = {}
    
    # Sort weights in descending order
    sorted_weights = np.sort(weights)[::-1]
    
    # Top N concentration
    metrics[f'top_{top_n}_weight'] = np.sum(sorted_weights[:top_n])
    
    # Herfindahl index (sum of squared weights)
    metrics['herfindahl'] = np.sum(weights ** 2)
    
    # Effective number of assets
    metrics['effective_n_assets'] = 1.0 / metrics['herfindahl']
    
    # Gini coefficient (measure of inequality)
    n = len(weights)
    sorted_w = np.sort(weights)
    cumsum = np.cumsum(sorted_w)
    metrics['gini'] = (2 * np.sum((np.arange(1, n + 1)) * sorted_w) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    return metrics


def get_seed_constraint_check(
    weights: np.ndarray,
    sector_portfolio_value: float,
    seed_total_assets: float,
    seed_max_pct: float = 0.08
) -> tuple:
    """
    Check if portfolio satisfies SEED fund constraints.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights within sector (N x 1)
    sector_portfolio_value : float
        Total value of sector portfolio (USD)
    seed_total_assets : float
        Total SEED fund assets (USD)
    seed_max_pct : float, default=0.08
        Maximum allowed % of SEED assets per position
        
    Returns
    -------
    tuple
        (is_compliant, details_dict)
    """
    details = {}
    is_compliant = True
    
    # Convert sector weights to SEED-level percentages
    seed_percentages = (weights * sector_portfolio_value) / seed_total_assets
    
    details['max_seed_pct'] = np.max(seed_percentages)
    details['max_allowed'] = seed_max_pct
    details['sector_pct_of_seed'] = sector_portfolio_value / seed_total_assets
    
    if np.max(seed_percentages) > seed_max_pct:
        is_compliant = False
        details['violation'] = f"Max position {np.max(seed_percentages):.2%} exceeds {seed_max_pct:.2%} SEED limit"
    else:
        details['violation'] = None
    
    # Show top positions as % of SEED
    details['seed_percentages'] = seed_percentages
    
    return is_compliant, details


if __name__ == "__main__":
    # Example usage
    import cvxpy as cp
    
    print("="*60)
    print("Constraints Module Example")
    print("="*60)
    
    # Create optimization variable
    n = 8
    w = cp.Variable(n)
    
    # Build constraints
    constraints = build_constraints(
        w,
        n_assets=n,
        long_only=True,
        fully_invested=True,
        max_weight=0.20,
        min_weight=0.0
    )
    
    print(f"\nBuilt {len(constraints)} constraints:")
    for i, c in enumerate(constraints):
        print(f"  {i+1}. {c}")
    
    # Validate example weights
    test_weights = np.array([0.15, 0.12, 0.18, 0.10, 0.15, 0.08, 0.12, 0.10])
    
    config = {
        'long_only': True,
        'fully_invested': True,
        'max_weight': 0.20,
        'min_weight': 0.0
    }
    
    is_valid, violations = validate_weights(test_weights, config)
    print(f"\nValidation result: {is_valid}")
    if violations:
        print(f"Violations: {violations}")
    
    # Check concentration
    conc = check_concentration(test_weights)
    print(f"\nConcentration metrics:")
    for k, v in conc.items():
        print(f"  {k}: {v:.4f}")
