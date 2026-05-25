"""
Mean-Variance Optimization Module

Implementation of Markowitz mean-variance portfolio optimization.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import List, Optional, Dict, Tuple
from .constraints import build_constraints


def optimize_mean_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    constraints_config: dict,
    rf: float = 0.0,
    lambda_grid: Optional[np.ndarray] = None,
    target_return: Optional[float] = None,
    target_vol: Optional[float] = None,
    w_before: Optional[np.ndarray] = None
) -> Dict:
    """
    Perform mean-variance portfolio optimization.
    
    Two approaches:
    1. Risk-aversion sweep: minimize 0.5*w'*Sigma*w - lambda*mu'*w
    2. Target return/vol: minimize w'*Sigma*w subject to mu'*w >= target
    
    Parameters
    ----------
    mu : np.ndarray
        Expected returns (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    constraints_config : dict
        Constraint parameters (long_only, max_weight, etc.)
    rf : float, default=0.0
        Risk-free rate for Sharpe ratio calculation
    lambda_grid : np.ndarray, optional
        Grid of risk-aversion parameters to sweep
        If None, defaults to log-spaced grid [0.1, 10]
    target_return : float, optional
        If specified, minimize variance subject to return >= target
    target_vol : float, optional
        If specified, maximize return subject to vol <= target
    w_before : np.ndarray, optional
        Baseline weights for turnover constraint
        
    Returns
    -------
    Dict
        Results including optimal weights, metrics, and efficient frontier
        
    Notes
    -----
    - Lambda (risk aversion): higher λ → more emphasis on returns
    - The optimal portfolio is selected by maximizing Sharpe ratio
    """
    n_assets = len(mu)
    
    # Default lambda grid
    if lambda_grid is None:
        lambda_grid = np.logspace(-1, 1, 30)  # [0.1, 10] with 30 points
    
    print(f"\n{'='*60}")
    print("Mean-Variance Optimization")
    print(f"{'='*60}")
    print(f"Number of assets: {n_assets}")
    print(f"Expected returns: {mu.min():.2%} to {mu.max():.2%}")
    print(f"Volatilities: {np.sqrt(np.diag(Sigma)).min():.2%} to {np.sqrt(np.diag(Sigma)).max():.2%}")
    
    # Build constraint list
    w = cp.Variable(n_assets)
    constraints = build_constraints(
        w=w,
        n_assets=n_assets,
        long_only=constraints_config.get('long_only', True),
        fully_invested=constraints_config.get('fully_invested', True),
        max_weight=constraints_config.get('max_weight'),
        min_weight=constraints_config.get('min_weight'),
        turnover_cap=constraints_config.get('turnover_cap'),
        w_before=w_before
    )
    
    # Efficient frontier storage
    frontier_results = []
    
    # Sweep over lambda (risk aversion parameter)
    print(f"\nSweeping {len(lambda_grid)} risk-aversion parameters...")
    
    for lam in lambda_grid:
        # Objective: minimize 0.5 * w' * Sigma * w - lambda * mu' * w
        # Higher lambda = more weight on returns
        portfolio_risk = 0.5 * cp.quad_form(w, Sigma)
        portfolio_return = mu @ w
        objective = cp.Minimize(portfolio_risk - lam * portfolio_return)
        
        # Solve (try multiple solvers)
        problem = cp.Problem(objective, constraints)
        try:
            # Try solvers in order of preference
            for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
                try:
                    problem.solve(solver=solver, verbose=False)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        break
                except:
                    continue
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                w_opt = w.value
                ret = mu @ w_opt
                vol = np.sqrt(w_opt @ Sigma @ w_opt)
                sharpe = (ret - rf) / vol if vol > 1e-8 else 0.0
                
                frontier_results.append({
                    'lambda': lam,
                    'weights': w_opt.copy(),
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe
                })
            else:
                print(f"  Warning: Optimization failed for lambda={lam:.4f} (status: {problem.status})")
        except Exception as e:
            print(f"  Error for lambda={lam:.4f}: {str(e)}")
    
    if not frontier_results:
        raise RuntimeError("All optimizations failed. Check constraints and data.")
    
    # Convert to DataFrame for analysis
    frontier_df = pd.DataFrame([
        {
            'lambda': r['lambda'],
            'return': r['return'],
            'volatility': r['volatility'],
            'sharpe': r['sharpe']
        }
        for r in frontier_results
    ])
    
    # Select portfolio with maximum Sharpe ratio
    best_idx = frontier_df['sharpe'].idxmax()
    best_result = frontier_results[best_idx]
    
    print(f"\n{'='*60}")
    print("Optimization Complete")
    print(f"{'='*60}")
    print(f"Efficient frontier computed: {len(frontier_results)} portfolios")
    print(f"Best lambda (risk aversion): {best_result['lambda']:.4f}")
    print(f"Expected return: {best_result['return']:.2%}")
    print(f"Expected volatility: {best_result['volatility']:.2%}")
    print(f"Sharpe ratio: {best_result['sharpe']:.4f}")
    
    # Package results
    results = {
        'weights': best_result['weights'],
        'return': best_result['return'],
        'volatility': best_result['volatility'],
        'sharpe': best_result['sharpe'],
        'lambda_opt': best_result['lambda'],
        'frontier': frontier_df,
        'frontier_results': frontier_results,
        'status': 'success'
    }
    
    return results


def optimize_target_return(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
    constraints_config: dict,
    w_before: Optional[np.ndarray] = None
) -> Dict:
    """
    Minimize variance subject to target return constraint.
    
    Parameters
    ----------
    mu : np.ndarray
        Expected returns (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    target_return : float
        Minimum required return
    constraints_config : dict
        Constraint parameters
    w_before : np.ndarray, optional
        Baseline weights for turnover constraint
        
    Returns
    -------
    Dict
        Optimization results
    """
    n_assets = len(mu)
    w = cp.Variable(n_assets)
    
    # Build constraints
    constraints = build_constraints(
        w=w,
        n_assets=n_assets,
        long_only=constraints_config.get('long_only', True),
        fully_invested=constraints_config.get('fully_invested', True),
        max_weight=constraints_config.get('max_weight'),
        min_weight=constraints_config.get('min_weight'),
        turnover_cap=constraints_config.get('turnover_cap'),
        w_before=w_before
    )
    
    # Add target return constraint
    constraints.append(mu @ w >= target_return)
    
    # Objective: minimize variance
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    
    # Solve (try multiple solvers)
    problem = cp.Problem(objective, constraints)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                break
        except:
            continue
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"Optimization failed: {problem.status}")
    
    w_opt = w.value
    ret = mu @ w_opt
    vol = np.sqrt(w_opt @ Sigma @ w_opt)
    
    return {
        'weights': w_opt,
        'return': ret,
        'volatility': vol,
        'status': problem.status
    }


def compute_efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    constraints_config: dict,
    n_points: int = 50,
    w_before: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute the efficient frontier by sweeping target returns.
    
    Parameters
    ----------
    mu : np.ndarray
        Expected returns (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    constraints_config : dict
        Constraint parameters
    n_points : int, default=50
        Number of points on the frontier
    w_before : np.ndarray, optional
        Baseline weights for turnover constraint
        
    Returns
    -------
    pd.DataFrame
        Efficient frontier with returns and volatilities
    """
    n_assets = len(mu)
    
    # Find feasible return range
    # Minimum variance portfolio (no return constraint)
    w = cp.Variable(n_assets)
    constraints = build_constraints(
        w=w, n_assets=n_assets,
        long_only=constraints_config.get('long_only', True),
        fully_invested=constraints_config.get('fully_invested', True),
        max_weight=constraints_config.get('max_weight'),
        min_weight=constraints_config.get('min_weight'),
        turnover_cap=constraints_config.get('turnover_cap'),
        w_before=w_before
    )
    
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    min_return = mu @ w.value
    
    # Maximum return portfolio
    problem = cp.Problem(cp.Maximize(mu @ w), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    max_return = mu @ w.value
    
    # Sweep target returns
    target_returns = np.linspace(min_return, max_return, n_points)
    frontier = []
    
    for target in target_returns:
        try:
            result = optimize_target_return(mu, Sigma, target, constraints_config, w_before)
            frontier.append({
                'return': result['return'],
                'volatility': result['volatility']
            })
        except:
            continue
    
    return pd.DataFrame(frontier)


def get_portfolio_statistics(
    weights: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.0
) -> Dict:
    """
    Calculate comprehensive portfolio statistics.
    
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
        
    Returns
    -------
    Dict
        Portfolio statistics
    """
    # Return and risk
    portfolio_return = mu @ weights
    portfolio_variance = weights @ Sigma @ weights
    portfolio_vol = np.sqrt(portfolio_variance)
    
    # Risk-adjusted metrics
    sharpe = (portfolio_return - rf) / portfolio_vol if portfolio_vol > 1e-8 else 0.0
    
    # Diversification
    individual_vols = np.sqrt(np.diag(Sigma))
    weighted_avg_vol = weights @ individual_vols
    diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 1e-8 else 1.0
    
    # Contribution to risk (marginal contribution * weight)
    marginal_contribution = Sigma @ weights / portfolio_vol if portfolio_vol > 1e-8 else np.zeros_like(weights)
    risk_contribution = weights * marginal_contribution
    
    # Effective number of bets
    effective_n = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0.0
    
    stats = {
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'variance': portfolio_variance,
        'sharpe_ratio': sharpe,
        'diversification_ratio': diversification_ratio,
        'effective_n_assets': effective_n,
        'max_weight': np.max(weights),
        'min_weight': np.min(weights),
        'marginal_contribution': marginal_contribution,
        'risk_contribution': risk_contribution
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Mean-Variance Optimizer Example")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n = 5
    mu = np.array([0.08, 0.12, 0.15, 0.10, 0.09])
    
    # Random covariance matrix
    L = np.random.randn(n, n) * 0.05
    Sigma = L @ L.T + np.eye(n) * 0.01
    
    # Constraints
    constraints_config = {
        'long_only': True,
        'fully_invested': True,
        'max_weight': 0.30
    }
    
    # Optimize
    results = optimize_mean_variance(mu, Sigma, constraints_config, rf=0.03)
    
    print(f"\nOptimal weights:")
    for i, w in enumerate(results['weights']):
        print(f"  Asset {i}: {w:.2%}")
