"""
CVaR (Conditional Value-at-Risk) Optimization Module

Implementation of CVaR portfolio optimization using the Rockafellar-Uryasev (2000) formulation.
CVaR minimizes expected shortfall (tail risk) rather than variance.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict
from .constraints import build_constraints


def optimize_cvar(
    returns: pd.DataFrame,
    constraints_config: dict,
    confidence_level: float = 0.95,
    w_before: Optional[np.ndarray] = None
) -> Dict:
    """
    Perform CVaR (Conditional Value-at-Risk) portfolio optimization.
    
    Minimizes expected shortfall in the tail of the return distribution using
    the Rockafellar-Uryasev (2000) linear programming formulation.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical daily returns (T x N), where T = number of scenarios, N = number of assets
    constraints_config : dict
        Constraint parameters (long_only, max_weight, etc.)
    confidence_level : float, default=0.95
        Confidence level for CVaR (e.g., 0.95 = 95% confidence, focus on worst 5%)
    w_before : np.ndarray, optional
        Baseline weights for turnover constraint (N x 1)
        
    Returns
    -------
    Dict
        Results including:
        - 'weights': optimal portfolio weights
        - 'cvar_value': minimized CVaR (expected shortfall)
        - 'var_value': Value-at-Risk threshold (alpha)
        - 'expected_return': portfolio expected return
        - 'volatility': portfolio volatility
        - 'confidence_level': confidence level used
        
    Notes
    -----
    Rockafellar-Uryasev (2000) formulation:
    - Minimize: alpha + (1 / ((1 - beta) * T)) * sum(z_t)
    - Where:
        - alpha = VaR (Value-at-Risk threshold)
        - beta = confidence level (0.95)
        - T = number of scenarios (trading days)
        - z_t = auxiliary variable (loss exceeding VaR in scenario t)
    
    Constraints:
    - z_t >= 0 for all t
    - z_t >= -r_t' * w - alpha for all t (loss in scenario t exceeding VaR)
    - Standard portfolio constraints (long-only, fully invested, position limits)
    
    CVaR represents the expected loss in the worst (1-beta)% of cases.
    Unlike variance, CVaR is asymmetric and focuses specifically on downside risk.
    """
    # Convert returns to numpy array
    returns_array = returns.values  # Shape: (T, N)
    T, n_assets = returns_array.shape
    
    print(f"\n{'='*60}")
    print("CVaR (Conditional Value-at-Risk) Optimization")
    print(f"{'='*60}")
    print(f"Number of assets: {n_assets}")
    print(f"Number of scenarios (trading days): {T}")
    print(f"Confidence level: {confidence_level:.1%}")
    print(f"Tail risk focus: worst {(1-confidence_level):.1%} of scenarios")
    
    # Decision variables
    w = cp.Variable(n_assets)  # Portfolio weights
    alpha = cp.Variable()  # VaR (Value-at-Risk threshold)
    z = cp.Variable(T)  # Auxiliary variables for CVaR calculation
    
    # Build standard portfolio constraints
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
    
    # Add CVaR-specific constraints
    # Constraint 1: z_t >= 0 for all t
    constraints.append(z >= 0)
    
    # Constraint 2: z_t >= -returns_t @ w - alpha for all t
    # This captures the loss in scenario t that exceeds VaR
    # Note: We use negative returns because we're measuring losses
    for t in range(T):
        portfolio_return_t = returns_array[t, :] @ w
        # Loss = -return, and z_t captures loss exceeding VaR (alpha)
        constraints.append(z[t] >= -portfolio_return_t - alpha)
    
    # Objective: Minimize CVaR using Rockafellar-Uryasev formulation
    # CVaR = alpha + (1 / ((1 - beta) * T)) * sum(z_t)
    tail_weight = 1.0 / ((1.0 - confidence_level) * T)
    cvar_objective = alpha + tail_weight * cp.sum(z)
    
    objective = cp.Minimize(cvar_objective)
    
    # Solve optimization problem
    print(f"\nSolving CVaR optimization problem...")
    problem = cp.Problem(objective, constraints)
    
    # Try multiple solvers in order of preference
    solved = False
    solver_status = None
    
    for solver in [cp.ECOS, cp.SCS, cp.OSQP]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solved = True
                solver_status = f"{solver} (status: {problem.status})"
                break
        except Exception as e:
            continue
    
    if not solved:
        raise RuntimeError(
            f"CVaR optimization failed. Status: {problem.status}. "
            f"Check constraints and data quality."
        )
    
    # Extract optimal solution
    w_opt = w.value
    alpha_opt = alpha.value
    cvar_opt = problem.value
    
    # Calculate portfolio statistics
    portfolio_returns = returns_array @ w_opt
    expected_return = np.mean(portfolio_returns) * 252  # Annualized
    volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
    
    # Calculate empirical CVaR and VaR for validation
    losses = -portfolio_returns  # Convert returns to losses
    sorted_losses = np.sort(losses)
    var_threshold_idx = int(np.ceil(confidence_level * T))
    empirical_var = sorted_losses[var_threshold_idx] if var_threshold_idx < T else sorted_losses[-1]
    tail_losses = sorted_losses[var_threshold_idx:]
    empirical_cvar = np.mean(tail_losses) if len(tail_losses) > 0 else empirical_var
    
    print(f"\n{'='*60}")
    print("Optimization Complete")
    print(f"{'='*60}")
    print(f"Solver: {solver_status}")
    print(f"CVaR ({confidence_level:.0%}): {cvar_opt:.6f} (daily loss)")
    print(f"VaR ({confidence_level:.0%}):  {alpha_opt:.6f} (daily loss)")
    print(f"Expected return: {expected_return:.2%} (annualized)")
    print(f"Volatility: {volatility:.2%} (annualized)")
    print(f"\nValidation (empirical from historical data):")
    print(f"  Empirical VaR:  {empirical_var:.6f}")
    print(f"  Empirical CVaR: {empirical_cvar:.6f}")
    print(f"  Difference: {abs(cvar_opt - empirical_cvar):.6f}")
    
    # Verify constraints
    print(f"\nConstraint Verification:")
    print(f"  Weights sum: {np.sum(w_opt):.6f} (should be 1.0)")
    print(f"  Min weight: {np.min(w_opt):.6f} (should be >= 0)")
    print(f"  Max weight: {np.max(w_opt):.6f} (should be <= {constraints_config.get('max_weight', 1.0)})")
    print(f"  CVaR >= VaR: {cvar_opt >= alpha_opt} (mathematical property)")
    
    # Build results dictionary
    results = {
        'weights': w_opt,
        'cvar_value': cvar_opt,
        'var_value': alpha_opt,
        'expected_return': expected_return,
        'volatility': volatility,
        'confidence_level': confidence_level,
        'solver': solver_status,
        'empirical_var': empirical_var,
        'empirical_cvar': empirical_cvar
    }
    
    print(f"{'='*60}\n")
    
    return results


def calculate_portfolio_cvar(
    weights: np.ndarray,
    returns: pd.DataFrame,
    confidence_level: float = 0.95
) -> Dict:
    """
    Calculate CVaR and VaR for a given portfolio.
    
    Useful for backtesting or analyzing existing portfolios.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N x 1)
    returns : pd.DataFrame
        Historical daily returns (T x N)
    confidence_level : float, default=0.95
        Confidence level
        
    Returns
    -------
    Dict
        CVaR metrics including VaR, CVaR, and tail statistics
    """
    returns_array = returns.values
    portfolio_returns = returns_array @ weights
    
    # Calculate losses (negative returns)
    losses = -portfolio_returns
    sorted_losses = np.sort(losses)
    
    T = len(losses)
    var_threshold_idx = int(np.ceil(confidence_level * T))
    
    # VaR: threshold loss at confidence level
    var = sorted_losses[var_threshold_idx] if var_threshold_idx < T else sorted_losses[-1]
    
    # CVaR: expected loss beyond VaR
    tail_losses = sorted_losses[var_threshold_idx:]
    cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
    
    # Additional statistics
    max_loss = np.max(losses)
    tail_size = len(tail_losses)
    
    return {
        'var': var,
        'cvar': cvar,
        'max_loss': max_loss,
        'tail_size': tail_size,
        'confidence_level': confidence_level
    }


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("CVaR Optimizer Module - Example Usage")
    print("="*70)
    
    # Generate synthetic returns data
    np.random.seed(42)
    T = 252  # One year of daily data
    n_assets = 5
    
    # Simulate returns with some fat tails
    returns_data = np.random.standard_t(df=5, size=(T, n_assets)) * 0.02
    returns_df = pd.DataFrame(returns_data, columns=[f'Asset{i+1}' for i in range(n_assets)])
    
    print(f"\nSynthetic data: {T} scenarios, {n_assets} assets")
    print(f"Return statistics:")
    print(returns_df.describe())
    
    # Define constraints
    constraints_config = {
        'long_only': True,
        'fully_invested': True,
        'max_weight': 0.40,
        'min_weight': 0.0
    }
    
    # Run CVaR optimization
    try:
        results = optimize_cvar(
            returns=returns_df,
            constraints_config=constraints_config,
            confidence_level=0.95
        )
        
        print("\nOptimal weights:")
        for i, w in enumerate(results['weights']):
            print(f"  Asset{i+1}: {w:.4f} ({w*100:.2f}%)")
    except Exception as e:
        print(f"\nError: {str(e)}")
