"""
Black-Litterman Model Module

Implementation of Black-Litterman portfolio optimization with analyst views.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
from .constraints import build_constraints


def compute_equilibrium_returns(
    Sigma: np.ndarray,
    w_market: np.ndarray,
    delta: float = 2.5
) -> np.ndarray:
    """
    Compute market-implied equilibrium returns (prior).
    
    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix (N x N)
    w_market : np.ndarray
        Market (or baseline) portfolio weights (N x 1)
    delta : float, default=2.5
        Risk aversion parameter
        
    Returns
    -------
    np.ndarray
        Equilibrium returns π = δ * Σ * w_market
        
    Notes
    -----
    - Delta represents market risk aversion (typical values: 2-4)
    - Higher delta implies investors require more return for risk
    - Equilibrium returns assume the market is in equilibrium
    """
    pi = delta * (Sigma @ w_market)
    
    print(f"\nEquilibrium (Prior) Returns:")
    print(f"  Risk aversion (delta): {delta}")
    print(f"  Equilibrium returns: {pi.min():.2%} to {pi.max():.2%}")
    print(f"  Mean equilibrium return: {pi.mean():.2%}")
    
    return pi


def encode_views(
    tickers: List[str],
    views_config: Dict,
    n_assets: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode analyst views into P, q, Ω matrices.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    views_config : Dict
        View specifications: {ticker: {'type': 'absolute', 'return': 0.05, 'confidence': 0.001}}
    n_assets : int
        Number of assets
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (P, q, Omega) where:
        - P: View matrix (K x N), picks out assets involved in views
        - q: View returns (K x 1), expected returns from views
        - Omega: View uncertainty (K x K), diagonal confidence matrix
        
    Notes
    -----
    Two types of views:
    1. Absolute: "Asset i will return X%"
       P[k, i] = 1, q[k] = X
    2. Relative: "Asset i will outperform asset j by X%"
       P[k, i] = 1, P[k, j] = -1, q[k] = X
       
    Confidence: Lower Ω values = higher confidence
    """
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    views = []
    for ticker, view_spec in views_config.items():
        if ticker not in ticker_to_idx:
            print(f"Warning: View specified for unknown ticker {ticker}, skipping")
            continue
        views.append((ticker, view_spec))
    
    K = len(views)  # Number of views
    
    # Initialize matrices
    P = np.zeros((K, n_assets))
    q = np.zeros(K)
    Omega = np.zeros((K, K))
    
    print(f"\nEncoding {K} analyst views:")
    
    for k, (ticker, view_spec) in enumerate(views):
        view_type = view_spec.get('type', 'absolute')
        view_return = view_spec['return']
        confidence = view_spec['confidence']
        
        if view_type == 'absolute':
            # Absolute view: asset i will return q[k]
            idx = ticker_to_idx[ticker]
            P[k, idx] = 1.0
            q[k] = view_return
            Omega[k, k] = confidence
            
            print(f"  {k+1}. {ticker}: absolute return = {view_return:+.2%}, confidence σ² = {confidence:.6f}")
            
        elif view_type == 'relative':
            # Relative view: asset i will outperform asset j by q[k]
            idx_i = ticker_to_idx[ticker]
            ticker_j = view_spec['relative_to']
            
            if ticker_j not in ticker_to_idx:
                print(f"Warning: Relative view references unknown ticker {ticker_j}, skipping")
                continue
            
            idx_j = ticker_to_idx[ticker_j]
            P[k, idx_i] = 1.0
            P[k, idx_j] = -1.0
            q[k] = view_return
            Omega[k, k] = confidence
            
            print(f"  {k+1}. {ticker} vs {ticker_j}: relative return = {view_return:+.2%}, confidence σ² = {confidence:.6f}")
        
        else:
            raise ValueError(f"Unknown view type: {view_type}")
    
    return P, q, Omega


def compute_bl_posterior(
    pi: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Black-Litterman posterior expected returns.
    
    Parameters
    ----------
    pi : np.ndarray
        Equilibrium (prior) returns (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    P : np.ndarray
        View matrix (K x N)
    q : np.ndarray
        View returns (K x 1)
    Omega : np.ndarray
        View uncertainty matrix (K x K)
    tau : float, default=0.05
        Scaling parameter for uncertainty in equilibrium
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (μ_BL, Σ_BL) - posterior expected returns and covariance
        
    Notes
    -----
    Black-Litterman formula:
    μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}q]
    
    Posterior covariance:
    Σ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} + Σ
    
    - Tau represents uncertainty in the equilibrium (typical: 0.01-0.05)
    - Lower tau = more confidence in equilibrium
    - The posterior is a weighted average of prior and views
    """
    n_assets = len(pi)
    
    # Scale covariance by tau (represents uncertainty in equilibrium)
    tau_Sigma = tau * Sigma
    
    # Compute precision matrices (inverse covariances)
    try:
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        # Handle singular matrices
        tau_Sigma_inv = np.linalg.pinv(tau_Sigma)
        Omega_inv = np.linalg.pinv(Omega)
        print("Warning: Using pseudo-inverse for singular matrices")
    
    # Posterior precision
    precision_posterior = tau_Sigma_inv + P.T @ Omega_inv @ P
    
    # Posterior covariance of returns
    try:
        cov_posterior = np.linalg.inv(precision_posterior)
    except np.linalg.LinAlgError:
        cov_posterior = np.linalg.pinv(precision_posterior)
    
    # Posterior expected returns (Black-Litterman formula)
    mu_BL = cov_posterior @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ q)
    
    # Posterior covariance (optional, usually we still use original Sigma for optimization)
    Sigma_BL = cov_posterior + Sigma
    
    print(f"\nBlack-Litterman Posterior:")
    print(f"  Tau (uncertainty scaling): {tau}")
    print(f"  Posterior returns: {mu_BL.min():.2%} to {mu_BL.max():.2%}")
    print(f"  Mean posterior return: {mu_BL.mean():.2%}")
    print(f"  Change from prior: {(mu_BL - pi).min():.2%} to {(mu_BL - pi).max():.2%}")
    
    return mu_BL, Sigma_BL


def optimize_black_litterman(
    pi: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
    constraints_config: dict,
    rf: float = 0.0,
    w_before: Optional[np.ndarray] = None,
    use_posterior_cov: bool = False
) -> Dict:
    """
    Perform Black-Litterman portfolio optimization.
    
    Parameters
    ----------
    pi : np.ndarray
        Equilibrium (prior) returns (N x 1)
    Sigma : np.ndarray
        Covariance matrix (N x N)
    P : np.ndarray
        View matrix (K x N)
    q : np.ndarray
        View returns (K x 1)
    Omega : np.ndarray
        View uncertainty matrix (K x K)
    tau : float
        Scaling parameter
    constraints_config : dict
        Constraint parameters
    rf : float, default=0.0
        Risk-free rate
    w_before : np.ndarray, optional
        Baseline weights for turnover constraint
    use_posterior_cov : bool, default=False
        If True, use posterior covariance; if False, use original Sigma
        
    Returns
    -------
    Dict
        Optimization results
        
    Notes
    -----
    - Computes BL posterior expected returns
    - Optimizes using mean-variance with posterior returns
    - Typically use original Sigma (not posterior) for optimization
    """
    print(f"\n{'='*60}")
    print("Black-Litterman Optimization")
    print(f"{'='*60}")
    
    # Compute BL posterior
    mu_BL, Sigma_BL = compute_bl_posterior(pi, Sigma, P, q, Omega, tau)
    
    # Choose covariance matrix
    cov_matrix = Sigma_BL if use_posterior_cov else Sigma
    
    n_assets = len(mu_BL)
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
    
    # Objective: maximize Sharpe ratio (equivalently, maximize return for given risk)
    # We'll use a fixed risk aversion parameter
    # Alternatively, sweep lambda like in MV optimization
    
    # For simplicity, we'll maximize: mu' * w - 0.5 * lambda * w' * Sigma * w
    # with lambda = 1 (can be tuned)
    lam = 1.0
    portfolio_return = mu_BL @ w
    portfolio_risk = 0.5 * cp.quad_form(w, cov_matrix)
    objective = cp.Maximize(portfolio_return - lam * portfolio_risk)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        # Try multiple solvers
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
        ret = mu_BL @ w_opt
        vol = np.sqrt(w_opt @ cov_matrix @ w_opt)
        sharpe = (ret - rf) / vol if vol > 1e-8 else 0.0
        
        print(f"\n{'='*60}")
        print("Optimization Complete")
        print(f"{'='*60}")
        print(f"Expected return: {ret:.2%}")
        print(f"Expected volatility: {vol:.2%}")
        print(f"Sharpe ratio: {sharpe:.4f}")
        
        results = {
            'weights': w_opt,
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'mu_posterior': mu_BL,
            'Sigma_posterior': Sigma_BL,
            'pi_prior': pi,
            'status': problem.status
        }
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Black-Litterman optimization failed: {str(e)}")


def analyze_view_impact(
    pi: np.ndarray,
    mu_BL: np.ndarray,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Analyze the impact of views on expected returns.
    
    Parameters
    ----------
    pi : np.ndarray
        Prior (equilibrium) returns
    mu_BL : np.ndarray
        Posterior (BL) returns
    tickers : List[str]
        Asset tickers
        
    Returns
    -------
    pd.DataFrame
        Comparison of prior vs posterior returns
    """
    df = pd.DataFrame({
        'Ticker': tickers,
        'Prior (π)': pi,
        'Posterior (μ_BL)': mu_BL,
        'Change': mu_BL - pi,
        'Change (%)': (mu_BL - pi) / np.abs(pi) * 100
    })
    
    return df


def optimize_black_litterman_sentiment(
    pi: np.ndarray,
    Sigma: np.ndarray,
    combined_views: Dict,
    tickers: List[str],
    tau: float,
    constraints_config: Dict,
    risk_free_rate: float,
    before_weights: np.ndarray,
    delta: float
) -> Dict:
    """
    Black-Litterman optimization with sentiment-enhanced views.
    
    This is a wrapper around the standard BL optimization that accepts
    pre-computed combined views from sentiment analysis.
    
    Parameters
    ----------
    pi : np.ndarray
        Equilibrium (prior) returns
    Sigma : np.ndarray
        Covariance matrix
    combined_views : Dict
        Combined views from sentiment + analyst in format:
        {ticker: {'type': 'absolute', 'return': float, 'confidence': float}}
    tickers : List[str]
        List of ticker symbols
    tau : float
        BL scaling parameter
    constraints_config : Dict
        Portfolio constraints
    risk_free_rate : float
        Risk-free rate
    before_weights : np.ndarray
        Current portfolio weights
    delta : float
        Risk aversion parameter
        
    Returns
    -------
    Dict
        Optimization results including weights, posterior returns, metrics
        
    Notes
    -----
    The combined_views should already be a blend of:
    - Sentiment-based views (from news analysis)
    - Analyst views (manual input)
    
    This function simply reformats them for the standard BL optimization.
    """
    print("\n" + "="*70)
    print("BLACK-LITTERMAN OPTIMIZATION (SENTIMENT-ENHANCED)")
    print("="*70)
    
    # Convert combined views to format expected by encode_views
    views_config = {}
    for ticker, view_data in combined_views.items():
        views_config[ticker] = {
            'type': view_data['type'],
            'return': view_data['return'],
            'confidence': view_data['confidence'],
            'description': view_data.get('description', '')
        }
    
    # Encode views
    n_assets = len(tickers)
    P, q, Omega = encode_views(tickers, views_config, n_assets)
    
    # Compute BL posterior
    mu_BL, Sigma_BL = compute_bl_posterior(pi, Sigma, P, q, Omega, tau)
    
    print(f"\nPosterior Returns (Sentiment-Enhanced):")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {mu_BL[i]:.2%} (prior: {pi[i]:.2%}, change: {mu_BL[i]-pi[i]:+.2%})")
    
    # Optimize portfolio using posterior returns
    print(f"\nOptimizing portfolio with {len(combined_views)} sentiment-enhanced views...")
    
    w = cp.Variable(n_assets)
    
    # Objective: maximize return - risk (with risk aversion delta)
    objective = cp.Maximize(mu_BL @ w - (delta / 2) * cp.quad_form(w, Sigma))
    
    # Build constraints
    constraint_list = build_constraints(
        w=w,
        n_assets=n_assets,
        long_only=constraints_config.get('long_only', True),
        fully_invested=constraints_config.get('fully_invested', True),
        max_weight=constraints_config.get('max_weight'),
        min_weight=constraints_config.get('min_weight'),
        turnover_cap=constraints_config.get('turnover_cap'),
        w_before=before_weights
    )
    
    # Solve
    problem = cp.Problem(objective, constraint_list)
    
    # Try multiple solvers for robustness
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            problem.solve(solver=solver)
            if problem.status == 'optimal':
                print(f"  ✓ Optimization successful (solver: {solver})")
                break
        except Exception as e:
            print(f"  Solver {solver} failed: {e}")
            continue
    
    if problem.status != 'optimal':
        raise RuntimeError("Sentiment BL optimization failed. Check constraints and data.")
    
    # Extract results
    weights = w.value
    expected_return = mu_BL @ weights
    expected_vol = np.sqrt(weights @ Sigma @ weights)
    sharpe = (expected_return - risk_free_rate) / expected_vol if expected_vol > 0 else 0
    
    print(f"\n{'='*70}")
    print("Optimization Complete (Sentiment-Enhanced)")
    print(f"{'='*70}")
    print(f"Expected return: {expected_return:.2%}")
    print(f"Expected volatility: {expected_vol:.2%}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    print(f"\nSentiment-Enhanced BL Optimal Weights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {weights[i]:.2%}")
    
    return {
        'weights': weights,
        'expected_return': expected_return,
        'expected_volatility': expected_vol,
        'sharpe_ratio': sharpe,
        'posterior_returns': mu_BL,
        'posterior_covariance': Sigma_BL,
        'views': combined_views,
        'P': P,
        'q': q,
        'Omega': Omega
    }


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Black-Litterman Model Example")
    print("="*60)
    
    # Sample data
    np.random.seed(42)
    n = 4
    tickers = ['A', 'B', 'C', 'D']
    
    # Market weights (baseline portfolio)
    w_market = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Covariance matrix
    L = np.random.randn(n, n) * 0.05
    Sigma = L @ L.T + np.eye(n) * 0.01
    
    # Compute equilibrium returns
    pi = compute_equilibrium_returns(Sigma, w_market, delta=2.5)
    
    # Define views
    views_config = {
        'A': {'type': 'absolute', 'return': 0.05, 'confidence': 0.0004},
        'C': {'type': 'absolute', 'return': -0.02, 'confidence': 0.0009}
    }
    
    P, q, Omega = encode_views(tickers, views_config, n)
    
    print(f"\nP matrix (view picks):\n{P}")
    print(f"\nq vector (view returns):\n{q}")
    print(f"\nOmega matrix (view confidence):\n{Omega}")
    
    # Compute posterior
    mu_BL, Sigma_BL = compute_bl_posterior(pi, Sigma, P, q, Omega, tau=0.05)
    
    # Analyze impact
    impact = analyze_view_impact(pi, mu_BL, tickers)
    print(f"\nView Impact:")
    print(impact.round(4))
