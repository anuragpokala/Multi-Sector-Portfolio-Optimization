"""
Macro-Driven Black-Litterman Module

Implementation of Black-Litterman optimization using macroeconomic views
(specifically interest rate forecasts) for the Financials sector.

Uses PyPortfolioOpt library for efficient BL calculations.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from pypfopt import BlackLittermanModel, EfficientFrontier
from pypfopt import risk_models, expected_returns


def get_rate_betas() -> Dict[str, float]:
    """
    Return heuristic interest rate sensitivity (beta) for financial stocks.
    
    Returns
    -------
    Dict[str, float]
        Ticker -> Rate Beta mapping
        Beta = approximate % stock price move per 1% change in interest rates
        
    Notes
    -----
    Rate Beta Interpretation:
    - Negative beta: Stock moves inversely to rates (rates down = stock up)
    - Positive beta: Stock moves with rates (rates up = stock up)
    
    Sector Categories:
    - REITs (O, PLD): High negative beta (-15) - very rate sensitive
    - Capital Markets (BX, MS, AMP): Medium negative beta (-8)
    - Payments (V): Low negative beta (-2) - less rate sensitive
    - Insurers (MKL): Slight positive beta (+2) - benefits from float income
    """
    return {
        'O': -15.0,      # Realty Income (REIT): High inverse sensitivity
        'PLD': -15.0,    # Prologis (REIT): High inverse sensitivity
        'BX': -8.0,      # Blackstone (Capital Markets): Medium sensitivity
        'MS': -8.0,      # Morgan Stanley (Capital Markets): Medium sensitivity
        'AMP': -8.0,     # Ameriprise (Asset Manager): Medium sensitivity
        'V': -2.0,       # Visa (Payments): Low sensitivity
        'MKL': 2.0       # Markel (Insurer): Slight positive (float benefit)
    }


def generate_macro_views(
    tickers: List[str],
    rate_change_forecast: float,
    confidence: float = 0.60,
    rate_betas: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate Black-Litterman views based on interest rate forecast.
    
    Parameters
    ----------
    tickers : List[str]
        Asset tickers
    rate_change_forecast : float
        Expected rate change in percentage points (e.g., -0.50 for 50bps cut)
    confidence : float, default=0.60
        Confidence level (0 to 1), higher = more confident in view
    rate_betas : Dict[str, float], optional
        Custom rate betas. If None, uses default get_rate_betas()
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (viewdict, omega) where:
        - viewdict: Dictionary of ticker -> expected excess return
        - omega: Diagonal covariance matrix of view uncertainties (as Series)
        
    Notes
    -----
    View Generation Logic:
    - Expected_Return = rate_change_forecast * rate_beta
    - Negative beta * negative rate change = positive return (inverse relationship)
    
    Confidence Mapping:
    - omega (uncertainty) = base_uncertainty / confidence
    - Higher confidence → lower omega → more weight on view
    - Base uncertainty = 0.001 (0.1% variance)
    
    Example:
    - rate_change = -0.50 (50bps cut)
    - beta = -15 (REIT)
    - Expected return = -0.50 * -15 = +7.5% (REITs benefit from rate cuts)
    """
    if rate_betas is None:
        rate_betas = get_rate_betas()
    
    print(f"\n{'='*70}")
    print("MACRO VIEW GENERATION")
    print(f"{'='*70}")
    print(f"Rate Forecast: {rate_change_forecast:+.2f}% change ({abs(rate_change_forecast*100):.0f}bps {'cut' if rate_change_forecast < 0 else 'hike'})")
    print(f"Confidence Level: {confidence:.0%}")
    print(f"\n{'='*70}")
    print("VIEWS BY TICKER")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Rate Beta':<12} {'Expected Return':<18} {'View Strength'}")
    print(f"{'='*70}")
    
    # Generate views for each ticker
    viewdict = {}
    omega_dict = {}
    
    # Base uncertainty (variance)
    base_uncertainty = 0.001  # 0.1% variance = ~3.2% std dev
    
    for ticker in tickers:
        if ticker not in rate_betas:
            print(f"Warning: {ticker} not in rate_betas, skipping")
            continue
        
        beta = rate_betas[ticker]
        
        # Calculate expected return from rate change
        # Negative beta * negative rate change = positive return (intuitive)
        expected_return = rate_change_forecast * beta / 100.0  # Convert to decimal
        
        # Calculate omega (view uncertainty)
        # Higher confidence → lower omega
        omega = base_uncertainty / confidence
        
        viewdict[ticker] = expected_return
        omega_dict[ticker] = omega
        
        # Determine view strength
        abs_return = abs(expected_return)
        if abs_return > 0.05:
            strength = "Strong"
        elif abs_return > 0.02:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "Bullish" if expected_return > 0 else "Bearish" if expected_return < 0 else "Neutral"
        
        print(f"{ticker:<8} {beta:>10.1f}  {expected_return:>+15.2%}  {strength} {direction}")
    
    print(f"{'='*70}\n")
    
    # Convert to pandas Series for PyPortfolioOpt
    viewdict_series = pd.Series(viewdict)
    omega_series = pd.Series(omega_dict)
    
    return viewdict_series, omega_series


def optimize_bl_macro(
    returns: pd.DataFrame,
    current_weights: np.ndarray,
    tickers: List[str],
    rate_change_forecast: float,
    confidence: float,
    constraints_config: dict,
    delta: float = 2.5,
    tau: float = 0.05,
    rf: float = 0.045,
    Sigma: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Run Black-Litterman optimization using macroeconomic (rate) views.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical daily returns (T x N)
    current_weights : np.ndarray
        Current portfolio weights as prior (N x 1)
    tickers : List[str]
        Asset tickers
    rate_change_forecast : float
        Expected rate change in percentage points (e.g., -0.50 for 50bps cut)
    confidence : float
        Confidence in rate forecast (0 to 1)
    constraints_config : dict
        Portfolio constraints
    delta : float, default=2.5
        Risk aversion parameter
    tau : float, default=0.05
        Uncertainty scaling parameter
    rf : float, default=0.045
        Risk-free rate for Sharpe calculation
    Sigma : pd.DataFrame, optional
        Pre-computed covariance matrix. If None, will be calculated from returns.
        
    Returns
    -------
    Dict
        Optimization results including:
        - 'weights': optimal portfolio weights
        - 'posterior_returns': BL posterior expected returns
        - 'prior_returns': implied equilibrium returns
        - 'views': view dictionary
        - 'expected_return': portfolio expected return
        - 'volatility': portfolio volatility
        - 'sharpe': Sharpe ratio
        
    Notes
    -----
    Uses PyPortfolioOpt's BlackLittermanModel:
    1. Calculate implied equilibrium returns from current weights
    2. Generate macro views based on rate betas
    3. Compute BL posterior returns
    4. Optimize with constraints using EfficientFrontier
    """
    print(f"\n{'='*70}")
    print("BLACK-LITTERMAN MACRO OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Number of assets: {len(tickers)}")
    print(f"Rate forecast: {rate_change_forecast:+.2f}% ({abs(rate_change_forecast*100):.0f}bps {'cut' if rate_change_forecast < 0 else 'hike'})")
    print(f"Confidence: {confidence:.0%}")
    
    # Step 1: Get or calculate covariance matrix
    print(f"\nStep 1: Getting covariance matrix...")
    if Sigma is not None:
        # Use pre-computed covariance matrix
        S = Sigma if isinstance(Sigma, pd.DataFrame) else pd.DataFrame(Sigma, index=tickers, columns=tickers)
        print(f"  Using pre-computed covariance matrix")
    else:
        # Calculate covariance matrix using PyPortfolioOpt
        # Clean returns: drop NaN and inf values
        returns_clean = returns.dropna()
        returns_clean = returns_clean.replace([np.inf, -np.inf], np.nan).dropna()
        S = risk_models.CovarianceShrinkage(returns_clean).ledoit_wolf()
        print(f"  Calculated covariance matrix using Ledoit-Wolf")
    
    print(f"  Covariance matrix shape: {S.shape}")
    
    # Step 2: Calculate implied equilibrium returns (pi) from current weights
    # pi = delta * Sigma * w_market
    print(f"\nStep 2: Computing implied equilibrium returns...")
    print(f"  Using current portfolio weights as market prior")
    print(f"  Risk aversion (delta): {delta}")
    
    # Convert current weights to Series for PyPortfolioOpt
    current_weights_series = pd.Series(current_weights, index=tickers)
    
    # Calculate equilibrium returns manually: pi = delta * Sigma * w
    # This is the reverse optimization assumption in Black-Litterman
    if isinstance(S, pd.DataFrame):
        pi_values = delta * (S.values @ current_weights)
        pi = pd.Series(pi_values, index=tickers)
    else:
        pi_values = delta * (S @ current_weights)
        pi = pd.Series(pi_values, index=tickers)
    
    print(f"  Implied equilibrium returns:")
    for ticker in tickers:
        print(f"    {ticker}: {pi[ticker]:>8.2%}")
    
    # Step 3: Generate macro views based on rate forecast
    print(f"\nStep 3: Generating macro views...")
    viewdict, omega_series = generate_macro_views(
        tickers=tickers,
        rate_change_forecast=rate_change_forecast,
        confidence=confidence
    )
    
    # Convert omega Series to diagonal matrix (as dict for PyPortfolioOpt)
    omega = pd.DataFrame(np.diag(omega_series.values), 
                         index=omega_series.index, 
                         columns=omega_series.index)
    
    # Step 4: Compute BL posterior returns
    print(f"Step 4: Computing Black-Litterman posterior...")
    bl = BlackLittermanModel(
        cov_matrix=S,
        pi=pi,
        absolute_views=viewdict,
        omega=omega,
        tau=tau
    )
    
    # Get posterior returns
    posterior_returns = bl.bl_returns()
    
    print(f"  Posterior returns (after incorporating views):")
    for ticker in tickers:
        change = posterior_returns[ticker] - pi[ticker]
        print(f"    {ticker}: {posterior_returns[ticker]:>8.2%} (change: {change:>+7.2%})")
    
    # Step 5: Optimize portfolio with constraints
    print(f"\nStep 5: Optimizing portfolio with constraints...")
    posterior_cov = bl.bl_cov()
    
    ef = EfficientFrontier(posterior_returns, posterior_cov)
    
    # Add constraints
    max_weight = constraints_config.get('max_weight', 0.20)
    min_weight = constraints_config.get('min_weight', 0.0)
    
    ef.add_constraint(lambda w: w >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)
    
    # Optimize for maximum Sharpe ratio
    try:
        raw_weights = ef.max_sharpe(risk_free_rate=rf)
        weights_dict = ef.clean_weights()
        
        # Convert to numpy array in ticker order
        optimal_weights = np.array([weights_dict[ticker] for ticker in tickers])
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(optimal_weights * posterior_returns.values)
        portfolio_vol = np.sqrt(optimal_weights @ posterior_cov.values @ optimal_weights)
        sharpe = (portfolio_return - rf) / portfolio_vol if portfolio_vol > 0 else 0.0
        
        print(f"\nOptimization successful!")
        print(f"  Expected return: {portfolio_return:.2%}")
        print(f"  Volatility: {portfolio_vol:.2%}")
        print(f"  Sharpe ratio: {sharpe:.3f}")
        
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        print(f"Falling back to equal weight...")
        optimal_weights = np.ones(len(tickers)) / len(tickers)
        portfolio_return = np.mean(posterior_returns.values)
        portfolio_vol = 0.20
        sharpe = 0.0
    
    # Step 6: Display weight comparison
    print(f"\n{'='*70}")
    print("WEIGHT COMPARISON")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Current':<12} {'Optimized':<12} {'Change'}")
    print(f"{'='*70}")
    
    for i, ticker in enumerate(tickers):
        current = current_weights[i]
        optimized = optimal_weights[i]
        change = optimized - current
        print(f"{ticker:<8} {current:>10.2%}  {optimized:>10.2%}  {change:>+10.2%}")
    
    print(f"{'='*70}")
    print(f"{'Total':<8} {np.sum(current_weights):>10.2%}  {np.sum(optimal_weights):>10.2%}")
    print(f"{'='*70}\n")
    
    # Build results dictionary
    results = {
        'weights': optimal_weights,
        'posterior_returns': posterior_returns.values,
        'prior_returns': pi.values,
        'views': viewdict.to_dict(),
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe': sharpe,
        'rate_forecast': rate_change_forecast,
        'confidence': confidence
    }
    
    return results


def calculate_implied_returns(
    Sigma: np.ndarray,
    weights: np.ndarray,
    delta: float = 2.5
) -> np.ndarray:
    """
    Calculate implied equilibrium returns from portfolio weights.
    
    This is a standalone helper function for manual calculation (not using PyPortfolioOpt).
    
    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix (N x N)
    weights : np.ndarray
        Portfolio weights (N x 1)
    delta : float
        Risk aversion parameter
        
    Returns
    -------
    np.ndarray
        Implied returns: pi = delta * Sigma * w
        
    Notes
    -----
    Uses reverse optimization: given weights, infer what returns would justify them.
    This assumes the portfolio is in equilibrium (optimal for some expected returns).
    """
    pi = delta * (Sigma @ weights)
    return pi


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Black-Litterman Macro Module - Example Usage")
    print("="*70)
    
    # Test rate beta retrieval
    betas = get_rate_betas()
    print("\nRate Betas:")
    for ticker, beta in betas.items():
        print(f"  {ticker}: {beta:>6.1f}")
    
    # Test view generation
    test_tickers = ['O', 'PLD', 'BX', 'MS', 'AMP', 'V', 'MKL']
    
    print("\n" + "="*70)
    print("TEST SCENARIO: 50bps Rate Cut")
    print("="*70)
    
    views, omega = generate_macro_views(
        tickers=test_tickers,
        rate_change_forecast=-0.50,
        confidence=0.60
    )
    
    print("\nGenerated Views:")
    for ticker in test_tickers:
        print(f"  {ticker}: {views[ticker]:>+8.2%} (omega: {omega[ticker]:.6f})")
    
    print("\n" + "="*70)
    print("Expected Behavior:")
    print("  - REITs (O, PLD) should have strong positive views (~+7.5%)")
    print("  - Cap Markets (BX, MS, AMP) moderate positive (~+4%)")
    print("  - Payments (V) slight positive (~+1%)")
    print("  - Insurer (MKL) slight negative (~-1%)")
    print("="*70)
