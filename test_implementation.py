#!/usr/bin/env python3
"""
Quick validation script to test the implementation.
Run this before executing the main notebook to verify modules work correctly.
"""

import sys
import numpy as np
import pandas as pd
import yaml

# Import custom modules
from src import data_loader, estimators, constraints, mv_optimizer, bl_model, metrics, reporting

def test_config_loading():
    """Test configuration loading"""
    print("="*60)
    print("Test 1: Configuration Loading")
    print("="*60)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert len(config['tickers']) == 8, "Should have 8 tickers"
    assert sum(config['before_weights'].values()) > 0.99, "Weights should sum to ~1"
    
    print("✓ Configuration loaded successfully")
    print(f"  Tickers: {', '.join(config['tickers'])}")
    print(f"  Before weights sum: {sum(config['before_weights'].values()):.4f}")
    return config


def test_estimators():
    """Test covariance and returns estimation"""
    print("\n" + "="*60)
    print("Test 2: Covariance and Returns Estimation")
    print("="*60)
    
    # Generate sample returns
    np.random.seed(42)
    n_days = 252 * 3
    n_assets = 8
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    # Test covariance estimation
    Sigma, shrinkage = estimators.estimate_covariance_shrinkage(returns)
    assert Sigma.shape == (n_assets, n_assets), "Covariance should be NxN"
    assert 0 <= shrinkage <= 1, "Shrinkage should be in [0,1]"
    
    # Test returns estimation
    mu = estimators.estimate_expected_returns(returns)
    assert len(mu) == n_assets, "Should have N expected returns"
    
    # Validate covariance
    is_valid, diagnostics = estimators.validate_covariance_matrix(Sigma.values)
    assert is_valid, "Covariance matrix should be valid"
    
    print("✓ Estimators working correctly")
    print(f"  Shrinkage intensity: {shrinkage:.4f}")
    print(f"  Covariance valid: {is_valid}")
    print(f"  Min eigenvalue: {diagnostics['min_eigenvalue']:.6f}")


def test_constraints():
    """Test constraint builders"""
    print("\n" + "="*60)
    print("Test 3: Constraint Building")
    print("="*60)
    
    import cvxpy as cp
    
    n = 8
    w = cp.Variable(n)
    
    constraints_config = {
        'long_only': True,
        'fully_invested': True,
        'max_weight': 0.20,
        'min_weight': 0.0
    }
    
    constraint_list = constraints.build_constraints(
        w, n, **constraints_config
    )
    
    assert len(constraint_list) >= 3, "Should have at least 3 constraints"
    
    # Test validation
    test_weights = np.array([0.15, 0.12, 0.13, 0.11, 0.14, 0.10, 0.12, 0.13])
    is_valid, violations = constraints.validate_weights(test_weights, constraints_config)
    assert is_valid, "Valid weights should pass validation"
    
    print("✓ Constraints working correctly")
    print(f"  Built {len(constraint_list)} constraints")
    print(f"  Test weights valid: {is_valid}")


def test_optimization():
    """Test Mean-Variance optimization"""
    print("\n" + "="*60)
    print("Test 4: Mean-Variance Optimization")
    print("="*60)
    
    # Sample data
    np.random.seed(42)
    n = 8
    mu = np.random.randn(n) * 0.02 + 0.10
    L = np.random.randn(n, n) * 0.05
    Sigma = L @ L.T + np.eye(n) * 0.01
    
    constraints_config = {
        'long_only': True,
        'fully_invested': True,
        'max_weight': 0.20
    }
    
    # Run optimization
    results = mv_optimizer.optimize_mean_variance(
        mu, Sigma, constraints_config, rf=0.045,
        lambda_grid=np.logspace(-1, 1, 10)
    )
    
    assert results['status'] == 'success', "Optimization should succeed"
    assert len(results['weights']) == n, "Should have N weights"
    assert abs(np.sum(results['weights']) - 1.0) < 1e-4, "Weights should sum to 1"
    assert np.all(results['weights'] >= -1e-6), "Weights should be non-negative"
    
    print("✓ Mean-Variance optimization working")
    print(f"  Optimal return: {results['return']:.2%}")
    print(f"  Optimal volatility: {results['volatility']:.2%}")
    print(f"  Sharpe ratio: {results['sharpe']:.4f}")


def test_black_litterman():
    """Test Black-Litterman model"""
    print("\n" + "="*60)
    print("Test 5: Black-Litterman Model")
    print("="*60)
    
    # Sample data
    np.random.seed(42)
    n = 4
    tickers = ['A', 'B', 'C', 'D']
    w_market = np.array([0.25, 0.25, 0.25, 0.25])
    
    L = np.random.randn(n, n) * 0.05
    Sigma = L @ L.T + np.eye(n) * 0.01
    
    # Compute equilibrium
    pi = bl_model.compute_equilibrium_returns(Sigma, w_market, delta=2.5)
    assert len(pi) == n, "Should have N equilibrium returns"
    
    # Encode views
    views_config = {
        'A': {'type': 'absolute', 'return': 0.05, 'confidence': 0.0004}
    }
    P, q, Omega = bl_model.encode_views(tickers, views_config, n)
    assert P.shape == (1, n), "Should have 1 view"
    assert len(q) == 1, "Should have 1 view return"
    
    # Compute posterior
    mu_BL, Sigma_BL = bl_model.compute_bl_posterior(pi, Sigma, P, q, Omega, tau=0.05)
    assert len(mu_BL) == n, "Should have N posterior returns"
    assert mu_BL[0] > pi[0], "View should increase expected return for asset A"
    
    print("✓ Black-Litterman model working")
    print(f"  Equilibrium returns: {pi.min():.2%} to {pi.max():.2%}")
    print(f"  Posterior returns: {mu_BL.min():.2%} to {mu_BL.max():.2%}")


def test_metrics():
    """Test metrics calculations"""
    print("\n" + "="*60)
    print("Test 6: Metrics Calculation")
    print("="*60)
    
    # Sample data
    np.random.seed(42)
    n = 5
    weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    mu = np.array([0.08, 0.10, 0.12, 0.09, 0.11])
    L = np.random.randn(n, n) * 0.05
    Sigma = L @ L.T + np.eye(n) * 0.01
    
    portfolio_metrics = metrics.compute_portfolio_metrics(
        weights, mu, Sigma, rf=0.045
    )
    
    assert 'expected_annual_return' in portfolio_metrics
    assert 'expected_annual_volatility' in portfolio_metrics
    assert 'sharpe_ratio' in portfolio_metrics
    assert portfolio_metrics['sharpe_ratio'] > 0, "Sharpe should be positive"
    
    print("✓ Metrics calculation working")
    print(f"  Expected return: {portfolio_metrics['expected_annual_return']:.2%}")
    print(f"  Volatility: {portfolio_metrics['expected_annual_volatility']:.2%}")
    print(f"  Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.4f}")


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# Consumer Portfolio Optimization - Implementation Test")
    print("#"*60 + "\n")
    
    try:
        config = test_config_loading()
        test_estimators()
        test_constraints()
        test_optimization()
        test_black_litterman()
        test_metrics()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe implementation is working correctly.")
        print("You can now run the main notebook: notebooks/portfolio_optimization.ipynb")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED ✗")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
