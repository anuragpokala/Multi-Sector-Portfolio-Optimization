"""
Hierarchical Risk Parity (HRP) Portfolio Optimization

Implements HRP as described in Marcos Lopez de Prado's work.
HRP uses hierarchical clustering and recursive bisection to allocate
risk across the portfolio.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional


def optimize_hrp(
    returns: pd.DataFrame,
    Sigma: np.ndarray,
    distance_metric: str = 'correlation'
) -> Dict:
    """
    Optimize portfolio using Hierarchical Risk Parity.
    
    HRP combines graph theory and machine learning to build
    diversified portfolios without the need for matrix inversion.
    
    Steps:
    1. Compute distance matrix from correlation matrix
    2. Perform hierarchical clustering
    3. Quasi-diagonalize the covariance matrix
    4. Recursive bisection to allocate weights
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns (dates x assets)
    Sigma : np.ndarray
        Covariance matrix
    distance_metric : str
        Distance metric ('correlation' or 'euclidean')
        
    Returns
    -------
    dict
        Dictionary with 'weights' (np.ndarray), 'linkage_matrix', and 'metrics'
    """
    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    
    # Step 1: Calculate correlation matrix from covariance
    volatilities = np.sqrt(np.diag(Sigma))
    corr_matrix = Sigma / np.outer(volatilities, volatilities)
    
    # Step 2: Convert correlation to distance matrix
    if distance_metric == 'correlation':
        # Distance = sqrt(0.5 * (1 - correlation))
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    else:
        # Use correlation matrix directly as distances
        dist_matrix = 1 - np.abs(corr_matrix)
    
    # Ensure diagonal is 0
    np.fill_diagonal(dist_matrix, 0)
    
    # Convert to condensed distance matrix for scipy
    dist_condensed = squareform(dist_matrix, checks=False)
    
    # Step 3: Hierarchical clustering
    link = linkage(dist_condensed, method='single')
    
    # Step 4: Quasi-diagonalization (get optimal leaf order)
    sorted_indices = _get_quasi_diag(link)
    
    # Step 5: Recursive bisection to get weights
    weights = _get_rec_bipart(Sigma, sorted_indices)
    
    # Reorder weights to match original ticker order
    weights_dict = {tickers[i]: weights[i] for i in range(n_assets)}
    weights_array = np.array([weights_dict[ticker] for ticker in tickers])
    
    # Calculate portfolio metrics
    portfolio_return = np.dot(weights_array, returns.mean() * 252)
    portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(Sigma, weights_array)))
    
    metrics = {
        'expected_annual_return': portfolio_return,
        'expected_annual_volatility': portfolio_vol,
        'diversification_ratio': _calculate_diversification_ratio(weights_array, Sigma)
    }
    
    return {
        'weights': weights_array,
        'linkage_matrix': link,
        'sorted_indices': sorted_indices,
        'metrics': metrics,
        'method': 'HRP'
    }


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """
    Reorder the covariance matrix based on hierarchical clustering.
    
    This creates a quasi-diagonal matrix where similar assets
    are placed next to each other.
    
    Parameters
    ----------
    link : np.ndarray
        Linkage matrix from hierarchical clustering
        
    Returns
    -------
    list
        Sorted indices representing optimal leaf order
    """
    link = link.astype(int)
    sorted_indices = _get_cluster_leaves(link, link.shape[0])
    return sorted_indices


def _get_cluster_leaves(link: np.ndarray, cluster_id: int) -> List[int]:
    """
    Recursively get the leaves (original indices) of a cluster.
    
    Parameters
    ----------
    link : np.ndarray
        Linkage matrix
    cluster_id : int
        ID of the cluster (indices >= n_samples refer to merged clusters)
        
    Returns
    -------
    list
        List of original indices in this cluster
    """
    n_samples = link.shape[0] + 1
    
    if cluster_id < n_samples:
        # This is an original sample (leaf node)
        return [cluster_id]
    
    # This is a merged cluster
    cluster_idx = cluster_id - n_samples
    left = int(link[cluster_idx, 0])
    right = int(link[cluster_idx, 1])
    
    # Recursively get leaves from both branches
    return _get_cluster_leaves(link, left) + _get_cluster_leaves(link, right)


def _get_rec_bipart(Sigma: np.ndarray, sorted_indices: List[int]) -> np.ndarray:
    """
    Compute HRP weights using recursive bisection.
    
    Allocate weight to each cluster based on inverse variance,
    then recursively split within clusters.
    
    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix
    sorted_indices : list
        Sorted indices from quasi-diagonalization
        
    Returns
    -------
    np.ndarray
        Portfolio weights
    """
    n_assets = len(sorted_indices)
    weights = pd.Series(1.0, index=sorted_indices)
    
    # Build clusters recursively
    clusters = [sorted_indices]
    
    while len(clusters) > 0:
        # Pop two clusters
        clusters = [c[int(len(c)/2):] + [c[:int(len(c)/2)]] 
                   if len(c) > 1 else c for c in clusters]
        
        # Flatten the list
        clusters = [item for sublist in clusters for item in 
                   ([sublist] if isinstance(sublist, list) and 
                    all(isinstance(x, int) for x in sublist) else sublist)]
        
        # Break if we can't split further
        if all(len(c) == 1 for c in clusters):
            break
        
        # Allocate weight to each cluster
        for i in range(0, len(clusters), 2):
            if i + 1 < len(clusters):
                cluster_0 = clusters[i]
                cluster_1 = clusters[i + 1]
                
                # Calculate cluster variances
                var_0 = _get_cluster_var(Sigma, cluster_0)
                var_1 = _get_cluster_var(Sigma, cluster_1)
                
                # Allocate inversely proportional to variance
                alpha = 1 - var_0 / (var_0 + var_1)
                
                # Update weights
                weights[cluster_0] *= alpha
                weights[cluster_1] *= (1 - alpha)
    
    return weights.values


def _get_cluster_var(Sigma: np.ndarray, cluster_indices: List[int]) -> float:
    """
    Calculate the variance of a cluster (subset of assets).
    
    Assumes equal weighting within the cluster.
    
    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix
    cluster_indices : list
        Indices of assets in the cluster
        
    Returns
    -------
    float
        Variance of equally-weighted cluster portfolio
    """
    # Extract submatrix for this cluster
    Sigma_cluster = Sigma[np.ix_(cluster_indices, cluster_indices)]
    
    # Equal weights within cluster
    n = len(cluster_indices)
    w = np.ones(n) / n
    
    # Portfolio variance
    return np.dot(w, np.dot(Sigma_cluster, w))


def _calculate_diversification_ratio(weights: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Calculate diversification ratio.
    
    DR = (weighted average volatility) / (portfolio volatility)
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    Sigma : np.ndarray
        Covariance matrix
        
    Returns
    -------
    float
        Diversification ratio (higher is more diversified)
    """
    volatilities = np.sqrt(np.diag(Sigma))
    weighted_avg_vol = np.dot(weights, volatilities)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
    
    return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
