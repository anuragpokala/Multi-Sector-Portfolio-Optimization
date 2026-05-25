"""
Reporting Module

Functions for visualization and exporting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_weights_comparison(
    weights_dict: Dict[str, np.ndarray],
    tickers: List[str],
    output_path: Optional[str] = None,
    title: str = "Portfolio Weights Comparison",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a grouped bar chart comparing portfolio weights.
    
    Parameters
    ----------
    weights_dict : Dict[str, np.ndarray]
        Dictionary mapping portfolio names to weight arrays
    tickers : List[str]
        Asset tickers
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Create DataFrame
    df = pd.DataFrame(weights_dict, index=tickers)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(tickers))
    width = 0.8 / len(weights_dict)
    
    for i, (name, weights) in enumerate(weights_dict.items()):
        offset = (i - len(weights_dict)/2 + 0.5) * width
        bars = ax.bar(x + offset, weights, width, label=name, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label if weight > 1%
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Assets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=0)
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved weights comparison plot to {output_path}")
    
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    optimal_point: Optional[Dict] = None,
    output_path: Optional[str] = None,
    title: str = "Mean-Variance Efficient Frontier",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the efficient frontier.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        DataFrame with 'volatility' and 'return' columns
    optimal_point : Dict, optional
        Optimal portfolio with 'volatility' and 'return' keys
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot frontier
    ax.plot(frontier_df['volatility'], frontier_df['return'], 
            'b-', linewidth=2, label='Efficient Frontier', alpha=0.7)
    
    # Highlight optimal portfolio
    if optimal_point:
        ax.scatter(optimal_point['volatility'], optimal_point['return'],
                  color='red', s=200, marker='*', zorder=5,
                  label=f"Optimal (Sharpe: {optimal_point.get('sharpe', 0):.3f})",
                  edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Expected Annual Volatility', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Annual Return', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved efficient frontier plot to {output_path}")
    
    return fig


def plot_correlation_heatmap(
    Sigma: np.ndarray,
    tickers: List[str],
    output_path: Optional[str] = None,
    title: str = "Asset Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot correlation heatmap.
    
    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix (N x N)
    tickers : List[str]
        Asset tickers
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Convert covariance to correlation
    std = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(std, std)
    
    # Create DataFrame
    corr_df = pd.DataFrame(corr, index=tickers, columns=tickers)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {output_path}")
    
    return fig


def plot_risk_contribution(
    risk_decomp: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Risk Contribution by Asset",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot risk contribution bar chart.
    
    Parameters
    ----------
    risk_decomp : pd.DataFrame
        Risk decomposition DataFrame (from metrics.compute_risk_decomposition)
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(risk_decomp))
    width = 0.35
    
    ax.bar(x - width/2, risk_decomp['Weight'], width, label='Weight', alpha=0.8)
    ax.bar(x + width/2, risk_decomp['% Contribution'], width, label='% Risk Contribution', alpha=0.8)
    
    ax.set_xlabel('Assets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(risk_decomp['Ticker'])
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved risk contribution plot to {output_path}")
    
    return fig


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Cumulative Returns",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot cumulative returns over time.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with returns (dates as index, assets as columns)
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Calculate cumulative returns
    cumulative = (1 + returns_df).cumprod()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=col, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative returns plot to {output_path}")
    
    return fig


def plot_view_impact(
    impact_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Black-Litterman View Impact",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the impact of views on expected returns.
    
    Parameters
    ----------
    impact_df : pd.DataFrame
        DataFrame with 'Prior (π)', 'Posterior (μ_BL)', and 'Change' columns
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Prior vs Posterior returns
    x = np.arange(len(impact_df))
    width = 0.35
    
    ax1.bar(x - width/2, impact_df['Prior (π)'], width, label='Prior (Equilibrium)', alpha=0.8)
    ax1.bar(x + width/2, impact_df['Posterior (μ_BL)'], width, label='Posterior (BL)', alpha=0.8)
    
    ax1.set_xlabel('Assets', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    ax1.set_title('Prior vs Posterior Returns', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(impact_df['Ticker'])
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Change from prior
    colors = ['green' if c > 0 else 'red' for c in impact_df['Change']]
    ax2.bar(impact_df['Ticker'], impact_df['Change'], color=colors, alpha=0.7)
    
    ax2.set_xlabel('Assets', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Change in Expected Return', fontsize=12, fontweight='bold')
    ax2.set_title('View Impact (Posterior - Prior)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved view impact plot to {output_path}")
    
    return fig


def export_results(
    weights_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict],
    tickers: List[str],
    output_dir: str = "outputs"
) -> None:
    """
    Export results to CSV files.
    
    Parameters
    ----------
    weights_dict : Dict[str, np.ndarray]
        Dictionary mapping portfolio names to weight arrays
    metrics_dict : Dict[str, Dict]
        Dictionary mapping portfolio names to metrics dictionaries
    tickers : List[str]
        Asset tickers
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Export weights
    weights_df = pd.DataFrame(weights_dict, index=tickers)
    weights_path = output_path / "portfolio_weights.csv"
    weights_df.to_csv(weights_path)
    print(f"\nExported weights to {weights_path}")
    
    # Export metrics summary
    metrics_rows = [
        'expected_annual_return',
        'expected_annual_volatility',
        'sharpe_ratio',
        'risk_adjusted_return',
        'diversification_ratio',
        'effective_n_assets',
        'max_weight',
        'turnover'
    ]
    
    metrics_data = {}
    for name, metrics in metrics_dict.items():
        metrics_data[name] = [metrics.get(key, np.nan) for key in metrics_rows]
    
    metrics_df = pd.DataFrame(metrics_data, index=metrics_rows)
    metrics_path = output_path / "portfolio_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"Exported metrics to {metrics_path}")
    
    print(f"\nAll results saved to {output_dir}/")


def create_summary_report(
    weights_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict],
    tickers: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Create a formatted text summary report.
    
    Parameters
    ----------
    weights_dict : Dict[str, np.ndarray]
        Portfolio weights
    metrics_dict : Dict[str, Dict]
        Portfolio metrics
    tickers : List[str]
        Asset tickers
    output_path : str, optional
        Path to save report text file
        
    Returns
    -------
    str
        Formatted report text
    """
    lines = []
    lines.append("="*80)
    lines.append("PORTFOLIO OPTIMIZATION SUMMARY REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Weights comparison
    lines.append("PORTFOLIO WEIGHTS")
    lines.append("-"*80)
    weights_df = pd.DataFrame(weights_dict, index=tickers)
    lines.append(weights_df.to_string(float_format=lambda x: f"{x:.2%}"))
    lines.append("")
    
    # Metrics comparison
    lines.append("PORTFOLIO METRICS")
    lines.append("-"*80)
    
    for name, metrics in metrics_dict.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Expected Annual Return:    {metrics['expected_annual_return']:>8.2%}")
        lines.append(f"  Expected Annual Volatility: {metrics['expected_annual_volatility']:>8.2%}")
        lines.append(f"  Sharpe Ratio:              {metrics['sharpe_ratio']:>8.4f}")
        lines.append(f"  Diversification Ratio:     {metrics['diversification_ratio']:>8.4f}")
        lines.append(f"  Effective # Assets:        {metrics['effective_n_assets']:>8.2f}")
        if metrics.get('turnover') is not None:
            lines.append(f"  Turnover:                  {metrics['turnover']:>8.2%}")
    
    lines.append("")
    lines.append("="*80)
    
    report_text = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nSaved summary report to {output_path}")
    
    return report_text


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Reporting Module Example")
    print("="*60)
    
    # Sample data
    tickers = ['A', 'B', 'C', 'D', 'E']
    weights_dict = {
        'Before': np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
        'MV': np.array([0.15, 0.25, 0.22, 0.28, 0.10]),
        'BL': np.array([0.18, 0.22, 0.20, 0.25, 0.15])
    }
    
    # Create plot
    fig = plot_weights_comparison(weights_dict, tickers)
    plt.show()
    
    print("\nReporting module example complete!")
