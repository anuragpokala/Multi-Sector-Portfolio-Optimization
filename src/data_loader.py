"""
Data Loading Module

Functions for downloading historical price data and computing returns.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple
import warnings


def load_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    price_col: str = "Adj Close"
) -> pd.DataFrame:
    """
    Load historical adjusted close prices for given tickers using yfinance.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    price_col : str, default='Adj Close'
        Price column to extract
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns
        
    Notes
    -----
    - Missing data is forward-filled
    - If a ticker fails to download, a warning is issued
    """
    print(f"Downloading price data for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Download data for all tickers
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            prices = data[[price_col]].copy()
            prices.columns = tickers
        else:
            prices = data[price_col].copy()
        
        # Forward fill missing values
        prices = prices.fillna(method='ffill')
        
        # Drop any remaining NaN rows (at the beginning)
        prices = prices.dropna()
        
        print(f"Successfully loaded {len(prices)} trading days of data")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Check for missing tickers
        missing = set(tickers) - set(prices.columns)
        if missing:
            warnings.warn(f"Missing data for tickers: {missing}")
        
        return prices
        
    except Exception as e:
        raise RuntimeError(f"Failed to download price data: {str(e)}")


def load_from_csv(filepath: str, date_col: str = "Date") -> pd.DataFrame:
    """
    Load price data from CSV file as a fallback.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    date_col : str, default='Date'
        Name of date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns
    """
    try:
        df = pd.read_csv(filepath, parse_dates=[date_col])
        df.set_index(date_col, inplace=True)
        print(f"Loaded price data from {filepath}")
        print(f"Shape: {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file: {str(e)}")


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple"
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with prices (dates as index, tickers as columns)
    method : str, default='simple'
        Return calculation method: 'log' or 'simple'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with daily returns
        
    Notes
    -----
    - Log returns: ln(P_t / P_{t-1})
    - Simple returns: (P_t - P_{t-1}) / P_{t-1}
    - First row will be NaN and is dropped
    - Default is 'simple' to match standard backtest implementations
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'.")
    
    # Drop first row (NaN)
    returns = returns.dropna()
    
    print(f"\nReturns Summary ({method} returns):")
    print(f"Shape: {returns.shape}")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    return returns


def get_summary_statistics(returns: pd.DataFrame, annualization_factor: int = 252) -> pd.DataFrame:
    """
    Calculate summary statistics for returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with daily returns
    annualization_factor : int, default=252
        Number of trading days per year
        
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    stats = pd.DataFrame({
        'Daily Mean': returns.mean(),
        'Daily Std': returns.std(),
        'Annual Mean': returns.mean() * annualization_factor,
        'Annual Vol': returns.std() * np.sqrt(annualization_factor),
        'Sharpe (Ann, rf=0)': (returns.mean() * annualization_factor) / (returns.std() * np.sqrt(annualization_factor)),
        'Min': returns.min(),
        'Max': returns.max(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })
    
    return stats


def check_data_quality(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    max_missing_pct: float = 0.05
) -> Tuple[bool, dict]:
    """
    Perform data quality checks.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Returns data
    max_missing_pct : float, default=0.05
        Maximum allowed percentage of missing data
        
    Returns
    -------
    Tuple[bool, dict]
        (passed, diagnostics) where diagnostics contains details
    """
    diagnostics = {}
    passed = True
    
    # Check for missing values
    missing_pct = prices.isnull().sum() / len(prices)
    diagnostics['missing_pct'] = missing_pct.to_dict()
    if (missing_pct > max_missing_pct).any():
        passed = False
        diagnostics['missing_check'] = 'FAILED'
    else:
        diagnostics['missing_check'] = 'PASSED'
    
    # Check for extreme returns (potential data errors)
    extreme_threshold = 0.25  # 25% daily return
    extreme_returns = (returns.abs() > extreme_threshold).sum()
    diagnostics['extreme_returns'] = extreme_returns.to_dict()
    if extreme_returns.sum() > 0:
        warnings.warn(f"Found {extreme_returns.sum()} extreme daily returns (>25%)")
    
    # Check for sufficient data
    min_observations = 252  # At least 1 year
    if len(returns) < min_observations:
        passed = False
        diagnostics['data_length_check'] = f'FAILED (only {len(returns)} days)'
    else:
        diagnostics['data_length_check'] = f'PASSED ({len(returns)} days)'
    
    return passed, diagnostics


if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = load_prices(tickers, "2023-01-01", "2024-01-01")
    returns = compute_returns(prices)
    stats = get_summary_statistics(returns)
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(stats.round(4))
