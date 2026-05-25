"""
Multi-Sector Portfolio Optimization

This package implements Mean-Variance, Black-Litterman, and CVaR portfolio optimization
for multiple sector sleeves (Consumer, Industrial, Healthcare, Financials).
"""

from . import cvar_optimizer
from . import bl_macro

__version__ = "1.0.0"
