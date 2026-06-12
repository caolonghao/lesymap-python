"""
Numba-compiled statistical functions for LESYMAP-Python.

Provides high-performance implementations of univariate statistical tests.
"""

from .bm import brunner_munzel_fast
from .ttest import ttest_fast, welch_fast
from .regression import regression_fast

__all__ = [
    'brunner_munzel_fast',
    'ttest_fast',
    'welch_fast',
    'regression_fast',
]
