"""
Numba-compiled t-test implementations for LESYMAP-Python.

Port of C++ implementation from LESYMAP/src/TTfast.cpp
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _sample_variance(arr: np.ndarray) -> float:
    """
    Compute sample variance with ddof=1 (Numba-compatible).

    This is equivalent to np.var(arr, ddof=1) but works in Numba nopython mode.
    """
    n = len(arr)
    if n <= 1:
        return 0.0
    mean = np.mean(arr)
    ss = 0.0
    for i in range(n):
        diff = arr[i] - mean
        ss += diff * diff
    return ss / (n - 1.0)


def ttest_fast(X: np.ndarray,
               y: np.ndarray,
               compute_dof: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Student's t-test using Numba JIT compilation.

    Tests whether the mean behavioral score differs between
    lesioned and non-lesioned groups at each voxel, assuming
    equal variances.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_voxels)
        Binary lesion matrix (1 = lesioned, 0 = not lesioned)
    y : ndarray, shape (n_subjects,)
        Behavioral scores
    compute_dof : bool
        Whether to compute degrees of freedom

    Returns
    -------
    statistic : ndarray, shape (n_voxels,)
        t-statistics for each voxel
    df : ndarray, shape (n_voxels,)
        Degrees of freedom (N-2 for all voxels)

    Notes
    -----
    The t-statistic is calculated as:
        t = (mean0 - mean1) / sqrt(var_pooled * (1/n0 + 1/n1))

    where var_pooled is the weighted average of variances.
    """
    n_subjects, n_voxels = X.shape
    mask = X != 0

    n1 = mask.sum(axis=0).astype(np.float64)
    n0 = n_subjects - n1
    valid = (n0 > 0) & (n1 > 0)

    y = np.asarray(y, dtype=np.float64)
    y_centered = y - np.mean(y)
    y2 = y_centered * y_centered
    total_sum = y_centered.sum()
    total_sumsq = y2.sum()

    sum1 = y_centered @ mask
    sumsq1 = y2 @ mask
    sum0 = total_sum - sum1
    sumsq0 = total_sumsq - sumsq1

    statistic = np.zeros(n_voxels, dtype=np.float64)
    df = np.zeros(n_voxels, dtype=np.float64)
    if compute_dof:
        df.fill(n_subjects - 2.0)

    mean0 = np.zeros(n_voxels, dtype=np.float64)
    mean1 = np.zeros(n_voxels, dtype=np.float64)
    np.divide(sum0, n0, out=mean0, where=n0 > 0)
    np.divide(sum1, n1, out=mean1, where=n1 > 0)

    ss0 = sumsq0 - (sum0 * sum0) / np.maximum(n0, 1.0)
    ss1 = sumsq1 - (sum1 * sum1) / np.maximum(n1, 1.0)
    ss0 = np.maximum(ss0, 0.0)
    ss1 = np.maximum(ss1, 0.0)

    var_pooled = (ss0 + ss1) / (n_subjects - 2.0)
    se = np.sqrt(var_pooled * (1.0 / np.maximum(n0, 1.0) + 1.0 / np.maximum(n1, 1.0)))
    np.divide(mean0 - mean1, se, out=statistic, where=valid & (se > 0))

    return statistic, df


def welch_fast(X: np.ndarray,
              y: np.ndarray,
              compute_dof: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Welch's t-test using Numba JIT compilation.

    Tests whether the mean behavioral score differs between
    lesioned and non-lesioned groups at each voxel, without
    assuming equal variances.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_voxels)
        Binary lesion matrix (1 = lesioned, 0 = not lesioned)
    y : ndarray, shape (n_subjects,)
        Behavioral scores
    compute_dof : bool
        Whether to compute degrees of freedom

    Returns
    -------
    statistic : ndarray, shape (n_voxels,)
        Welch d-statistics for each voxel
    df : ndarray, shape (n_voxels,)
        Degrees of freedom (varies by voxel)

    Notes
    -----
    The Welch statistic is calculated as:
        d = (mean0 - mean1) / sqrt(var0/n0 + var1/n1)

    Degrees of freedom are computed using the Welch-Satterthwaite equation.
    """
    n_subjects, n_voxels = X.shape
    mask = X != 0

    n1 = mask.sum(axis=0).astype(np.float64)
    n0 = n_subjects - n1
    valid = (n0 > 0) & (n1 > 0)

    y = np.asarray(y, dtype=np.float64)
    y_centered = y - np.mean(y)
    y2 = y_centered * y_centered
    total_sum = y_centered.sum()
    total_sumsq = y2.sum()

    sum1 = y_centered @ mask
    sumsq1 = y2 @ mask
    sum0 = total_sum - sum1
    sumsq0 = total_sumsq - sumsq1

    statistic = np.zeros(n_voxels, dtype=np.float64)
    df = np.zeros(n_voxels, dtype=np.float64)
    df[~valid] = 1.0

    mean0 = np.zeros(n_voxels, dtype=np.float64)
    mean1 = np.zeros(n_voxels, dtype=np.float64)
    np.divide(sum0, n0, out=mean0, where=n0 > 0)
    np.divide(sum1, n1, out=mean1, where=n1 > 0)

    ss0 = sumsq0 - (sum0 * sum0) / np.maximum(n0, 1.0)
    ss1 = sumsq1 - (sum1 * sum1) / np.maximum(n1, 1.0)
    ss0 = np.maximum(ss0, 0.0)
    ss1 = np.maximum(ss1, 0.0)

    var0 = np.zeros(n_voxels, dtype=np.float64)
    var1 = np.zeros(n_voxels, dtype=np.float64)
    np.divide(ss0, n0 - 1.0, out=var0, where=n0 > 1)
    np.divide(ss1, n1 - 1.0, out=var1, where=n1 > 1)

    se2 = var0 / np.maximum(n0, 1.0) + var1 / np.maximum(n1, 1.0)
    se = np.sqrt(se2)
    np.divide(mean0 - mean1, se, out=statistic, where=valid & (se > 0))

    if compute_dof:
        df_fallback = n0 + n1 - 2.0
        df[valid] = df_fallback[valid]

        df_valid = valid & (var0 > 0) & (var1 > 0) & (n0 > 1) & (n1 > 1)
        term0 = var0 / np.maximum(n0, 1.0)
        term1 = var1 / np.maximum(n1, 1.0)
        numerator = (term0 + term1) ** 2
        denominator = (
            (term0 * term0) / np.maximum(n0 - 1.0, 1.0)
            + (term1 * term1) / np.maximum(n1 - 1.0, 1.0)
        )
        np.divide(numerator, denominator, out=df, where=df_valid & (denominator > 0))

    return statistic, df
