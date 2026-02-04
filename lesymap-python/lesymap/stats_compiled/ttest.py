"""
Numba-compiled t-test implementations for LESYMAP-Python.

Port of C++ implementation from LESYMAP/src/TTfast.cpp
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(parallel=True, cache=True)
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

    # Initialize output
    statistic = np.zeros(n_voxels, dtype=np.float64)
    df = np.zeros(n_voxels, dtype=np.float64)

    if compute_dof:
        # Student t has constant degrees of freedom
        df.fill(n_subjects - 2.0)

    # Loop through voxels
    for vox in prange(n_voxels):
        thisvox = X[:, vox]

        # Find indices for each group
        indx0 = np.where(thisvox == 0)[0]
        indx1 = np.where(thisvox != 0)[0]

        n0 = len(indx0)
        n1 = len(indx1)

        if n0 == 0 or n1 == 0:
            statistic[vox] = 0.0
            continue

        # Extract behavior for each group
        y0 = y[indx0]
        y1 = y[indx1]

        # Compute means
        mean0 = np.mean(y0)
        mean1 = np.mean(y1)

        # Compute variances (sample variance, n-1 denominator)
        var0 = np.var(y0, ddof=1) if n0 > 1 else 0.0
        var1 = np.var(y1, ddof=1) if n1 > 1 else 0.0

        # Pooled variance (weighted by n-1)
        var_pooled = ((var0 * (n0 - 1.0)) + (var1 * (n1 - 1.0))) / (n_subjects - 2.0)

        # t-statistic
        se = np.sqrt(var_pooled * (1.0 / n0 + 1.0 / n1))

        if se > 0:
            statistic[vox] = (mean0 - mean1) / se
        else:
            statistic[vox] = 0.0

    return statistic, df


@njit(parallel=True, cache=True)
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

    # Initialize output
    statistic = np.zeros(n_voxels, dtype=np.float64)
    df = np.zeros(n_voxels, dtype=np.float64)

    # Loop through voxels
    for vox in prange(n_voxels):
        thisvox = X[:, vox]

        # Find indices for each group
        indx0 = np.where(thisvox == 0)[0]
        indx1 = np.where(thisvox != 0)[0]

        n0 = len(indx0)
        n1 = len(indx1)

        if n0 == 0 or n1 == 0:
            statistic[vox] = 0.0
            df[vox] = 1.0
            continue

        # Extract behavior for each group
        y0 = y[indx0]
        y1 = y[indx1]

        # Compute means
        mean0 = np.mean(y0)
        mean1 = np.mean(y1)

        # Compute variances
        var0 = np.var(y0, ddof=1) if n0 > 1 else 0.0
        var1 = np.var(y1, ddof=1) if n1 > 1 else 0.0

        # Welch statistic
        se = np.sqrt(var0 / n0 + var1 / n1)

        if se > 0:
            statistic[vox] = (mean0 - mean1) / se
        else:
            statistic[vox] = 0.0

        # Degrees of freedom (Welch-Satterthwaite)
        if compute_dof:
            if var0 > 0 and var1 > 0 and n0 > 1 and n1 > 1:
                num = (var0 / n0 + var1 / n1) ** 2
                den = (var0 / n0) ** 2 / (n0 - 1.0) + (var1 / n1) ** 2 / (n1 - 1.0)

                if den > 0:
                    df[vox] = num / den
                else:
                    df[vox] = n0 + n1 - 2
            else:
                df[vox] = n0 + n1 - 2

    return statistic, df
