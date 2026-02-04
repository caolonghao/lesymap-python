"""
Numba-compiled Brunner-Munzel test for LESYMAP-Python.

Port of C++ implementation from LESYMAP/src/BMfast2.cpp
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(parallel=True, cache=True)
def brunner_munzel_fast(X: np.ndarray,
                        y: np.ndarray,
                        compute_dof: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Brunner-Munzel test using Numba JIT compilation.

    This is a Python port of the C++ implementation from LESYMAP.
    Tests whether the distribution of behavioral scores differs
    between lesioned and non-lesioned groups at each voxel.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_voxels)
        Binary lesion matrix (1 = lesioned, 0 = not lesioned)
    y : ndarray, shape (n_subjects,)
        Behavioral scores
    compute_dof : bool
        Whether to compute degrees of freedom (can skip during permutations)

    Returns
    -------
    statistic : ndarray, shape (n_voxels,)
        BM statistics for each voxel (negative values indicate
        lower behavior in lesioned group)
    df : ndarray, shape (n_voxels,)
        Degrees of freedom for each voxel

    Notes
    -----
    The Brunner-Munzel test is a non-parametric test that is
    robust to heteroscedasticity. It tests the hypothesis that
    the probability that a randomly selected value from one group
    is greater than a randomly selected value from the other group
    equals 0.5.

    The test statistic is calculated as:
        BM = -(p - 0.5) * sqrt(N / s)

    where p is the relative effect, N is total sample size, and s
    is the standardized rank variance.

    References
    ----------
    Brunner, E., & Munzel, U. (2000). The nonparametric Behrens-Fisher
    problem: Asymptotic theory and a small-sample approximation.
    Biometrical Journal, 42(1), 17-25.
    """
    n_subjects, n_voxels = X.shape

    # Initialize output
    statistic = np.zeros(n_voxels, dtype=np.float64)
    df = np.zeros(n_voxels, dtype=np.float64)

    # Pre-compute global ranks with tie handling (JIT-compatible)
    r = _compute_ranks_with_ties(y)

    # Check for duplicates in y (JIT-compatible implementation)
    has_duplicates = _has_duplicates(y)

    # Loop through voxels
    for vox in prange(n_voxels):
        thisvox = X[:, vox]

        # Use boolean indexing instead of np.where() for better performance
        mask0 = thisvox == 0
        mask1 = ~mask0  # or thisvox != 0

        n1 = 0
        n2 = 0
        # Count using boolean masks
        for m in mask0:
            if m:
                n1 += 1
        n2 = n_subjects - n1

        if n1 == 0 or n2 == 0:
            statistic[vox] = 0.0
            df[vox] = 1.0
            continue

        # Create arrays for each group using boolean indexing
        y0 = np.empty(n1, dtype=y.dtype)
        y1 = np.empty(n2, dtype=y.dtype)
        r_at_indx0 = np.empty(n1, dtype=r.dtype)
        r_at_indx1 = np.empty(n2, dtype=r.dtype)

        idx0 = 0
        idx1 = 0
        for i in range(n_subjects):
            if mask0[i]:
                y0[idx0] = y[i]
                r_at_indx0[idx0] = r[i]
                idx0 += 1
            else:
                y1[idx1] = y[i]
                r_at_indx1[idx1] = r[i]
                idx1 += 1

        # Compute ranks within each group
        r1 = _compute_ranks_with_ties(y0)
        r2 = _compute_ranks_with_ties(y1)

        # Relative effect (probability that random observation from
        # group 1 is greater than from group 0)
        p = (1.0 / n1) * (np.mean(r_at_indx1) - (n2 + 1.0) / 2.0)

        # Handle edge cases
        if p == 0:
            p = 1e-5
        elif p == 1:
            p = 1 - 1e-5

        # Compute rank variances
        mean_r0 = np.mean(r_at_indx0)
        mean_r1 = np.mean(r_at_indx1)

        S1 = (1.0 / (n1 - 1.0)) * (
            _sum_squared_diff(r_at_indx0, r1) -
            n1 * (mean_r0 - (n1 + 1.0) / 2.0) ** 2
        )

        S2 = (1.0 / (n2 - 1.0)) * (
            _sum_squared_diff(r_at_indx1, r2) -
            n2 * (mean_r1 - (n2 + 1.0) / 2.0) ** 2
        )

        # Handle edge cases
        if S1 == 0:
            S1 = 1.0 / (4.0 * n1)
        if S2 == 0:
            S2 = 1.0 / (4.0 * n2)

        # Standardized rank variance
        s = n_subjects / (n1 * n2) * (S1 / n2 + S2 / n1)

        # BM statistic (negated so negative values mean lower behavior in lesioned group)
        statistic[vox] = -(p - 0.5) * np.sqrt(n_subjects / s)

        # Degrees of freedom
        if compute_dof:
            m1 = mean_r0
            m2 = mean_r1

            v1 = np.sum((r_at_indx0 - r1 - m1 + (n1 + 1) / 2.0) ** 2) / (n1 - 1.0)
            v2 = np.sum((r_at_indx1 - r2 - m2 + (n2 + 1) / 2.0) ** 2) / (n2 - 1.0)

            dfbm_num = (n1 * v1 + n2 * v2) ** 2
            dfbm_den = (n1 * v1) ** 2 / (n1 - 1.0) + (n2 * v2) ** 2 / (n2 - 1.0)

            if dfbm_den > 0:
                df[vox] = dfbm_num / dfbm_den
            else:
                df[vox] = n1 + n2 - 2
        else:
            df[vox] = 1.0

    return statistic, df


@njit(cache=True)
def _has_duplicates(values: np.ndarray) -> bool:
    """
    Check if array has duplicate values (JIT-compatible).

    Parameters
    ----------
    values : ndarray
        Values to check

    Returns
    -------
    bool
        True if duplicates exist
    """
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] == values[j]:
                return True
    return False


@njit(cache=True)
def _sum_squared_diff(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute sum of squared differences (JIT-compatible helper).

    Parameters
    ----------
    a, b : ndarray
        Arrays to compare

    Returns
    -------
    float
        Sum of (a - b)^2
    """
    result = 0.0
    n = len(a)
    for i in range(n):
        diff = a[i] - b[i]
        result += diff * diff
    return result


@njit(cache=True)
def _compute_ranks_with_ties(values: np.ndarray) -> np.ndarray:
    """
    Compute ranks with average rank for ties (O(n log n) algorithm).

    Parameters
    ----------
    values : ndarray
        Values to rank

    Returns
    -------
    ndarray
        Ranks (1-indexed) with ties averaged
    """
    n = len(values)

    # Get sorted indices
    sorted_indices = np.argsort(values)

    # Compute ranks (1-indexed)
    ranks = np.empty(n, dtype=np.float64)
    for i, idx in enumerate(sorted_indices):
        ranks[idx] = i + 1

    # Handle ties: replace tied ranks with their average
    # O(n log n) single-pass algorithm
    i = 0
    while i < n:
        # Find all indices with same value
        j = i
        while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
            j += 1

        # If there are ties (j > i + 1), compute average rank
        if j > i + 1:
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank

        i = j

    return ranks
