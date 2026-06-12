"""
Multiple comparison correction methods for LESYMAP-Python.

Implements various correction methods including FDR, FWER, and
permutation-based thresholding.
"""

from typing import Callable, Optional
import warnings


__all__ = [
    'correct_pvalues',
    'fwer_permutation_threshold',
    'cluster_permutation_threshold',
    'compute_fdr',
]

import numpy as np
import statsmodels.stats.multitest as smm


def correct_pvalues(pvals: np.ndarray,
                   method: str = 'fdr') -> tuple:
    """
    Apply multiple comparison correction to p-values.

    Parameters
    ----------
    pvals : ndarray
        P-values to correct
    method : str
        Correction method: 'fdr', 'bonferroni', 'holm', 'BY', 'none'

    Returns
    -------
    reject : ndarray
        Boolean array of rejected hypotheses
    pvals_corr : ndarray
        Corrected p-values
    _, _ : tuple
        Additional outputs (for compatibility with statsmodels)

    Examples
    --------
    >>> pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    >>> reject, pvals_corr, _, _ = correct_pvalues(pvals, method='fdr')
    >>> print(reject)
    [ True False False False False]
    """
    # Map method names to statsmodels methods
    method_map = {
        'fdr': 'fdr_bh',
        'bonferroni': 'bonferroni',
        'holm': 'holm',
        'BY': 'fdr_by',
        'hochberg': 'hochberg',
        'hommel': 'hommel',
        'none': 'identity',
    }

    if method not in method_map:
        warnings.warn(f"Unknown correction method: {method}. Using 'fdr'.")
        method = 'fdr'

    if method == 'none' or method_map[method] == 'identity':
        # No correction
        alpha = 0.05
        reject = pvals < alpha
        return reject, pvals, alpha, None

    # Apply correction
    return smm.multipletests(pvals, alpha=0.05, method=method_map[method])


def fwer_permutation_threshold(stat_func: Callable,
                               nperm: int = 1000,
                               alpha: float = 0.95,
                               n_jobs: int = 1) -> float:
    """
    Compute family-wise error rate threshold via permutation.

    Parameters
    ----------
    stat_func : callable
        Function that returns statistic array (e.g., BM, t-test result)
    nperm : int
        Number of permutations
    alpha : float
        Percentile for threshold (default 0.95 = 5% FWER)
    n_jobs : int
        Number of parallel jobs (default 1 = serial). Use -1 for all cores.
        Uses threading backend (safe with Numba-compiled stat_func).

    Returns
    -------
    float
        Threshold value at the specified percentile
    """
    from joblib import Parallel, delayed

    def _one_perm(_):
        return float(np.max(np.abs(stat_func())))

    max_stats = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_one_perm)(i) for i in range(nperm)
    )

    return float(np.percentile(max_stats, alpha * 100))


def cluster_permutation_threshold(stat_img: np.ndarray,
                                 p_threshold: float = 0.05,
                                 nperm: int = 1000,
                                 min_cluster_size: int = 10) -> tuple:
    """
    Cluster-based permutation thresholding.

    Parameters
    ----------
    stat_img : ndarray
        Statistical map (flattened or 3D)
    p_threshold : float
        Initial cluster-forming threshold
    nperm : int
        Number of permutations
    min_cluster_size : int
        Minimum cluster size in voxels

    Returns
    -------
    clustered_stats : ndarray
        Clustered statistical map
    cluster_sizes : ndarray
        Sizes of significant clusters
    """
    from scipy import ndimage

    # Threshold at p-value
    z_thresh = abs_norm_ppf(1 - p_threshold / 2)

    # Find clusters above threshold
    above_thresh = np.abs(stat_img) > z_thresh

    # Label clusters
    labeled, n_clusters = ndimage.label(above_thresh)

    # Get cluster sizes
    cluster_sizes = ndimage.sum(above_thresh, labeled, range(n_clusters + 1))

    # Filter small clusters
    clustered_stats = stat_img.copy()
    for i in range(1, n_clusters + 1):
        if cluster_sizes[i] < min_cluster_size:
            clustered_stats[labeled == i] = 0

    return clustered_stats, cluster_sizes[1:]  # Exclude background


def abs_norm_ppf(p: float) -> float:
    """
    Absolute value of normal distribution percent point function.

    Parameters
    ----------
    p : float
        Probability (0 < p < 1)

    Returns
    -------
    float
        |Z_p| where Z_p is the p-th quantile of standard normal
    """
    from scipy.stats import norm
    return abs(norm.ppf(p))


def compute_fdr(pvals: np.ndarray,
               method: str = 'bh') -> np.ndarray:
    """
    Compute false discovery rate adjusted p-values.

    Parameters
    ----------
    pvals : ndarray
        Original p-values
    method : str
        FDR method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)

    Returns
    -------
    ndarray
        FDR-adjusted p-values (q-values)

    Notes
    -----
    The Benjamini-Hochberg procedure controls the FDR at level q:
        q * i / m

    where i is the rank and m is the number of tests.
    """
    n = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # Compute adjusted p-values
    if method == 'bh':
        # Benjamini-Hochberg
        adj_pvals = sorted_pvals * n / np.arange(1, n + 1)
    elif method == 'by':
        # Benjamini-Yekutieli (more conservative)
        harmonic = np.sum(1.0 / np.arange(1, n + 1))
        adj_pvals = sorted_pvals * n * harmonic / np.arange(1, n + 1)
    else:
        raise ValueError(f"Unknown FDR method: {method}")

    # Enforce monotonicity: each adjusted p-value ≤ the next larger one
    adj_pvals = np.minimum.accumulate(adj_pvals[::-1])[::-1]

    # Cap at 1
    adj_pvals = np.minimum(adj_pvals, 1.0)

    # Unsort
    final_adj = np.empty(n)
    final_adj[sorted_indices] = adj_pvals

    return final_adj
