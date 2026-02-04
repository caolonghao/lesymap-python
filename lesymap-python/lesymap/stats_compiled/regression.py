"""
Numba-compiled regression for LESYMAP-Python.

Port of C++ implementation from LESYMAP/src/regresfast.cpp
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional


@njit(parallel=True, cache=True)
def regression_fast(X: np.ndarray,
                   y: np.ndarray,
                   covariates: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, int]:
    """
    Fast linear regression per voxel using Numba JIT compilation.

    Fits a linear regression model at each voxel with behavioral score
    as the dependent variable and lesion status (plus optional covariates)
    as predictors. Returns the t-statistic for the lesion coefficient.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_voxels)
        Lesion matrix (can be binary or continuous)
    y : ndarray, shape (n_subjects,)
        Behavioral scores
    covariates : ndarray, shape (n_subjects, n_covariates), optional
        Covariate matrix (e.g., age, lesion size)

    Returns
    -------
    statistic : ndarray, shape (n_voxels,)
        t-statistics for lesion coefficient at each voxel
    n : int
        Number of subjects
    kxmat : int
        Number of predictors (1 for intercept + 1 for lesion + n_covariates)

    Notes
    -----
    The model fitted at each voxel is:
        y = beta0 + beta1 * X[:, vox] + covariates @ gamma + epsilon

    The returned statistic is the t-statistic for beta1 (lesion effect).
    """
    n_subjects, n_voxels = X.shape

    # Initialize output
    statistic = np.zeros(n_voxels, dtype=np.float64)

    # Build design matrix structure
    if covariates is None:
        # Just intercept + lesion
        kxmat = 2
    else:
        n_covariates = covariates.shape[1] if covariates.ndim > 1 else 1
        kxmat = 2 + n_covariates

    # Loop through voxels
    for vox in prange(n_voxels):
        # Build design matrix for this voxel
        xmat = _build_design_matrix(X[:, vox], covariates)

        # Solve OLS: (X'X)^-1 X'y
        coef, resid, sig2 = _solve_ols(xmat, y, kxmat)

        # Compute standard errors
        stderrest = _compute_standard_errors(xmat, sig2, kxmat)

        # t-statistic for first predictor (lesion effect)
        if stderrest[0] > 0:
            statistic[vox] = coef[0] / stderrest[0]
        else:
            statistic[vox] = 0.0

    return statistic, n_subjects, kxmat


@njit(cache=True)
def _build_design_matrix(voxel_data: np.ndarray,
                        covariates: Optional[np.ndarray]) -> np.ndarray:
    """
    Build design matrix for regression at a single voxel.

    Parameters
    ----------
    voxel_data : ndarray, shape (n_subjects,)
        Lesion data for this voxel
    covariates : ndarray or None
        Covariate matrix

    Returns
    -------
    ndarray, shape (n_subjects, n_predictors)
        Design matrix with intercept, lesion, and covariates
    """
    n = len(voxel_data)

    if covariates is None:
        # Just intercept + lesion
        xmat = np.empty((n, 2), dtype=np.float64)
        xmat[:, 0] = voxel_data  # Lesion predictor
        xmat[:, 1] = 1.0  # Intercept
    else:
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        n_covariates = covariates.shape[1]
        xmat = np.empty((n, 2 + n_covariates), dtype=np.float64)

        xmat[:, 0] = voxel_data  # Lesion predictor
        xmat[:, 1] = 1.0  # Intercept

        # Add covariates
        for i in range(n_covariates):
            xmat[:, 2 + i] = covariates[:, i]

    return xmat


@njit(cache=True)
def _solve_ols(xmat: np.ndarray,
              y: np.ndarray,
              kxmat: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve OLS regression using normal equations.

    Parameters
    ----------
    xmat : ndarray, shape (n, k)
        Design matrix
    y : ndarray, shape (n,)
        Dependent variable
    kxmat : int
        Number of predictors

    Returns
    -------
    coef : ndarray, shape (k,)
        Regression coefficients
    resid : ndarray, shape (n,)
        Residuals
    sig2 : float
        Residual variance (MSE)
    """
    # X'X and X'y
    xt_x = xmat.T @ xmat
    xt_y = xmat.T @ y

    # Solve (X'X) * coef = X'y
    # Use simple LU decomposition via solve
    coef = np.linalg.solve(xt_x, xt_y)

    # Residuals
    y_pred = xmat @ coef
    resid = y - y_pred

    # Residual variance
    n = len(y)
    sig2 = np.sum(resid ** 2) / (n - kxmat)

    return coef, resid, sig2


@njit(cache=True)
def _compute_standard_errors(xmat: np.ndarray,
                            sig2: float,
                            kxmat: int) -> np.ndarray:
    """
    Compute standard errors of regression coefficients.

    Parameters
    ----------
    xmat : ndarray, shape (n, k)
        Design matrix
    sig2 : float
        Residual variance
    kxmat : int
        Number of predictors

    Returns
    -------
    ndarray, shape (k,)
        Standard errors for each coefficient
    """
    # Inverse of X'X (diagonal only for efficiency)
    xt_x = xmat.T @ xmat
    xt_x_inv = np.linalg.inv(xt_x)

    # Variance of coefficients: sig2 * diag((X'X)^-1)
    var_coef = sig2 * np.diag(xt_x_inv)

    # Standard errors
    stderrest = np.sqrt(var_coef)

    return stderrest
