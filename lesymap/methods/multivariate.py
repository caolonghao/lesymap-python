"""
Multivariate statistical methods for LESYMAP-Python.

Implements SCCAN (Sparse Canonical Correlation Analysis) and SVR
(Support Vector Regression) for lesion-symptom mapping.

Based on R implementation in LESYMAP/R/lsm_sccan.R
"""

from typing import Optional, Dict, List, Tuple
import warnings
import inspect


__all__ = [
    'lsm_sccan',
    'lsm_svr',
]

import numpy as np
import nibabel as nib
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, t as t_dist, rankdata
import scipy.ndimage as ndimage
from joblib import Parallel, delayed

from ..core.result import LesymapResult
from ..core.patch import patches_to_voxels
from ..core.image_utils import matrix_to_image


def _import_antspy():
    """Try to import ANTsPy from multiple possible package names."""
    for pkg_name in ['ants', 'antspyx', 'antspy', 'antspyt']:
        try:
            mod = __import__(pkg_name)
            if hasattr(mod, 'sparse_decom2'):
                return mod
        except (ImportError, AttributeError):
            continue
    return None


def _call_sparse_decom2(antspyt, inmats, **kwargs):
    """Call sparse_decom2 across ANTsPy variants with small API differences."""
    sparse_decom2 = antspyt.sparse_decom2
    try:
        signature = inspect.signature(sparse_decom2)
    except (TypeError, ValueError):
        return sparse_decom2(inmats, **kwargs)

    parameters = signature.parameters
    if 'maxBased' in kwargs and 'maxBased' not in parameters and 'max_based' in parameters:
        kwargs = kwargs.copy()
        kwargs['max_based'] = kwargs.pop('maxBased')

    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}

    return sparse_decom2(inmats, **kwargs)


def _rank_transform_columns(matrix: np.ndarray) -> np.ndarray:
    """Column-wise average-rank transform followed by z-scoring."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    ranked = np.empty_like(matrix, dtype=float)
    for col_idx in range(matrix.shape[1]):
        ranked[:, col_idx] = rankdata(matrix[:, col_idx], method='average')

    centered = ranked - np.mean(ranked, axis=0)
    scale = np.std(ranked, axis=0, ddof=0)
    constant = scale == 0
    scale[constant] = 1.0
    transformed = centered / scale
    transformed[:, constant] = 0.0
    return transformed


def _normalize_robust_rank_fallback(policy: str) -> str:
    valid = {'auto', 'never', 'force'}
    if policy not in valid:
        raise ValueError(
            f"robust_rank_fallback must be one of {sorted(valid)}, got {policy!r}"
        )
    return policy


def _is_robust_not_implemented(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return 'robust' in text and ('not implemented' in text or 'not currently implemented' in text)


def _rank_transform_sccan_inputs(inmats: List[np.ndarray]) -> List[np.ndarray]:
    return [_rank_transform_columns(np.asarray(mat, dtype=float)) for mat in inmats]


def _call_sparse_decom2_with_robust_fallback(
    antspyt,
    inmats: List[np.ndarray],
    robust: int,
    robust_rank_fallback: str = 'auto',
    **kwargs,
):
    """Call sparse_decom2, optionally emulating ANTsR robust rank behavior."""
    robust_rank_fallback = _normalize_robust_rank_fallback(robust_rank_fallback)
    info = {
        'robust_requested': robust,
        'robust_backend_used': robust,
        'robust_rank_fallback': False,
        'rank_transform': None,
        'rank_transform_applied_to': [],
        'backend_robust_error': None,
    }

    if robust <= 0:
        call_kwargs = kwargs.copy()
        call_kwargs['robust'] = robust
        return _call_sparse_decom2(antspyt, inmats, **call_kwargs), inmats, info

    if robust_rank_fallback == 'force':
        transformed = _rank_transform_sccan_inputs(inmats)
        call_kwargs = kwargs.copy()
        call_kwargs['robust'] = 0
        info.update({
            'robust_backend_used': 0,
            'robust_rank_fallback': True,
            'rank_transform': 'column_average_rank_then_zscore',
            'rank_transform_applied_to': ['lesmat', 'behavior'],
        })
        return _call_sparse_decom2(antspyt, transformed, **call_kwargs), transformed, info

    call_kwargs = kwargs.copy()
    call_kwargs['robust'] = robust
    try:
        return _call_sparse_decom2(antspyt, inmats, **call_kwargs), inmats, info
    except Exception as exc:
        if robust_rank_fallback != 'auto' or not _is_robust_not_implemented(exc):
            raise

        transformed = _rank_transform_sccan_inputs(inmats)
        call_kwargs['robust'] = 0
        info.update({
            'robust_backend_used': 0,
            'robust_rank_fallback': True,
            'rank_transform': 'column_average_rank_then_zscore',
            'rank_transform_applied_to': ['lesmat', 'behavior'],
            'backend_robust_error': f"{type(exc).__name__}: {exc}",
        })
        return _call_sparse_decom2(antspyt, transformed, **call_kwargs), transformed, info


def lsm_sccan(lesmat: np.ndarray,
              behavior: np.ndarray,
              mask_img: nib.Nifti1Image,
              patchinfo: Optional[Dict] = None,
              multiple_comparison: str = 'fdr',
              p_threshold: float = 0.05,
              nperm: int = 0,
              optimize_sparseness: bool = True,
              validate_sparseness: bool = False,
              sparseness: Optional[float] = None,
              sparseness_behav: float = -0.99,
              nvecs: int = 1,
              cthresh: int = 150,
              its: int = 20,
              smooth: float = 0.4,
              mycoption: int = 1,
              robust: int = 1,
              max_based: bool = False,
              directional_sccan: bool = True,
              min_cluster_size: Optional[int] = None,
              sparseness_range: Optional[List[float]] = None,
              n_jobs: int = 1,
              robust_rank_fallback: str = 'auto',
              show_info: bool = True,
              **kwargs) -> LesymapResult:
    """
    Sparse Canonical Correlation Analysis for lesion-symptom mapping.

    Uses ANTsPy's sparse_decom2() function to find sparse linear
    combinations of lesion voxels that correlate with behavior.

    Parameters
    ----------
    lesmat : np.ndarray
        Lesion matrix (n_subjects, n_features)
    behavior : np.ndarray
        Behavioral scores (n_subjects,)
    mask_img : nibabel image
        Analysis mask
    patchinfo : dict, optional
        Patch information if patches were used
    multiple_comparison : str
        Multiple comparison correction method
    p_threshold : float
        P-value threshold for CV correlation significance
    nperm : int
        Number of SCCAN permutations (default 0)
    optimize_sparseness : bool
        Whether to optimize sparseness via cross-validation
    validate_sparseness : bool
        If sparseness is manually defined, whether to validate via CV
    sparseness : float, optional
        Fixed sparseness value (default 0.045 as in R)
    sparseness_behav : float
        Sparseness for behavior side (default -0.99)
    nvecs : int
        Number of canonical vectors
    cthresh : int
        Cluster threshold for sparseDecom2 (default 150)
    its : int
        Number of iterations (default 20)
    smooth : float
        Smoothing parameter (default 0.4)
    mycoption : int
        SCCAN optimization option (default 1)
    robust : int
        Whether to use ranks (default 1)
    max_based : bool
        Whether to use max-based filtering in SCCAN (default False)
    directional_sccan : bool
        If True, allow positive and negative weights (default True)
    min_cluster_size : int, optional
        Minimum cluster size for post-processing (defaults to cthresh)
    sparseness_range : list of float, optional
        Sparseness grid to test during optimization
    n_jobs : int
        Number of parallel jobs for sparseness optimization
    robust_rank_fallback : {'auto', 'never', 'force'}
        How to handle ANTsPy backends that expose robust but do not implement it.
    show_info : bool
        Print progress information
    **kwargs
        Additional ANTsPy parameters

    Returns
    -------
    LesymapResult
        Result object with weights and statistical maps
    """
    antspyt = _import_antspy()
    if antspyt is None:
        raise ImportError(
            "ANTsPy is required for SCCAN method. "
            "Install with: pip install lesymap[sccan]\n"
            "Or install ANTsPy directly:\n"
            "  pip install antspyx (recommended)\n"
            "  conda install -c conda-forge antspy\n"
            "\n"
            "For univariate methods (BMfast, t-test, etc.), "
            "ANTsPy is not required."
        )

    robust_rank_fallback = _normalize_robust_rank_fallback(robust_rank_fallback)

    n_subjects, n_features = lesmat.shape
    behavior_orig = behavior.copy()

    # Default sparseness (as in R version: 0.045)
    if sparseness is None:
        sparseness = 0.045

    # Default min_cluster_size to cthresh
    if min_cluster_size is None:
        min_cluster_size = cthresh

    # Scale and center data (as in R version)
    # R: behavior = scale(behavior, scale=T, center=T)
    # R: lesmat = scale(lesmat, scale=T, center=T)
    behavior_mean = np.mean(behavior)
    behavior_std = np.std(behavior, ddof=0)
    behavior_scaled = (behavior - behavior_mean) / behavior_std

    lesmat_mean = np.mean(lesmat, axis=0)
    lesmat_std = np.std(lesmat, axis=0, ddof=0)
    # Avoid division by zero for constant columns
    lesmat_std[lesmat_std == 0] = 1.0
    lesmat_scaled = (lesmat - lesmat_mean) / lesmat_std

    # Prepare sparseness vector (for lesion and behavior)
    sparseness_vec = [sparseness, sparseness_behav]
    cthresh_vec = [cthresh, 0]

    # Check if user specified sparseness manually
    user_specified_sparseness = sparseness is not None

    # Cross-validation for sparseness optimization/validation
    cv_correlation_stat = None
    cv_correlation_pval = None
    optimal_sparseness = sparseness

    if optimize_sparseness or validate_sparseness:
        if show_info:
            if optimize_sparseness:
                print("  Optimizing sparseness via cross-validation...")
            else:
                print(f"  Validating sparseness={sparseness} via cross-validation...")

        cv_result = _optimize_sccan_sparseness(
            lesmat_scaled, behavior_scaled,
            mask=None,  # Not using mask for matrix data
            cthresh=cthresh_vec,
            mycoption=mycoption,
            robust=robust,
            nvecs=nvecs,
            its=its,
            nperm=nperm,
            smooth=smooth,
            sparseness_behav=sparseness_behav,
            show_info=show_info,
            max_based=max_based,
            sparseness=sparseness,
            just_validate=validate_sparseness,
            directional_sccan=directional_sccan,
            sparseness_range=sparseness_range,
            n_jobs=n_jobs,
            antspyt=antspyt,
            robust_rank_fallback=robust_rank_fallback,
        )

        optimal_sparseness = cv_result['optimal_sparseness']
        cv_correlation_stat = cv_result['cv_correlation']
        cv_robust_info = cv_result.get('robust_info', {})
        actual_sparseness_range = cv_result.get('sparseness_range', sparseness_range)
        sparseness_vec = [optimal_sparseness, sparseness_behav]

        # Compute p-value for CV correlation
        # R: tstat = (r*sqrt(n-2))/(sqrt(1 - r^2))
        # R: CVcorrelation.pval = pt(-abs(tstat), n-2)*2
        r = abs(cv_correlation_stat) if np.isfinite(cv_correlation_stat) else np.nan
        n = len(behavior)
        if not np.isfinite(r):
            cv_correlation_pval = 1.0
        elif r < 1.0:
            tstat = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)
            cv_correlation_pval = t_dist.sf(abs(tstat), n - 2) * 2
        else:
            cv_correlation_pval = 0.0
        cv_correlation_pval = min(cv_correlation_pval, 1.0)

        if show_info:
            if optimize_sparseness:
                print(f"  Found optimal sparseness {optimal_sparseness:.3f} "
                      f"(CV corr={cv_correlation_stat:.3f} p={cv_correlation_pval:.3g})")
            else:
                print(f"  Validated sparseness {optimal_sparseness:.3f} "
                      f"(CV corr={cv_correlation_stat:.3f} p={cv_correlation_pval:.3g})")

        # If poor result, return null result
        if cv_correlation_pval > p_threshold:
            if show_info:
                print("  WARNING: Poor cross-validated accuracy, returning NULL result.")

            # Create empty result
            stat_img = matrix_to_image(np.zeros(n_features), mask_img)
            result = LesymapResult(
                stat_img=stat_img,
                mask_img=mask_img,
                method='sccan',
                model_params={
                    'sparseness': optimal_sparseness,
                    'robust': robust,
                    'robust_rank_fallback_policy': robust_rank_fallback,
                    'cv_correlation': cv_correlation_stat,
                    'cv_pvalue': cv_correlation_pval,
                    'sparseness_range': actual_sparseness_range,
                    'null_result': True,
                    **cv_robust_info,
                },
                patchinfo=patchinfo,
            )
            return result
    else:
        cv_robust_info = {}
        actual_sparseness_range = sparseness_range

    if show_info:
        print(f"  Calling SCCAN with:")
        print(f"       Components:         {nvecs}")
        print(f"       Use ranks:          {robust}")
        print(f"       Sparseness:         {optimal_sparseness:.3f}")
        print(f"       Cluster threshold:  {cthresh}")
        print(f"       Smooth sigma:       {smooth}")
        print(f"       Iterations:         {its}")
        print(f"       maxBased:           {max_based}")
        print(f"       directionalSCCAN:   {directional_sccan}")

    # Prepare data for ANTsPy
    inmats = [lesmat_scaled, behavior_scaled.reshape(-1, 1)]
    inmask = [None, None]

    # Run SCCAN
    try:
        sccan_result, fitted_inmats, robust_info = _call_sparse_decom2_with_robust_fallback(
            antspyt,
            inmats,
            robust=robust,
            robust_rank_fallback=robust_rank_fallback,
            inmask=inmask,
            sparseness=sparseness_vec,
            nvecs=nvecs,
            cthresh=cthresh_vec,
            its=its,
            smooth=smooth,
            mycoption=mycoption,
            perms=nperm,
            maxBased=max_based,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"SCCAN failed: {e}")

    # Extract weights (eig1)
    eig1 = sccan_result['eig1'] if 'eig1' in sccan_result else sccan_result['weights'][0]
    eig1 = np.asarray(eig1).flatten()

    # Extract behavior weights (eig2)
    eig2 = sccan_result['eig2'] if 'eig2' in sccan_result else sccan_result['weights'][1]
    eig2 = np.asarray(eig2).flatten()

    # Get correlation from ccasummary if available
    raw_correlation = None
    if 'ccasummary' in sccan_result and 'corrs' in sccan_result['ccasummary']:
        raw_correlation = sccan_result['ccasummary']['corrs'][0]

    # Normalize weights to [-1, 1] (R: statistic = sccan$eig1 / max(abs(sccan$eig1)))
    max_abs_weight = np.max(np.abs(eig1))
    if max_abs_weight > 0:
        statistic = eig1 / max_abs_weight
    else:
        statistic = eig1.copy()

    # Flip weights if necessary (R: directionalSCCAN logic)
    if directional_sccan:
        # R: posbehav = ifelse(sccan$eig2[1,1] < 0, -1, 1)
        posbehav = -1 if eig2[0] < 0 else 1
        # R: poscor = ifelse(sccan$ccasummary$corrs[1] < 0, -1, 1)
        if raw_correlation is not None:
            poscor = -1 if raw_correlation < 0 else 1
        else:
            # Compute correlation manually
            weighted_scores = fitted_inmats[0] @ eig1
            corr, _ = pearsonr(weighted_scores, fitted_inmats[1].ravel())
            poscor = -1 if corr < 0 else 1

        flipval = posbehav * poscor
        statistic = statistic * flipval
    else:
        # R: statistic = abs(statistic)
        statistic = np.abs(statistic)

    # Shave away weights < 0.1 (R: if (!maxBased) statistic[statistic < 0.1 & statistic > -0.1] = 0)
    if not max_based:
        statistic[(statistic < 0.1) & (statistic > -0.1)] = 0

    # Save raw weights before cluster thresholding
    raw_weights = eig1.copy()

    # Map back to voxel space if using patches
    if patchinfo is not None and 'patchindx' in patchinfo:
        statistic_full = patches_to_voxels(statistic, patchinfo['patchindx'])
        raw_weights_full = patches_to_voxels(raw_weights, patchinfo['patchindx'])
    else:
        statistic_full = statistic
        raw_weights_full = raw_weights

    # Create raw weights image (before cluster thresholding)
    raw_weights_img = matrix_to_image(raw_weights_full, mask_img)

    # Apply cluster thresholding (R: labelClusters + thresholdImage)
    stat_img = matrix_to_image(statistic_full, mask_img)
    stat_data = stat_img.get_fdata().copy()

    if min_cluster_size > 0:
        # Label connected components
        # Use very small threshold to find non-zero regions
        binary_mask = np.abs(stat_data) > np.finfo(float).eps
        labeled, n_clusters = ndimage.label(binary_mask)

        # Remove small clusters
        for i in range(1, n_clusters + 1):
            cluster_size = np.sum(labeled == i)
            if cluster_size < min_cluster_size:
                stat_data[labeled == i] = 0

        if show_info and np.sum(stat_data != 0) == 0:
            print("  WARNING: Post-SCCAN cluster thresholding removed all voxels.")

    stat_img = nib.Nifti1Image(stat_data, mask_img.affine, mask_img.header)

    # Linear calibration for prediction (R: lsm_sccan.R:251-256)
    # R: predbehav = lesmat %*% t(sccan$eig1) %*% sccan$eig2
    predbehav_scaled = lesmat_scaled @ eig1.reshape(-1, 1) @ eig2.reshape(1, -1)
    predbehav_scaled = predbehav_scaled.flatten()

    # R: predbehav.raw = predbehav * output$sccan.behavior.scaleval + output$sccan.behavior.centerval
    predbehav_raw = predbehav_scaled * behavior_std + behavior_mean

    # R: output$sccan.predictlm = lm(behavior.orig ~ predbehav.raw, ...)
    calibrate_lm = LinearRegression()
    calibrate_lm.fit(predbehav_raw.reshape(-1, 1), behavior_orig)

    # Compute calibrated correlation
    calibrated_pred = calibrate_lm.predict(predbehav_raw.reshape(-1, 1))
    calibrated_correlation, p_cal = pearsonr(calibrated_pred, behavior_orig)

    if show_info:
        print(f"  Calibrated correlation: {calibrated_correlation:.4f}")

    # Create result object
    result = LesymapResult(
        stat_img=stat_img,
        mask_img=mask_img,
        method='sccan',
        raw_weights_img=raw_weights_img,
        sccan_weights=eig1,
        sccan_eig2=eig2,
        sccan_behavior_scale=behavior_std,
        sccan_behavior_center=behavior_mean,
        sccan_lesmat_scale=lesmat_std,
        sccan_lesmat_center=lesmat_mean,
        sccan_predict_lm=calibrate_lm,
        model_params={
            'sparseness': optimal_sparseness,
            'nvecs': nvecs,
            'its': its,
            'smooth': smooth,
            'cthresh': cthresh,
            'robust': robust,
            'robust_rank_fallback_policy': robust_rank_fallback,
            'prediction_rank_transform': (
                'none; match LESYMAP R calibration/prediction with saved lesion center/scale'
                if robust_info['robust_rank_fallback'] else None
            ),
            'mycoption': mycoption,
            'max_based': max_based,
            'directional_sccan': directional_sccan,
            'correlation': calibrated_correlation,
            'p_value': p_cal,
            'cv_correlation': cv_correlation_stat,
            'cv_pvalue': cv_correlation_pval,
            'sparseness_range': actual_sparseness_range,
            'n_jobs': n_jobs,
            **cv_robust_info,
            **robust_info,
        },
        patchinfo=patchinfo,
    )

    return result


def lsm_svr(lesmat: np.ndarray,
            behavior: np.ndarray,
            mask_img: nib.Nifti1Image,
            patchinfo: Optional[Dict] = None,
            multiple_comparison: str = 'fdr',
            p_threshold: float = 0.05,
            nperm: int = 100,
            C: float = 1.0,
            kernel: str = 'linear',
            epsilon: float = 0.1,
            n_perm: int = 50,
            max_features: Optional[int] = None,
            show_info: bool = True,
            **kwargs) -> LesymapResult:
    """
    Support Vector Regression for lesion-symptom mapping.

    Uses sklearn's SVR to find support vectors for predicting behavior
    from lesion patterns.

    Parameters
    ----------
    lesmat : np.ndarray
        Lesion matrix (n_subjects, n_features)
    behavior : np.ndarray
        Behavioral scores (n_subjects,)
    mask_img : nibabel image
        Analysis mask
    patchinfo : dict, optional
        Patch information if patches were used
    multiple_comparison : str
        Multiple comparison correction method
    p_threshold : float
        P-value threshold
    nperm : int
        Number of permutations for significance testing
    C : float
        SVR regularization parameter
    kernel : str
        SVR kernel ('linear', 'rbf', 'poly')
    epsilon : float
        SVR epsilon parameter
    n_perm : int
        Number of permutations per feature for non-linear kernels
    max_features : int, optional
        Maximum number of features to evaluate for non-linear permutation
        importance. Required for non-linear kernels to avoid unbounded work.
    show_info : bool
        Print progress information
    **kwargs
        Additional SVR parameters

    Returns
    -------
    LesymapResult
        Result object with weights and statistical maps
    """
    if show_info:
        print(f"  Running SVR (kernel={kernel}, C={C})...")

    n_subjects, n_features = lesmat.shape

    # Fit SVR
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
    svr.fit(lesmat, behavior)

    # Get predictions and correlation
    predictions = svr.predict(lesmat)
    correlation, p_value = pearsonr(predictions, behavior)

    if show_info:
        print(f"  SVR correlation: {correlation:.4f}")

    # Extract weights (for linear kernel, these are the coefficients)
    if kernel == 'linear':
        weights = svr.coef_.flatten()
    else:
        # For non-linear kernels, use permutation importance
        if max_features is None:
            raise ValueError(
                "Non-linear SVR permutation importance requires max_features "
                "to bound runtime. Pass max_features=<int> and optionally "
                "n_perm=<int>, or use kernel='linear' for coefficient weights."
            )
        warnings.warn("Weight extraction not supported for non-linear kernels. "
                      "Using permutation importance instead.")
        weights = _compute_svr_importance(
            svr, lesmat, behavior, n_perm=n_perm, max_features=max_features
        )

    # Map back to voxel space if using patches
    if patchinfo is not None and 'patchindx' in patchinfo:
        weights_full = patches_to_voxels(weights, patchinfo['patchindx'])
    else:
        weights_full = weights

    # Create statistical maps
    stat_img = matrix_to_image(weights_full, mask_img)
    raw_weights_img = matrix_to_image(weights_full, mask_img)

    result = LesymapResult(
        stat_img=stat_img,
        mask_img=mask_img,
        method='svr',
        raw_weights_img=raw_weights_img,
        svr_model=svr,
        model_params={
            'C': C,
            'kernel': kernel,
            'epsilon': epsilon,
            'correlation': correlation,
            'p_value': p_value,
            'n_perm': n_perm,
            'max_features': max_features,
        },
        patchinfo=patchinfo,
    )

    return result


def _optimize_sccan_sparseness(lesmat: np.ndarray,
                               behavior: np.ndarray,
                               mask=None,
                               cthresh: List[int] = None,
                               mycoption: int = 1,
                               robust: int = 1,
                               nvecs: int = 1,
                               its: int = 5,
                               nperm: int = 0,
                               smooth: float = 0.4,
                               sparseness_behav: float = -0.99,
                               show_info: bool = True,
                               max_based: bool = False,
                               sparseness: float = 0.045,
                               just_validate: bool = False,
                               directional_sccan: bool = True,
                               sparseness_range: List[float] = None,
                               n_folds: int = 4,
                               n_reps: int = 1,
                               n_jobs: int = 1,
                               antspyt=None,
                               robust_rank_fallback: str = 'auto') -> Dict:
    """
    Optimize SCCAN sparseness parameter via cross-validation.

    Implements R's optimize_SCCANsparseness function.

    Parameters
    ----------
    lesmat : np.ndarray
        Scaled lesion matrix
    behavior : np.ndarray
        Scaled behavioral scores
    mask : optional
        Not used for matrix data
    cthresh : list of int
        Cluster thresholds [lesion, behavior]
    mycoption : int
        SCCAN optimization option
    robust : int
        Whether to use ranks
    nvecs : int
        Number of canonical vectors
    its : int
        Number of iterations
    nperm : int
        Number of permutations
    smooth : float
        Smoothing parameter
    sparseness_behav : float
        Sparseness for behavior side
    show_info : bool
        Print progress
    max_based : bool
        Whether to use max-based filtering
    sparseness : float
        Initial/fixed sparseness value
    just_validate : bool
        If True, only validate the given sparseness
    directional_sccan : bool
        Whether to allow directional weights
    sparseness_range : list of float
        Sparseness values to try
    n_folds : int
        Number of CV folds
    n_reps : int
        Number of CV repetitions
    n_jobs : int
        Number of parallel jobs across sparseness candidates
    antspyt : module
        ANTsPy module
    robust_rank_fallback : {'auto', 'never', 'force'}
        How to handle ANTsPy backends that expose robust but do not implement it.

    Returns
    -------
    dict
        Dictionary with 'optimal_sparseness' and 'cv_correlation'
    """
    if antspyt is None:
        antspyt = _import_antspy()

    if antspyt is None:
        warnings.warn(
            "ANTsPy not available for sparseness optimization. "
            "Using default sparseness=0.045."
        )
        return {'optimal_sparseness': 0.045, 'cv_correlation': np.nan}

    if cthresh is None:
        cthresh = [150, 0]

    robust_rank_fallback = _normalize_robust_rank_fallback(robust_rank_fallback)

    # Default sparseness range (as in R: use negative for directional)
    if sparseness_range is None:
        if directional_sccan:
            # R uses negative sparseness for bidirectional
            sparseness_range = [-0.01, -0.02, -0.03, -0.045, -0.05, -0.1, -0.15, -0.2]
        else:
            sparseness_range = [0.01, 0.02, 0.03, 0.045, 0.05, 0.1, 0.15, 0.2]

    if just_validate:
        # Only validate the given sparseness
        sparseness_range = [sparseness if directional_sccan else abs(sparseness)]
        if directional_sccan and sparseness > 0:
            sparseness_range = [-sparseness]

    inmask = [None, None]

    best_sparseness = sparseness_range[0]
    best_correlation = -np.inf

    def evaluate_sparseness(test_sparseness):
        sparseness_vec = [test_sparseness, sparseness_behav]

        cv_correlations = []
        robust_infos = []

        for rep in range(n_reps):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + rep)

            # Storage for predictions across folds
            behavior_predicted = np.zeros_like(behavior)

            for train_idx, test_idx in kf.split(lesmat):
                lesmat_train = lesmat[train_idx]
                lesmat_test = lesmat[test_idx]
                behavior_train = behavior[train_idx]

                # Train SCCAN on training fold
                inmats_train = [lesmat_train, behavior_train.reshape(-1, 1)]

                result, _, robust_info = _call_sparse_decom2_with_robust_fallback(
                    antspyt,
                    inmats_train,
                    robust=robust,
                    robust_rank_fallback=robust_rank_fallback,
                    inmask=inmask,
                    sparseness=sparseness_vec,
                    nvecs=nvecs,
                    its=its,
                    smooth=smooth,
                    mycoption=mycoption,
                    cthresh=cthresh,
                    maxBased=max_based,
                )
                robust_infos.append(robust_info)

                # Extract weights
                eig1 = result['eig1'] if 'eig1' in result else result['weights'][0]
                eig1 = np.asarray(eig1).flatten()
                eig2 = result['eig2'] if 'eig2' in result else result['weights'][1]
                eig2 = np.asarray(eig2).flatten()

                # Predict on test fold
                # R: behavior.predicted[fold] = lesmat[fold,] %*% t(trainsccan$eig1) %*% trainsccan$eig2
                pred = lesmat_test @ eig1.reshape(-1, 1) @ eig2.reshape(1, -1)
                behavior_predicted[test_idx] = pred.flatten()

            # Compute CV correlation for this rep
            valid_mask = ~np.isnan(behavior_predicted)
            if np.sum(valid_mask) > 2:
                corr = np.abs(np.corrcoef(behavior[valid_mask], behavior_predicted[valid_mask])[0, 1])
                cv_correlations.append(corr)
            else:
                cv_correlations.append(-np.inf)

        mean_corr = np.mean(cv_correlations)
        return test_sparseness, mean_corr, robust_infos

    if n_jobs == 1 or len(sparseness_range) == 1:
        grid_results = [evaluate_sparseness(s) for s in sparseness_range]
    else:
        grid_results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(evaluate_sparseness)(s) for s in sparseness_range
        )

    all_robust_infos = [
        info
        for _, _, infos in grid_results
        for info in infos
    ]
    cv_robust_info = {
        'cv_robust_requested': robust,
        'cv_robust_backend_used': 0
        if any(info['robust_backend_used'] == 0 for info in all_robust_infos)
        else robust,
        'cv_robust_rank_fallback': any(
            info['robust_rank_fallback'] for info in all_robust_infos
        ),
        'cv_rank_transform': (
            'column_average_rank_then_zscore'
            if any(info['robust_rank_fallback'] for info in all_robust_infos)
            else None
        ),
        'cv_backend_robust_error': next(
            (
                info['backend_robust_error']
                for info in all_robust_infos
                if info['backend_robust_error']
            ),
            None,
        ),
    }

    for test_sparseness, mean_corr, _ in grid_results:
        if show_info and not just_validate:
            print(f"    Sparseness={test_sparseness:.3f}: CV correlation={mean_corr:.4f}")

        if np.isfinite(mean_corr) and mean_corr > best_correlation:
            best_correlation = mean_corr
            best_sparseness = test_sparseness

    if best_correlation == -np.inf:
        best_correlation = np.nan

    # Return absolute value of sparseness
    return {
        'optimal_sparseness': abs(best_sparseness),
        'cv_correlation': best_correlation,
        'sparseness_range': sparseness_range,
        'robust_info': cv_robust_info,
    }


def _compute_svr_importance(svr: SVR,
                            lesmat: np.ndarray,
                            behavior: np.ndarray,
                            n_perm: int = 50,
                            max_features: Optional[int] = None) -> np.ndarray:
    """
    Compute feature importance for SVR via permutation.

    Parameters
    ----------
    svr : fitted SVR
        Trained SVR model
    lesmat : np.ndarray
        Lesion matrix
    behavior : np.ndarray
        Behavioral scores
    n_perm : int
        Number of permutations per feature
    max_features : int, optional
        Maximum number of features to permute. If smaller than the full feature
        count, features with highest variance are evaluated and the rest remain 0.

    Returns
    -------
    np.ndarray
        Feature importance scores
    """
    baseline_score = svr.score(lesmat, behavior)
    n_features = lesmat.shape[1]
    importance = np.zeros(n_features)

    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    if max_features is not None:
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        if max_features < n_features:
            feature_indices = np.argsort(np.var(lesmat, axis=0))[::-1][:max_features]
        else:
            feature_indices = np.arange(n_features)
    else:
        feature_indices = np.arange(n_features)

    for i in feature_indices:
        scores = []
        for _ in range(n_perm):
            lesmat_perm = lesmat.copy()
            lesmat_perm[:, i] = np.random.permutation(lesmat_perm[:, i])
            score = svr.score(lesmat_perm, behavior)
            scores.append(score)

        importance[i] = baseline_score - np.mean(scores)

    return importance
