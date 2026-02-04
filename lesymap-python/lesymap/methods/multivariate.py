"""
Multivariate statistical methods for LESYMAP-Python.

Implements SCCAN (Sparse Canonical Correlation Analysis) and SVR
(Support Vector Regression) for lesion-symptom mapping.
"""

from typing import Optional, Dict, List
import warnings


__all__ = [
    'lsm_sccan',
    'lsm_svr',
]

import numpy as np
import nibabel as nib
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import pearsonr

from ..core.result import LesymapResult
from ..core.patch import patches_to_voxels
from ..core.image_utils import matrix_to_image


def lsm_sccan(lesmat: np.ndarray,
              behavior: np.ndarray,
              mask_img: nib.Nifti1Image,
              patchinfo: Optional[Dict] = None,
              multiple_comparison: str = 'fdr',
              p_threshold: float = 0.05,
              nperm: int = 1000,
              optimize_sparseness: bool = True,
              sparseness: Optional[float] = None,
              nvecs: int = 1,
              cthresh: float = 0.0,
              its: int = 10,
              smooth: float = 0.0,
              cluster_threshold: Optional[float] = None,
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
        P-value threshold
    nperm : int
        Number of permutations (if using permutation testing)
    optimize_sparseness : bool
        Whether to optimize sparseness via cross-validation
    sparseness : float, optional
        Fixed sparseness value (if not optimizing)
    nvecs : int
        Number of canonical vectors
    cthresh : float
        Cluster threshold
    its : int
        Number of iterations
    smooth : float
        Smoothing parameter
    cluster_threshold : float, optional
        Cluster size threshold for final weights
    show_info : bool
        Print progress information
    **kwargs
        Additional ANTsPy parameters

    Returns
    -------
    LesymapResult
        Result object with weights and statistical maps
    """
    # Try to import ANTsPy (try multiple possible package names)
    antspyt = None
    for pkg_name in ['antspyx', 'antspy', 'antspyt']:
        try:
            mod = __import__(pkg_name)
            # Check if sparse_decom2 exists
            if hasattr(mod, 'sparse_decom2'):
                antspyt = mod
                break
        except (ImportError, AttributeError):
            continue

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

    n_subjects, n_features = lesmat.shape

    # Scale and center data (as in R version)
    scaler_lesion = StandardScaler()
    scaler_behavior = StandardScaler()

    lesmat_scaled = scaler_lesion.fit_transform(lesmat)
    behavior_scaled = scaler_behavior.fit_transform(behavior.reshape(-1, 1)).flatten()

    # Optimize sparseness if requested
    if optimize_sparseness and sparseness is None:
        if show_info:
            print("  Optimizing sparseness via cross-validation...")
        sparseness = _optimize_sccan_sparseness(
            lesmat_scaled, behavior_scaled,
            nvecs=nvecs,
            its=its,
            smooth=smooth,
            show_info=show_info
        )
    elif sparseness is None:
        sparseness = 0.1  # Default

    if show_info:
        print(f"  Running SCCAN with sparseness={sparseness}...")

    # Prepare data for ANTsPy
    # ANTsPy expects matrices as lists
    inmats = [lesmat_scaled, behavior_scaled.reshape(-1, 1)]

    # Create mask for ANTsPy (for matrix data, this is just which features to use)
    inmask = [np.ones(n_features, dtype=int), None]

    # Run SCCAN
    try:
        sccan_result = antspyt.sparse_decom2(
            inmats,
            inmask=inmask,
            sparseness=sparseness,
            nvecs=nvecs,
            cthresh=cthresh,
            its=its,
            smooth=smooth,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"SCCAN failed: {e}")

    # Extract weights
    weights = sccan_result['weights'][0]  # Lesion weights
    weights = weights.flatten()

    # Compute correlation
    weighted_scores = lesmat_scaled @ weights
    raw_correlation, p_raw = pearsonr(weighted_scores, behavior_scaled)

    # Linear calibration (correct for correlation-optimization bias)
    if show_info:
        print(f"  Raw correlation: {raw_correlation:.4f}")

    calibrate_lm = LinearRegression()
    calibrate_lm.fit(weighted_scores.reshape(-1, 1), behavior)

    # Compute calibrated predictions
    predictions = calibrate_lm.predict(weighted_scores.reshape(-1, 1))
    calibrated_correlation, p_cal = pearsonr(predictions, behavior)

    if show_info:
        print(f"  Calibrated correlation: {calibrated_correlation:.4f}")

    # Map back to voxel space if using patches
    if patchinfo is not None and 'patchindx' in patchinfo:
        weights_full = patches_to_voxels(weights, patchinfo['patchindx'])
    else:
        weights_full = weights

    # Create statistical map (use correlation as statistic)
    stat_img = matrix_to_image(weights_full, mask_img)

    # Create weights image
    raw_weights_img = matrix_to_image(weights_full, mask_img)

    # Apply cluster threshold if requested
    if cluster_threshold is not None:
        from ..core.image_utils import label_clusters
        # Binrize weights first
        import scipy.ndimage as ndimage
        weights_data = raw_weights_img.get_fdata()
        labeled, n_clusters = ndimage.label(np.abs(weights_data) > cluster_threshold)

        # Zero out small clusters
        for i in range(1, n_clusters + 1):
            cluster_size = np.sum(labeled == i)
            if cluster_size < cluster_threshold:
                weights_data[labeled == i] = 0

        raw_weights_img = nib.Nifti1Image(weights_data, mask_img.affine, mask_img.header)

    # Compute p-values via permutation if requested
    pval_img = None
    zmap_img = None

    if nperm > 0 and multiple_comparison in ['FWERperm', 'clusterPerm']:
        if show_info:
            print(f"  Running {nperm} permutations for FWER control...")

        # Simplified permutation test
        max_stats = []
        for i in range(nperm):
            perm_behavior = np.random.permutation(behavior)
            perm_behavior_scaled = scaler_behavior.transform(perm_behavior.reshape(-1, 1)).flatten()

            perm_inmats = [lesmat_scaled, perm_behavior_scaled.reshape(-1, 1)]
            perm_result = antspyt.sparse_decom2(
                perm_inmats,
                inmask=inmask,
                sparseness=sparseness,
                nvecs=nvecs,
                its=its,
            )

            perm_weights = perm_result['weights'][0].flatten()
            max_stats.append(np.max(np.abs(perm_weights)))

        # Establish threshold
        threshold = np.percentile(max_stats, 95)

        # Create p-value map based on permutation
        if patchinfo is not None and 'patchindx' in patchinfo:
            weights_thresholded = patches_to_voxels(weights, patchinfo['patchindx'])
        else:
            weights_thresholded = weights.copy()

        # Apply threshold
        weights_thresholded[np.abs(weights_thresholded) < threshold] = 0
        stat_img = matrix_to_image(weights_thresholded, mask_img)

    # Create result object
    result = LesymapResult(
        stat_img=stat_img,
        mask_img=mask_img,
        method='sccan',
        raw_weights_img=raw_weights_img,
        pval_img=pval_img,
        zmap_img=zmap_img,
        sccan_weights=weights,
        sccan_behavior_scale=scaler_behavior.scale_[0],
        sccan_behavior_center=scaler_behavior.mean_[0],
        sccan_lesmat_scale=scaler_lesion.scale_,
        sccan_lesmat_center=scaler_lesion.mean_,
        sccan_predict_lm=calibrate_lm,
        model_params={
            'sparseness': sparseness,
            'nvecs': nvecs,
            'its': its,
            'correlation': calibrated_correlation,
            'p_value': p_cal,
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
        warnings.warn("Weight extraction not supported for non-linear kernels. "
                      "Using permutation importance instead.")
        weights = _compute_svr_importance(svr, lesmat, behavior)

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
        },
        patchinfo=patchinfo,
    )

    return result


def _optimize_sccan_sparseness(lesmat: np.ndarray,
                                behavior: np.ndarray,
                                sparseness_range: List[float] = None,
                                n_folds: int = 5,
                                nvecs: int = 1,
                                its: int = 5,
                                smooth: float = 0.0,
                                show_info: bool = True) -> float:
    """
    Optimize SCCAN sparseness parameter via cross-validation.

    Parameters
    ----------
    lesmat : np.ndarray
        Scaled lesion matrix
    behavior : np.ndarray
        Scaled behavioral scores
    sparseness_range : list of float, optional
        Sparseness values to try
    n_folds : int
        Number of cross-validation folds
    nvecs : int
        Number of canonical vectors
    its : int
        Number of iterations (reduced for speed during CV)
    smooth : float
        Smoothing parameter
    show_info : bool
        Print progress

    Returns
    -------
    float
        Optimal sparseness value
    """
    # Try to import ANTsPy (try multiple possible package names)
    antspyt = None
    for pkg_name in ['antspyx', 'antspy', 'antspyt']:
        try:
            mod = __import__(pkg_name)
            if hasattr(mod, 'sparse_decom2'):
                antspyt = mod
                break
        except (ImportError, AttributeError):
            continue

    if antspyt is None:
        warnings.warn(
            "ANTsPy not available for sparseness optimization. "
            "Using default sparseness=0.1. "
            "Install with: pip install lesymap[sccan]"
        )
        return 0.1

    if sparseness_range is None:
        sparseness_range = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]

    best_sparseness = sparseness_range[0]
    best_correlation = -np.inf

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for sparseness in sparseness_range:
        correlations = []

        for train_idx, test_idx in kf.split(lesmat):
            lesmat_train, lesmat_test = lesmat[train_idx], lesmat[test_idx]
            behavior_train, behavior_test = behavior[train_idx], behavior[test_idx]

            # Run SCCAN on training data
            inmats = [lesmat_train, behavior_train.reshape(-1, 1)]
            inmask = [np.ones(lesmat.shape[1], dtype=int), None]

            try:
                result = antspyt.sparse_decom2(
                    inmats,
                    inmask=inmask,
                    sparseness=sparseness,
                    nvecs=nvecs,
                    its=its,
                    smooth=smooth,
                )

                weights = result['weights'][0].flatten()
                scores_test = lesmat_test @ weights
                corr, _ = pearsonr(scores_test, behavior_test)
                correlations.append(corr)
            except Exception:
                correlations.append(-np.inf)

        mean_corr = np.mean(correlations)

        if show_info:
            print(f"    Sparseness={sparseness}: CV correlation={mean_corr:.4f}")

        if mean_corr > best_correlation:
            best_correlation = mean_corr
            best_sparseness = sparseness

    if show_info:
        print(f"  Optimal sparseness: {best_sparseness}")

    return best_sparseness


def _compute_svr_importance(svr: SVR,
                            lesmat: np.ndarray,
                            behavior: np.ndarray,
                            n_perm: int = 50) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Feature importance scores
    """
    baseline_score = svr.score(lesmat, behavior)
    n_features = lesmat.shape[1]
    importance = np.zeros(n_features)

    for i in range(n_features):
        scores = []
        for _ in range(n_perm):
            lesmat_perm = lesmat.copy()
            lesmat_perm[:, i] = np.random.permutation(lesmat_perm[:, i])
            score = svr.score(lesmat_perm, behavior)
            scores.append(score)

        importance[i] = baseline_score - np.mean(scores)

    return importance
