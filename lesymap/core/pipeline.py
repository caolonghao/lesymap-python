"""
Main pipeline function for LESYMAP-Python.

Implements the core lesymap() analysis workflow.
"""

import warnings
from typing import Union, List, Optional
from pathlib import Path

import numpy as np
import nibabel as nib

from .io import (
    check_input_type,
    load_lesions,
    load_behavior,
    check_binary_values,
)
from .image_utils import (
    mask_from_average,
    images_to_matrix,
    matrix_to_image,
    get_lesion_load,
)
from .patch import (
    get_unique_lesion_patches,
    patches_to_voxels,
    filter_patches_by_prevalence,
)
from .result import LesymapResult


# Valid method-specific parameters for validation
VALID_METHOD_PARAMS = {
    'sccan': {
        'optimize_sparseness', 'sparseness', 'nvecs', 'cthresh',
        'its', 'smooth', 'cluster_threshold', 'sparseness_range', 'n_jobs',
        'robust', 'robust_rank_fallback'
    },
    'svr': {
        'C', 'kernel', 'epsilon', 'gamma', 'degree', 'coef0', 'n_perm', 'max_features'
    },
    'BMfast': set(),
    'ttest': set(),
    'welch': set(),
    'regresfast': {
        'covariates'
    },
    'chisq': set(),
}


def _validate_method_params(method: str, kwargs: dict) -> None:
    """
    Validate method-specific parameters.

    Parameters
    ----------
    method : str
        Analysis method
    kwargs : dict
        Keyword arguments to validate

    Raises
    ------
    ValueError
        If invalid parameters are provided
    """
    valid_params = VALID_METHOD_PARAMS.get(method, set())
    invalid_params = set(kwargs.keys()) - valid_params

    if invalid_params:
        raise ValueError(
            f"Invalid parameters for method '{method}': {invalid_params}\n"
            f"Valid parameters for '{method}' are: {sorted(valid_params) if valid_params else 'None'}"
        )


def _validate_global_params(p_threshold: float,
                            nperm: int,
                            correct_by_les_size: str,
                            min_subject_per_voxel: Union[str, int],
                            n_subjects: int) -> None:
    """
    Validate global parameters.

    Parameters
    ----------
    p_threshold : float
        P-value threshold
    nperm : int
        Number of permutations
    correct_by_les_size : str
        Lesion size correction type
    min_subject_per_voxel : str or int
        Minimum subjects per voxel
    n_subjects : int
        Number of subjects (for validation)

    Raises
    ------
    ValueError
        If parameters are out of valid range
    """
    # p_threshold validation
    if not 0 < p_threshold < 1:
        raise ValueError(
            f"p_threshold must be between 0 and 1 (exclusive), got {p_threshold}"
        )

    # nperm validation
    if nperm < 0:
        raise ValueError(f"nperm must be >= 0, got {nperm}")
    if nperm > 10000:
        warnings.warn(
            f"nperm={nperm} is very large and will be slow. "
            f"Consider using nperm <= 1000 for faster analysis."
        )

    # correct_by_les_size validation
    valid_corrections = ['none', 'voxel', 'behavior', 'both']
    if correct_by_les_size not in valid_corrections:
        raise ValueError(
            f"correct_by_les_size must be one of {valid_corrections}, "
            f"got '{correct_by_les_size}'"
        )

    # min_subject_per_voxel validation
    if isinstance(min_subject_per_voxel, str):
        if not min_subject_per_voxel.endswith('%'):
            raise ValueError(
                f"min_subject_per_voxel as string must end with '%', "
                f"got '{min_subject_per_voxel}'"
            )
        pct = float(min_subject_per_voxel.rstrip('%'))
        if not 0 < pct < 100:
            raise ValueError(
                f"Percentage must be between 0 and 100, got {pct}"
            )
    elif isinstance(min_subject_per_voxel, int):
        if min_subject_per_voxel <= 0:
            raise ValueError("min_subject_per_voxel must be > 0")
        if min_subject_per_voxel >= n_subjects:
            raise ValueError(
                f"min_subject_per_voxel ({min_subject_per_voxel}) "
                f"must be less than n_subjects ({n_subjects})"
            )


def run_lesymap(lesions,
                behavior,
                method='sccan',
                mask=None,
                patchinfo=None,
                correct_by_les_size='none',
                multiple_comparison='fdr',
                p_threshold=0.05,
                min_subject_per_voxel='10%',
                nperm=1000,
                save_dir=None,
                binary_check=True,
                no_patch=False,
                show_info=True,
                **kwargs):
    """
    Main lesion-symptom mapping pipeline.

    Parameters
    ----------
    lesions : list of str or nibabel images
        Lesion maps (must be in template space)
    behavior : array-like or str
        Behavioral scores or path to CSV file
    method : str
        Analysis method: 'sccan', 'svr', 'BMfast', 'ttest', 'welch',
        'regresfast', 'chisq'
    mask : nibabel image or str, optional
        Analysis mask
    patchinfo : dict, optional
        Pre-computed patch information
    correct_by_les_size : str
        Lesion size correction: 'none', 'voxel', 'behavior', 'both'
    multiple_comparison : str
        Multiple comparison correction method
    p_threshold : float
        P-value threshold
    min_subject_per_voxel : str or int
        Minimum subjects per voxel
    nperm : int
        Number of permutations
    save_dir : str, optional
        Output directory
    binary_check : bool
        Check and rebinarize lesion maps
    no_patch : bool
        Skip patch computation
    show_info : bool
        Print progress information
    **kwargs
        Method-specific parameters

    Returns
    -------
    LesymapResult
        Result object with statistical maps
    """
    # Print info
    if show_info:
        print_info("Starting LESYMAP analysis...")

    # Step 1: Input validation and loading
    if show_info:
        print_info("Loading and validating input data...")

    # Detect input type
    input_type = check_input_type(lesions)
    if show_info:
        print_info(f"  Input type: {input_type.value}")

    # Load lesions
    if binary_check:
        images = load_lesions(lesions, check_headers=True)
        binary_info = check_binary_values(images, verbose=show_info)

        # Rebinarize if using 255 format
        if binary_info['is_255_format']:
            if show_info:
                print_info("  Rebinarizing images (255 -> 1)...")
            from .image_utils import threshold_image
            images = [threshold_image(img, 0.5, binarize=True) for img in images]
    else:
        images = load_lesions(lesions, check_headers=True)

    n_subjects = len(images)
    if show_info:
        print_info(f"  Loaded {n_subjects} lesion maps")

    # Load behavior
    behavior = load_behavior(behavior, n_subjects=n_subjects)
    if show_info:
        print_info(f"  Loaded {len(behavior)} behavioral scores")

    # Validate global parameters
    _validate_global_params(
        p_threshold, nperm, correct_by_les_size,
        min_subject_per_voxel, n_subjects
    )

    # Validate method-specific parameters
    _validate_method_params(method, kwargs)

    # Validate method name
    valid_methods = ['sccan', 'svr', 'BMfast', 'ttest', 'welch', 'regresfast', 'chisq']
    if method not in valid_methods:
        import difflib
        suggestions = difflib.get_close_matches(method, valid_methods, n=2, cutoff=0.3)
        raise ValueError(
            f"Unknown method: '{method}'\n"
            f"Valid methods: {', '.join(valid_methods)}\n"
            f"{f'Did you mean: {suggestions[0]}?' if suggestions else ''}"
        )

    # Step 2: Prepare mask
    if show_info:
        print_info("Preparing analysis mask...")

    if mask is None:
        # Auto-generate mask from average lesion map
        if show_info:
            print_info("  Auto-generating mask from average lesion map...")
        mask_img = mask_from_average(images, threshold=0.05, min_voxels=10)
    else:
        if isinstance(mask, str):
            mask_img = nib.load(mask)
        else:
            mask_img = mask

    mask_data = mask_img.get_fdata()
    n_voxels = int(np.sum(mask_data > 0))
    if show_info:
        print_info(f"  Mask contains {n_voxels} voxels")

    # Step 3: Build lesion matrix
    if show_info:
        print_info("Building lesion matrix...")

    lesmat = images_to_matrix(images, mask_img)

    # Step 4: Compute patches (if not disabled)
    if no_patch:
        if show_info:
            print_info("Skipping patch computation (no_patch=True)")
        patchinfo = None
        lesmat_for_analysis = lesmat
    else:
        if patchinfo is None:
            if show_info:
                print_info("Computing unique lesion patches...")
            patchinfo = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

            if show_info:
                print_info(f"  Found {patchinfo['npatches']} unique patches "
                          f"(compression: {patchinfo['compression_ratio']:.2f}x)")
        else:
            if show_info:
                print_info("Using provided patch information...")

        # Filter by prevalence
        if 'patchmatrix' in patchinfo:
            keep_mask = filter_patches_by_prevalence(
                patchinfo, patchinfo['patchmatrix'], min_subject_per_voxel
            )
            n_filtered = np.sum(~keep_mask)
            if n_filtered > 0 and show_info:
                print_info(f"  Filtered {n_filtered} patches below prevalence threshold")

            # Update patchinfo for filtered patches
            # (This is a simplified version - proper implementation would reindex)
            lesmat_for_analysis = patchinfo['patchmatrix'][:, keep_mask]
        else:
            lesmat_for_analysis = patchinfo['patchmatrix']

    # Step 5: Lesion size correction
    if correct_by_les_size != 'none':
        if show_info:
            print_info(f"Applying lesion size correction ({correct_by_les_size})...")

        lesion_volumes = get_lesion_load(images)

        if correct_by_les_size in ['voxel', 'both']:
            # Voxel correction: divide by 1/sqrt(lesion_size)
            correction_factors = 1 / np.sqrt(lesion_volumes)
            lesmat_for_analysis = lesmat_for_analysis * correction_factors[:, np.newaxis]

        if correct_by_les_size in ['behavior', 'both']:
            # Behavior correction: residualize by lesion size
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(lesion_volumes, behavior)
            behavior = behavior - (slope * lesion_volumes + intercept)
            if show_info:
                print_info(f"  Residualized behavior by lesion size")

    # Step 6: Run statistical analysis
    if show_info:
        print_info(f"Running {method} analysis...")

    if method == 'sccan':
        from .methods.multivariate import lsm_sccan
        result = lsm_sccan(
            lesmat_for_analysis,
            behavior,
            mask_img,
            patchinfo=patchinfo,
            multiple_comparison=multiple_comparison,
            p_threshold=p_threshold,
            nperm=nperm,
            show_info=show_info,
            **kwargs
        )
    elif method == 'svr':
        from .methods.multivariate import lsm_svr
        result = lsm_svr(
            lesmat_for_analysis,
            behavior,
            mask_img,
            patchinfo=patchinfo,
            multiple_comparison=multiple_comparison,
            p_threshold=p_threshold,
            nperm=nperm,
            show_info=show_info,
            **kwargs
        )
    elif method == 'BMfast':
        from .methods.univariate import lsm_bmfast
        result = lsm_bmfast(
            lesmat_for_analysis,
            behavior,
            mask_img,
            patchinfo=patchinfo,
            multiple_comparison=multiple_comparison,
            p_threshold=p_threshold,
            nperm=nperm,
            show_info=show_info,
            **kwargs
        )
    elif method in ['ttest', 'welch']:
        from .methods.univariate import lsm_ttest, lsm_welch
        if method == 'ttest':
            result = lsm_ttest(
                lesmat_for_analysis,
                behavior,
                mask_img,
                patchinfo=patchinfo,
                multiple_comparison=multiple_comparison,
                p_threshold=p_threshold,
                nperm=nperm,
                show_info=show_info,
                welch=False,
                **kwargs
            )
        else:
            result = lsm_welch(
                lesmat_for_analysis,
                behavior,
                mask_img,
                patchinfo=patchinfo,
                multiple_comparison=multiple_comparison,
                p_threshold=p_threshold,
                nperm=nperm,
                show_info=show_info,
                **kwargs
            )
    elif method == 'regresfast':
        from .methods.univariate import lsm_regresfast
        result = lsm_regresfast(
            lesmat_for_analysis,
            behavior,
            mask_img,
            patchinfo=patchinfo,
            multiple_comparison=multiple_comparison,
            p_threshold=p_threshold,
            nperm=nperm,
            show_info=show_info,
            **kwargs
        )
    elif method == 'chisq':
        from .methods.univariate import lsm_chisq
        result = lsm_chisq(
            lesmat_for_analysis,
            behavior,
            mask_img,
            patchinfo=patchinfo,
            multiple_comparison=multiple_comparison,
            p_threshold=p_threshold,
            show_info=show_info,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Add call info
    result.callinfo = {
        'method': method,
        'n_subjects': n_subjects,
        'n_voxels': n_voxels,
        'correct_by_les_size': correct_by_les_size,
        'multiple_comparison': multiple_comparison,
        'p_threshold': p_threshold,
        'min_subject_per_voxel': min_subject_per_voxel,
        'nperm': nperm,
        'no_patch': no_patch,
    }

    if show_info:
        print_info("Analysis complete!")

    # Save if requested
    if save_dir is not None:
        if show_info:
            print_info(f"Saving results to {save_dir}...")
        result.save(save_dir, save_model=True)

    return result


def print_info(message: str) -> None:
    """Print info message with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
