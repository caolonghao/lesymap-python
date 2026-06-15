"""
LESYMAP-Python: Lesion-Symptom Mapping for Neuroimaging Research

A Python implementation of LESYMAP for mapping brain areas responsible
for cognitive deficits by analyzing lesion maps (NIfTI images) and
behavioral scores from stroke patients.

Main entry point:
    lesymap() - Main lesion-symptom mapping function

Example:
    >>> import lesymap
    >>> result = lesymap.lesymap(
    ...     lesions=lesion_files,
    ...     behavior=behavior_scores,
    ...     method='sccan'
    ... )
    >>> result.save('output_dir/')
"""

__version__ = '0.1.0'

import numpy as np

# Import main functions
from .core.io import (
    check_input_type,
    load_lesions,
    load_behavior,
    check_headers_match,
)
from .core.image_utils import (
    mask_from_average,
    threshold_image,
    average_images,
    images_to_matrix,
    matrix_to_image,
)
from .core.patch import get_unique_lesion_patches
from .core.registration import (
    register_lesion_to_template,
    register_batch,
    get_mni152_template_path,
)
from .methods.univariate import (
    lsm_bmfast,
    lsm_ttest,
    lsm_welch,
    lsm_regresfast,
    lsm_chisq,
)
from .methods.multivariate import lsm_sccan, lsm_svr
from .methods.correction import (
    correct_pvalues,
    fwer_permutation_threshold,
    cluster_permutation_threshold,
)
from .core.result import LesymapResult
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

__all__ = [
    # Main function
    'lesymap',
    # Registration
    'register_lesion_to_template',
    'register_batch',
    'get_mni152_template_path',
    # I/O
    'check_input_type',
    'load_lesions',
    'load_behavior',
    'check_headers_match',
    # Image utils
    'mask_from_average',
    'threshold_image',
    'average_images',
    'images_to_matrix',
    'matrix_to_image',
    # Patch
    'get_unique_lesion_patches',
    # Univariate methods
    'lsm_bmfast',
    'lsm_ttest',
    'lsm_welch',
    'lsm_regresfast',
    'lsm_chisq',
    # Multivariate methods
    'lsm_sccan',
    'lsm_svr',
    # Correction
    'correct_pvalues',
    'fwer_permutation_threshold',
    'cluster_permutation_threshold',
    # Result class
    'LesymapResult',
]


def lesymap(
    lesions: Union[List[str], List['nibabel.nifti1.Nifti1Image'], 'nibabel.nifti1.Nifti1Image', str],
    behavior: Union[np.ndarray, List[float], str],
    method: str = 'sccan',
    mask: Union[str, 'nibabel.nifti1.Nifti1Image', None] = None,
    patchinfo: Optional[Dict[str, Any]] = None,
    correct_by_les_size: str = 'none',
    multiple_comparison: str = 'fdr',
    p_threshold: float = 0.05,
    min_subject_per_voxel: Union[str, int] = '10%',
    nperm: int = 1000,
    save_dir: Union[str, Path, None] = None,
    binary_check: bool = True,
    no_patch: bool = False,
    show_info: bool = True,
    **kwargs: Any
) -> LesymapResult:
    """
    Main lesion-symptom mapping function.

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
        Analysis mask. If None, auto-generated from lesions.
    patchinfo : dict, optional
        Pre-computed patch information
    correct_by_les_size : str
        Lesion size correction: 'none', 'voxel', 'behavior', 'both'
    multiple_comparison : str
        Multiple comparison correction: 'fdr', 'bonferroni', 'holm',
        'FWERperm', 'clusterPerm', 'none'
    p_threshold : float
        P-value threshold for significance
    min_subject_per_voxel : str or int
        Minimum subjects per voxel (e.g., '10%' or 5)
    nperm : int
        Number of permutations for permutation-based methods
    save_dir : str, optional
        Directory to save results
    binary_check : bool
        Check and rebinarize lesion maps if needed
    no_patch : bool
        Skip patch computation (use full voxel-wise analysis)
    show_info : bool
        Print progress information
    **kwargs
        Additional method-specific parameters

    Returns
    -------
    result : LesymapResult
        Object containing statistical maps and metadata

    Examples
    --------
    >>> import lesymap
    >>> result = lesymap.lesymap(
    ...     lesions=['sub1.nii.gz', 'sub2.nii.gz'],
    ...     behavior=[1.2, 3.4, 2.1],
    ...     method='sccan'
    ... )
    """
    from .core.pipeline import run_lesymap
    return run_lesymap(
        lesions=lesions,
        behavior=behavior,
        method=method,
        mask=mask,
        patchinfo=patchinfo,
        correct_by_les_size=correct_by_les_size,
        multiple_comparison=multiple_comparison,
        p_threshold=p_threshold,
        min_subject_per_voxel=min_subject_per_voxel,
        nperm=nperm,
        save_dir=save_dir,
        binary_check=binary_check,
        no_patch=no_patch,
        show_info=show_info,
        **kwargs
    )
