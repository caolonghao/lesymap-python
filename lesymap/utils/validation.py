"""
Input validation utilities for LESYMAP-Python.
"""

from typing import List, Union, Optional
import warnings


__all__ = [
    'validate_lesions',
    'validate_behavior',
]

import numpy as np
import nibabel as nib


def validate_lesions(lesions: Union[List[str], List[nib.Nifti1Image]],
                     n_subjects: Optional[int] = None,
                     min_voxels: int = 1) -> dict:
    """
    Validate lesion maps.

    Parameters
    ----------
    lesions : list of str or nibabel images
        Lesion maps to validate
    n_subjects : int, optional
        Expected number of subjects
    min_voxels : int
        Minimum number of lesioned voxels required

    Returns
    -------
    dict
        Validation report with warnings and errors
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'n_subjects': len(lesions),
    }

    # Check number of subjects
    if n_subjects is not None and len(lesions) != n_subjects:
        report['errors'].append(
            f"Expected {n_subjects} lesions, got {len(lesions)}"
        )
        report['valid'] = False

    # Check each lesion
    lesion_volumes = []
    for i, les in enumerate(lesions):
        if isinstance(les, str):
            img = nib.load(les)
        else:
            img = les

        data = img.get_fdata()
        n_lesioned = np.sum(data > 0)
        lesion_volumes.append(n_lesioned)

        if n_lesioned == 0:
            report['warnings'].append(f"Subject {i}: No lesioned voxels found")
        elif n_lesioned < min_voxels:
            report['warnings'].append(
                f"Subject {i}: Only {n_lesioned} lesioned voxels "
                f"(minimum: {min_voxels})"
            )

    # Check for empty lesions
    if all(v == 0 for v in lesion_volumes):
        report['errors'].append("All lesion maps are empty")
        report['valid'] = False

    report['lesion_volumes'] = lesion_volumes

    return report


def validate_behavior(behavior: np.ndarray,
                     n_subjects: Optional[int] = None,
                     binary: bool = False) -> dict:
    """
    Validate behavioral scores.

    Parameters
    ----------
    behavior : ndarray
        Behavioral scores
    n_subjects : int, optional
        Expected number of subjects
    binary : bool
        Whether behavior should be binary (0/1)

    Returns
    -------
    dict
        Validation report
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'n_subjects': len(behavior),
    }

    # Check number of subjects
    if n_subjects is not None and len(behavior) != n_subjects:
        report['errors'].append(
            f"Expected {n_subjects} behavior scores, got {len(behavior)}"
        )
        report['valid'] = False

    # Check for missing values
    if np.any(np.isnan(behavior)):
        n_nan = np.sum(np.isnan(behavior))
        report['errors'].append(f"{n_nan} NaN values in behavior")
        report['valid'] = False

    # Check for infinite values
    if np.any(np.isinf(behavior)):
        n_inf = np.sum(np.isinf(behavior))
        report['errors'].append(f"{n_inf} infinite values in behavior")
        report['valid'] = False

    # Check binary if requested
    if binary:
        unique_vals = np.unique(behavior[~np.isnan(behavior)])
        if not set(unique_vals).issubset({0, 1}):
            report['warnings'].append(
                f"Behavior should be binary (0/1), but contains: {unique_vals}"
            )

    # Check variance
    if len(behavior) > 1:
        behavior_var = np.var(behavior[~np.isnan(behavior)])
        if behavior_var == 0:
            report['warnings'].append("Behavior has zero variance")

    return report
