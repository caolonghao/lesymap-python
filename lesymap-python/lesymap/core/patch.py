"""
Patch computation for LESYMAP-Python.

Implements unique lesion patch computation for efficient univariate analysis.
Based on R implementation in LESYMAP/R/getUniqueLesionPatches.R
"""

from typing import Dict, Optional, Union
import warnings


__all__ = [
    'get_unique_lesion_patches',
    'patches_to_voxels',
    'filter_patches_by_prevalence',
    'get_patch_indices',
    'reconstruct_from_patches',
]

import numpy as np


def get_unique_lesion_patches(lesmat: np.ndarray,
                               return_patch_matrix: bool = True) -> Dict:
    """
    Compute unique patches of voxels with identical lesion patterns.

    This function groups voxels that have identical lesion patterns across
    subjects, significantly reducing the number of comparisons for univariate
    analysis.

    Algorithm (from R implementation):
        add = 1
        summed = rep(0, ncol(lesmat))
        for (i in 1:nrow(lesmat)):
            summed = summed + (lesmat[i, ]*add)
            summed = match(summed, unique(summed))
            add = max(summed)+1

    Parameters
    ----------
    lesmat : np.ndarray
        Lesion matrix of shape (n_subjects, n_voxels)
        Binary values: 1 = lesioned, 0 = not lesioned
    return_patch_matrix : bool
        If True, return the reduced patch matrix

    Returns
    -------
    dict
        Dictionary containing:
        - 'patchindx': Patch membership for each voxel (1 to npatches)
        - 'npatches': Number of unique patches
        - 'nvoxels': Total number of voxels
        - 'patchmatrix': Reduced matrix (n_subjects, npatches), if requested
        - 'voxels_per_patch': Number of voxels in each patch
        - 'compression_ratio': Ratio of voxels to patches

    Examples
    --------
    >>> lesmat = np.array([
    ...     [1, 1, 0, 0, 1],
    ...     [1, 1, 0, 0, 0],
    ...     [0, 0, 1, 1, 0]
    ... ])
    >>> patches = get_unique_lesion_patches(lesmat)
    >>> print(patches['npatches'])
    3
    """
    n_subjects, n_voxels = lesmat.shape

    # Initialize
    add = 1
    summed = np.zeros(n_voxels, dtype=np.int64)

    # Main algorithm (optimized version without dictionary comprehension)
    for i in range(n_subjects):
        # Add weighted current row to cumulative sum
        summed = summed + (lesmat[i, :] * add)

        # Remap to consecutive integers using vectorized search
        # This is more efficient than dictionary comprehension
        unique_vals = np.unique(summed)
        remapped = np.empty_like(summed)
        for new_val, old_val in enumerate(unique_vals, start=1):
            remapped[summed == old_val] = new_val
        summed = remapped

        # Update add for next iteration
        add = summed.max() + 1

    # Final result is the patch assignment
    patchindx = summed
    npatches = len(np.unique(patchindx))

    # Count voxels per patch
    voxels_per_patch = np.bincount(patchindx, minlength=npatches+1)[1:]  # Skip 0

    result = {
        'patchindx': patchindx,
        'npatches': npatches,
        'nvoxels': n_voxels,
        'voxels_per_patch': voxels_per_patch,
        'compression_ratio': n_voxels / npatches,
    }

    # Generate patch matrix if requested
    if return_patch_matrix:
        # For each patch, compute the mean lesion pattern across its voxels
        # (Since all voxels in a patch have the same pattern, mean = pattern)
        patchmatrix = np.zeros((n_subjects, npatches))

        for patch_id in range(1, npatches + 1):
            # Get all voxels in this patch
            voxel_mask = patchindx == patch_id

            # Get lesion pattern (should be same for all voxels in patch)
            pattern = lesmat[:, voxel_mask]

            # Use first voxel as representative (all should be identical)
            patchmatrix[:, patch_id - 1] = pattern[:, 0]

        result['patchmatrix'] = patchmatrix

    # Print info
    n_subjects_per_patch = np.sum(patchmatrix, axis=0) if 'patchmatrix' in result else None

    if n_subjects_per_patch is not None:
        min_subjects = int(n_subjects_per_patch.min())
        max_subjects = int(n_subjects_per_patch.max())
    else:
        min_subjects = max_subjects = None

    result['min_subjects_per_patch'] = min_subjects
    result['max_subjects_per_patch'] = max_subjects

    return result


def patches_to_voxels(statistic: np.ndarray,
                      patchindx: np.ndarray,
                      fill_value: float = 0) -> np.ndarray:
    """
    Map patch-level statistics back to voxel-level.

    Parameters
    ----------
    statistic : np.ndarray
        Statistics at patch level (n_patches,)
    patchindx : np.ndarray
        Patch assignment for each voxel (n_voxels,)
    fill_value : float
        Value for voxels not in any patch

    Returns
    -------
    np.ndarray
        Statistics at voxel level (n_voxels,)
    """
    n_voxels = len(patchindx)
    voxel_stats = np.full(n_voxels, fill_value, dtype=statistic.dtype)

    # Map patch statistics to voxels
    npatches = len(statistic)
    for patch_id in range(1, npatches + 1):
        voxel_mask = patchindx == patch_id
        voxel_stats[voxel_mask] = statistic[patch_id - 1]

    return voxel_stats


def filter_patches_by_prevalence(patchinfo: Dict,
                                  lesmat: np.ndarray,
                                  min_subjects: Union[int, str]) -> np.ndarray:
    """
    Filter patches based on minimum subject prevalence.

    Parameters
    ----------
    patchinfo : dict
        Patch information from get_unique_lesion_patches
    lesmat : np.ndarray
        Lesion matrix (n_subjects, n_patches)
    min_subjects : int or str
        Minimum subjects per patch. Can be:
        - int: absolute number
        - str: percentage (e.g., '10%')

    Returns
    -------
    np.ndarray
        Boolean mask of patches to keep
    """
    # Compute subjects per patch (lesioned or total)
    subjects_per_patch = np.sum(lesmat > 0, axis=0)

    # Parse threshold
    if isinstance(min_subjects, str):
        if min_subjects.endswith('%'):
            pct = float(min_subjects.rstrip('%'))
            n_subjects = lesmat.shape[0]
            threshold = np.ceil(n_subjects * pct / 100)
        else:
            raise ValueError(f"Invalid min_subjects format: {min_subjects}")
    else:
        threshold = min_subjects

    # Create mask
    keep_mask = subjects_per_patch >= threshold

    return keep_mask


def get_patch_indices(patchindx: np.ndarray,
                      patch_id: int) -> np.ndarray:
    """
    Get voxel indices for a specific patch.

    Parameters
    ----------
    patchindx : np.ndarray
        Patch assignment for each voxel
    patch_id : int
        Patch ID (1-indexed)

    Returns
    -------
    np.ndarray
        Indices of voxels in the patch
    """
    return np.where(patchindx == patch_id)[0]


def reconstruct_from_patches(patch_values: np.ndarray,
                             patchindx: np.ndarray,
                             n_voxels: int,
                             fill_value: float = 0) -> np.ndarray:
    """
    Reconstruct full voxel array from patch values.

    Parameters
    ----------
    patch_values : np.ndarray
        Values for each patch (n_patches,)
    patchindx : np.ndarray
        Patch assignment for each voxel (n_voxels,)
    n_voxels : int
        Total number of voxels
    fill_value : float
        Value for unassigned voxels

    Returns
    -------
    np.ndarray
        Reconstructed voxel array
    """
    reconstructed = np.full(n_voxels, fill_value, dtype=patch_values.dtype)

    for patch_id, value in enumerate(patch_values, start=1):
        reconstructed[patchindx == patch_id] = value

    return reconstructed
