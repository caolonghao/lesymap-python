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

    # Main algorithm - faithful implementation of R's match() logic
    # CRITICAL: Must preserve first-occurrence order, not sort unique values
    for i in range(n_subjects):
        # Add weighted current row to cumulative sum
        summed = summed + (lesmat[i, :] * add)

        # Implement R's match() function: match(summed, unique(summed))
        # R's unique() preserves first-occurrence order (does NOT sort)
        # R's match() returns position in unique array

        # Get unique values with their first occurrence indices
        unique_vals, first_index, inverse = np.unique(
            summed,
            return_index=True,
            return_inverse=True
        )

        # Sort by first occurrence to preserve R's behavior
        order = np.argsort(first_index)

        # Create mapping: unique_vals[order] -> 1, 2, 3, ...
        mapping = np.empty(len(order), dtype=np.int64)
        mapping[order] = np.arange(1, len(order) + 1)

        # Apply mapping to get match() result
        summed = mapping[inverse]

        # Update add for next iteration
        add = int(summed.max() + 1)

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
        # All voxels in a patch share the identical lesion pattern, so we
        # just need one representative voxel per patch.
        # np.unique returns sorted unique values; since patchindx is 1..npatches,
        # first_occ[i] is the first position where patchindx == (i+1).
        _, first_occ = np.unique(patchindx, return_index=True)
        patchmatrix = lesmat[:, first_occ]

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
        Patch assignment for each voxel (n_voxels,), 1-indexed (0 = no patch)
    fill_value : float
        Value for voxels not in any patch (patchindx == 0)

    Returns
    -------
    np.ndarray
        Statistics at voxel level (n_voxels,)
    """
    voxel_stats = np.full(len(patchindx), fill_value, dtype=statistic.dtype)
    valid = patchindx > 0
    voxel_stats[valid] = statistic[patchindx[valid] - 1]
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
        Patch assignment for each voxel (n_voxels,), 1-indexed (0 = no patch)
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
    valid = patchindx > 0
    reconstructed[valid] = patch_values[patchindx[valid] - 1]
    return reconstructed
