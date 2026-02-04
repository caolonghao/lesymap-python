"""
Masking utilities for LESYMAP-Python.
"""

from typing import List, Union, Optional
import numpy as np


__all__ = [
    'generate_mask_from_lesions',
    'apply_mask_to_images',
    'create_roi_mask',
    'combine_masks',
    'invert_mask',
]
import nibabel as nib

from ..core.image_utils import mask_from_average, threshold_image


def generate_mask_from_lesions(lesions: List[nib.Nifti1Image],
                               prevalence_threshold: float = 0.05,
                               min_voxels: int = 10) -> nib.Nifti1Image:
    """
    Generate analysis mask from lesion maps.

    Creates a mask that includes voxels lesioned in at least a
    certain proportion of subjects.

    Parameters
    ----------
    lesions : list of nibabel images
        Lesion maps
    prevalence_threshold : float
        Minimum proportion of subjects with lesion at each voxel
    min_voxels : int
        Minimum cluster size in voxels

    Returns
    -------
    nibabel image
        Binary mask

    Examples
    --------
    >>> mask = generate_mask_from_lesions(lesions, prevalence_threshold=0.1)
    """
    return mask_from_average(lesions, threshold=prevalence_threshold, min_voxels=min_voxels)


def apply_mask_to_images(images: List[nib.Nifti1Image],
                        mask: nib.Nifti1Image,
                        fill_value: float = 0) -> List[nib.Nifti1Image]:
    """
    Apply mask to a list of images.

    Parameters
    ----------
    images : list of nibabel images
        Images to mask
    mask : nibabel image
        Binary mask
    fill_value : float
        Value for voxels outside mask

    Returns
    -------
    list of nibabel images
        Masked images
    """
    from ..core.image_utils import apply_mask
    return [apply_mask(img, mask, fill_value) for img in images]


def create_roi_mask(atlas_img: nib.Nifti1Image,
                   roi_labels: Union[int, List[int]]) -> nib.Nifti1Image:
    """
    Create mask from atlas ROI labels.

    Parameters
    ----------
    atlas_img : nibabel image
        Atlas image with integer labels
    roi_labels : int or list of int
        ROI label(s) to include

    Returns
    -------
    nibabel image
        Binary mask for specified ROI(s)
    """
    atlas_data = atlas_img.get_fdata()

    if isinstance(roi_labels, int):
        roi_labels = [roi_labels]

    mask_data = np.zeros_like(atlas_data)
    for label in roi_labels:
        mask_data[atlas_data == label] = 1

    return nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)


def combine_masks(*masks: nib.Nifti1Image,
                 method: str = 'intersection') -> nib.Nifti1Image:
    """
    Combine multiple masks.

    Parameters
    ----------
    *masks : nibabel images
        Masks to combine
    method : str
        Combination method: 'intersection' (AND) or 'union' (OR)

    Returns
    -------
    nibabel image
        Combined mask
    """
    if len(masks) == 0:
        raise ValueError("At least one mask required")

    # Get reference affine and header
    ref_img = masks[0]

    # Load all mask data
    mask_data_list = [mask.get_fdata() for mask in masks]

    # Combine
    if method == 'intersection':
        combined = np.ones_like(mask_data_list[0])
        for mask_data in mask_data_list:
            combined = combined & (mask_data > 0)
    elif method == 'union':
        combined = np.zeros_like(mask_data_list[0])
        for mask_data in mask_data_list:
            combined = combined | (mask_data > 0)
    else:
        raise ValueError(f"Unknown combination method: {method}")

    return nib.Nifti1Image(combined.astype(np.float64), ref_img.affine, ref_img.header)


def invert_mask(mask: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Invert a binary mask.

    Parameters
    ----------
    mask : nibabel image
        Binary mask

    Returns
    -------
    nibabel image
        Inverted mask
    """
    mask_data = mask.get_fdata()
    inverted = 1 - mask_data
    return nib.Nifti1Image(inverted, mask.affine, mask.header)
