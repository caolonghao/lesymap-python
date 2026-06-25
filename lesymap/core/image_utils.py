"""
Image processing utilities for LESYMAP-Python.

Provides functions for NIfTI image manipulation, masking, and conversion.
"""

from typing import List, Union, Optional, Tuple
from pathlib import Path


__all__ = [
    'mask_from_average',
    'threshold_image',
    'average_images',
    'images_to_matrix',
    'matrix_to_image',
    'get_lesion_load',
    'apply_mask',
    'dilate_mask',
    'erode_mask',
    'label_clusters',
]

import numpy as np
import nibabel as nib
from scipy import ndimage


def threshold_image(img: nib.Nifti1Image,
                   threshold: float,
                   binarize: bool = False,
                   above: bool = True) -> nib.Nifti1Image:
    """
    Apply threshold to a NIfTI image.

    Parameters
    ----------
    img : nibabel image
        Input image
    threshold : float
        Threshold value
    binarize : bool
        If True, set values above threshold to 1, else to 0
    above : bool
        If True, keep values above threshold; if False, keep values below

    Returns
    -------
    nibabel image
        Thresholded image
    """
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    if above:
        mask = data > threshold
    else:
        mask = data < threshold

    if binarize:
        data = mask.astype(np.float64)
    else:
        data = data * mask

    return nib.Nifti1Image(data, affine, header)


def average_images(images: List[nib.Nifti1Image],
                   weights: Optional[np.ndarray] = None) -> nib.Nifti1Image:
    """
    Average multiple NIfTI images.

    Parameters
    ----------
    images : list of nibabel images
        Images to average
    weights : array-like, optional
        Weights for each image (must sum to 1)

    Returns
    -------
    nibabel image
        Averaged image
    """
    if len(images) == 0:
        raise ValueError("At least one image is required")

    # Set weights
    if weights is None:
        weights = np.ones(len(images)) / len(images)
    else:
        weights = np.asarray(weights)
        if len(weights) != len(images):
            raise ValueError("Number of weights must match number of images")
        weights = weights / weights.sum()  # Normalize

    # Compute weighted average without retaining all image data at once.
    avg_data = np.zeros(images[0].shape, dtype=np.float64)
    for img, w in zip(images, weights):
        avg_data += np.asanyarray(img.dataobj) * w

    # Create output image
    return nib.Nifti1Image(avg_data, images[0].affine, images[0].header)


def mask_from_average(images: List[nib.Nifti1Image],
                      threshold: float = 0.1,
                      min_voxels: int = 10) -> nib.Nifti1Image:
    """
    Create a mask from the average of multiple images.

    Parameters
    ----------
    images : list of nibabel images
        Input images (e.g., lesion maps)
    threshold : float
        Threshold for average image (proportion of subjects)
    min_voxels : int
        Minimum cluster size in voxels

    Returns
    -------
    nibabel image
        Binary mask image
    """
    if len(images) == 0:
        raise ValueError("At least one image is required")

    # Compute average with streaming accumulation to avoid retaining per-image arrays.
    avg_data = np.zeros(images[0].shape, dtype=np.float64)
    scale = 1.0 / len(images)
    for img in images:
        avg_data += np.asanyarray(img.dataobj) * scale

    # Threshold
    mask_data = (avg_data > threshold).astype(np.float64)

    # Remove small clusters
    if min_voxels > 0:
        labeled, num_features = ndimage.label(mask_data)
        sizes = ndimage.sum(mask_data, labeled, range(num_features + 1))

        # Keep only large enough clusters
        mask_data = np.zeros_like(mask_data)
        for i, size in enumerate(sizes):
            if size >= min_voxels:
                mask_data[labeled == i] = 1

    return nib.Nifti1Image(mask_data, images[0].affine, images[0].header)


def label_clusters(img: nib.Nifti1Image,
                   min_size: int = 10) -> Tuple[nib.Nifti1Image, int]:
    """
    Label connected components in a binary image.

    Parameters
    ----------
    img : nibabel image
        Binary image
    min_size : int
        Minimum cluster size to keep

    Returns
    -------
    labeled_img : nibabel image
        Labeled image (0 = background)
    n_clusters : int
        Number of clusters found
    """
    data = img.get_fdata()
    labeled, n_clusters = ndimage.label(data > 0)

    # Filter by size if requested
    if min_size > 0:
        sizes = ndimage.sum(data > 0, labeled, range(n_clusters + 1))
        labeled_filtered = np.zeros_like(labeled)

        for i in range(1, n_clusters + 1):
            if sizes[i] >= min_size:
                labeled_filtered[labeled == i] = i

        labeled = labeled_filtered
        n_clusters = len(np.unique(labeled)) - 1  # Exclude background

    return nib.Nifti1Image(labeled.astype(np.int32), img.affine, img.header), n_clusters


def images_to_matrix(images: List[nib.Nifti1Image],
                     mask: nib.Nifti1Image,
                     dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Convert list of NIfTI images to matrix (subjects x voxels).

    Parameters
    ----------
    images : list of nibabel images
        Input images
    mask : nibabel image
        Binary mask defining voxels to extract
    dtype : np.dtype
        Data type for output matrix

    Returns
    -------
    np.ndarray
        Matrix of shape (n_subjects, n_voxels_in_mask)
    """
    mask_data = np.asanyarray(mask.dataobj)
    voxel_mask = mask_data > 0
    n_voxels = int(np.count_nonzero(voxel_mask))

    matrix = np.empty((len(images), n_voxels), dtype=dtype)
    for i, img in enumerate(images):
        data = np.asanyarray(img.dataobj)
        matrix[i, :] = data[voxel_mask].astype(dtype, copy=False)

    return matrix


def matrix_to_image(vector: np.ndarray,
                    mask: nib.Nifti1Image,
                    fill_value: float = 0) -> nib.Nifti1Image:
    """
    Convert vector back to NIfTI image using mask.

    Parameters
    ----------
    vector : array-like
        1D array of values for masked voxels
    mask : nibabel image
        Binary mask defining voxel locations
    fill_value : float
        Value for voxels outside mask

    Returns
    -------
    nibabel image
        NIfTI image with values placed in mask
    """
    mask_data = mask.get_fdata()
    voxel_indices = np.where(mask_data > 0)

    # Create empty image
    img_data = np.full(mask_data.shape, fill_value, dtype=vector.dtype)

    # Fill in values
    img_data[voxel_indices] = vector

    return nib.Nifti1Image(img_data, mask.affine, mask.header)


def get_lesion_load(images: List[nib.Nifti1Image],
                    voxel_volume: Optional[float] = None) -> np.ndarray:
    """
    Calculate lesion load (volume) for each subject.

    Parameters
    ----------
    images : list of nibabel images
        Binary lesion maps
    voxel_volume : float, optional
        Volume of each voxel in mm^3. If None, calculates from header.

    Returns
    -------
    np.ndarray
        Lesion volumes for each subject
    """
    loads = []

    for img in images:
        data = img.get_fdata()
        n_lesioned_voxels = np.sum(data > 0)

        # Get voxel volume from header if not provided
        if voxel_volume is None:
            zooms = img.header.get_zooms()[:3]
            voxel_volume = np.prod(zooms)

        volume = n_lesioned_voxels * voxel_volume
        loads.append(volume)

    return np.array(loads)


def apply_mask(img: nib.Nifti1Image,
               mask: nib.Nifti1Image,
               fill_value: float = 0) -> nib.Nifti1Image:
    """
    Apply a binary mask to an image.

    Parameters
    ----------
    img : nibabel image
        Input image
    mask : nibabel image
        Binary mask
    fill_value : float
        Value for voxels outside mask

    Returns
    -------
    nibabel image
        Masked image
    """
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    masked_data = img_data.copy()
    masked_data[mask_data == 0] = fill_value

    return nib.Nifti1Image(masked_data, img.affine, img.header)


def dilate_mask(mask: nib.Nifti1Image,
                iterations: int = 1,
                connect_diag: bool = True) -> nib.Nifti1Image:
    """
    Dilate a binary mask.

    Parameters
    ----------
    mask : nibabel image
        Binary mask
    iterations : int
        Number of dilation iterations
    connect_diag : bool
        Whether to use diagonal connectivity (26-connectivity in 3D)

    Returns
    -------
    nibabel image
        Dilated mask
    """
    data = mask.get_fdata().astype(np.uint8)

    # Define structuring element
    if connect_diag:
        structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    else:
        structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity

    # Dilate
    dilated = ndimage.binary_dilation(data, structure=structure, iterations=iterations)

    return nib.Nifti1Image(dilated.astype(np.float64), mask.affine, mask.header)


def erode_mask(mask: nib.Nifti1Image,
               iterations: int = 1,
               connect_diag: bool = True) -> nib.Nifti1Image:
    """
    Erode a binary mask.

    Parameters
    ----------
    mask : nibabel image
        Binary mask
    iterations : int
        Number of erosion iterations
    connect_diag : bool
        Whether to use diagonal connectivity

    Returns
    -------
    nibabel image
        Eroded mask
    """
    data = mask.get_fdata().astype(np.uint8)

    if connect_diag:
        structure = ndimage.generate_binary_structure(3, 3)
    else:
        structure = ndimage.generate_binary_structure(3, 1)

    eroded = ndimage.binary_erosion(data, structure=structure, iterations=iterations)

    return nib.Nifti1Image(eroded.astype(np.float64), mask.affine, mask.header)
