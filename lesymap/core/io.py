"""
Input validation and I/O functions for LESYMAP-Python.

Handles loading and validation of NIfTI images and behavioral data.
"""

import os
import warnings
from enum import Enum
from typing import List, Union, Optional
from pathlib import Path


__all__ = [
    'check_input_type',
    'load_lesions',
    'load_behavior',
    'check_headers_match',
    'InputType',
]

import numpy as np
import nibabel as nib


class InputType(Enum):
    """Types of input data for lesion maps."""
    NIFTI_FILES = 'nifti_files'
    NIFTI_IMAGE = 'nifti_image'
    NIFTI_LIST = 'nifti_list'
    NIFTI_4D = 'nifti_4d'


def check_input_type(lesions, check_headers=False) -> InputType:
    """
    Detect the type of input for lesion maps.

    Parameters
    ----------
    lesions : various
        Input data (file paths, nibabel images, or 4D image)
    check_headers : bool
        Whether to check if headers match across images

    Returns
    -------
    InputType
        The detected input type

    Raises
    ------
    TypeError
        If input type is not recognized
    """
    # Check if it's a single string (file path)
    if isinstance(lesions, str):
        img = nib.load(lesions)
        if img.ndim == 4:
            return InputType.NIFTI_4D
        return InputType.NIFTI_IMAGE

    # Check if it's a Path object
    if isinstance(lesions, Path):
        img = nib.load(str(lesions))
        if img.ndim == 4:
            return InputType.NIFTI_4D
        return InputType.NIFTI_IMAGE

    # Check if it's a single nibabel image
    if isinstance(lesions, nib.nifti1.Nifti1Image):
        if lesions.ndim == 4:
            return InputType.NIFTI_4D
        return InputType.NIFTI_IMAGE

    # Check if it's a list
    if isinstance(lesions, list):
        if len(lesions) == 0:
            raise TypeError("Empty list provided")

        # Check all elements
        first_is_str = isinstance(lesions[0], (str, Path))
        first_is_img = isinstance(lesions[0], nib.nifti1.Nifti1Image)

        if first_is_str:
            # All should be file paths
            if not all(isinstance(x, (str, Path)) for x in lesions):
                raise TypeError("Mixed types in list: all elements must be file paths or nibabel images")
            return InputType.NIFTI_FILES
        elif first_is_img:
            # All should be nibabel images
            if not all(isinstance(x, nib.nifti1.Nifti1Image) for x in lesions):
                raise TypeError("Mixed types in list: all elements must be file paths or nibabel images")
            return InputType.NIFTI_LIST
        else:
            raise TypeError("List elements must be file paths or nibabel images")

    # Check if it's a numpy array (unusual but possible)
    if isinstance(lesions, np.ndarray):
        if lesions.ndim == 4:
            return InputType.NIFTI_4D
        raise TypeError("NumPy arrays must be 4D for 4D NIfTI representation")

    raise TypeError(f"Unrecognized input type: {type(lesions)}")


def check_headers_match(images: List[nib.Nifti1Image],
                        raise_error: bool = False) -> bool:
    """
    Check if all images have matching spatial dimensions and spacing.

    Parameters
    ----------
    images : list of nibabel images
        Images to check
    raise_error : bool
        Whether to raise an error if headers don't match

    Returns
    -------
    bool
        True if all headers match

    Raises
    ------
    ValueError
        If headers don't match and raise_error=True
    """
    if len(images) <= 1:
        return True

    # Get reference header from first image
    ref_shape = images[0].shape
    ref_affine = images[0].affine
    ref_zooms = images[0].header.get_zooms()[:3]

    mismatches = []

    for i, img in enumerate(images[1:], start=1):
        # Check shape
        if img.shape != ref_shape:
            mismatches.append(
                f"Image {i}: shape mismatch {img.shape} vs {ref_shape}"
            )

        # Check affine (approximate check for orientation/spacing)
        if not np.allclose(img.affine, ref_affine, atol=1e-3):
            mismatches.append(
                f"Image {i}: affine mismatch"
            )

        # Check voxel sizes
        zooms = img.header.get_zooms()[:3]
        if not np.allclose(zooms, ref_zooms, atol=1e-3):
            mismatches.append(
                f"Image {i}: voxel size mismatch {zooms} vs {ref_zooms}"
            )

    if mismatches:
        msg = "Header mismatches found:\n" + "\n".join(mismatches)
        if raise_error:
            raise ValueError(msg)
        warnings.warn(msg)
        return False

    return True


def load_lesions(lesions,
                 check_headers: bool = True,
                 rebinarize_threshold: Optional[float] = None) -> List[nib.Nifti1Image]:
    """
    Load lesion maps from various input types.

    Parameters
    ----------
    lesions : various
        Input data (file paths, nibabel images, or 4D image)
    check_headers : bool
        Whether to check if headers match across images
    rebinarize_threshold : float, optional
        If provided, binarize images using this threshold

    Returns
    -------
    list of nibabel images
        Loaded 3D lesion maps

    Raises
    ------
    ValueError
        If input is invalid or headers don't match
    """
    input_type = check_input_type(lesions)

    images = []

    if input_type == InputType.NIFTI_FILES:
        # Load from file paths
        images = [nib.load(str(f)) for f in lesions]

    elif input_type == InputType.NIFTI_LIST:
        images = lesions.copy()

    elif input_type == InputType.NIFTI_IMAGE:
        images = [lesions]

    elif input_type == InputType.NIFTI_4D:
        # Split 4D image into list of 3D images
        data = lesions.get_fdata()
        if data.ndim != 4:
            raise ValueError(f"Expected 4D image, got {data.ndim}D")

        n_volumes = data.shape[3]
        affine = lesions.affine
        header = lesions.header

        for i in range(n_volumes):
            img_data = data[:, :, :, i]
            new_img = nib.Nifti1Image(img_data, affine, header)
            images.append(new_img)

    # Check headers
    if check_headers:
        check_headers_match(images, raise_error=True)

    # Rebinarize if requested
    if rebinarize_threshold is not None:
        images = [threshold_image(img, rebinarize_threshold, binarize=True)
                  for img in images]

    return images


def load_behavior(behavior,
                  n_subjects: Optional[int] = None) -> np.ndarray:
    """
    Load behavioral scores from file or array.

    Parameters
    ----------
    behavior : str, array-like, or np.ndarray
        Behavioral scores (file path or array)
    n_subjects : int, optional
        Expected number of subjects (for validation)

    Returns
    -------
    np.ndarray
        Behavioral scores as 1D array

    Raises
    ------
    ValueError
        If number of subjects doesn't match
    """
    # Load from file if needed
    if isinstance(behavior, (str, Path)):
        behavior = np.loadtxt(behavior)

    # Convert to numpy array
    behavior = np.asarray(behavior)

    # Ensure 1D
    if behavior.ndim != 1:
        if behavior.ndim == 2 and behavior.shape[1] == 1:
            behavior = behavior.flatten()
        else:
            raise ValueError(f"Behavior must be 1D array, got shape {behavior.shape}")

    # Check length
    if n_subjects is not None and len(behavior) != n_subjects:
        raise ValueError(
            f"Number of behavior scores ({len(behavior)}) "
            f"doesn't match number of lesions ({n_subjects})"
        )

    return behavior


def check_binary_values(images: List[nib.Nifti1Image],
                        allow_255: bool = True,
                        verbose: bool = True) -> dict:
    """
    Check if images contain binary values (0 and 1, or 0 and 255).

    Parameters
    ----------
    images : list of nibabel images
        Images to check
    allow_255 : bool
        Whether to allow 255 as binary value (MRIcron format)
    verbose : bool
        Whether to print warnings

    Returns
    -------
    dict
        Information about binary values found
    """
    observed_values = set()
    invalid_examples = []
    is_binary = True
    is_255_format = False

    for i, img in enumerate(images):
        data = np.asanyarray(img.dataobj)

        has_zero = bool(np.any(data == 0))
        has_one = bool(np.any(data == 1))
        has_255 = bool(np.any(data == 255)) if allow_255 else False

        if has_zero:
            observed_values.add(0)
        if has_one:
            observed_values.add(1)
        if has_255:
            observed_values.add(255)

        if np.all((data == 0) | (data == 1)):
            pass  # Standard binary
        elif allow_255 and np.all((data == 0) | (data == 255)):
            is_255_format = True
            if verbose:
                warnings.warn(
                    f"Image {i} uses 255 values (MRIcron format). "
                    "Consider rebinarizing with rebinarize_threshold=0.5"
                )
        else:
            is_binary = False
            valid_mask = (data == 0) | (data == 1)
            if allow_255:
                valid_mask |= data == 255
            invalid = data[~valid_mask]
            if invalid_examples == [] and invalid.size > 0:
                invalid_examples = invalid.flat[:5].tolist()
            if verbose:
                warnings.warn(
                    f"Image {i} contains non-binary values: {invalid_examples}..."
                )

    valid_values = {0, 1}
    if allow_255:
        valid_values.add(255)
    global_binary = is_binary and observed_values <= valid_values

    if invalid_examples:
        unique_values = np.asarray((sorted(observed_values) + invalid_examples)[:5])
    else:
        unique_values = np.asarray(sorted(observed_values))

    return {
        'unique_values': unique_values,
        'is_binary': global_binary,
        'is_255_format': is_255_format,
    }
