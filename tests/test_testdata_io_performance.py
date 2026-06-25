"""Smoke tests for real test_data NIfTI I/O paths."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lesymap.core.image_utils import images_to_matrix, mask_from_average
from lesymap.core.io import check_binary_values, check_headers_match, load_lesions


TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "test_data"
MASK_DIR = TEST_DATA_DIR / "normalized_masks"
IMAGE_DIR = TEST_DATA_DIR / "normalized_images"


def test_real_testdata_loads_headers_mask_and_uint8_matrix():
    """Real symlinked test_data can be streamed into a compact lesion matrix."""
    files = sorted(MASK_DIR.glob("*.nii.gz"))
    if not files:
        pytest.skip("test_data/normalized_masks is not available")
    assert len(files) == 17

    images = load_lesions(files, check_headers=True)

    assert len(images) == 17
    assert check_headers_match(images, raise_error=True)
    assert len({img.shape for img in images}) == 1

    mask = mask_from_average(images, threshold=0.05, min_voxels=10)
    matrix = images_to_matrix(images, mask, dtype=np.uint8)

    assert matrix.shape[0] == 17
    assert matrix.shape[1] == int(np.count_nonzero(mask.get_fdata() > 0))
    assert matrix.dtype == np.uint8

    reference = np.asarray(images[0].dataobj)[np.asarray(mask.dataobj) > 0]
    np.testing.assert_array_equal(matrix[0], reference.astype(np.uint8, copy=False))


def test_continuous_images_default_matrix_dtype_does_not_truncate():
    """Default matrix conversion keeps float64 compatibility for continuous images."""
    files = sorted(IMAGE_DIR.glob("*.nii.gz"))[:2]
    if not files:
        pytest.skip("test_data/normalized_images is not available")
    assert len(files) == 2

    images = load_lesions(files, check_headers=True)
    mask = mask_from_average(images, threshold=0.05, min_voxels=10)
    matrix = images_to_matrix(images, mask)

    assert matrix.dtype == np.float64
    assert np.any((matrix > 0) & (matrix < 1))


def test_check_binary_values_handles_large_nonbinary_without_listing_all_values():
    """Non-binary reporting should not build a huge sorted unique-value payload."""
    data = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    img = nib.Nifti1Image(data, np.eye(4))

    info = check_binary_values([img], verbose=False)

    assert info["is_binary"] is False
    assert info["is_255_format"] is False
    assert len(info["unique_values"]) <= 5


def test_check_binary_values_accepts_mixed_01_and_0255_masks():
    img_01 = nib.Nifti1Image(
        np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        np.eye(4),
    )
    img_0255 = nib.Nifti1Image(
        np.array([[[0, 255], [255, 0]]], dtype=np.uint8),
        np.eye(4),
    )

    info = check_binary_values([img_01, img_0255], verbose=False)

    assert info["is_binary"] is True
    assert info["is_255_format"] is True
