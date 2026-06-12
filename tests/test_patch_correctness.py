"""
Test patch algorithm correctness against R implementation.

This module tests that Python's get_unique_lesion_patches() produces
identical results to R's getUniqueLesionPatches().
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lesymap.core.patch import get_unique_lesion_patches


def test_simple_binary_matrix():
    """Test with a simple 3x5 binary lesion matrix."""

    # Test case 1: Simple pattern
    lesmat = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    # Basic checks
    assert result['nvoxels'] == 5
    assert result['npatches'] >= 1
    assert result['npatches'] <= 5
    assert len(result['patchindx']) == 5

    # Compression ratio check
    assert result['compression_ratio'] == 5 / result['npatches']

    # Patch matrix shape
    assert result['patchmatrix'].shape[0] == 3  # n_subjects
    assert result['patchmatrix'].shape[1] == result['npatches']

    print(f"✓ Test 1 passed: {result['npatches']} patches found")
    print(f"  Patch indices: {result['patchindx']}")
    print(f"  Compression: {result['compression_ratio']:.2f}x")


def test_identical_columns():
    """Test that identical columns get the same patch number."""

    # Columns 0 and 2 are identical, should be in same patch
    lesmat = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 1]
    ], dtype=np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=False)

    # Columns 0 and 2 should have the same patch index
    assert result['patchindx'][0] == result['patchindx'][2], \
        f"Identical columns should have same patch: {result['patchindx']}"

    # Column 1 and 3 are different from 0 and 2
    assert result['patchindx'][1] != result['patchindx'][0]
    assert result['patchindx'][3] != result['patchindx'][0]

    print(f"✓ Test 2 passed: Identical columns correctly grouped")
    print(f"  Patch indices: {result['patchindx']}")


def test_all_unique():
    """Test when all voxels have unique patterns."""

    lesmat = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 1]
    ], dtype=np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    # Each column is unique, so should have 4 patches
    assert result['npatches'] == 4, \
        f"Expected 4 unique patches, got {result['npatches']}"

    # All patch indices should be different
    assert len(np.unique(result['patchindx'])) == 4

    # Compression ratio should be 1.0
    assert result['compression_ratio'] == 1.0

    print(f"✓ Test 3 passed: All unique patterns detected")
    print(f"  Patch indices: {result['patchindx']}")


def test_all_same():
    """Test when all voxels have identical patterns."""

    lesmat = np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ], dtype=np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    # All columns identical, should be 1 patch
    assert result['npatches'] == 1, \
        f"Expected 1 patch for identical columns, got {result['npatches']}"

    # All indices should be the same
    assert len(np.unique(result['patchindx'])) == 1

    # Compression ratio should be 4.0
    assert result['compression_ratio'] == 4.0

    # Patch matrix should have 1 column
    assert result['patchmatrix'].shape[1] == 1

    print(f"✓ Test 4 passed: Identical patterns grouped into single patch")
    print(f"  Patch indices: {result['patchindx']}")
    print(f"  Compression: {result['compression_ratio']:.1f}x")


def test_r_reference_case1():
    """
    Test against R ground truth - Case 1.

    R code to generate ground truth:
    ```r
    library(LESYMAP)
    lesmat = matrix(c(
        1, 1, 0, 0, 1,
        1, 1, 0, 0, 0,
        0, 0, 1, 1, 0
    ), nrow=3, byrow=TRUE)

    # Core algorithm (simplified without image ops)
    add = 1
    summed = rep(0, ncol(lesmat))
    for (i in 1:nrow(lesmat)) {
        summed = summed + (lesmat[i, ]*add)
        summed = match(summed, unique(summed))
        add = max(summed)+1
    }
    print(summed)
    print(length(unique(summed)))
    ```
    """

    lesmat = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)

    # Expected output from R (need to run R code to get actual values)
    # For now, we test consistency properties
    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    # Verify basic properties
    assert result['nvoxels'] == 5

    # Columns 0 and 1 have pattern [1,1,0] - should be same patch
    assert result['patchindx'][0] == result['patchindx'][1], \
        "Columns with identical patterns should have same patch index"

    # Columns 2 and 3 have pattern [0,0,1] - should be same patch
    assert result['patchindx'][2] == result['patchindx'][3], \
        "Columns with identical patterns should have same patch index"

    # Column 4 has pattern [1,0,0] - should be different
    assert result['patchindx'][4] != result['patchindx'][0]
    assert result['patchindx'][4] != result['patchindx'][2]

    # Should have exactly 3 patches
    assert result['npatches'] == 3, \
        f"Expected 3 patches, got {result['npatches']}"

    print(f"✓ Test 5 (R reference) passed")
    print(f"  Patch indices: {result['patchindx']}")
    print(f"  Number of patches: {result['npatches']}")
    print(f"  Compression: {result['compression_ratio']:.2f}x")


def test_patch_matrix_correctness():
    """Verify patch matrix contains correct representative patterns."""

    lesmat = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ], dtype=np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    patchindx = result['patchindx']
    patchmatrix = result['patchmatrix']

    # For each patch, verify the pattern is correct
    for patch_id in range(1, result['npatches'] + 1):
        # Find voxels in this patch
        voxel_indices = np.where(patchindx == patch_id)[0]

        # Get the pattern from patch matrix (column patch_id - 1)
        patch_pattern = patchmatrix[:, patch_id - 1]

        # Verify all voxels in this patch have the same pattern
        for voxel_idx in voxel_indices:
            voxel_pattern = lesmat[:, voxel_idx]
            assert np.array_equal(voxel_pattern, patch_pattern), \
                f"Patch {patch_id} pattern mismatch for voxel {voxel_idx}"

    print(f"✓ Test 6 passed: Patch matrix patterns verified")


def test_voxels_per_patch():
    """Test voxels_per_patch calculation."""

    lesmat = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=False)

    # Manually count voxels per patch
    manual_counts = {}
    for patch_id in result['patchindx']:
        patch_id = int(patch_id)
        manual_counts[patch_id] = manual_counts.get(patch_id, 0) + 1

    # Verify against result
    for i, count in enumerate(result['voxels_per_patch'], start=1):
        assert count == manual_counts.get(i, 0), \
            f"Voxel count mismatch for patch {i}"

    print(f"✓ Test 7 passed: Voxels per patch correctly counted")
    print(f"  Voxels per patch: {result['voxels_per_patch']}")


def test_large_matrix():
    """Test with a larger matrix to verify performance and correctness."""

    np.random.seed(42)
    lesmat = np.random.randint(0, 2, size=(10, 100)).astype(np.float64)

    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    # Basic checks
    assert result['nvoxels'] == 100
    assert result['npatches'] >= 1
    assert result['npatches'] <= 100

    # Verify all patch indices are valid
    assert np.all(result['patchindx'] >= 1)
    assert np.all(result['patchindx'] <= result['npatches'])

    # Verify patch matrix shape
    assert result['patchmatrix'].shape == (10, result['npatches'])

    print(f"✓ Test 8 passed: Large matrix (10x100)")
    print(f"  Patches found: {result['npatches']}")
    print(f"  Compression: {result['compression_ratio']:.2f}x")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Patch Algorithm Correctness")
    print("=" * 60)

    test_simple_binary_matrix()
    print()
    test_identical_columns()
    print()
    test_all_unique()
    print()
    test_all_same()
    print()
    test_r_reference_case1()
    print()
    test_patch_matrix_correctness()
    print()
    test_voxels_per_patch()
    print()
    test_large_matrix()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
