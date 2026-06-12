"""
Test Python patch computation against R reference data.

Validates that get_unique_lesion_patches() produces identical results
to R's getUniqueLesionPatches() function.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add the lesymap-python package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lesymap-python'))

from lesymap.core.patch import get_unique_lesion_patches


class TestPatchRComparison:
    """Test patch computation against R reference."""

    @pytest.fixture
    def r_reference_data(self):
        """Load R reference data."""
        data_dir = os.path.join(
            os.path.dirname(__file__),
            'fixtures', 'r_reference_results'
        )

        # Load lesion matrix
        lesmat_df = pd.read_csv(os.path.join(data_dir, 'patch_lesmat.csv'))
        lesmat = lesmat_df.values

        # Load patch index
        patchindx_df = pd.read_csv(os.path.join(data_dir, 'patch_patchindx.csv'))
        patchindx = patchindx_df.values.flatten()

        # Load patch matrix
        patchmatrix_df = pd.read_csv(os.path.join(data_dir, 'patch_patchmatrix.csv'))
        patchmatrix = patchmatrix_df.values

        # Load stats
        stats_df = pd.read_csv(os.path.join(data_dir, 'patch_stats.csv'))
        npatches = int(stats_df['npatches'].iloc[0])
        nvoxels = int(stats_df['nvoxels'].iloc[0])

        return {
            'lesmat': lesmat,
            'patchindx': patchindx,
            'patchmatrix': patchmatrix,
            'npatches': npatches,
            'nvoxels': nvoxels,
        }

    def test_number_of_patches_matches(self, r_reference_data):
        """Test that number of unique patches matches R."""
        lesmat = r_reference_data['lesmat']
        r_npatches = r_reference_data['npatches']

        # Run Python implementation
        result = get_unique_lesion_patches(lesmat)
        py_npatches = result['npatches']

        print(f"\nR npatches: {r_npatches}")
        print(f"Python npatches: {py_npatches}")

        assert py_npatches == r_npatches, (
            f"Number of patches mismatch: Python={py_npatches}, R={r_npatches}"
        )

    def test_patch_index_mapping(self, r_reference_data):
        """Test that patch index assignments are consistent with R."""
        lesmat = r_reference_data['lesmat']
        r_patchindx = r_reference_data['patchindx']

        # Run Python implementation
        result = get_unique_lesion_patches(lesmat)
        py_patchindx = result['patchindx']

        print(f"\nR patchindx: {r_patchindx[:10]}...")
        print(f"Python patchindx: {py_patchindx[:10]}...")

        # The actual patch numbers may differ, but the grouping should be the same
        # Two voxels should be in the same patch iff they have the same R index
        # Check this by verifying: for all pairs (i,j),
        # (py[i]==py[j]) iff (r[i]==r[j])

        n_voxels = len(r_patchindx)

        # Sample check for efficiency (full check would be O(n^2))
        np.random.seed(42)
        sample_size = min(100, n_voxels * (n_voxels - 1) // 2)

        mismatches = 0
        for _ in range(sample_size):
            i, j = np.random.choice(n_voxels, 2, replace=False)
            r_same = r_patchindx[i] == r_patchindx[j]
            py_same = py_patchindx[i] == py_patchindx[j]
            if r_same != py_same:
                mismatches += 1

        print(f"Pairwise grouping mismatches: {mismatches}/{sample_size}")
        assert mismatches == 0, f"Patch grouping differs: {mismatches} mismatches"

    def test_patch_index_exact_match(self, r_reference_data):
        """Test that patch indices match exactly (same numbering)."""
        lesmat = r_reference_data['lesmat']
        r_patchindx = r_reference_data['patchindx']

        # Run Python implementation
        result = get_unique_lesion_patches(lesmat)
        py_patchindx = result['patchindx']

        # Check exact match
        exact_match = np.array_equal(py_patchindx, r_patchindx)

        if not exact_match:
            diff_count = np.sum(py_patchindx != r_patchindx)
            print(f"\nExact index mismatch at {diff_count} positions")
            print(f"R patchindx: {r_patchindx}")
            print(f"Python patchindx: {py_patchindx}")

        assert exact_match, "Patch indices do not match exactly"

    def test_patch_matrix_content(self, r_reference_data):
        """Test that patch matrix content matches R."""
        lesmat = r_reference_data['lesmat']
        r_patchmatrix = r_reference_data['patchmatrix']

        # Run Python implementation
        result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)
        py_patchmatrix = result['patchmatrix']

        print(f"\nR patchmatrix shape: {r_patchmatrix.shape}")
        print(f"Python patchmatrix shape: {py_patchmatrix.shape}")

        # Shapes should match
        assert py_patchmatrix.shape == r_patchmatrix.shape, (
            f"Patch matrix shape mismatch: Python={py_patchmatrix.shape}, "
            f"R={r_patchmatrix.shape}"
        )

        # Content should match
        content_match = np.allclose(py_patchmatrix, r_patchmatrix)

        if not content_match:
            diff = np.abs(py_patchmatrix - r_patchmatrix)
            max_diff = np.max(diff)
            diff_count = np.sum(diff > 0)
            print(f"Max difference: {max_diff}")
            print(f"Differing elements: {diff_count}")

        assert content_match, "Patch matrix content does not match"

    def test_nvoxels_matches(self, r_reference_data):
        """Test that number of voxels matches."""
        lesmat = r_reference_data['lesmat']
        r_nvoxels = r_reference_data['nvoxels']

        result = get_unique_lesion_patches(lesmat)
        py_nvoxels = result['nvoxels']

        assert py_nvoxels == r_nvoxels, (
            f"Number of voxels mismatch: Python={py_nvoxels}, R={r_nvoxels}"
        )

    def test_compression_ratio(self, r_reference_data):
        """Test that compression ratio is computed correctly."""
        lesmat = r_reference_data['lesmat']
        r_npatches = r_reference_data['npatches']
        r_nvoxels = r_reference_data['nvoxels']
        expected_ratio = r_nvoxels / r_npatches

        result = get_unique_lesion_patches(lesmat)
        py_ratio = result['compression_ratio']

        print(f"\nExpected compression ratio: {expected_ratio:.4f}")
        print(f"Python compression ratio: {py_ratio:.4f}")

        assert np.isclose(py_ratio, expected_ratio), (
            f"Compression ratio mismatch: Python={py_ratio}, expected={expected_ratio}"
        )


def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("PATCH COMPUTATION: Python vs R Reference Comparison")
    print("=" * 60)

    # Load reference data
    data_dir = os.path.join(
        os.path.dirname(__file__),
        'fixtures', 'r_reference_results'
    )

    # Check if reference data exists
    if not os.path.exists(os.path.join(data_dir, 'patch_lesmat.csv')):
        print("ERROR: R reference data not found!")
        print(f"Expected at: {data_dir}")
        print("Run generate_r_patch_reference.R first.")
        return False

    # Load data
    lesmat_df = pd.read_csv(os.path.join(data_dir, 'patch_lesmat.csv'))
    lesmat = lesmat_df.values

    patchindx_df = pd.read_csv(os.path.join(data_dir, 'patch_patchindx.csv'))
    r_patchindx = patchindx_df.values.flatten()

    patchmatrix_df = pd.read_csv(os.path.join(data_dir, 'patch_patchmatrix.csv'))
    r_patchmatrix = patchmatrix_df.values

    stats_df = pd.read_csv(os.path.join(data_dir, 'patch_stats.csv'))
    r_npatches = int(stats_df['npatches'].iloc[0])
    r_nvoxels = int(stats_df['nvoxels'].iloc[0])

    print(f"\nR Reference Data:")
    print(f"  Lesion matrix: {lesmat.shape}")
    print(f"  Number of patches: {r_npatches}")
    print(f"  Number of voxels: {r_nvoxels}")
    print(f"  Patch matrix: {r_patchmatrix.shape}")

    # Run Python implementation
    print("\nRunning Python implementation...")
    result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

    print(f"\nPython Results:")
    print(f"  Number of patches: {result['npatches']}")
    print(f"  Number of voxels: {result['nvoxels']}")
    print(f"  Patch matrix: {result['patchmatrix'].shape}")
    print(f"  Compression ratio: {result['compression_ratio']:.4f}")

    # Run comparisons
    all_passed = True
    tests = []

    # Test 1: Number of patches
    test1_pass = result['npatches'] == r_npatches
    tests.append(("Number of patches matches", test1_pass))

    # Test 2: Number of voxels
    test2_pass = result['nvoxels'] == r_nvoxels
    tests.append(("Number of voxels matches", test2_pass))

    # Test 3: Patch index exact match
    test3_pass = np.array_equal(result['patchindx'], r_patchindx)
    tests.append(("Patch indices match exactly", test3_pass))

    # Test 4: Patch matrix shape
    test4_pass = result['patchmatrix'].shape == r_patchmatrix.shape
    tests.append(("Patch matrix shape matches", test4_pass))

    # Test 5: Patch matrix content
    test5_pass = np.allclose(result['patchmatrix'], r_patchmatrix)
    tests.append(("Patch matrix content matches", test5_pass))

    # Test 6: Compression ratio
    expected_ratio = r_nvoxels / r_npatches
    test6_pass = np.isclose(result['compression_ratio'], expected_ratio)
    tests.append(("Compression ratio correct", test6_pass))

    # Report results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test_name, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
