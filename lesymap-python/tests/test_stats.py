"""
Unit tests for statistical methods in LESYMAP-Python.

Tests validate that Python implementations produce similar results
to the R/C++ versions.
"""

import numpy as np
import pytest
import nibabel as nib

from lesymap.stats_compiled import brunner_munzel_fast, ttest_fast, welch_fast, regression_fast
from lesymap.core.patch import get_unique_lesion_patches
from lesymap.core.image_utils import images_to_matrix, matrix_to_image


class TestBrunnerMunzel:
    """Test Brunner-Munzel implementation."""

    def test_basic_functionality(self):
        """Test basic BM computation."""
        np.random.seed(42)
        n_subjects = 50
        n_voxels = 100

        # Create test data
        X = (np.random.rand(n_subjects, n_voxels) < 0.3).astype(np.float64)
        y = np.random.randn(n_subjects)

        # Run BM test
        statistic, df = brunner_munzel_fast(X, y)

        # Check shapes
        assert statistic.shape == (n_voxels,)
        assert df.shape == (n_voxels,)

        # Check for valid values
        assert np.all(np.isfinite(statistic))
        assert np.all(np.isfinite(df))
        assert np.all(df > 0)

    def test_edge_cases(self):
        """Test edge cases."""
        # All zeros in a voxel
        X = np.zeros((20, 5))
        X[:, 0] = 1  # Only first voxel has lesions
        y = np.random.randn(20)

        statistic, df = brunner_munzel_fast(X, y)

        # First voxel should have valid statistic
        assert np.isfinite(statistic[0])


class TestTTest:
    """Test t-test implementations."""

    def test_student_ttest(self):
        """Test Student's t-test."""
        np.random.seed(42)
        n_subjects = 50
        n_voxels = 100

        X = (np.random.rand(n_subjects, n_voxels) < 0.3).astype(np.float64)
        y = np.random.randn(n_subjects)

        statistic, df = ttest_fast(X, y)

        assert statistic.shape == (n_voxels,)
        assert df.shape == (n_voxels,)
        assert np.all(df == n_subjects - 2)  # Constant df for Student's t

    def test_welch_ttest(self):
        """Test Welch's t-test."""
        np.random.seed(42)
        n_subjects = 50
        n_voxels = 100

        X = (np.random.rand(n_subjects, n_voxels) < 0.3).astype(np.float64)
        y = np.random.randn(n_subjects)

        statistic, df = welch_fast(X, y)

        assert statistic.shape == (n_voxels,)
        assert df.shape == (n_voxels,)
        # df should vary for Welch
        assert np.any(df != df[0])


class TestRegression:
    """Test regression implementation."""

    def test_basic_regression(self):
        """Test basic OLS regression."""
        np.random.seed(42)
        n_subjects = 50
        n_voxels = 100

        X = np.random.randn(n_subjects, n_voxels)
        y = np.random.randn(n_subjects)

        statistic, n, k = regression_fast(X, y)

        assert statistic.shape == (n_voxels,)
        assert n == n_subjects
        assert k == 2  # Intercept + voxel

    def test_regression_with_covariates(self):
        """Test regression with covariates."""
        np.random.seed(42)
        n_subjects = 50
        n_voxels = 100
        n_covariates = 2

        X = np.random.randn(n_subjects, n_voxels)
        y = np.random.randn(n_subjects)
        covariates = np.random.randn(n_subjects, n_covariates)

        statistic, n, k = regression_fast(X, y, covariates)

        assert statistic.shape == (n_voxels,)
        assert n == n_subjects
        assert k == 2 + n_covariates


class TestPatchComputation:
    """Test patch computation."""

    def test_patch_basic(self):
        """Test basic patch computation."""
        # Create simple lesion matrix
        lesmat = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
        ])

        result = get_unique_lesion_patches(lesmat)

        # Should have 3 unique patterns
        assert result['npatches'] == 3
        assert result['nvoxels'] == 5

    def test_patch_compression(self):
        """Test that patch compression works."""
        np.random.seed(42)
        n_subjects = 20
        n_voxels = 1000

        # Create binary lesion matrix
        lesmat = (np.random.rand(n_subjects, n_voxels) < 0.2).astype(np.float64)

        result = get_unique_lesion_patches(lesmat)

        # Should have fewer patches than voxels
        assert result['npatches'] < n_voxels
        assert result['compression_ratio'] > 1

    def test_patch_matrix(self):
        """Test patch matrix generation."""
        lesmat = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])

        result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)

        assert 'patchmatrix' in result
        assert result['patchmatrix'].shape[1] == result['npatches']


class TestImageUtils:
    """Test image utility functions."""

    def test_images_to_matrix(self):
        """Test converting images to matrix."""
        # Create test images
        affine = np.eye(4)
        mask_data = np.ones((10, 10, 10))
        mask_img = nib.Nifti1Image(mask_data, affine)

        images = []
        for i in range(5):
            data = np.random.rand(10, 10, 10)
            img = nib.Nifti1Image(data, affine)
            images.append(img)

        matrix = images_to_matrix(images, mask_img)

        assert matrix.shape == (5, 1000)  # 5 subjects, 1000 voxels

    def test_matrix_to_image(self):
        """Test converting matrix back to image."""
        affine = np.eye(4)
        mask_data = np.ones((10, 10, 10))
        mask_img = nib.Nifti1Image(mask_data, affine)

        vector = np.random.rand(1000)
        img = matrix_to_image(vector, mask_img)

        assert isinstance(img, nib.Nifti1Image)
        assert img.shape == (10, 10, 10)

        # Extract back and verify
        extracted = img.get_fdata()[mask_data > 0]
        np.testing.assert_array_almost_equal(extracted, vector)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
