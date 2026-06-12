"""
Tests for the three performance optimizations:
1. patches_to_voxels vectorization
2. lsm_chisq vectorization
3. regression XtX single computation
"""

import numpy as np
import pytest
import nibabel as nib
from scipy import stats

from lesymap.core.patch import patches_to_voxels, reconstruct_from_patches
from lesymap.stats_compiled.regression import _solve_ols_with_xtx


# ─────────────────────────────────────────────────────────────
# Optimization 1: patches_to_voxels vectorization
# ─────────────────────────────────────────────────────────────

class TestPatchesToVoxelsVectorized:
    """patches_to_voxels should use fancy indexing, not a Python loop."""

    def test_basic_mapping(self):
        """Patch stats correctly map back to voxels."""
        patchindx = np.array([1, 1, 2, 3, 3, 2])
        statistic = np.array([10.0, 20.0, 30.0])

        result = patches_to_voxels(statistic, patchindx)

        expected = np.array([10.0, 10.0, 20.0, 30.0, 30.0, 20.0])
        np.testing.assert_array_equal(result, expected)

    def test_fill_value_respected(self):
        """Voxels with patchindx == 0 get fill_value."""
        patchindx = np.array([0, 1, 2, 0])
        statistic = np.array([1.0, 2.0])

        result = patches_to_voxels(statistic, patchindx, fill_value=-999.0)

        assert result[0] == -999.0
        assert result[3] == -999.0
        assert result[1] == 1.0
        assert result[2] == 2.0

    def test_single_patch(self):
        """Single patch: all voxels get same value."""
        patchindx = np.array([1, 1, 1, 1])
        statistic = np.array([42.0])

        result = patches_to_voxels(statistic, patchindx)

        np.testing.assert_array_equal(result, [42.0, 42.0, 42.0, 42.0])

    def test_large_array_correctness(self):
        """Vectorized result matches reference loop on large input."""
        np.random.seed(0)
        n_patches = 500
        n_voxels = 10000
        # each voxel belongs to a random patch [1..n_patches]
        patchindx = np.random.randint(1, n_patches + 1, size=n_voxels)
        statistic = np.random.randn(n_patches)

        result = patches_to_voxels(statistic, patchindx)

        # Reference: original Python-loop behaviour
        expected = np.zeros(n_voxels, dtype=statistic.dtype)
        for pid in range(1, n_patches + 1):
            expected[patchindx == pid] = statistic[pid - 1]

        np.testing.assert_array_almost_equal(result, expected)

    def test_reconstruct_from_patches_basic(self):
        """reconstruct_from_patches maps patch values to voxels correctly."""
        patchindx = np.array([1, 2, 1, 3, 3])
        patch_values = np.array([100.0, 200.0, 300.0])

        result = reconstruct_from_patches(patch_values, patchindx, n_voxels=5)

        expected = np.array([100.0, 200.0, 100.0, 300.0, 300.0])
        np.testing.assert_array_equal(result, expected)


# ─────────────────────────────────────────────────────────────
# Optimization 2: lsm_chisq vectorized chi2 computation
# ─────────────────────────────────────────────────────────────

def _chi2_vectorized(lesmat: np.ndarray, behavior: np.ndarray):
    """
    Vectorized chi-square statistic for all voxels at once.

    lesmat : (n_subjects, n_voxels) binary
    behavior: (n_subjects,) binary (0/1)

    Returns (statistic, pvals) both shape (n_voxels,).
    Uses the exact formula: chi2 = N*(ad - bc)^2 / (R1*R0*C1*C0)
    with correction=False, matching scipy's chi2_contingency.
    """
    n = lesmat.shape[0]
    b_int = behavior.astype(np.int64)

    n11 = lesmat.T @ b_int           # les=1, beh=1
    n10 = lesmat.T @ (1 - b_int)     # les=1, beh=0
    n01 = (1 - lesmat).T @ b_int     # les=0, beh=1
    n00 = (1 - lesmat).T @ (1 - b_int)  # les=0, beh=0

    R1 = n11 + n10  # row: les=1
    R0 = n01 + n00  # row: les=0
    C1 = n11 + n01  # col: beh=1
    C0 = n10 + n00  # col: beh=0

    denom = R1.astype(np.float64) * R0 * C1 * C0
    chi2 = np.where(
        denom > 0,
        n * (n11.astype(np.float64) * n00 - n10.astype(np.float64) * n01) ** 2 / denom,
        0.0
    )
    pvals = stats.chi2.sf(chi2, df=1)
    return chi2, pvals


class TestChisqVectorized:
    """Vectorized chi2 must match scipy per-voxel results."""

    def _scipy_reference(self, lesmat, behavior):
        """Per-voxel scipy reference implementation."""
        n_voxels = lesmat.shape[1]
        stat_ref = np.zeros(n_voxels)
        pval_ref = np.zeros(n_voxels)
        for v in range(n_voxels):
            table = np.zeros((2, 2), dtype=int)
            for les, beh in zip(lesmat[:, v], behavior):
                table[int(les > 0), int(beh > 0)] += 1
            try:
                chi2, p = stats.chi2_contingency(table, correction=False)[:2]
                stat_ref[v] = chi2
                pval_ref[v] = p
            except ValueError:
                stat_ref[v] = 0.0
                pval_ref[v] = 1.0
        return stat_ref, pval_ref

    def test_matches_scipy_small(self):
        """Vectorized chi2 matches scipy on small dataset."""
        np.random.seed(7)
        lesmat = (np.random.rand(30, 20) < 0.4).astype(np.float64)
        behavior = (np.random.rand(30) < 0.5).astype(np.float64)

        chi2_vec, pval_vec = _chi2_vectorized(lesmat, behavior)
        chi2_ref, pval_ref = self._scipy_reference(lesmat, behavior)

        np.testing.assert_allclose(chi2_vec, chi2_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(pval_vec, pval_ref, rtol=1e-10, atol=1e-10)

    def test_matches_scipy_larger(self):
        """Vectorized chi2 matches scipy on larger dataset."""
        np.random.seed(99)
        lesmat = (np.random.rand(80, 200) < 0.3).astype(np.float64)
        behavior = (np.random.rand(80) < 0.6).astype(np.float64)

        chi2_vec, pval_vec = _chi2_vectorized(lesmat, behavior)
        chi2_ref, pval_ref = self._scipy_reference(lesmat, behavior)

        np.testing.assert_allclose(chi2_vec, chi2_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(pval_vec, pval_ref, rtol=1e-10, atol=1e-10)

    def test_edge_case_all_lesioned(self):
        """When all subjects are lesioned at a voxel, chi2 = 0."""
        lesmat = np.ones((20, 5))
        behavior = (np.random.rand(20) < 0.5).astype(np.float64)

        chi2_vec, pval_vec = _chi2_vectorized(lesmat, behavior)

        np.testing.assert_array_equal(chi2_vec, 0.0)
        np.testing.assert_array_equal(pval_vec, 1.0)

    def test_edge_case_no_lesioned(self):
        """When no subject is lesioned at a voxel, chi2 = 0."""
        lesmat = np.zeros((20, 5))
        behavior = (np.random.rand(20) < 0.5).astype(np.float64)

        chi2_vec, pval_vec = _chi2_vectorized(lesmat, behavior)

        np.testing.assert_array_equal(chi2_vec, 0.0)


# ─────────────────────────────────────────────────────────────
# Optimization 3: regression XtX computed only once
# ─────────────────────────────────────────────────────────────

class TestRegressionSingleXtX:
    """_solve_ols_with_xtx must return same results as original _solve_ols."""

    def test_same_result_no_covariates(self):
        """Single XtX computation gives same t-stat as current regression_fast."""
        from lesymap.stats_compiled.regression import regression_fast

        np.random.seed(42)
        n_subjects = 50
        n_voxels = 100
        X = np.random.randn(n_subjects, n_voxels)
        y = np.random.randn(n_subjects)

        statistic_orig, n, k = regression_fast(X, y)
        statistic_new = _solve_ols_with_xtx(X, y)

        np.testing.assert_allclose(statistic_new, statistic_orig, rtol=1e-10)

    def test_same_result_known_signal(self):
        """t-stat is consistent for data with a known voxel signal."""
        np.random.seed(5)
        n = 60
        n_vox = 10
        X = (np.random.rand(n, n_vox) < 0.3).astype(np.float64)
        # Plant signal in voxel 0
        y = X[:, 0] * 3.0 + np.random.randn(n) * 0.5

        from lesymap.stats_compiled.regression import regression_fast
        statistic_orig, _, _ = regression_fast(X, y)
        statistic_new = _solve_ols_with_xtx(X, y)

        np.testing.assert_allclose(statistic_new, statistic_orig, rtol=1e-8)
        # Voxel 0 should have the largest t-stat
        assert np.argmax(np.abs(statistic_new)) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
