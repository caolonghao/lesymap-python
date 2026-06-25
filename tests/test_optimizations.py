"""
Tests for the three performance optimizations:
1. patches_to_voxels vectorization
2. lsm_chisq vectorization
3. regression XtX single computation
"""

import time
import numpy as np
import pytest
import nibabel as nib
from scipy import stats

from lesymap.core.patch import (
    analysis_patches_to_voxels,
    patches_to_voxels,
    reconstruct_from_patches,
)
from lesymap.methods.univariate import lsm_chisq
from lesymap.stats_compiled.ttest import ttest_fast, welch_fast
from lesymap.stats_compiled.regression import regression_fast, _solve_ols_with_xtx


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

    def test_analysis_patches_to_voxels_expands_filtered_patches(self):
        """Filtered analysis patch stats expand before voxel mapping."""
        statistic = np.array([10.0, 30.0])
        patchinfo = {
            "patchindx": np.array([1, 1, 2, 3, 3]),
            "analysis_keep_mask": np.array([True, False, True]),
        }

        result = analysis_patches_to_voxels(statistic, patchinfo, fill_value=-1.0)

        expected = np.array([10.0, 10.0, -1.0, 30.0, 30.0])
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

    def test_lsm_chisq_expands_filtered_patch_direction(self):
        """Filtered patch direction vector expands before z-map creation."""
        lesmat = np.array(
            [
                [1, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 1],
                [0, 0],
            ],
            dtype=float,
        )
        behavior = np.array([1, 1, 0, 0, 1, 0], dtype=float)
        mask = nib.Nifti1Image(np.ones((3, 1, 1), dtype=float), np.eye(4))
        patchinfo = {
            "patchindx": np.array([1, 2, 3]),
            "analysis_keep_mask": np.array([True, False, True]),
        }

        result = lsm_chisq(
            lesmat,
            behavior,
            mask,
            patchinfo=patchinfo,
            multiple_comparison="none",
            show_info=False,
        )

        assert result.zmap_img.get_fdata().reshape(-1).shape == (3,)
        assert result.zmap_img.get_fdata().reshape(-1)[1] == 0


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

    def test_no_covariates_is_stable_with_large_behavior_offset(self):
        rng = np.random.default_rng(77)
        X = (rng.random((40, 8)) < 0.4).astype(np.float64)
        y = 1e8 + rng.normal(scale=0.3, size=40)

        stat, _, _ = regression_fast(X, y)
        expected = np.array([
            stats.linregress(X[:, i], y).slope / stats.linregress(X[:, i], y).stderr
            if np.std(X[:, i]) > 0 else 0.0
            for i in range(X.shape[1])
        ])

        np.testing.assert_allclose(stat, expected, rtol=1e-6, atol=1e-6)

    def test_one_dimensional_covariate_vector_is_supported(self):
        rng = np.random.default_rng(88)
        X = (rng.random((32, 6)) < 0.35).astype(np.float64)
        y = rng.normal(size=32)
        covariate = rng.normal(size=32)

        stat_1d, n_1d, k_1d = regression_fast(X, y, covariate)
        stat_2d, n_2d, k_2d = regression_fast(X, y, covariate.reshape(-1, 1))

        assert n_1d == n_2d == 32
        assert k_1d == k_2d == 3
        np.testing.assert_allclose(stat_1d, stat_2d, rtol=1e-10, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Optimization 8: t-test group moments avoid per-voxel slicing
# ─────────────────────────────────────────────────────────────

def _ttest_reference_loop(X, y, welch=False):
    """Reference implementation that mirrors the original per-voxel formulas."""
    n_subjects, n_voxels = X.shape
    statistic = np.zeros(n_voxels)
    df = np.zeros(n_voxels)

    if not welch:
        df.fill(n_subjects - 2.0)

    for vox in range(n_voxels):
        lesioned = X[:, vox] != 0
        y1 = y[lesioned]
        y0 = y[~lesioned]
        n0 = len(y0)
        n1 = len(y1)

        if n0 == 0 or n1 == 0:
            statistic[vox] = 0.0
            df[vox] = 1.0 if welch else n_subjects - 2.0
            continue

        mean0 = np.mean(y0)
        mean1 = np.mean(y1)
        var0 = np.var(y0, ddof=1) if n0 > 1 else 0.0
        var1 = np.var(y1, ddof=1) if n1 > 1 else 0.0

        if welch:
            se = np.sqrt(var0 / n0 + var1 / n1)
            if se > 0:
                statistic[vox] = (mean0 - mean1) / se
            if var0 > 0 and var1 > 0 and n0 > 1 and n1 > 1:
                num = (var0 / n0 + var1 / n1) ** 2
                den = (var0 / n0) ** 2 / (n0 - 1.0) + (var1 / n1) ** 2 / (n1 - 1.0)
                df[vox] = num / den if den > 0 else n0 + n1 - 2
            else:
                df[vox] = n0 + n1 - 2
        else:
            pooled = ((var0 * (n0 - 1.0)) + (var1 * (n1 - 1.0))) / (n_subjects - 2.0)
            se = np.sqrt(pooled * (1.0 / n0 + 1.0 / n1))
            if se > 0:
                statistic[vox] = (mean0 - mean1) / se

    return statistic, df


class TestTTestMomentVectorization:
    """t-test kernels match loop formulas and run fast on matrix-shaped inputs."""

    def test_ttest_and_welch_match_reference_loop(self):
        rng = np.random.default_rng(1234)
        X = (rng.random((48, 240)) < 0.32).astype(np.float64)
        y = rng.normal(size=48).astype(np.float64)
        X[:, 0] = 0.0
        X[:, 1] = 1.0

        stat_t, df_t = ttest_fast(X, y, compute_dof=True)
        ref_t, ref_df_t = _ttest_reference_loop(X, y, welch=False)
        stat_w, df_w = welch_fast(X, y, compute_dof=True)
        ref_w, ref_df_w = _ttest_reference_loop(X, y, welch=True)

        np.testing.assert_allclose(stat_t, ref_t, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(df_t, ref_df_t, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(stat_w, ref_w, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(df_w, ref_df_w, rtol=1e-12, atol=1e-12)

    def test_ttest_and_welch_are_stable_with_large_behavior_offset(self):
        rng = np.random.default_rng(4321)
        X = (rng.random((48, 40)) < 0.32).astype(np.float64)
        y = 1e8 + rng.normal(scale=0.5, size=48)

        stat_t, df_t = ttest_fast(X, y, compute_dof=True)
        ref_t, ref_df_t = _ttest_reference_loop(X, y, welch=False)
        stat_w, df_w = welch_fast(X, y, compute_dof=True)
        ref_w, ref_df_w = _ttest_reference_loop(X, y, welch=True)

        np.testing.assert_allclose(stat_t, ref_t, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(df_t, ref_df_t, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(stat_w, ref_w, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(df_w, ref_df_w, rtol=1e-6, atol=1e-6)

    def test_ttest_and_welch_large_matrix_runtime(self):
        rng = np.random.default_rng(5678)
        X = (rng.random((80, 12000)) < 0.35).astype(np.float64)
        y = rng.normal(size=80).astype(np.float64)

        t0 = time.perf_counter()
        stat_t, _ = ttest_fast(X, y, compute_dof=False)
        ttest_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        stat_w, _ = welch_fast(X, y, compute_dof=False)
        welch_elapsed = time.perf_counter() - t0

        assert np.isfinite(stat_t).all()
        assert np.isfinite(stat_w).all()
        assert ttest_elapsed < 2.0, f"ttest_fast took {ttest_elapsed:.3f}s on 80x12000"
        assert welch_elapsed < 2.0, f"welch_fast took {welch_elapsed:.3f}s on 80x12000"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
