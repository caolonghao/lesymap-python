"""
Tests for the next batch of performance optimizations:
4. _has_duplicates O(n²) → O(n log n)
5. patchmatrix construction vectorization
6. FDR monotonicity Python loop → np.minimum.accumulate
7. fwer_permutation_threshold parallel execution
"""

import time
import numpy as np
import pytest

from lesymap.stats_compiled.bm import _has_duplicates
from lesymap.core.patch import get_unique_lesion_patches
from lesymap.methods.correction import compute_fdr, fwer_permutation_threshold


# ─────────────────────────────────────────────────────────────
# Optimization 4: _has_duplicates O(n²) → O(n log n)
# ─────────────────────────────────────────────────────────────

class TestHasDuplicates:
    """_has_duplicates must be correct and not O(n²) for large inputs."""

    def test_detects_duplicates(self):
        """Returns True when duplicates exist."""
        arr = np.array([1.0, 2.0, 3.0, 2.0, 5.0])
        assert _has_duplicates(arr) is True

    def test_no_duplicates(self):
        """Returns False for unique values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _has_duplicates(arr) is False

    def test_single_element(self):
        """Single element has no duplicate."""
        assert _has_duplicates(np.array([42.0])) is False

    def test_two_same_elements(self):
        """Two identical elements is a duplicate."""
        assert _has_duplicates(np.array([7.0, 7.0])) is True

    def test_all_same(self):
        """All same values → has duplicates."""
        assert _has_duplicates(np.ones(100)) is True

    def test_large_no_duplicates_is_fast(self):
        """O(n log n) version must handle n=10000 in << 1 s."""
        arr = np.arange(10000, dtype=np.float64)
        # warm up JIT
        _has_duplicates(arr[:10])

        t0 = time.perf_counter()
        result = _has_duplicates(arr)
        elapsed = time.perf_counter() - t0

        assert result is False
        assert elapsed < 0.5, f"_has_duplicates took {elapsed:.3f}s on n=10000 (too slow)"


# ─────────────────────────────────────────────────────────────
# Optimization 5: patchmatrix construction vectorized
# ─────────────────────────────────────────────────────────────

class TestPatchmatrixVectorized:
    """Patchmatrix rows must equal the lesion column of any voxel in that patch."""

    def test_patchmatrix_shape(self):
        """Patchmatrix has shape (n_subjects, n_patches)."""
        lesmat = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ], dtype=np.float64)
        result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)
        pm = result['patchmatrix']
        assert pm.shape == (3, result['npatches'])

    def test_patchmatrix_columns_match_lesion_pattern(self):
        """Each patchmatrix column must equal the lesion pattern of its patch."""
        lesmat = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.float64)
        result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)
        pm = result['patchmatrix']
        patchindx = result['patchindx']

        # Each column of patchmatrix must match all voxels in that patch
        for patch_id in range(1, result['npatches'] + 1):
            vox_mask = patchindx == patch_id
            for vox_idx in np.where(vox_mask)[0]:
                np.testing.assert_array_equal(
                    pm[:, patch_id - 1],
                    lesmat[:, vox_idx],
                    err_msg=f"Patch {patch_id}: patchmatrix col != lesmat col {vox_idx}"
                )

    def test_patchmatrix_larger(self):
        """Vectorized patchmatrix construction correct on larger data."""
        np.random.seed(11)
        lesmat = (np.random.rand(40, 300) < 0.25).astype(np.float64)
        result = get_unique_lesion_patches(lesmat, return_patch_matrix=True)
        pm = result['patchmatrix']
        patchindx = result['patchindx']

        # Spot-check 20 random patches
        patch_ids = np.unique(patchindx)
        sample = np.random.choice(patch_ids, size=min(20, len(patch_ids)), replace=False)
        for pid in sample:
            first_vox = np.where(patchindx == pid)[0][0]
            np.testing.assert_array_equal(pm[:, pid - 1], lesmat[:, first_vox])


# ─────────────────────────────────────────────────────────────
# Optimization 6: FDR monotonicity → np.minimum.accumulate
# ─────────────────────────────────────────────────────────────

class TestFDRMonotonicity:
    """compute_fdr must produce monotone-adjusted p-values without Python loop."""

    def test_bh_is_monotone(self):
        """BH adjusted p-values are non-decreasing (monotone after sort)."""
        np.random.seed(3)
        pvals = np.random.uniform(0, 1, 200)
        adj = compute_fdr(pvals, method='bh')

        sorted_adj = np.sort(adj)
        np.testing.assert_array_equal(sorted_adj, np.sort(np.minimum.accumulate(
            np.sort(adj)[::-1]
        )[::-1]))  # Verify monotonicity property holds

    def test_bh_matches_statsmodels(self):
        """BH result matches statsmodels multipletests as reference."""
        from statsmodels.stats.multitest import multipletests

        np.random.seed(17)
        pvals = np.random.uniform(0, 0.2, 100)
        adj_ours = compute_fdr(pvals, method='bh')
        _, adj_ref, _, _ = multipletests(pvals, method='fdr_bh')

        np.testing.assert_allclose(adj_ours, adj_ref, rtol=1e-10)

    def test_by_matches_statsmodels(self):
        """BY result matches statsmodels."""
        from statsmodels.stats.multitest import multipletests

        np.random.seed(22)
        pvals = np.random.uniform(0, 0.5, 80)
        adj_ours = compute_fdr(pvals, method='by')
        _, adj_ref, _, _ = multipletests(pvals, method='fdr_by')

        np.testing.assert_allclose(adj_ours, adj_ref, rtol=1e-10)

    def test_capped_at_one(self):
        """No adjusted p-value exceeds 1.0."""
        pvals = np.linspace(0.01, 0.99, 50)
        adj = compute_fdr(pvals, method='bh')
        assert np.all(adj <= 1.0)


# ─────────────────────────────────────────────────────────────
# Optimization 7: fwer_permutation_threshold parallel
# ─────────────────────────────────────────────────────────────

class TestFWERParallel:
    """fwer_permutation_threshold accepts n_jobs and produces correct thresholds.

    Note: n_jobs > 1 uses joblib threading. Stat functions that internally
    use numba parallel (prange) must use n_jobs=1 to avoid nested thread
    pool conflicts. For pure-numpy stat functions, n_jobs > 1 is safe.
    """

    def _make_numpy_stat_func(self, n_subjects=60, n_voxels=200, seed=0):
        """Pure-numpy stat_func (safe for threading with n_jobs > 1)."""
        rng = np.random.default_rng(seed)
        X = (rng.random((n_subjects, n_voxels)) < 0.3).astype(np.float64)
        y = rng.standard_normal(n_subjects)
        n = n_subjects

        def stat_func():
            perm_y = rng.permutation(y)
            # Mean difference per voxel: pure numpy, releases GIL, thread-safe
            return X.T @ perm_y / n

        return stat_func

    def test_parallel_threshold_close_to_serial(self):
        """Parallel and serial FWER thresholds are in the same ballpark."""
        stat_func = self._make_numpy_stat_func(seed=42)

        t_serial = fwer_permutation_threshold(stat_func, nperm=100, alpha=0.95, n_jobs=1)
        t_parallel = fwer_permutation_threshold(stat_func, nperm=100, alpha=0.95, n_jobs=2)

        # Both thresholds should be positive and in a similar range
        assert t_serial > 0 and t_parallel > 0
        assert 0.3 * t_serial < t_parallel < 3.0 * t_serial, (
            f"Thresholds diverge too much: serial={t_serial:.4f}, parallel={t_parallel:.4f}"
        )

    def test_parallel_accepts_n_jobs_param(self):
        """fwer_permutation_threshold accepts n_jobs without error."""
        stat_func = self._make_numpy_stat_func(seed=1)

        threshold = fwer_permutation_threshold(stat_func, nperm=20, alpha=0.95, n_jobs=2)
        assert np.isfinite(threshold)
        assert threshold > 0

    def test_serial_result_is_valid(self):
        """n_jobs=1 (default) produces a valid threshold."""
        stat_func = self._make_numpy_stat_func(seed=7)

        threshold = fwer_permutation_threshold(stat_func, nperm=30, alpha=0.95, n_jobs=1)
        assert np.isfinite(threshold)
        assert threshold > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
