"""
Test Python T-test implementation against R LESYMAP reference results.

This test compares:
- ttest_fast() (Student's t-test, equal variance) against R's TTfast with varEqual=TRUE
- welch_fast() (Welch's t-test, unequal variance) against R's TTfast with varEqual=FALSE
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import os
import sys

# Add the lesymap-python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lesymap-python'))

from lesymap.stats_compiled.ttest import ttest_fast, welch_fast


# Path to R reference results
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'r_reference_results')


def load_r_reference_data():
    """Load R reference input data and results."""
    # Load input data
    lesmat = pd.read_csv(os.path.join(REFERENCE_DIR, 'ttest_lesmat.csv')).values
    behavior = pd.read_csv(os.path.join(REFERENCE_DIR, 'ttest_behavior.csv'))['behavior'].values

    # Load T-test results (Student's t-test)
    ttest_results = pd.read_csv(os.path.join(REFERENCE_DIR, 'ttest_results.csv'))

    # Load Welch results
    welch_results = pd.read_csv(os.path.join(REFERENCE_DIR, 'welch_results.csv'))

    return {
        'lesmat': lesmat,
        'behavior': behavior,
        'ttest': ttest_results,
        'welch': welch_results
    }


class TestTtestRComparison:
    """Test Python T-test implementations against R reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load reference data before each test."""
        self.data = load_r_reference_data()
        self.tolerance = 1e-6

    def test_student_ttest_statistics(self):
        """Test Student's t-test statistics match R TTfast with varEqual=TRUE."""
        lesmat = self.data['lesmat'].astype(np.float64)
        behavior = self.data['behavior'].astype(np.float64)

        # Run Python implementation
        py_stat, py_df = ttest_fast(lesmat, behavior, compute_dof=True)

        # Get R reference
        r_stat = self.data['ttest']['statistic'].values
        r_df = self.data['ttest']['df'].values

        # Compare statistics
        stat_diff = np.abs(py_stat - r_stat)
        max_stat_diff = np.max(stat_diff)

        print(f"\n=== Student's T-test Statistics ===")
        print(f"Python statistic range: [{py_stat.min():.6f}, {py_stat.max():.6f}]")
        print(f"R statistic range: [{r_stat.min():.6f}, {r_stat.max():.6f}]")
        print(f"Max absolute difference: {max_stat_diff:.2e}")
        print(f"Mean absolute difference: {np.mean(stat_diff):.2e}")

        assert max_stat_diff < self.tolerance, \
            f"Student's t-test statistics differ by {max_stat_diff:.2e} (tolerance: {self.tolerance})"

        print(f"PASS: Statistics match within tolerance {self.tolerance}")

    def test_student_ttest_dof(self):
        """Test Student's t-test degrees of freedom match R."""
        lesmat = self.data['lesmat'].astype(np.float64)
        behavior = self.data['behavior'].astype(np.float64)

        # Run Python implementation
        py_stat, py_df = ttest_fast(lesmat, behavior, compute_dof=True)

        # Get R reference
        r_df = self.data['ttest']['df'].values

        # Compare DOF
        df_diff = np.abs(py_df - r_df)
        max_df_diff = np.max(df_diff)

        print(f"\n=== Student's T-test Degrees of Freedom ===")
        print(f"Python DOF: {py_df[0]:.1f} (constant for all voxels)")
        print(f"R DOF: {r_df[0]:.1f}")
        print(f"Max absolute difference: {max_df_diff:.2e}")

        assert max_df_diff < self.tolerance, \
            f"Student's t-test DOF differs by {max_df_diff:.2e} (tolerance: {self.tolerance})"

        print(f"PASS: DOF matches within tolerance {self.tolerance}")

    def test_student_ttest_pvalues(self):
        """Test Student's t-test p-values match R."""
        lesmat = self.data['lesmat'].astype(np.float64)
        behavior = self.data['behavior'].astype(np.float64)

        # Run Python implementation
        py_stat, py_df = ttest_fast(lesmat, behavior, compute_dof=True)

        # Compute p-values (two-sided)
        py_pvals = 2 * stats.t.sf(np.abs(py_stat), py_df)

        # Get R reference
        r_pvals = self.data['ttest']['pvalue_twosided'].values

        # Compare p-values
        pval_diff = np.abs(py_pvals - r_pvals)
        max_pval_diff = np.max(pval_diff)

        print(f"\n=== Student's T-test P-values ===")
        print(f"Python p-value range: [{py_pvals.min():.6f}, {py_pvals.max():.6f}]")
        print(f"R p-value range: [{r_pvals.min():.6f}, {r_pvals.max():.6f}]")
        print(f"Max absolute difference: {max_pval_diff:.2e}")
        print(f"Mean absolute difference: {np.mean(pval_diff):.2e}")

        assert max_pval_diff < self.tolerance, \
            f"Student's t-test p-values differ by {max_pval_diff:.2e} (tolerance: {self.tolerance})"

        print(f"PASS: P-values match within tolerance {self.tolerance}")

    def test_welch_ttest_statistics(self):
        """Test Welch's t-test statistics match R TTfast with varEqual=FALSE."""
        lesmat = self.data['lesmat'].astype(np.float64)
        behavior = self.data['behavior'].astype(np.float64)

        # Run Python implementation
        py_stat, py_df = welch_fast(lesmat, behavior, compute_dof=True)

        # Get R reference
        r_stat = self.data['welch']['statistic'].values

        # Compare statistics
        stat_diff = np.abs(py_stat - r_stat)
        max_stat_diff = np.max(stat_diff)

        print(f"\n=== Welch's T-test Statistics ===")
        print(f"Python statistic range: [{py_stat.min():.6f}, {py_stat.max():.6f}]")
        print(f"R statistic range: [{r_stat.min():.6f}, {r_stat.max():.6f}]")
        print(f"Max absolute difference: {max_stat_diff:.2e}")
        print(f"Mean absolute difference: {np.mean(stat_diff):.2e}")

        assert max_stat_diff < self.tolerance, \
            f"Welch's t-test statistics differ by {max_stat_diff:.2e} (tolerance: {self.tolerance})"

        print(f"PASS: Statistics match within tolerance {self.tolerance}")

    def test_welch_ttest_dof(self):
        """Test Welch's t-test degrees of freedom match R."""
        lesmat = self.data['lesmat'].astype(np.float64)
        behavior = self.data['behavior'].astype(np.float64)

        # Run Python implementation
        py_stat, py_df = welch_fast(lesmat, behavior, compute_dof=True)

        # Get R reference
        r_df = self.data['welch']['df'].values

        # Compare DOF
        df_diff = np.abs(py_df - r_df)
        max_df_diff = np.max(df_diff)

        print(f"\n=== Welch's T-test Degrees of Freedom ===")
        print(f"Python DOF range: [{py_df.min():.4f}, {py_df.max():.4f}]")
        print(f"R DOF range: [{r_df.min():.4f}, {r_df.max():.4f}]")
        print(f"Max absolute difference: {max_df_diff:.2e}")
        print(f"Mean absolute difference: {np.mean(df_diff):.2e}")

        assert max_df_diff < self.tolerance, \
            f"Welch's t-test DOF differs by {max_df_diff:.2e} (tolerance: {self.tolerance})"

        print(f"PASS: DOF matches within tolerance {self.tolerance}")

    def test_welch_ttest_pvalues(self):
        """Test Welch's t-test p-values match R."""
        lesmat = self.data['lesmat'].astype(np.float64)
        behavior = self.data['behavior'].astype(np.float64)

        # Run Python implementation
        py_stat, py_df = welch_fast(lesmat, behavior, compute_dof=True)

        # Compute p-values (two-sided)
        py_pvals = 2 * stats.t.sf(np.abs(py_stat), py_df)

        # Get R reference
        r_pvals = self.data['welch']['pvalue_twosided'].values

        # Compare p-values
        pval_diff = np.abs(py_pvals - r_pvals)
        max_pval_diff = np.max(pval_diff)

        print(f"\n=== Welch's T-test P-values ===")
        print(f"Python p-value range: [{py_pvals.min():.6f}, {py_pvals.max():.6f}]")
        print(f"R p-value range: [{r_pvals.min():.6f}, {r_pvals.max():.6f}]")
        print(f"Max absolute difference: {max_pval_diff:.2e}")
        print(f"Mean absolute difference: {np.mean(pval_diff):.2e}")

        assert max_pval_diff < self.tolerance, \
            f"Welch's t-test p-values differ by {max_pval_diff:.2e} (tolerance: {self.tolerance})"

        print(f"PASS: P-values match within tolerance {self.tolerance}")


def test_summary():
    """Run all tests and print summary."""
    data = load_r_reference_data()
    lesmat = data['lesmat'].astype(np.float64)
    behavior = data['behavior'].astype(np.float64)
    tolerance = 1e-6

    print("\n" + "=" * 60)
    print("T-TEST VALIDATION: Python vs R LESYMAP")
    print("=" * 60)
    print(f"Input shape: {lesmat.shape[0]} subjects x {lesmat.shape[1]} voxels")
    print(f"Tolerance: {tolerance}")
    print("=" * 60)

    all_passed = True

    # Student's t-test
    print("\n--- Student's T-test (equal variance) ---")
    py_stat, py_df = ttest_fast(lesmat, behavior, compute_dof=True)
    r_stat = data['ttest']['statistic'].values
    r_df = data['ttest']['df'].values
    r_pvals = data['ttest']['pvalue_twosided'].values
    py_pvals = 2 * stats.t.sf(np.abs(py_stat), py_df)

    stat_diff = np.max(np.abs(py_stat - r_stat))
    df_diff = np.max(np.abs(py_df - r_df))
    pval_diff = np.max(np.abs(py_pvals - r_pvals))

    stat_pass = stat_diff < tolerance
    df_pass = df_diff < tolerance
    pval_pass = pval_diff < tolerance

    print(f"  Statistics: max diff = {stat_diff:.2e} {'PASS' if stat_pass else 'FAIL'}")
    print(f"  DOF:        max diff = {df_diff:.2e} {'PASS' if df_pass else 'FAIL'}")
    print(f"  P-values:   max diff = {pval_diff:.2e} {'PASS' if pval_pass else 'FAIL'}")

    all_passed = all_passed and stat_pass and df_pass and pval_pass

    # Welch's t-test
    print("\n--- Welch's T-test (unequal variance) ---")
    py_stat, py_df = welch_fast(lesmat, behavior, compute_dof=True)
    r_stat = data['welch']['statistic'].values
    r_df = data['welch']['df'].values
    r_pvals = data['welch']['pvalue_twosided'].values
    py_pvals = 2 * stats.t.sf(np.abs(py_stat), py_df)

    stat_diff = np.max(np.abs(py_stat - r_stat))
    df_diff = np.max(np.abs(py_df - r_df))
    pval_diff = np.max(np.abs(py_pvals - r_pvals))

    stat_pass = stat_diff < tolerance
    df_pass = df_diff < tolerance
    pval_pass = pval_diff < tolerance

    print(f"  Statistics: max diff = {stat_diff:.2e} {'PASS' if stat_pass else 'FAIL'}")
    print(f"  DOF:        max diff = {df_diff:.2e} {'PASS' if df_pass else 'FAIL'}")
    print(f"  P-values:   max diff = {pval_diff:.2e} {'PASS' if pval_pass else 'FAIL'}")

    all_passed = all_passed and stat_pass and df_pass and pval_pass

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL RESULT: ALL TESTS PASSED")
    else:
        print("OVERALL RESULT: SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = test_summary()
    exit(0 if success else 1)
