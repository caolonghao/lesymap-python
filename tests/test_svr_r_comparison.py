"""
Test SVR implementation against R reference data.

This test validates the Python SVR implementation by comparing results
with the R LESYMAP package's lsm_svr function.

Key differences between R and Python implementations:
- R uses e1071::svm with defaults: kernel='radial', cost=30, gamma=5, epsilon=0.1
- Python uses sklearn.SVR with defaults: kernel='linear', C=1.0, epsilon=0.1
- Both use libsvm backend, so with matched parameters they should produce similar results

R lsm_svr workflow:
1. Scale and center both lesmat and behavior
2. Fit SVM with e1071::svm
3. Get weights: w = t(svr$coefs) %*% svr$SV
4. Scale weights: statistic = w * (10/max(abs(w)))
5. Run permutations for p-values

Python lsm_svr workflow:
1. Fit sklearn.SVR (no scaling by default)
2. For linear kernel: weights = svr.coef_
3. For non-linear: use permutation importance
"""

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.svm import SVR

from lesymap.methods.multivariate import lsm_svr

# Path to R reference data
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "r_reference_results"


def load_r_reference_data():
    """Load R reference data from CSV files."""
    data = {}

    # Load input data
    data['lesmat'] = pd.read_csv(FIXTURES_DIR / "svr_lesmat.csv").values
    data['lesmat_scaled'] = pd.read_csv(FIXTURES_DIR / "svr_lesmat_scaled.csv").values
    data['behavior'] = pd.read_csv(FIXTURES_DIR / "svr_behavior.csv")['behavior'].values
    data['behavior_scaled'] = pd.read_csv(FIXTURES_DIR / "svr_behavior_scaled.csv")['behavior_scaled'].values

    # Load linear kernel results
    linear_pred = pd.read_csv(FIXTURES_DIR / "svr_linear_predictions.csv")
    data['linear_predictions'] = linear_pred['predictions'].values
    data['linear_correlation'] = linear_pred['correlation'].values[0]
    data['linear_weights'] = pd.read_csv(FIXTURES_DIR / "svr_linear_weights.csv")['weights'].values

    # Load radial kernel results
    radial_pred = pd.read_csv(FIXTURES_DIR / "svr_radial_predictions.csv")
    data['radial_predictions'] = radial_pred['predictions'].values
    data['radial_correlation'] = radial_pred['correlation'].values[0]
    data['radial_weights'] = pd.read_csv(FIXTURES_DIR / "svr_radial_weights.csv")['weights'].values

    lsm_linear = FIXTURES_DIR / "svr_lsm_linear_results.csv"
    lsm_radial = FIXTURES_DIR / "svr_lsm_radial_results.csv"
    if lsm_linear.exists():
        data['lsm_linear'] = pd.read_csv(lsm_linear)
    if lsm_radial.exists():
        data['lsm_radial'] = pd.read_csv(lsm_radial)

    return data


def _vector_mask(n_voxels):
    return nib.Nifti1Image(np.ones((n_voxels, 1, 1), dtype=np.uint8), np.eye(4))


def _r_beta_scaled(weights):
    max_abs = np.max(np.abs(weights))
    return weights * (10.0 / max_abs)


def load_tiny_lsm_svr_reference_data():
    """Load small true R lsm_svr fixtures."""
    return {
        'lesmat': pd.read_csv(FIXTURES_DIR / "svr_tiny_lesmat.csv").values,
        'behavior': pd.read_csv(FIXTURES_DIR / "svr_tiny_behavior.csv")['behavior'].values,
        'linear': pd.read_csv(FIXTURES_DIR / "svr_tiny_lsm_linear_results.csv"),
        'radial': pd.read_csv(FIXTURES_DIR / "svr_tiny_lsm_radial_results.csv"),
    }


@pytest.fixture(scope="module")
def r_data():
    """Fixture to load R reference data once per module."""
    if not FIXTURES_DIR.exists():
        pytest.skip("R reference data not found. Run generate_r_svr_reference.R first.")

    try:
        return load_r_reference_data()
    except Exception as e:
        pytest.skip(f"Failed to load R reference data: {e}")


@pytest.fixture(scope="module")
def tiny_lsm_svr_data():
    required = [
        "svr_tiny_lesmat.csv",
        "svr_tiny_behavior.csv",
        "svr_tiny_lsm_linear_results.csv",
        "svr_tiny_lsm_radial_results.csv",
    ]
    missing = [f for f in required if not (FIXTURES_DIR / f).exists()]
    if missing:
        pytest.skip(
            "Missing tiny true R lsm_svr fixtures: "
            + ", ".join(missing)
            + ". Run tests/generate_r_svr_tiny_reference.R first."
        )
    return load_tiny_lsm_svr_reference_data()


@pytest.fixture(scope="module")
def sklearn_svr_models(r_data):
    """Fit direct sklearn SVR models once for the module."""
    linear = SVR(kernel='linear', C=1.0, epsilon=0.1)
    linear.fit(r_data['lesmat_scaled'], r_data['behavior_scaled'])

    rbf = SVR(kernel='rbf', C=30, gamma=5, epsilon=0.1)
    rbf.fit(r_data['lesmat_scaled'], r_data['behavior_scaled'])

    return {'linear': linear, 'rbf': rbf}


@pytest.fixture(scope="module")
def python_lsm_svr_results(r_data):
    """Run Python lsm_svr r_compatible models once for the module."""
    lesmat = r_data['lesmat']
    behavior = r_data['behavior']
    mask = _vector_mask(lesmat.shape[1])

    linear = lsm_svr(
        lesmat,
        behavior,
        mask,
        r_compatible=True,
        kernel='linear',
        C=1.0,
        epsilon=0.1,
        show_info=False,
    )
    rbf = lsm_svr(
        lesmat,
        behavior,
        mask,
        r_compatible=True,
        show_info=False,
    )
    return {'linear': linear, 'rbf': rbf}


@pytest.mark.slow
class TestSVRLinearKernel:
    """Test SVR with linear kernel against R reference."""

    def test_predictions_correlation(self, r_data, sklearn_svr_models):
        """Test that Python SVR predictions correlate highly with R predictions."""
        # Use pre-scaled data (as R does)
        lesmat_scaled = r_data['lesmat_scaled']
        behavior_scaled = r_data['behavior_scaled']

        # Get predictions
        py_predictions = sklearn_svr_models['linear'].predict(lesmat_scaled)
        r_predictions = r_data['linear_predictions']

        # Compare predictions
        corr, p_value = pearsonr(py_predictions, r_predictions)

        print(f"\n=== Linear Kernel SVR Comparison ===")
        print(f"Python predictions range: [{py_predictions.min():.6f}, {py_predictions.max():.6f}]")
        print(f"R predictions range: [{r_predictions.min():.6f}, {r_predictions.max():.6f}]")
        print(f"Prediction correlation: {corr:.6f} (p={p_value:.2e})")

        # Both use libsvm, so predictions should be nearly identical
        assert corr > 0.95, f"Prediction correlation {corr:.4f} < 0.95"
        print("PASS: Prediction correlation > 0.95")

    def test_weights_correlation(self, r_data, sklearn_svr_models):
        """Test that Python SVR weights correlate highly with R weights."""
        lesmat_scaled = r_data['lesmat_scaled']
        # Get Python weights (for linear kernel)
        py_weights = sklearn_svr_models['linear'].coef_.flatten()

        # R computes weights as: w = t(svr$coefs) %*% svr$SV
        # sklearn linear kernel gives coef_ directly
        r_weights = r_data['linear_weights']

        # Compare weights
        corr, p_value = pearsonr(py_weights, r_weights)

        print(f"\n=== Linear Kernel Weights Comparison ===")
        print(f"Python weights range: [{py_weights.min():.6f}, {py_weights.max():.6f}]")
        print(f"R weights range: [{r_weights.min():.6f}, {r_weights.max():.6f}]")
        print(f"Weight correlation: {corr:.6f} (p={p_value:.2e})")

        # Weights should be very similar
        assert corr > 0.95, f"Weight correlation {corr:.4f} < 0.95"
        print("PASS: Weight correlation > 0.95")

    def test_behavior_correlation(self, r_data, sklearn_svr_models):
        """Test that Python SVR achieves similar correlation with behavior."""
        lesmat_scaled = r_data['lesmat_scaled']
        behavior_scaled = r_data['behavior_scaled']

        # Get correlations
        py_predictions = sklearn_svr_models['linear'].predict(lesmat_scaled)
        py_corr, _ = pearsonr(py_predictions, behavior_scaled)
        r_corr = r_data['linear_correlation']

        print(f"\n=== Behavior Correlation Comparison ===")
        print(f"Python correlation with behavior: {py_corr:.6f}")
        print(f"R correlation with behavior: {r_corr:.6f}")
        print(f"Difference: {abs(py_corr - r_corr):.6f}")

        # Correlations should be very close
        assert abs(py_corr - r_corr) < 0.01, f"Correlation difference {abs(py_corr - r_corr):.4f} >= 0.01"
        print("PASS: Correlation difference < 0.01")


@pytest.mark.slow
class TestSVRRadialKernel:
    """Test SVR with radial (RBF) kernel against R reference."""

    def test_predictions_correlation(self, r_data, sklearn_svr_models):
        """Test that Python SVR RBF predictions correlate with R predictions."""
        lesmat_scaled = r_data['lesmat_scaled']
        behavior_scaled = r_data['behavior_scaled']

        # Get predictions
        py_predictions = sklearn_svr_models['rbf'].predict(lesmat_scaled)
        r_predictions = r_data['radial_predictions']

        # Compare predictions
        corr, p_value = pearsonr(py_predictions, r_predictions)

        print(f"\n=== Radial Kernel SVR Comparison ===")
        print(f"Python predictions range: [{py_predictions.min():.6f}, {py_predictions.max():.6f}]")
        print(f"R predictions range: [{r_predictions.min():.6f}, {r_predictions.max():.6f}]")
        print(f"Prediction correlation: {corr:.6f} (p={p_value:.2e})")

        # RBF kernel should also produce very similar results
        assert corr > 0.95, f"Prediction correlation {corr:.4f} < 0.95"
        print("PASS: Prediction correlation > 0.95")

    def test_behavior_correlation(self, r_data, sklearn_svr_models):
        """Test that Python RBF SVR achieves similar correlation with behavior."""
        lesmat_scaled = r_data['lesmat_scaled']
        behavior_scaled = r_data['behavior_scaled']

        # Get correlations
        py_predictions = sklearn_svr_models['rbf'].predict(lesmat_scaled)
        py_corr, _ = pearsonr(py_predictions, behavior_scaled)
        r_corr = r_data['radial_correlation']

        print(f"\n=== RBF Behavior Correlation Comparison ===")
        print(f"Python correlation with behavior: {py_corr:.6f}")
        print(f"R correlation with behavior: {r_corr:.6f}")
        print(f"Difference: {abs(py_corr - r_corr):.6f}")

        # Correlations should be very close
        assert abs(py_corr - r_corr) < 0.01, f"Correlation difference {abs(py_corr - r_corr):.4f} >= 0.01"
        print("PASS: Correlation difference < 0.01")


@pytest.mark.slow
class TestPythonLSMSVREndToEnd:
    """Test Python lsm_svr r_compatible mode against R-style reference outputs."""

    def test_lsm_svr_linear_matches_r_reference(self, r_data, python_lsm_svr_results):
        result = python_lsm_svr_results['linear']

        py_weights = result.raw_weights_img.get_fdata().reshape(-1)
        py_statistic = result.stat_img.get_fdata().reshape(-1)
        r_weights = r_data['linear_weights']
        r_statistic = _r_beta_scaled(r_weights)

        assert result.model_params['r_compatible'] is True
        assert result.model_params['kernel'] == 'linear'
        assert abs(result.model_params['correlation'] - r_data['linear_correlation']) < 1e-10
        np.testing.assert_allclose(py_weights, r_weights, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(py_statistic, r_statistic, rtol=1e-8, atol=1e-8)

    def test_lsm_svr_rbf_matches_r_reference(self, r_data, python_lsm_svr_results):
        result = python_lsm_svr_results['rbf']

        py_weights = result.raw_weights_img.get_fdata().reshape(-1)
        py_statistic = result.stat_img.get_fdata().reshape(-1)
        r_weights = r_data['radial_weights']
        r_statistic = _r_beta_scaled(r_weights)

        weight_corr, _ = pearsonr(py_weights, r_weights)
        stat_corr, _ = pearsonr(py_statistic, r_statistic)

        assert result.model_params['r_compatible'] is True
        assert result.model_params['kernel'] == 'rbf'
        assert result.model_params['C'] == 30.0
        assert result.model_params['gamma'] == 5.0
        assert abs(result.model_params['correlation'] - r_data['radial_correlation']) < 1e-5
        assert weight_corr > 0.999999
        assert stat_corr > 0.999999
        np.testing.assert_allclose(py_statistic, r_statistic, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
class TestTrueRLSMSVRReference:
    """Compare Python lsm_svr with true R LESYMAP lsm_svr outputs when present."""

    def test_lsm_svr_linear_matches_true_r_lsm_svr(self, r_data, python_lsm_svr_results):
        if 'lsm_linear' not in r_data:
            pytest.skip("Missing svr_lsm_linear_results.csv; regenerate R SVR reference fixtures")

        result = python_lsm_svr_results['linear']

        r_statistic = r_data['lsm_linear']['statistic'].values
        r_pvalue = r_data['lsm_linear']['pvalue'].values
        py_statistic = result.stat_img.get_fdata().reshape(-1)

        np.testing.assert_allclose(py_statistic, r_statistic, rtol=1e-8, atol=1e-8)
        assert np.all((r_pvalue >= 0) & (r_pvalue <= 1))

    def test_lsm_svr_rbf_matches_true_r_lsm_svr(self, r_data, python_lsm_svr_results):
        if 'lsm_radial' not in r_data:
            pytest.skip("Missing svr_lsm_radial_results.csv; regenerate R SVR reference fixtures")

        result = python_lsm_svr_results['rbf']

        r_statistic = r_data['lsm_radial']['statistic'].values
        r_pvalue = r_data['lsm_radial']['pvalue'].values
        py_statistic = result.stat_img.get_fdata().reshape(-1)
        stat_corr, _ = pearsonr(py_statistic, r_statistic)

        assert stat_corr > 0.999999
        np.testing.assert_allclose(py_statistic, r_statistic, rtol=1e-3, atol=1e-3)
        assert np.all((r_pvalue >= 0) & (r_pvalue <= 1))


class TestTinyTrueRLSMSVRReference:
    """Fast true R lsm_svr end-to-end comparison on a tiny fixture."""

    def test_lsm_svr_linear_matches_tiny_true_r_lsm_svr(self, tiny_lsm_svr_data):
        lesmat = tiny_lsm_svr_data['lesmat']
        behavior = tiny_lsm_svr_data['behavior']

        result = lsm_svr(
            lesmat,
            behavior,
            _vector_mask(lesmat.shape[1]),
            r_compatible=True,
            kernel='linear',
            C=1.0,
            epsilon=0.1,
            show_info=False,
        )

        r_statistic = tiny_lsm_svr_data['linear']['statistic'].values
        r_pvalue = tiny_lsm_svr_data['linear']['pvalue'].values
        py_statistic = result.stat_img.get_fdata().reshape(-1)

        np.testing.assert_allclose(py_statistic, r_statistic, rtol=1e-8, atol=1e-8)
        assert np.all((r_pvalue >= 0) & (r_pvalue <= 1))

    def test_lsm_svr_rbf_matches_tiny_true_r_lsm_svr(self, tiny_lsm_svr_data):
        lesmat = tiny_lsm_svr_data['lesmat']
        behavior = tiny_lsm_svr_data['behavior']

        result = lsm_svr(
            lesmat,
            behavior,
            _vector_mask(lesmat.shape[1]),
            r_compatible=True,
            show_info=False,
        )

        r_statistic = tiny_lsm_svr_data['radial']['statistic'].values
        r_pvalue = tiny_lsm_svr_data['radial']['pvalue'].values
        py_statistic = result.stat_img.get_fdata().reshape(-1)

        np.testing.assert_allclose(py_statistic, r_statistic, rtol=1e-8, atol=1e-8)
        assert np.all((r_pvalue >= 0) & (r_pvalue <= 1))


@pytest.mark.slow
class TestSVRDataLoading:
    """Test that R reference data was loaded correctly."""

    def test_data_dimensions(self, r_data):
        """Verify data dimensions match expected values."""
        n_subjects = 50
        n_voxels = 378254  # From R output

        print(f"\n=== Data Dimensions ===")
        print(f"lesmat shape: {r_data['lesmat'].shape}")
        print(f"lesmat_scaled shape: {r_data['lesmat_scaled'].shape}")
        print(f"behavior shape: {r_data['behavior'].shape}")
        print(f"Expected: ({n_subjects}, {n_voxels})")

        assert r_data['lesmat'].shape == (n_subjects, n_voxels), \
            f"lesmat shape mismatch: {r_data['lesmat'].shape}"
        assert r_data['lesmat_scaled'].shape == (n_subjects, n_voxels), \
            f"lesmat_scaled shape mismatch: {r_data['lesmat_scaled'].shape}"
        assert r_data['behavior'].shape == (n_subjects,), \
            f"behavior shape mismatch: {r_data['behavior'].shape}"

        print("PASS: All dimensions correct")

    def test_scaling_verification(self, r_data):
        """Verify that scaled data has mean~0 and std~1."""
        behavior_scaled = r_data['behavior_scaled']

        print(f"\n=== Scaling Verification ===")
        print(f"Behavior scaled mean: {behavior_scaled.mean():.6f}")
        print(f"Behavior scaled std: {behavior_scaled.std():.6f}")

        assert abs(behavior_scaled.mean()) < 0.01, "Behavior scaled mean not ~0"
        assert abs(behavior_scaled.std() - 1.0) < 0.1, "Behavior scaled std not ~1"

        print("PASS: Scaling verified")


@pytest.mark.slow
def test_summary(r_data):
    """Print summary of all tests."""
    print("\n" + "=" * 60)
    print("SVR R Comparison Test Summary")
    print("=" * 60)
    print(f"Data: {r_data['lesmat'].shape[0]} subjects x {r_data['lesmat'].shape[1]} voxels")
    print(f"R Linear correlation: {r_data['linear_correlation']:.6f}")
    print(f"R Radial correlation: {r_data['radial_correlation']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
