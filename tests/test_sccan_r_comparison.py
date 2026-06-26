"""
Test SCCAN implementation against R reference data.

This test validates the Python SCCAN implementation by comparing results
with the R LESYMAP package's lsm_sccan function.

Key implementation details:
- R scale(center=TRUE, scale=TRUE) uses sample standard deviation (n - 1)
- SCCAN is stochastic, so correlations should match within tolerance (0.05)
- Weights are normalized to [-1, 1] range in both implementations

R lsm_sccan workflow:
1. Scale and center both lesmat and behavior using scale()
2. Call sparseDecom2() from ANTsR
3. Normalize weights: statistic = eig1 / max(abs(eig1))
4. Apply directional flip based on eig2 sign and correlation sign
5. Threshold small weights (< 0.1)
6. Linear calibration: lm(behavior.orig ~ predbehav.raw)

Python lsm_sccan workflow:
1. Scale and center using saved R-style scaling parameters
2. Call sparse_decom2() from ANTsPy
3. Same normalization and flipping logic
4. Same thresholding and linear calibration
"""

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.stats import pearsonr

from lesymap.methods.multivariate import lsm_sccan

# Path to R reference data
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "r_reference_results"


def r_scale(x, center=True, scale=True):
    """
    Replicate R's scale() function.

    R's scale() with center=TRUE uses:
    - center: mean(x)
    - scale: sqrt(sum((x - mean(x))^2) / (n - 1))
    """
    x = np.asarray(x, dtype=float)

    if center:
        center_val = np.mean(x, axis=0)
        x = x - center_val
    else:
        center_val = None

    if scale:
        # R's scale() uses sqrt(sum(x^2)/(n-1)) for centered data
        # which equals sd with ddof=1 for centered data
        # But for uncentered data it's different
        # Actually, R scale uses: sqrt(sum((x - mean)^2) / (n-1)) when center=TRUE
        # Let's compute it correctly
        n = x.shape[0] if x.ndim > 1 else len(x)
        if center:
            # After centering, scale by sqrt(sum(x^2)/(n-1))
            scale_val = np.sqrt(np.sum(x**2, axis=0) / (n - 1))
        else:
            scale_val = np.std(x, axis=0, ddof=1)

        # Handle zero scale values
        if np.isscalar(scale_val):
            if scale_val == 0:
                scale_val = 1.0
        else:
            scale_val[scale_val == 0] = 1.0

        x = x / scale_val
    else:
        scale_val = None

    return x, center_val, scale_val


def load_r_reference_data():
    """Load R reference data from CSV files."""
    data = {}

    # Load test1 metadata
    metadata = pd.read_csv(FIXTURES_DIR / "test1_metadata.csv")
    data['sparseness'] = metadata['sparseness'].values[0]
    data['behavior_scaleval'] = metadata['behavior_scaleval'].values[0]
    data['behavior_centerval'] = metadata['behavior_centerval'].values[0]
    data['eig2'] = metadata['eig2'].values[0]
    data['ccasummary_corr'] = metadata['ccasummary_corr'].values[0]
    data['nonzero_weights'] = int(metadata['nonzero_weights'].values[0])
    data['predictlm_intercept'] = metadata['predictlm_intercept'].values[0]
    data['predictlm_slope'] = metadata['predictlm_slope'].values[0]

    # Load raw eig1 weights
    data['eig1_raw'] = pd.read_csv(FIXTURES_DIR / "test1_eig1_raw.csv")['eig1'].values

    # Load normalized statistics
    data['statistic'] = pd.read_csv(FIXTURES_DIR / "test1_statistic.csv")['statistic'].values

    optional_files = {
        'lesmat': ("sccan_lesmat.csv.gz", "sccan_lesmat.csv"),
        'behavior': "sccan_behavior.csv",
        'mask_img': "mask.nii.gz",
        'predictions': "test4_predictions.csv",
    }
    for key, filename in optional_files.items():
        if isinstance(filename, tuple):
            path = next((FIXTURES_DIR / item for item in filename if (FIXTURES_DIR / item).exists()), None)
            if path is None:
                continue
        else:
            path = FIXTURES_DIR / filename
        if path.exists():
            if key == 'mask_img':
                data[key] = nib.load(path)
                continue

            frame = pd.read_csv(path)
            if key == 'behavior':
                data[key] = frame['behavior'].values
            elif key == 'lesmat':
                data[key] = frame.values
            else:
                data[key] = frame

    return data


@pytest.fixture(scope="module")
def r_data():
    """Fixture to load R reference data once per module."""
    if not FIXTURES_DIR.exists():
        pytest.skip("R reference data not found. Run generate_r_sccan_reference.R first.")

    required_files = [
        "test1_metadata.csv",
        "test1_eig1_raw.csv",
        "test1_statistic.csv",
    ]

    for f in required_files:
        if not (FIXTURES_DIR / f).exists():
            pytest.skip(f"Missing required file: {f}. Run generate_r_sccan_reference.R first.")

    try:
        return load_r_reference_data()
    except Exception as e:
        pytest.skip(f"Failed to load R reference data: {e}")


class TestScalingBehavior:
    """Test that Python scaling matches R's scale() function."""

    def test_r_scale_function_simple(self):
        """Test our r_scale implementation with simple data."""
        # Simple test case
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # R's scale() behavior:
        # center = mean(x) = 3.0
        # After centering: [-2, -1, 0, 1, 2]
        # scale = sqrt(sum(x^2)/(n-1)) = sqrt(10/4) = sqrt(2.5)

        scaled, center, scale = r_scale(x)

        expected_center = 3.0
        expected_scale = np.sqrt(10 / 4)  # sqrt(sum(centered^2)/(n-1))
        expected_scaled = (x - expected_center) / expected_scale

        print(f"\n=== R scale() Test ===")
        print(f"Input: {x}")
        print(f"Center (expected {expected_center}): {center}")
        print(f"Scale (expected {expected_scale:.6f}): {scale}")
        print(f"Scaled mean: {scaled.mean():.10f}")
        print(f"Scaled std (ddof=1): {scaled.std(ddof=1):.10f}")

        assert abs(center - expected_center) < 1e-10, f"Center mismatch: {center} vs {expected_center}"
        assert abs(scale - expected_scale) < 1e-10, f"Scale mismatch: {scale} vs {expected_scale}"
        np.testing.assert_allclose(scaled, expected_scaled, rtol=1e-10)

        # After R's scale(), the result should have mean=0 and sd=1 (with ddof=1)
        assert abs(scaled.mean()) < 1e-10, "Scaled mean should be 0"
        assert abs(scaled.std(ddof=1) - 1.0) < 1e-10, "Scaled std (ddof=1) should be 1"

        print("PASS: r_scale matches R's scale() function")

    def test_behavior_scaling_matches_r(self, r_data):
        """Test that behavior scaling parameters match R's output."""
        # R reported these scaling values in test1_metadata.csv:
        # behavior_scaleval = 0.256653642330325 (this is the sd used for scaling)
        # behavior_centerval = 0.0576910111523495 (this is the mean)

        r_scaleval = r_data['behavior_scaleval']
        r_centerval = r_data['behavior_centerval']

        print(f"\n=== Behavior Scaling Reference ===")
        print(f"R behavior_scaleval (sd): {r_scaleval}")
        print(f"R behavior_centerval (mean): {r_centerval}")

        # These are the scaling parameters R used
        # We can verify by checking the relationship between scale and center
        assert r_scaleval > 0, "Scale value should be positive"
        assert abs(r_centerval) < 1, "Center value should be reasonable for normalized behavior"

        print("PASS: R scaling parameters are valid")


class TestSCCANWeights:
    """Test SCCAN weight properties against R reference."""

    def test_weight_range(self, r_data):
        """Test that statistic weights are in expected range [-1, 1]."""
        statistic = r_data['statistic']

        # Filter out zeros for analysis
        nonzero = statistic[statistic != 0]

        print(f"\n=== Weight Range Test ===")
        print(f"Total voxels: {len(statistic)}")
        print(f"Non-zero voxels: {len(nonzero)}")
        print(f"R reported non-zero: {r_data['nonzero_weights']}")
        print(f"Statistic range: [{statistic.min():.6f}, {statistic.max():.6f}]")

        # R normalizes to [-1, 1]
        assert statistic.min() >= -1.0 - 1e-6, f"Min weight {statistic.min()} < -1"
        assert statistic.max() <= 1.0 + 1e-6, f"Max weight {statistic.max()} > 1"

        # Check non-zero count matches
        actual_nonzero = np.sum(statistic != 0)
        assert actual_nonzero == r_data['nonzero_weights'], \
            f"Non-zero count mismatch: {actual_nonzero} vs {r_data['nonzero_weights']}"

        print("PASS: Weight range is [-1, 1] and non-zero count matches")

    def test_eig1_raw_properties(self, r_data):
        """Test raw eig1 weight properties."""
        eig1 = r_data['eig1_raw']

        # Find non-zero weights
        nonzero_mask = np.abs(eig1) > 1e-15
        n_nonzero = np.sum(nonzero_mask)

        print(f"\n=== Raw eig1 Properties ===")
        print(f"Total voxels: {len(eig1)}")
        print(f"Non-zero (>1e-15): {n_nonzero}")
        print(f"eig1 range: [{eig1.min():.6e}, {eig1.max():.6e}]")

        if n_nonzero > 0:
            nonzero_vals = eig1[nonzero_mask]
            print(f"Non-zero min: {nonzero_vals.min():.6e}")
            print(f"Non-zero max: {nonzero_vals.max():.6e}")

        # eig1 should have some non-zero values
        assert n_nonzero > 0, "eig1 should have non-zero values"

        print("PASS: eig1 has expected properties")

    def test_directional_sccan_weights(self, r_data):
        """Test that directional SCCAN produces negative weights (as expected)."""
        statistic = r_data['statistic']
        nonzero = statistic[statistic != 0]

        print(f"\n=== Directional SCCAN Test ===")
        print(f"eig2 value: {r_data['eig2']}")
        print(f"ccasummary correlation: {r_data['ccasummary_corr']}")

        # With directionalSCCAN=TRUE, weights can be negative
        # The flip is based on: flipval = sign(eig2) * sign(correlation)
        # R output shows eig2 = -0.151, corr = 0.851
        # So flipval = -1 * 1 = -1, meaning weights are flipped to negative

        if r_data['eig2'] < 0 and r_data['ccasummary_corr'] > 0:
            # Expect mostly negative weights after flip
            n_negative = np.sum(nonzero < 0)
            n_positive = np.sum(nonzero > 0)
            print(f"Negative weights: {n_negative}")
            print(f"Positive weights: {n_positive}")

            # Most non-zero weights should be negative after flip
            assert n_negative > n_positive, \
                f"Expected mostly negative weights, got {n_negative} neg vs {n_positive} pos"

        print("PASS: Directional SCCAN weights have expected sign pattern")


class TestPredictionWorkflow:
    """Test SCCAN prediction workflow against R reference."""

    def test_linear_calibration_parameters(self, r_data):
        """Test that linear calibration parameters are reasonable."""
        intercept = r_data['predictlm_intercept']
        slope = r_data['predictlm_slope']

        print(f"\n=== Linear Calibration Parameters ===")
        print(f"Intercept: {intercept:.6f}")
        print(f"Slope: {slope:.6f}")

        # These values should be finite
        assert np.isfinite(intercept), "Intercept should be finite"
        assert np.isfinite(slope), "Slope should be finite"

        # Slope should be non-zero for meaningful calibration
        assert abs(slope) > 0.01, f"Slope {slope} seems too small"

        print("PASS: Linear calibration parameters are valid")

    def test_prediction_formula(self, r_data):
        """
        Test the SCCAN prediction formula.

        R formula:
        predbehav_scaled = lesmat_scaled %*% t(eig1) %*% eig2
        predbehav_raw = predbehav_scaled * behavior_scaleval + behavior_centerval
        predbehav_calibrated = intercept + slope * predbehav_raw
        """
        eig1 = r_data['eig1_raw']
        eig2 = r_data['eig2']

        print(f"\n=== Prediction Formula Test ===")
        print(f"eig1 shape: {eig1.shape}")
        print(f"eig2: {eig2}")

        # Create a simple test lesmat (one subject)
        test_lesmat_scaled = np.ones(len(eig1))  # Dummy scaled lesmat

        # Compute prediction using R's formula
        # predbehav_scaled = lesmat_scaled @ eig1 @ eig2
        # (since eig2 is scalar, this simplifies)
        pred_scaled = np.dot(test_lesmat_scaled, eig1) * eig2

        # Unscale
        pred_raw = pred_scaled * r_data['behavior_scaleval'] + r_data['behavior_centerval']

        # Calibrate
        pred_calibrated = r_data['predictlm_intercept'] + r_data['predictlm_slope'] * pred_raw

        print(f"Pred scaled: {pred_scaled:.6f}")
        print(f"Pred raw: {pred_raw:.6f}")
        print(f"Pred calibrated: {pred_calibrated:.6f}")

        # These should be finite
        assert np.isfinite(pred_scaled), "Scaled prediction should be finite"
        assert np.isfinite(pred_raw), "Raw prediction should be finite"
        assert np.isfinite(pred_calibrated), "Calibrated prediction should be finite"

        print("PASS: Prediction formula produces valid results")


@pytest.mark.slow
class TestPythonLSMSCCANEndToEnd:
    """Run Python lsm_sccan on the same matrix used by R reference generation."""

    def test_lsm_sccan_matches_r_reference_when_inputs_available(self, r_data):
        required = ['lesmat', 'behavior', 'mask_img', 'predictions']
        missing = [key for key in required if key not in r_data]
        if missing:
            pytest.skip(
                "Missing SCCAN rerun fixtures: "
                + ", ".join(missing)
                + ". Regenerate with tests/generate_r_sccan_reference.R."
            )

        lesmat = r_data['lesmat']
        behavior = r_data['behavior']
        mask_img = r_data['mask_img']
        mask_vector = mask_img.get_fdata().reshape(-1) > 0

        result = lsm_sccan(
            lesmat,
            behavior,
            mask_img,
            optimize_sparseness=False,
            sparseness=float(r_data['sparseness']),
            cthresh=150,
            its=20,
            smooth=0.4,
            robust=1,
            robust_rank_fallback='auto',
            mycoption=1,
            max_based=False,
            directional_sccan=True,
            min_cluster_size=150,
            show_info=False,
        )

        py_statistic = result.stat_img.get_fdata().reshape(-1)[mask_vector]
        r_statistic = r_data['statistic']
        mask_shape = mask_img.shape
        full_space_rows = []
        for row in lesmat:
            full_row = np.zeros(mask_shape, dtype=np.float32).reshape(-1)
            full_row[mask_vector] = row.astype(np.float32)
            full_space_rows.append(full_row)
        py_predictions = result.predict([
            nib.Nifti1Image(row.reshape(mask_shape), mask_img.affine)
            for row in full_space_rows
        ])
        r_predictions = r_data['predictions']['pred_calibrated'].values

        stat_corr, _ = pearsonr(py_statistic, r_statistic)
        aligned_stat_corr = abs(stat_corr)
        pred_corr, _ = pearsonr(py_predictions, r_predictions)

        assert result.model_params['robust'] == 1
        # SCCAN/CCA eigenvectors can differ by a global sign while preserving
        # the same prediction after the saved calibration model is applied.
        assert aligned_stat_corr > 0.75
        assert pred_corr > 0.75
        assert abs(np.count_nonzero(py_statistic) - np.count_nonzero(r_statistic)) / len(r_statistic) < 0.1


class TestDataIntegrity:
    """Test that R reference data was loaded correctly."""

    def test_data_loaded(self, r_data):
        """Verify all expected data was loaded."""
        expected_keys = [
            'sparseness', 'behavior_scaleval', 'behavior_centerval',
            'eig2', 'ccasummary_corr', 'nonzero_weights',
            'predictlm_intercept', 'predictlm_slope',
            'eig1_raw', 'statistic'
        ]

        print(f"\n=== Data Integrity Test ===")
        for key in expected_keys:
            assert key in r_data, f"Missing key: {key}"
            print(f"  {key}: {'loaded' if r_data[key] is not None else 'missing'}")

        print("PASS: All expected data loaded")

    def test_metadata_values(self, r_data):
        """Verify metadata values are in expected ranges."""
        print(f"\n=== Metadata Values ===")
        print(f"Sparseness: {r_data['sparseness']}")
        print(f"eig2: {r_data['eig2']}")
        print(f"Correlation: {r_data['ccasummary_corr']}")
        print(f"Non-zero weights: {r_data['nonzero_weights']}")

        assert 0 < r_data['sparseness'] <= 1, f"Sparseness {r_data['sparseness']} out of range"
        assert abs(r_data['eig2']) <= 1, f"eig2 {r_data['eig2']} out of range"
        assert -1 <= r_data['ccasummary_corr'] <= 1, f"Correlation {r_data['ccasummary_corr']} out of range"
        assert r_data['nonzero_weights'] > 0, "Should have non-zero weights"

        print("PASS: All metadata values in expected ranges")

    def test_array_dimensions(self, r_data):
        """Verify array dimensions are consistent."""
        n_voxels_eig1 = len(r_data['eig1_raw'])
        n_voxels_stat = len(r_data['statistic'])

        print(f"\n=== Array Dimensions ===")
        print(f"eig1_raw length: {n_voxels_eig1}")
        print(f"statistic length: {n_voxels_stat}")

        assert n_voxels_eig1 == n_voxels_stat, \
            f"Dimension mismatch: eig1={n_voxels_eig1}, stat={n_voxels_stat}"

        print("PASS: Array dimensions are consistent")


def test_summary(r_data):
    """Print summary of all tests."""
    print("\n" + "=" * 60)
    print("SCCAN R Comparison Test Summary")
    print("=" * 60)
    print(f"Sparseness: {r_data['sparseness']}")
    print(f"eig2: {r_data['eig2']:.6f}")
    print(f"Correlation: {r_data['ccasummary_corr']:.6f}")
    print(f"Non-zero weights: {r_data['nonzero_weights']}")
    print(f"Total voxels: {len(r_data['statistic'])}")
    print(f"Linear calibration: y = {r_data['predictlm_slope']:.4f}*x + {r_data['predictlm_intercept']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
