"""End-to-end prediction tests using real masked NIfTI test_data when present."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lesymap import LesymapResult, lesymap
from lesymap.core.image_utils import images_to_matrix, mask_from_average, matrix_to_image
from lesymap.core.io import load_lesions


def _find_test_data_dir():
    candidates = [
        Path(__file__).resolve().parents[1] / "test_data",
        Path("/Users/tonycao/mycode/lesymap-python/test_data"),
    ]
    for candidate in candidates:
        if (candidate / "normalized_masks").exists():
            return candidate
    pytest.skip("test_data/normalized_masks is not available")


def _real_mask_inputs(max_voxels=256):
    data_dir = _find_test_data_dir()
    files = sorted((data_dir / "normalized_masks").glob("*.nii.gz"))
    if len(files) < 8:
        pytest.skip("Need at least 8 real normalized masks for prediction e2e")

    images = load_lesions(files, check_headers=True)
    full_mask = mask_from_average(images, threshold=0.05, min_voxels=10)
    full_matrix = images_to_matrix(images, full_mask, dtype=np.uint8)

    variance = full_matrix.var(axis=0)
    variable = np.flatnonzero(variance > 0)
    if len(variable) < max_voxels:
        pytest.skip("Real masks do not contain enough variable lesion voxels")

    selected = variable[np.argsort(variance[variable])[-max_voxels:]]
    full_mask_vector = full_mask.get_fdata().reshape(-1) > 0
    full_selected_indices = np.flatnonzero(full_mask_vector)[selected]
    roi_data = np.zeros(full_mask.shape, dtype=np.uint8).reshape(-1)
    roi_data[full_selected_indices] = 1
    roi_mask = nib.Nifti1Image(roi_data.reshape(full_mask.shape), full_mask.affine, full_mask.header)

    roi_matrix = full_matrix[:, selected].astype(float)
    lesion_score = roi_matrix.sum(axis=1)
    behavior = (lesion_score - lesion_score.mean()) / (lesion_score.std(ddof=1) or 1.0)

    return files, roi_mask, behavior


@pytest.mark.slow
def test_real_mask_svr_training_checkpoint_and_prediction_roundtrip(tmp_path):
    """Real mask NIfTI files can train, save, load, and predict with SVR."""
    files, roi_mask, behavior = _real_mask_inputs(max_voxels=256)
    train_files = files[:12]
    train_behavior = behavior[:12]
    test_files = files[12:17]

    result = lesymap(
        train_files,
        train_behavior,
        method="svr",
        mask=roi_mask,
        no_patch=True,
        binary_check=True,
        r_compatible=True,
        kernel="linear",
        C=1.0,
        epsilon=0.1,
        show_info=False,
    )

    direct_predictions = result.predict(test_files)
    result.save_checkpoint(tmp_path / "real-mask-svr.pkl")
    loaded = LesymapResult.load_checkpoint(tmp_path / "real-mask-svr.pkl")
    loaded_predictions = loaded.predict(test_files)

    assert result.method == "svr"
    assert result.model_params["r_compatible"] is True
    assert result.stat_img.shape == roi_mask.shape
    assert direct_predictions.shape == (len(test_files),)
    assert np.all(np.isfinite(direct_predictions))
    np.testing.assert_allclose(loaded_predictions, direct_predictions, rtol=0, atol=0)


@pytest.mark.slow
def test_real_mask_sccan_training_checkpoint_and_prediction_smoke(monkeypatch, tmp_path):
    """Real mask NIfTI files exercise SCCAN training and checkpoint prediction."""
    pytest.importorskip("ants")

    files, roi_mask, behavior = _real_mask_inputs(max_voxels=64)
    train_files = files[:12]
    train_behavior = behavior[:12]
    test_files = files[12:15]

    result = lesymap(
        train_files,
        train_behavior,
        method="sccan",
        mask=roi_mask,
        no_patch=True,
        binary_check=True,
        optimize_sparseness=False,
        sparseness=0.25,
        smooth=0.0,
        its=5,
        cthresh=0,
        robust=0,
        max_based=False,
        min_cluster_size=0,
        show_info=False,
    )

    direct_predictions = result.predict(test_files)
    result.save_checkpoint(tmp_path / "real-mask-sccan.pkl")
    loaded = LesymapResult.load_checkpoint(tmp_path / "real-mask-sccan.pkl")
    loaded_predictions = loaded.predict(test_files)

    assert result.method == "sccan"
    assert result.stat_img.shape == roi_mask.shape
    assert result.sccan_weights is not None
    assert direct_predictions.shape == (len(test_files),)
    assert np.all(np.isfinite(direct_predictions))
    np.testing.assert_allclose(loaded_predictions, direct_predictions, rtol=0, atol=0)
