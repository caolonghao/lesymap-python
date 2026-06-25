import numpy as np
import nibabel as nib

import lesymap
from lesymap.core.result import LesymapResult
from lesymap.methods import multivariate


class FixedAntsPy:
    def sparse_decom2(self, inmats, **kwargs):
        n_features = inmats[0].shape[1]
        weights = np.linspace(1.0, 2.0, n_features, dtype=float)
        return {
            "eig1": weights,
            "eig2": np.array([1.0]),
            "ccasummary": {"corrs": [0.75]},
        }


def _toy_mask():
    return nib.Nifti1Image(np.ones((4, 1, 1), dtype=float), np.eye(4))


def _toy_images():
    patterns = [
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
    ]
    return [
        nib.Nifti1Image(np.asarray(pattern, dtype=float).reshape(4, 1, 1), np.eye(4))
        for pattern in patterns
    ]


def test_sccan_patch_prediction_roundtrips_through_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(multivariate, "_import_antspy", lambda: FixedAntsPy())

    images = _toy_images()
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)

    result = lesymap.lesymap(
        images,
        behavior,
        method="sccan",
        mask=_toy_mask(),
        no_patch=False,
        min_subject_per_voxel=1,
        optimize_sparseness=False,
        sparseness=0.5,
        cthresh=0,
        min_cluster_size=0,
        robust=0,
        nperm=0,
        show_info=False,
    )

    direct_predictions = result.predict(images[:3])
    result.save_checkpoint(tmp_path / "sccan.pkl")
    loaded = LesymapResult.load_checkpoint(tmp_path / "sccan.pkl")
    loaded_predictions = loaded.predict(images[:3])

    assert result.patchinfo is not None
    assert result.sccan_weights.shape[0] == result.patchinfo["npatches"]
    assert direct_predictions.shape == (3,)
    np.testing.assert_allclose(loaded_predictions, direct_predictions, rtol=0, atol=0)


def test_sccan_no_patch_prediction_matches_saved_formula(monkeypatch):
    monkeypatch.setattr(multivariate, "_import_antspy", lambda: FixedAntsPy())

    images = _toy_images()
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)

    result = lesymap.lesymap(
        images,
        behavior,
        method="sccan",
        mask=_toy_mask(),
        no_patch=True,
        optimize_sparseness=False,
        sparseness=0.5,
        cthresh=0,
        min_cluster_size=0,
        robust=0,
        nperm=0,
        show_info=False,
    )

    new_images = images[:2]
    predictions = result.predict(new_images)
    lesmat = np.vstack([
        np.asarray(img.dataobj).reshape(-1)
        for img in new_images
    ])
    lesmat_scaled = (
        (lesmat - result.sccan_lesmat_center)
        / result.sccan_lesmat_scale
    )
    pred_scaled = (
        lesmat_scaled
        @ result.sccan_weights.reshape(-1, 1)
        @ result.sccan_eig2.reshape(1, -1)
    ).ravel()
    pred_raw = (
        pred_scaled * result.sccan_behavior_scale
        + result.sccan_behavior_center
    )
    expected = result.sccan_predict_lm.predict(pred_raw.reshape(-1, 1))

    np.testing.assert_allclose(predictions, expected, rtol=0, atol=0)


def test_svr_patch_prediction_roundtrips_through_checkpoint(tmp_path):
    images = _toy_images()
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)

    result = lesymap.lesymap(
        images,
        behavior,
        method="svr",
        mask=_toy_mask(),
        no_patch=False,
        min_subject_per_voxel=1,
        kernel="linear",
        C=1.0,
        epsilon=0.01,
        nperm=0,
        show_info=False,
    )

    direct_predictions = result.predict(images[:3])
    result.save_checkpoint(tmp_path / "svr.pkl")
    loaded = LesymapResult.load_checkpoint(tmp_path / "svr.pkl")
    loaded_predictions = loaded.predict(images[:3])

    assert result.patchinfo is not None
    assert result.svr_model.n_features_in_ == result.patchinfo["npatches"]
    assert direct_predictions.shape == (3,)
    np.testing.assert_allclose(loaded_predictions, direct_predictions, rtol=0, atol=0)
