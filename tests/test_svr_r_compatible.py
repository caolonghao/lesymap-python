import numpy as np
import nibabel as nib

from lesymap.methods.multivariate import lsm_svr


def test_r_compatible_linear_svr_uses_r_statistic_scaling():
    lesmat = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=float,
    )
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)
    mask = nib.Nifti1Image(np.ones((3, 1, 1), dtype=float), np.eye(4))

    result = lsm_svr(
        lesmat,
        behavior,
        mask,
        kernel="linear",
        C=30.0,
        epsilon=0.1,
        r_compatible=True,
        show_info=False,
    )

    coef = result.svr_model.coef_.flatten()
    expected = coef * (10.0 / np.max(np.abs(coef)))
    statistic = result.stat_img.get_fdata().reshape(-1)

    assert result.model_params["r_compatible"] is True
    assert result.model_params["svr_statistic_scale"] == "r_beta_scale_10_over_max_abs"
    np.testing.assert_allclose(statistic, expected)


def test_r_compatible_svr_uses_r_defaults_when_not_overridden():
    lesmat = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=float,
    )
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)
    mask = nib.Nifti1Image(np.ones((3, 1, 1), dtype=float), np.eye(4))

    result = lsm_svr(
        lesmat,
        behavior,
        mask,
        r_compatible=True,
        show_info=False,
    )

    support_projection = (
        result.svr_model.dual_coef_ @ result.svr_model.support_vectors_
    ).ravel()
    expected = support_projection * (10.0 / np.max(np.abs(support_projection)))
    statistic = result.stat_img.get_fdata().reshape(-1)

    assert result.model_params["kernel"] == "rbf"
    assert result.model_params["C"] == 30.0
    assert result.model_params["epsilon"] == 0.1
    assert result.model_params["gamma"] == 5.0
    assert result.model_params["svr_statistic_scale"] == "r_beta_scale_10_over_max_abs"
    assert result.svr_model.kernel == "rbf"
    assert result.svr_model.C == 30.0
    assert result.svr_model.gamma == 5.0
    np.testing.assert_allclose(statistic, expected)


def test_standard_svr_defaults_are_independent_of_show_info():
    lesmat = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=float,
    )
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)
    mask = nib.Nifti1Image(np.ones((3, 1, 1), dtype=float), np.eye(4))

    result = lsm_svr(lesmat, behavior, mask)

    assert result.model_params["kernel"] == "linear"
    assert result.model_params["C"] == 1.0
    assert result.model_params["epsilon"] == 0.1
    assert result.svr_model.kernel == "linear"
    assert result.svr_model.C == 1.0


def test_svr_patch_mapping_expands_filtered_patches():
    lesmat = np.array(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    behavior = np.array([0.0, 0.2, 0.7, 0.9, 1.1, 0.4], dtype=float)
    mask = nib.Nifti1Image(np.ones((4, 1, 1), dtype=float), np.eye(4))
    patchinfo = {
        "patchindx": np.array([1, 1, 2, 3]),
        "npatches": 3,
        "analysis_keep_mask": np.array([True, False, True]),
    }

    result = lsm_svr(
        lesmat[:, [0, 3]],
        behavior,
        mask,
        patchinfo=patchinfo,
        kernel="linear",
        show_info=False,
    )

    stat = result.stat_img.get_fdata().reshape(-1)

    assert stat.shape == (4,)
    assert stat[0] == stat[1]
    assert stat[2] == 0
    assert stat[3] != 0
