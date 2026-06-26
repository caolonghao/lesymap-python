import numpy as np
import nibabel as nib
import pytest
from sklearn.svm import SVR

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


def test_r_compatible_svr_pvalues_are_explicit_and_use_r_directional_formula():
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
        compute_pvalues=True,
        nperm=3,
        random_state=7,
        show_info=False,
    )

    lesmat_fit = (lesmat - result.svr_lesmat_center) / result.svr_lesmat_scale
    behavior_fit = (
        (behavior - result.svr_behavior_center) / result.svr_behavior_scale
    )
    statistic = result.stat_img.get_fdata().reshape(-1)
    exceed = np.ones_like(statistic)
    rng = np.random.default_rng(7)

    for _ in range(3):
        permuted = behavior_fit[rng.permutation(len(behavior_fit))]
        svr = SVR(kernel="linear", C=30.0, epsilon=0.1)
        svr.fit(lesmat_fit, permuted)
        weights = (svr.dual_coef_ @ svr.support_vectors_).ravel()
        perm_stat = weights * (10.0 / np.max(np.abs(weights)))
        exceed[statistic >= 0] += perm_stat[statistic >= 0] >= statistic[statistic >= 0]
        exceed[statistic < 0] += perm_stat[statistic < 0] <= statistic[statistic < 0]

    expected = exceed / 4.0

    assert result.pval_img is not None
    assert result.model_params["compute_pvalues"] is True
    assert result.model_params["svr_pvalue_method"] == "r_permutation"
    assert result.model_params["pvalue_method"] == "r_permutation"
    np.testing.assert_allclose(result.pval_img.get_fdata().reshape(-1), expected)


def test_r_compatible_svr_does_not_compute_permutation_pvalues_by_default():
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

    assert result.pval_img is None
    assert result.model_params["compute_pvalues"] is False


def test_svr_r_permutation_pvalues_require_r_compatible_statistic():
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

    with pytest.raises(ValueError, match="r_compatible=True"):
        lsm_svr(
            lesmat,
            behavior,
            mask,
            svr_pvalue_method="r_permutation",
            show_info=False,
        )


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


def test_svr_patch_pvalue_mapping_fills_filtered_patches_with_one():
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
        r_compatible=True,
        kernel="linear",
        C=30.0,
        epsilon=0.1,
        svr_pvalue_method="r_permutation",
        nperm=1,
        random_state=7,
        show_info=False,
    )

    pvals = result.pval_img.get_fdata().reshape(-1)

    assert pvals.shape == (4,)
    assert pvals[0] == pvals[1]
    assert pvals[2] == 1
    assert 0.5 <= pvals[3] <= 1
