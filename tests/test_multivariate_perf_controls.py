import numpy as np
import nibabel as nib
import pytest

from lesymap.core.pipeline import VALID_METHOD_PARAMS
from lesymap.methods import multivariate


class FakeAntsPy:
    def __init__(self):
        self.calls = []

    def sparse_decom2(self, inmats, **kwargs):
        self.calls.append(kwargs)
        n_features = inmats[0].shape[1]
        return {
            'eig1': np.ones(n_features),
            'eig2': np.array([1.0]),
            'ccasummary': {'corrs': [0.5]},
        }


class RobustNotImplementedAntsPy(FakeAntsPy):
    def sparse_decom2(self, inmats, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get('robust', 0) > 0:
            raise NotImplementedError('robust > 0 not currently implemented')

        n_features = inmats[0].shape[1]
        return {
            'eig1': np.ones(n_features),
            'eig2': np.array([1.0]),
            'ccasummary': {'corrs': [0.5]},
        }


class ZeroWeightAntsPy(FakeAntsPy):
    def sparse_decom2(self, inmats, **kwargs):
        self.calls.append(kwargs)
        n_features = inmats[0].shape[1]
        return {
            'eig1': np.zeros(n_features),
            'eig2': np.array([1.0]),
            'ccasummary': {'corrs': [0.0]},
        }


def _mask_img(n_features):
    data = np.ones((n_features, 1, 1), dtype=float)
    return nib.Nifti1Image(data, np.eye(4))


def test_lsm_sccan_passes_sparseness_range_and_n_jobs_to_optimizer(monkeypatch):
    fake_antspy = FakeAntsPy()
    seen = {}

    def fake_optimize(*args, **kwargs):
        seen['sparseness_range'] = kwargs.get('sparseness_range')
        seen['n_jobs'] = kwargs.get('n_jobs')
        return {'optimal_sparseness': 0.2, 'cv_correlation': 0.99}

    monkeypatch.setattr(multivariate, '_import_antspy', lambda: fake_antspy)
    monkeypatch.setattr(multivariate, '_optimize_sccan_sparseness', fake_optimize)

    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.4, 0.8, 1.2])

    result = multivariate.lsm_sccan(
        lesmat,
        behavior,
        _mask_img(3),
        optimize_sparseness=True,
        sparseness_range=[0.1, 0.2],
        n_jobs=2,
        min_cluster_size=0,
        show_info=False,
    )

    assert seen == {'sparseness_range': [0.1, 0.2], 'n_jobs': 2}
    assert result.model_params['sparseness'] == 0.2


def test_optimize_sccan_sparseness_accepts_parallel_n_jobs_with_fake_antspy():
    fake_antspy = FakeAntsPy()
    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.3, 0.6, 0.8, 1.0, 1.2])

    result = multivariate._optimize_sccan_sparseness(
        lesmat,
        behavior,
        sparseness_range=[0.05, 0.1],
        n_jobs=2,
        n_folds=3,
        antspyt=fake_antspy,
        show_info=False,
    )

    assert result['optimal_sparseness'] in {0.05, 0.1}
    assert np.isfinite(result['cv_correlation'])
    assert len(fake_antspy.calls) == 6


def test_non_linear_svr_requires_feature_limit_for_default_permutation_importance():
    lesmat = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.0, 1.0, 0.5, 1.5])

    with pytest.raises(ValueError, match='max_features'):
        multivariate.lsm_svr(
            lesmat,
            behavior,
            _mask_img(4),
            kernel='rbf',
            show_info=False,
        )


def test_public_pipeline_allows_new_multivariate_parameters():
    assert {
        'sparseness_range',
        'n_jobs',
        'robust',
        'robust_rank_fallback',
        'max_based',
    } <= VALID_METHOD_PARAMS['sccan']
    assert {'n_perm', 'max_features'} <= VALID_METHOD_PARAMS['svr']


def test_rank_transform_binary_columns_matches_zscore_for_nonconstant_columns():
    matrix = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=float,
    )
    transformed = multivariate._rank_transform_columns(matrix)

    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale[scale == 0] = 1.0
    zscore = (matrix - mean) / scale
    zscore[:, 1] = 0.0

    assert np.allclose(transformed, zscore)
    assert np.all(transformed[:, 1] == 0.0)


def test_lsm_sccan_auto_rank_fallback_records_metadata(monkeypatch):
    fake_antspy = RobustNotImplementedAntsPy()
    monkeypatch.setattr(multivariate, '_import_antspy', lambda: fake_antspy)

    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.4, 0.8, 1.2])

    result = multivariate.lsm_sccan(
        lesmat,
        behavior,
        _mask_img(3),
        optimize_sparseness=False,
        sparseness=0.2,
        robust=1,
        robust_rank_fallback='auto',
        min_cluster_size=0,
        show_info=False,
    )

    assert [call['robust'] for call in fake_antspy.calls] == [1, 0]
    assert result.model_params['robust_requested'] == 1
    assert result.model_params['robust_backend_used'] == 0
    assert result.model_params['robust_rank_fallback'] is True
    assert result.model_params['rank_transform'] == 'column_average_rank_then_zscore'
    assert result.model_params['rank_transform_applied_to'] == ['lesmat', 'behavior']
    assert 'not currently implemented' in result.model_params['backend_robust_error']


def test_lsm_sccan_never_rank_fallback_preserves_backend_error(monkeypatch):
    fake_antspy = RobustNotImplementedAntsPy()
    monkeypatch.setattr(multivariate, '_import_antspy', lambda: fake_antspy)

    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.4, 0.8, 1.2])

    with pytest.raises(RuntimeError, match='robust > 0 not currently implemented'):
        multivariate.lsm_sccan(
            lesmat,
            behavior,
            _mask_img(3),
            optimize_sparseness=False,
            sparseness=0.2,
            robust=1,
            robust_rank_fallback='never',
            min_cluster_size=0,
            show_info=False,
        )

    assert [call['robust'] for call in fake_antspy.calls] == [1]


def test_optimize_sccan_sparseness_uses_rank_fallback_in_cv():
    fake_antspy = RobustNotImplementedAntsPy()
    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.3, 0.6, 0.8, 1.0, 1.2])

    result = multivariate._optimize_sccan_sparseness(
        lesmat,
        behavior,
        sparseness_range=[0.05],
        n_jobs=1,
        n_folds=3,
        antspyt=fake_antspy,
        robust=1,
        robust_rank_fallback='auto',
        show_info=False,
    )

    assert result['optimal_sparseness'] == 0.05
    assert np.isfinite(result['cv_correlation'])
    assert result['robust_info']['cv_robust_rank_fallback'] is True
    assert result['robust_info']['cv_robust_backend_used'] == 0
    assert [call['robust'] for call in fake_antspy.calls] == [1, 0] * 3


def test_optimize_sccan_sparseness_does_not_swallow_never_fallback_error():
    fake_antspy = RobustNotImplementedAntsPy()
    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.3, 0.6, 0.8, 1.0, 1.2])

    with pytest.raises(NotImplementedError, match='robust > 0 not currently implemented'):
        multivariate._optimize_sccan_sparseness(
            lesmat,
            behavior,
            sparseness_range=[0.05],
            n_jobs=1,
            n_folds=3,
            antspyt=fake_antspy,
            robust=1,
            robust_rank_fallback='never',
            show_info=False,
        )


def test_lsm_sccan_nonfinite_cv_returns_null_with_robust_metadata(monkeypatch):
    fake_antspy = ZeroWeightAntsPy()
    monkeypatch.setattr(multivariate, '_import_antspy', lambda: fake_antspy)

    lesmat = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.3, 0.6, 0.8, 1.0, 1.2])

    result = multivariate.lsm_sccan(
        lesmat,
        behavior,
        _mask_img(3),
        optimize_sparseness=True,
        sparseness_range=[0.05],
        n_jobs=1,
        robust=0,
        min_cluster_size=0,
        show_info=False,
    )

    assert result.model_params['null_result'] is True
    assert np.isnan(result.model_params['cv_correlation'])
    assert result.model_params['cv_pvalue'] == 1.0
    assert result.model_params['robust'] == 0
    assert result.model_params['robust_rank_fallback_policy'] == 'auto'
    assert result.model_params['cv_robust_requested'] == 0
    assert result.model_params['cv_robust_rank_fallback'] is False


def test_lsm_sccan_null_result_expands_filtered_patches(monkeypatch):
    fake_antspy = ZeroWeightAntsPy()
    monkeypatch.setattr(multivariate, '_import_antspy', lambda: fake_antspy)

    lesmat = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )
    behavior = np.array([0.1, 0.3, 0.6, 0.8, 1.0, 1.2])
    patchinfo = {
        "patchindx": np.array([1, 2, 3]),
        "analysis_keep_mask": np.array([True, False, True]),
    }

    result = multivariate.lsm_sccan(
        lesmat,
        behavior,
        _mask_img(3),
        patchinfo=patchinfo,
        optimize_sparseness=True,
        sparseness_range=[0.05],
        n_jobs=1,
        robust=0,
        min_cluster_size=0,
        show_info=False,
    )

    assert result.model_params["null_result"] is True
    np.testing.assert_array_equal(result.stat_img.get_fdata().reshape(-1), [0, 0, 0])


def test_import_antspy_finds_installed_ants_backend():
    ants = pytest.importorskip("ants")
    assert hasattr(ants, "sparse_decom2")

    assert multivariate._import_antspy() is ants


def test_lsm_sccan_runs_with_installed_ants_backend_when_robust_disabled():
    pytest.importorskip("ants")
    lesmat = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=float,
    )
    behavior = np.array([0.2, 0.5, 0.8, 1.0, 1.3, 1.6], dtype=float)

    result = multivariate.lsm_sccan(
        lesmat,
        behavior,
        _mask_img(4),
        optimize_sparseness=False,
        sparseness=0.2,
        its=2,
        cthresh=0,
        robust=0,
        min_cluster_size=0,
        show_info=False,
    )

    assert result.method == "sccan"
    assert result.sccan_weights.shape == (4,)
