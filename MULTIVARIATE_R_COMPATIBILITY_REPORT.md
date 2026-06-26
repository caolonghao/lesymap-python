# Multivariate R Compatibility Report

Date: 2026-06-26
Scope: SCCAN, SVR/SVM, prediction API, and binary behavior evaluation.

## Executive Summary

Current status:

- SCCAN is the closest path for R-like predictive behavior. Python now has patch-aware checkpoint/prediction tests, a Python-side rank fallback for ANTsPy versions where `robust > 0` is not implemented, and a true R rerun fixture showing highly correlated calibrated predictions. Exact voxel-weight equality is not claimed.
- SVR/SVM now has an explicit `r_compatible=True` mode. This mode matches the main R LESYMAP preprocessing/default-parameter shape more closely: R-style centering/scaling, R defaults (`kernel='rbf'`, `C=30`, `gamma=5`, `epsilon=0.1`), prediction unscaling, and R-style `10 / max(abs(w))` statistic scaling for linear SVR.
- Standard Python SVR remains intentionally Pythonic by default: `kernel='linear'`, `C=1.0`, `epsilon=0.1`, no internal R scaling. This avoids silently changing existing users' results.
- Prediction API coverage has improved. SCCAN and SVR now have checkpoint roundtrip tests, including patch-compressed training.
- Real masked NIfTI prediction coverage now exists for both SVR and SCCAN:
  train from `test_data/normalized_masks`, save checkpoint, reload, and predict
  held-out subjects.
- Binary behavior such as mutism `0/1` should be treated as a continuous-risk prediction problem first. A fixed `0.5` cutoff is not generally valid unless the output has been calibrated to probability scale.

Fresh verification on `main` after merge:

```text
pytest -q
115 passed, 15 deselected, 266 warnings in 7.63s
pytest -q tests/test_prediction_roundtrip.py tests/test_multivariate_perf_controls.py
18 passed, 40 warnings in 1.90s
pytest -q -m slow tests/test_testdata_prediction_e2e.py -rs
2 passed, 4 warnings in 8.24s
pytest -q -m slow tests/test_svr_r_comparison.py::TestPythonLSMSVREndToEnd
2 passed, 5 warnings in 98.17s (0:01:38)
pytest -q -m slow tests/test_svr_r_comparison.py
12 passed, 2 deselected, 29 warnings in 102.28s (0:01:42)
pytest -q -m slow tests/test_sccan_r_comparison.py::TestPythonLSMSCCANEndToEnd
1 passed, 2 warnings in 75.48s (0:01:15)
```

## SCCAN

### What Is Verified

- SCCAN prediction roundtrip now covers both direct `result.predict()` and `save_checkpoint()` / `load_checkpoint()` prediction.
- Patch-aware prediction is covered: when training used patch compression, prediction now extracts the full mask vector and projects it back to the training patch feature space using saved representative voxel indices.
- No-patch SCCAN prediction is tested against the saved formula:
  1. scale new lesion matrix using saved lesion center/scale,
  2. compute `lesmat_scaled @ eig1 @ eig2`,
  3. reverse behavior scaling,
  4. apply saved linear calibration.
- Robust fallback behavior is covered at the Python control-flow level: if ANTsPy raises `NotImplementedError` for `robust > 0`, Python rank-transforms the data and reruns with backend `robust=0`.
- `tests/generate_r_sccan_reference.R` now saves the missing rerun inputs
  (`sccan_lesmat.csv.gz`, `sccan_behavior.csv`, `mask.nii.gz`) in addition to R maps,
  scaling metadata, and prediction outputs.
- The SCCAN R reference fixtures were regenerated with the `dorianps/lesymap`
  Docker image. `tests/test_sccan_r_comparison.py::TestPythonLSMSCCANEndToEnd`
  now runs Python `lsm_sccan()` on the same R input matrix and mask.
- The SCCAN slow comparison passes: calibrated prediction correlation is about
  0.998 against R predictions. The statistic map has high full-mask
  sign-aligned correlation, but nonzero-support agreement is looser; this is
  reported as coarse map agreement rather than exact map parity. The sign
  alignment is intentional because CCA/SCCAN eigenvectors are identifiable only
  up to a global sign.

### What Is Not Yet Strictly Verified

- SCCAN fixed-sparseness prediction behavior and coarse map agreement are now
  covered by true R-vs-Python fixtures. CV-selected sparseness remains optional
  because the R CV generator is slow; run
  `RUN_SCCAN_CV=1 tests/generate_r_sccan_reference.R` when CV fixtures are
  specifically needed.
- Exact SCCAN statistic equality is not expected because ANTsR and ANTsPy can
  differ in robust-rank implementation and CCA sign orientation. Current gating
  uses prediction correlation and sign-aligned spatial correlation.

### Robust Rank Behavior

R LESYMAP SCCAN defaults to `robust=1`, which rank-transforms inputs inside ANTsR `sparseDecom2()`.

Current ANTsPy/antspyx versions may expose `robust` but raise:

```text
NotImplementedError: robust > 0 not currently implemented
```

Python now handles that by:

- trying backend `robust=1`,
- falling back to Python average-rank transform when the backend cannot do it,
- recording fallback metadata in `model_params`.

For binary lesion columns and binary behavior labels, rank transform mostly preserves the two-level structure. For continuous behavior, rank transform changes spacing and can change weights, so this remains an approximation until validated against ANTsR.

## SVR / SVM

### Standard Python Mode

Default Python behavior remains:

```text
kernel = linear
C = 1.0
epsilon = 0.1
no internal R-style scaling
```

This mode is useful as a conventional scikit-learn SVR model and preserves previous Python behavior.

### R-Compatible Mode

New behavior with `r_compatible=True`:

```text
kernel = rbf
C = 30.0
gamma = 5.0
epsilon = 0.1
scale lesmat and behavior using R scale(center=TRUE, scale=TRUE)
fit SVR on scaled data
unscale predictions back to behavior units
```

For `r_compatible=True`, Python computes a support-vector projection
statistic and applies R's beta-scale convention:

```text
weights = dual_coef @ support_vectors
statistic = weights * (10 / max(abs(weights)))
```

This covers the default RBF kernel and explicit linear kernel in the same
shape as R LESYMAP.

### What Is Verified

- `tests/test_svr_r_compatible.py` verifies:
  - R-compatible defaults are `rbf`, `C=30`, `gamma=5`, `epsilon=0.1`.
  - standard Python defaults remain `linear`, `C=1`, `epsilon=0.1` and are independent of `show_info`.
  - R-compatible statistic maps use support-vector projection followed by `10 / max(abs(weights))`.
  - filtered analysis patches expand back to the original patch index before voxel-map reconstruction.
- `tests/test_prediction_roundtrip.py` verifies:
  - SVR prediction survives checkpoint save/load,
  - SVR prediction works when training used patch compression,
  - R-compatible SVR stores scaling parameters and unscales predictions correctly after checkpoint load.
- `tests/test_svr_r_comparison.py::TestPythonLSMSVREndToEnd` now calls project
  `lsm_svr()` directly on the existing R-style reference matrix:
  - linear `r_compatible=True` matches R-style predictions, raw weights, and beta-scaled statistics to tight tolerance,
  - default RBF `r_compatible=True` matches behavior correlation and support-vector-projection statistic to high correlation with small numerical tolerance.
- `tests/test_svr_r_comparison.py::TestTinyTrueRLSMSVRReference` now compares
  Python `lsm_svr()` against true R `lsm_svr()` outputs on a small generated
  fixture for both linear and default RBF kernels. This runs in the default test
  suite and checks statistic vectors plus valid R permutation p-values.
- `tests/test_svr_r_comparison.py::TestTrueRLSMSVRReference` now compares
  Python `lsm_svr()` against large true R `lsm_svr()` statistic vectors for
  both linear and default RBF kernels. The fixture set includes compressed
  `svr_lesmat.csv.gz`, `svr_lesmat_scaled.csv.gz`,
  `svr_lsm_linear_results.csv.gz`, `svr_lsm_radial_results.csv.gz`, and a
  manifest with subject/voxel dimensions and R package versions. This is marked
  `slow` and must be run explicitly with `-m slow`.

### What Is Not Yet Strictly Verified

- Large-fixture R permutation p-values are not yet reproduced by Python
  `lsm_svr()`. The current true-R tests verify statistic-map parity and only
  check that the R p-values are valid probabilities.
- R output object metadata beyond statistic vectors and valid p-value ranges
  still needs fixture comparison if exact R object parity becomes required.

## Real Test Data E2E

`tests/test_testdata_prediction_e2e.py` uses the real masked NIfTI files under
`test_data/normalized_masks` when they are available. To keep the test bounded,
it builds a deterministic small ROI from variable lesion voxels and derives a
synthetic continuous behavior score from lesion load within that ROI.

What this verifies:

- public `lesymap.lesymap()` accepts real NIfTI mask paths and an explicit ROI mask,
- SVR `r_compatible=True` trains on real masked NIfTI data,
- SCCAN trains on real masked NIfTI data with fixed sparseness,
- both result types can `save_checkpoint()`, `load_checkpoint()`, and `predict()`
  held-out real NIfTI subjects,
- `LesymapResult.predict()` accepts `pathlib.Path` inputs as well as strings.

This is an end-to-end API and I/O smoke test, not an R-vs-Python statistical
parity test for the local `test_data/` cohort.

## Binary Behavior Evaluation

For behavior labels such as mutism `0/1`, SCCAN/SVR outputs should be interpreted as continuous scores unless an explicit calibration step has been fitted.

Recommended evaluation:

- ROC-AUC for ranking performance.
- PR-AUC when positive cases are rare.
- Sensitivity, specificity, balanced accuracy, F1, MCC, and confusion matrix at a selected threshold.
- Threshold selection inside training folds only, for example Youden J, max MCC, or a fixed-sensitivity target.
- Probability calibration only if needed, using logistic/Platt or isotonic calibration inside training folds.

A fixed `0.5` threshold is only defensible if the output is calibrated to probability scale. SCCAN calibrated scores and SVR predictions are not guaranteed to be probabilities and may fall outside `[0, 1]`.

The branch includes `evaluate_binary_predictions()` for this evaluation pattern.

## Current Recommendation

Use SCCAN as the primary R-like predictive method, especially for continuity with LESYMAP publications and R behavior. Use SVR in two modes:

- `r_compatible=True` when comparing with or migrating from R LESYMAP.
- default Python SVR when you want a conventional scikit-learn linear SVR baseline.

Before claiming full R replacement equivalence, add CV-sparseness SCCAN fixtures
and decide whether Python should reproduce R SVR permutation p-values.

## Remaining Validation Checklist

- Decide whether to implement Python-side SVR permutation p-value images that
  match R `lsm_svr()` or keep p-values out of the Python prediction API scope.
- Generate optional SCCAN CV-sparseness fixtures with `RUN_SCCAN_CV=1`.
