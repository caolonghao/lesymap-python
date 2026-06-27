# Multivariate R Compatibility Report

Date: 2026-06-26
Scope: SCCAN, SVR/SVM, prediction API, and binary behavior evaluation.

## Executive Summary

Current status:

- SCCAN: implementation intent and prediction formula are close to R LESYMAP, and the ANTsPy `robust=1` gap now has a Python-side rank fallback. However, current tests do not prove strict Python-vs-R SCCAN numerical equivalence.
- SVR/SVM: current Python `lsm_svr()` is not functionally equivalent to R `lsm_svr()` by default. Existing R comparison tests validate sklearn/e1071 behavior on pre-scaled fixture data, but they do not exercise Python `lsm_svr()` end to end.
- Prediction API: Python adds checkpoint/predict support for SCCAN, SVR, and regression. R public `lesymap.predict()` only supports SCCAN. SCCAN prediction formula is similar to R, but prediction roundtrip tests are currently missing.
- Binary behavior, e.g. mutism `0/1`: SCCAN/SVR predictions should be treated as continuous risk scores. A fixed `0.5` threshold is not enough for evaluation unless calibrated and selected inside training folds.

Last full test status after performance/robust integration:

```text
pytest -q
106 passed, 216 warnings in 113.41s
```

## SCCAN

### What Is Verified

- R SCCAN fixture files are loaded and sanity checked in `tests/test_sccan_r_comparison.py`.
- The current tests verify R fixture properties: weight range, nonzero count, directional sign pattern, finite calibration parameters, and finite formula outputs.
- Python SCCAN control tests cover `sparseness_range`, `n_jobs`, robust fallback metadata, backend error handling, and non-finite CV behavior in `tests/test_multivariate_perf_controls.py`.
- Installed ANTsPy/antspyx smoke tests confirm `robust=0` can run, and `robust=1` falls back through the Python rank path when ANTsPy raises `NotImplementedError`.

### What Is Not Yet Verified

- `tests/test_sccan_r_comparison.py` does not call Python `lsm_sccan()`. It mainly validates R fixture self-consistency.
- There is no strict test that runs Python SCCAN on the same input as R and compares `eig1`, statistic map, calibrated predictions, or CV-selected sparseness.
- The R generator script contains prediction fixture generation, but current fixtures do not appear to include all generated prediction outputs such as `test4_predictions.csv`.

### Robust Behavior

R LESYMAP default SCCAN passes `robust=1` into ANTsR `sparseDecom2()`, where it means rank-transforming input matrices.

Python ANTsPy/antspyx exposes `robust`, but current versions raise:

```text
NotImplementedError: robust > 0 not currently implemented
```

The Python port now uses this policy:

- Try backend `robust=1`.
- If backend says robust is not implemented, rank-transform lesion and behavior columns using average ranks, z-score them, then call ANTsPy with `robust=0`.
- Record this in `model_params` as fallback metadata.

This is a practical approximation, not proven bitwise equivalence to ANTsRCore. For binary lesion columns, rank-z and ordinary z-score are effectively the same for nonconstant columns. For continuous behavior, rank transform can change spacing and therefore model weights.

For binary behavior `0/1`, rank transform is also nearly equivalent to a two-level z-score: all 0s receive one rank and all 1s receive another rank. Therefore robust behavior ranking should have little effect for mutism labels, except through class imbalance and tie handling.

### SCCAN Prediction

R `lsm_sccan()` saves:

- raw SCCAN weights,
- `eig2`,
- behavior center/scale,
- lesion matrix center/scale,
- a linear calibration model.

R `lesymap.predict()` then:

1. extracts lesion voxels with the training mask,
2. scales new lesion data with saved lesion center/scale,
3. computes `lesmat %*% eig1 %*% eig2`,
4. reverses behavior scaling,
5. applies the saved linear calibration model.

Python `_predict_sccan()` follows the same formula shape. The key missing validation is a checkpoint/predict roundtrip test that compares direct predictions with loaded-checkpoint predictions and, ideally, R-generated calibrated predictions.

Important risk: default Python pipeline uses patch compression. If SCCAN stores patch-space weights but prediction extracts full-mask voxel vectors, prediction may fail or be semantically wrong unless weights/centers are expanded back to voxel space or prediction is restricted to `no_patch=True`.

## SVR / SVM

### What Is Verified

`tests/test_svr_r_comparison.py` verifies that direct sklearn SVR runs are highly correlated with R-generated intermediate fixtures:

- linear kernel predictions correlate with R fixture predictions,
- linear kernel weights correlate with R fixture weights,
- linear behavior correlation is close to R fixture correlation,
- RBF predictions correlate with R fixture predictions,
- RBF behavior correlation is close to R fixture correlation.

This is useful as a libsvm sanity check.

### What Is Not Yet Verified

The SVR comparison test does not call `lesymap.methods.multivariate.lsm_svr()`. It directly instantiates `sklearn.svm.SVR` on R pre-scaled data.

The R fixture generator also hand-runs `e1071::svm(...)` instead of calling the real R `lsm_svr()` pipeline. Therefore current tests do not verify:

- Python `lsm_svr()` end-to-end behavior,
- Python `LesymapResult` output maps,
- Python checkpoint/predict roundtrip,
- R `statistic` scaling,
- R permutation p-values,
- NIfTI mask extraction and voxel mapping.

### Known Differences From R

R `lsm_svr()` defaults:

```text
kernel = radial
cost = 30
gamma = 5
epsilon = 0.1
scale lesmat and behavior before fitting
```

Python `lsm_svr()` defaults:

```text
kernel = linear
C = 1.0
epsilon = 0.1
no internal R-style scaling
```

R computes weights as `t(coefs) %*% SV` and scales statistic by `10 / max(abs(w))`. Python linear SVR uses `svr.coef_` directly, and nonlinear SVR uses permutation importance with an explicit `max_features` limit. These are not the same statistic semantics.

Conclusion: current Python SVR is a useful Python implementation, but it should not yet be described as R-equivalent.

## Binary Behavior Evaluation

For mutism labels `0/1`, SCCAN and SVR should be evaluated as continuous score models first, not as calibrated probability models.

Recommended evaluation:

- report ROC-AUC,
- report PR-AUC, especially if positive mutism cases are rare,
- report sensitivity, specificity, balanced accuracy, F1, MCC, and confusion matrix,
- choose thresholds only inside the training fold, not on the test fold,
- consider threshold strategies such as Youden J, max MCC, or fixed-sensitivity target,
- if probabilities are needed, calibrate using logistic/Platt or isotonic calibration inside training folds.

A fixed `0.5` threshold is only reasonable if the output has been calibrated to probability scale. SCCAN's linear calibrated output and SVR's raw prediction are not guaranteed to be probabilities and can be outside `[0, 1]`.

## Recommended Next Validation Work

Priority 0:

- Add SCCAN prediction roundtrip test: direct result predict vs `save_checkpoint()` / `load_checkpoint()` predict.
- Verify/fix prediction with patch compression. Either expand patch-space parameters back to voxel space or document/enforce `no_patch=True` for predictive SCCAN/SVR.
- Add SVR prediction roundtrip test using `lsm_svr()` rather than direct sklearn.

Priority 1:

- Generate true R `lsm_sccan()` prediction fixtures and compare Python formula/calibrated predictions.
- Generate true R `lsm_svr()` fixtures by calling R `lsm_svr()`, not hand-coded `e1071::svm()`.
- Decide whether Python SVR should support an `r_compatible=True` mode with R-style scaling, R default parameters, and R statistic scaling.

Priority 2:

- Add binary behavior evaluation tests/benchmarks for ROC-AUC, PR-AUC, F1, MCC, balanced accuracy.
- Add fixture manifest tests so generated R reference files and Python tests do not drift.
- Update `VALIDATION_SUMMARY.md` wording: SCCAN is currently property/algorithm-level validated, not strict output-equivalence validated.

## Bottom Line

For the current project goal, SCCAN is the more plausible path for R-like predictive behavior, but it still needs strict prediction validation and patch-aware prediction handling before we can claim production-level R compatibility.

SVR/SVM currently has weaker R compatibility. It can be used as a Python model, but matching R LESYMAP would require explicit scaling/default/statistic changes and new end-to-end R fixture tests.
