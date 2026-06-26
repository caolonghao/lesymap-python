# LESYMAP-Python

A Python implementation of [LESYMAP](https://github.com/dorianps/LESYMAP) — lesion-symptom mapping analysis for neuroimaging research.

LESYMAP identifies brain regions associated with cognitive deficits by analyzing lesion maps (NIfTI images) and behavioral scores from stroke patients. This Python port implements the core statistical methods from the original R package with enhanced performance through Numba JIT compilation, and adds a model checkpointing and prediction API.

> **Original R package:** [dorianps/LESYMAP](https://github.com/dorianps/LESYMAP) by Dorian Pustina et al. (Penn Memory Center, University of Pennsylvania)

## Prerequisites: Template Space Registration

LESYMAP requires all lesion maps to be registered to a standard template space (MNI152) before analysis. If your lesion maps are already in MNI space, you can skip this step.

```python
from lesymap.core.registration import register_lesion_to_template, register_batch

# Single subject
result = register_lesion_to_template(
    subject_anatomical='sub-001_T1w.nii.gz',
    subject_lesion='sub-001_lesion.nii.gz',
    skull_strip=True,
    type_of_transform='SyN',       # 'SyNCC' for higher accuracy (~2h)
    output_prefix='output/sub-001',
)
lesion_mni = result['lesion_template']

# Batch
registered_lesions, _ = register_batch(
    subject_anatomicals=['sub-001_T1w.nii.gz', 'sub-002_T1w.nii.gz'],
    subject_lesions=['sub-001_lesion.nii.gz', 'sub-002_lesion.nii.gz'],
    output_dir='output/registered/',
)
```

**Template:** MNI152 2009c templates are stored in this repo via [Git LFS](https://git-lfs.com/) and downloaded automatically with `git clone`. Registration requires ANTsPy (`pip install antspyx`).

See `examples/register_lesions.py` for a full walkthrough.

## Installation

```bash
pip install -e .
```

For SCCAN support (requires ANTsPy):

```bash
pip install -e .[sccan]
```

## Quick Start

```python
import lesymap

# Run Brunner-Munzel voxel-wise analysis
result = lesymap.lesymap(
    lesions=['sub01_lesion.nii.gz', 'sub02_lesion.nii.gz', ...],
    behavior=[1.2, 3.4, 2.1, ...],
    method='BMfast',
    multiple_comparison='fdr',
    p_threshold=0.05,
)

# Save NIfTI statistical maps
result.save('output/')
```

```python
# SCCAN multivariate analysis with sparseness optimization
result = lesymap.lesymap(
    lesions=lesion_files,
    behavior=behavior_scores,
    method='sccan',
    optimize_sparseness=True,
)

# Predict on new subjects
predictions = result.predict(['new_subject.nii.gz'])
```

## Predictive Modeling Workflow

For R-like predictive LESYMAP workflows, use SCCAN as the primary method and
SVR with `r_compatible=True` when migrating or comparing against the R package.

### SCCAN Training and Inference

```python
import lesymap

result = lesymap.lesymap(
    lesions=train_lesion_files,
    behavior=train_behavior_scores,
    mask=roi_mask_file,              # optional, but recommended for fixed ROI work
    method='sccan',
    sparseness=0.045,                # fixed sparseness for reproducible runs
    optimize_sparseness=False,       # or True for cross-validation search
    robust=1,                        # matches R LESYMAP's robust SCCAN intent
)

result.save('sccan_model/', save_model=True)

model = lesymap.LesymapResult.load_checkpoint(
    'sccan_model/model_checkpoint.pkl'
)
predicted_scores = model.predict(test_lesion_files)
```

SCCAN predictions are continuous behavior scores. For binary outcomes such as
mutism `0/1`, treat these as risk scores unless you fit a separate probability
calibration model.

### R-Compatible SVR Training and Inference

```python
result = lesymap.lesymap(
    lesions=train_lesion_files,
    behavior=train_behavior_scores,
    mask=roi_mask_file,
    method='svr',
    r_compatible=True,               # R-style scaling and R default SVR params
)

result.save('svr_model/', save_model=True)

model = lesymap.LesymapResult.load_checkpoint(
    'svr_model/model_checkpoint.pkl'
)
predicted_scores = model.predict(test_lesion_files)
```

With `r_compatible=True`, SVR uses the R LESYMAP defaults unless overridden:
RBF kernel, `C=30`, `gamma=5`, `epsilon=0.1`, R-style centering/scaling of the
lesion matrix and behavior, and prediction unscaling back to behavior units.

R-style SVR permutation p-value maps are available, but they are intentionally
opt-in because they refit the SVR once per permutation:

```python
result = lesymap.lesymap(
    lesions=train_lesion_files,
    behavior=train_behavior_scores,
    mask=roi_mask_file,
    method='svr',
    r_compatible=True,
    svr_pvalue_method='r_permutation',
    nperm=1000,
    random_state=123,
)
```

Exact permutation p-values are random-sequence dependent. The Python
implementation follows the R directional exceedance rule, but NumPy and R do
not produce identical permutation sequences unless those sequences are saved
and replayed.

### Binary Outcome Evaluation

```python
from lesymap.utils.metrics import evaluate_binary_predictions

metrics = evaluate_binary_predictions(
    y_true=heldout_binary_labels,
    scores=predicted_scores,
    threshold='youden',              # choose thresholds inside training folds
)
```

Do not threshold SCCAN/SVR scores at `0.5` unless you have calibrated them to
probabilities. Prefer ROC-AUC, PR-AUC, balanced accuracy, F1, MCC, sensitivity,
and specificity, with threshold selection performed inside the training data or
cross-validation folds.

## Methods

### Univariate (voxel-wise, patch-accelerated)

| Method | Description |
|--------|-------------|
| `BMfast` | Brunner-Munzel non-parametric test |
| `ttest` | Student's t-test |
| `welch` | Welch's t-test (unequal variance) |
| `regresfast` | Linear regression (supports covariates) |
| `chisq` | Chi-square test (binary outcomes) |

### Multivariate

| Method | Description |
|--------|-------------|
| `sccan` | Sparse Canonical Correlation Analysis (requires ANTsPy) |
| `svr` | Support Vector Regression |

### Multiple Comparison Correction

`fdr`, `bonferroni`, `holm`, `BH`, `BY`, `FWERperm`, `clusterPerm`, `none`

## Result Object

```python
result.stat_img        # statistical map (nibabel image)
result.pval_img        # p-value map
result.zmap_img        # z-score map
result.raw_weights_img # raw model weights (SCCAN/SVR)

result.save('output/', save_model=True)          # save maps + model
result.save_checkpoint('model.pkl')              # model only
result.predict(new_lesion_files)                 # inference
lesymap.LesymapResult.load_checkpoint('model.pkl')  # reload
```

## Lesion Size Correction

```python
result = lesymap.lesymap(
    ...,
    correct_by_les_size='voxel',   # 'none' | 'voxel' | 'behavior' | 'both'
)
```

## Differences from the R Version

| Feature | R (original) | Python (this package) |
|---------|-------------|----------------------|
| Performance | C++/Rcpp | Numba JIT |
| Compilation | C++ toolchain required | None |
| SCCAN | ANTsR (required) | ANTsPy (optional) |
| SVR R compatibility | e1071 defaults | `r_compatible=True` mode |
| Model saving | R `.rds` format | pickle/joblib checkpoint |
| Prediction API | `lesymap.predict()` | `result.predict()` |

See `MULTIVARIATE_R_COMPATIBILITY_REPORT.md` for the current SCCAN/SVR
R-compatibility validation status and tested limitations.

## Citation

If you use this software in research, please cite the original LESYMAP paper:

> Pustina, D., Coslett, H. B., Turkeltaub, P. E., Tustison, N., Schwartz, M. F., & Avants, B. (2016). Automated segmentation of chronic stroke lesions using LINDA: Lesion identification with neighborhood data analysis. *Human Brain Mapping*, 37(4), 1405–1421.

and acknowledge both the original R implementation and this Python port:

> Pustina, D. et al. LESYMAP R package. https://github.com/dorianps/LESYMAP

> LESYMAP-Python. https://github.com/caolonghao/lesymap-python

## License

MIT
