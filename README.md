# LESYMAP-Python

A Python implementation of [LESYMAP](https://github.com/dorianps/LESYMAP) — lesion-symptom mapping analysis for neuroimaging research.

LESYMAP identifies brain regions associated with cognitive deficits by analyzing lesion maps (NIfTI images) and behavioral scores from stroke patients. This Python port implements the core statistical methods from the original R package with enhanced performance through Numba JIT compilation, and adds a model checkpointing and prediction API.

> **Original R package:** [dorianps/LESYMAP](https://github.com/dorianps/LESYMAP) by Dorian Pustina et al. (Penn Memory Center, University of Pennsylvania)

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
| Model saving | R `.rds` format | pickle/joblib checkpoint |
| Prediction API | `lesymap.predict()` | `result.predict()` |

## Citation

If you use this software in research, please cite the original LESYMAP paper:

> Pustina, D., Coslett, H. B., Turkeltaub, P. E., Tustison, N., Schwartz, M. F., & Avants, B. (2016). Automated segmentation of chronic stroke lesions using LINDA: Lesion identification with neighborhood data analysis. *Human Brain Mapping*, 37(4), 1405–1421.

and acknowledge the original R implementation:

> Pustina, D. et al. LESYMAP R package. https://github.com/dorianps/LESYMAP

## License

MIT
