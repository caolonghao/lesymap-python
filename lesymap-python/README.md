# LESYMAP-Python

Lesion-Symptom Mapping for Neuroimaging Research in Python.

A Python implementation of [LESYMAP](https://github.com/dorianps/LESYMAP) for mapping brain areas responsible for cognitive deficits by analyzing lesion maps (NIfTI images) and behavioral scores from stroke patients.

## Features

- **Univariate Methods:** Brunner-Munzel, Student's t-test, Welch's t-test, Linear Regression, Chi-square
- **Multivariate Methods:** SCCAN (Sparse Canonical Correlation Analysis), SVR (Support Vector Regression)
- **Patch-based Analysis:** Efficient computation by grouping voxels with identical lesion patterns
- **Multiple Comparison Correction:** FDR, Bonferroni, FWER permutation, Cluster permutation
- **Model Checkpointing:** Save and load trained models for inference on new data
- **Registration Pipeline:** ANTsPy-based registration to template space (coming soon)

## Installation

```bash
pip install lesymap
```

For SCCAN support, install with the sccan extra:

```bash
pip install lesymap[sccan]
```

Or install from source:

```bash
git clone https://github.com/yourusername/lesymap-python.git
cd lesymap-python
pip install -e .
```

### Dependencies

**Core (required):**
- numpy >= 1.21
- nibabel >= 5.0
- scipy >= 1.9
- statsmodels >= 0.13
- numba >= 0.56 (JIT compilation for performance)
- scikit-learn >= 1.0
- pandas >= 1.4
- joblib >= 1.2
- tqdm >= 4.64

**Optional (for SCCAN method):**
- antsyx >= 0.3.6 (or conda `antspy`)

## Quick Start

```python
import lesymap

# Prepare your data
lesion_files = ['subject1_lesion.nii.gz', 'subject2_lesion.nii.gz', ...]
behavior_scores = [1.2, 3.4, 2.1, ...]  # or path to CSV file

# Run SCCAN analysis
result = lesymap.lesymap(
    lesions=lesion_files,
    behavior=behavior_scores,
    method='sccan',
    multiple_comparison='fdr',
    p_threshold=0.05
)

# Save results
result.save('output_dir/', save_model=True)

# Make predictions on new data
new_lesions = ['new_subject.nii.gz']
predictions = result.predict(new_lesions)
print(f"Predicted behavior: {predictions}")
```

## Available Methods

### Univariate (Voxel-wise)

| Method | Description | Function |
|--------|-------------|----------|
| BMfast | Brunner-Munzel test (non-parametric) | `lsm_bmfast()` |
| ttest | Student's t-test (equal variance) | `lsm_ttest()` |
| welch | Welch's t-test (unequal variance) | `lsm_welch()` |
| regresfast | Linear regression | `lsm_regresfast()` |
| chisq | Chi-square test (binary outcomes) | `lsm_chisq()` |

### Multivariate (All voxels simultaneously)

| Method | Description | Function |
|--------|-------------|----------|
| SCCAN | Sparse Canonical Correlation Analysis | `lsm_sccan()` |
| SVR | Support Vector Regression | `lsm_svr()` |

## Examples

### SCCAN Analysis

```python
import lesymap

result = lesymap.lesymap(
    lesions=lesion_files,
    behavior=behavior_scores,
    method='sccan',
    optimize_sparseness=True,  # Auto-optimize sparseness via CV
    multiple_comparison='fdr',
)

# Access results
stat_map = result.stat_img  # Z-scores
weights = result.sccan_weights  # Voxel weights
correlation = result.model_params['correlation']
```

### Brunner-Munzel Test with Permutation

```python
result = lesymap.lesymap(
    lesions=lesion_files,
    behavior=behavior_scores,
    method='BMfast',
    multiple_comparison='FWERperm',
    nperm=1000,
    p_threshold=0.05
)
```

### Lesion Size Correction

```python
result = lesymap.lesymap(
    lesions=lesion_files,
    behavior=behavior_scores,
    method='regresfast',
    correct_by_les_size='voxel',  # or 'behavior' or 'both'
)
```

## Result Object

The `LesymapResult` object provides:

- **Statistical maps:** `stat_img`, `pval_img`, `zmap_img`
- **Model weights:** `raw_weights_img`, `sccan_weights`
- **Prediction:** `predict(new_lesions)` for SCCAN, SVR, and regression
- **Save/Load:** `save()`, `save_checkpoint()`, `load_checkpoint()`

```python
# Save full results
result.save('output/', save_model=True)

# Save just the model checkpoint
result.save_checkpoint('model.pkl')

# Load and predict later
loaded = lesymap.LesymapResult.load_checkpoint('model.pkl')
predictions = loaded.predict(new_lesions)
```

## License

MIT License

## Citation

If you use this software, please cite the original LESYMAP paper:

> Pustina, D., et al. (2017). LESYMAP: A user-friendly, stand-alone MATLAB toolbox for lesion mapping. Frontiers in Neurology, 8.

## Acknowledgments

This is a Python port of the [R version of LESYMAP](https://github.com/dorianps/LESYMAP) by Dorian Pustina et al.
