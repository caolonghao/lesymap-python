# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LESYMAP is a lesion-symptom mapping analysis tool for neuroimaging research. It maps brain areas responsible for cognitive deficits by analyzing lesion maps (NIfTI images) and behavioral scores from stroke patients. The package supports both univariate (voxel-wise) and multivariate statistical methods.

**This repository contains two implementations:**
1. **LESYMAP/** - Original R package (production-ready, ~4,181 lines of R code)
2. **lesymap-python/** - Python port (in development, ~4,733 lines of Python code)

## Git Worktrees

This project uses git worktrees for managing multiple branches simultaneously. All worktrees should be created in the `.worktrees/` directory at the project root, which is automatically ignored by git.

**Worktree directory:** `.worktrees/`

**Example usage:**
```bash
# Create new worktree for feature branch
git worktree add .worktrees/feature-name -b feature-name

# List all worktrees
git worktree list

# Remove worktree when done
git worktree remove feature-name
```

---

# R Package (LESYMAP/)

## Build and Installation

### Installation
```r
if (! 'devtools' %in% installed.packages()) install.packages('devtools')
devtools::install_github('dorianps/LESYMAP')
```

This installs all dependencies including ANTsR (medical image processing library), which may take up to an hour.

### Building the Package
The package uses R's standard build system with C++ compilation:

```bash
# Build from source (in R)
R CMD build LESYMAP

# Install from built tarball
R CMD INSTALL LESYMAP_*.tar.gz
```

### Documentation Generation
Documentation uses roxygen2. To regenerate man pages from source:
```r
# In R, from package root
devtools::document()
```

### Testing
```r
# Run built-in example
library(LESYMAP)
example(lesymap)

# Check package (including tests)
devtools::check()
```

Travis CI configuration exists for continuous integration testing on Linux and macOS across multiple R versions.

## Architecture

### Directory Structure
- **`R/`** - R source code (~31 files, 4,181 lines)
- **`src/`** - C++ performance-critical code (~7 files, 934 lines)
- **`man/`** - Roxygen2-generated documentation
- **`inst/extdata/`** - Example data: lesions (131 NIfTI files), behavioral scores, brain templates

### Core Analysis Pipeline

1. **Input Validation** (`checkAntsInput`, `checkImageList`, `checkFilenameHeaders`)
   - Accepts antsImage objects, file paths, or 4D images
   - Validates spatial alignment across all lesion maps
   - Checks for binary/continuous voxel values

2. **Patch Computation** (`getUniqueLesionPatches`)
   - Groups voxels with identical lesion patterns across subjects
   - Reduces multiple comparisons and speeds up univariate analyses
   - Creates patch matrix (subjects × unique patterns)

3. **Statistical Analysis** (method-specific `lsm_*` functions)
   - Returns statistic, p-value, and z-score vectors
   - Handles multiple comparison correction

4. **Image Reconstruction** (`makeImage`, `antsImageWrite`)
   - Converts statistical vectors back to NIfTI brain images
   - Applies thresholding and masking

5. **Output Management** (`save.lesymap`, `print.lesymap`)
   - Saves statistical maps as NIfTI files
   - Generates visualization overlays
   - Creates detailed info files with analysis parameters

### Statistical Methods

**Univariate (voxel-wise, using patches):**
- `BM`/`BMfast` - Brunner-Munzel non-parametric test (default in MRIcron)
- `ttest`/`welch` - T-tests for group comparisons
- `regres`/`regresfast` - Linear regression (supports covariates via Freedman-Lane method)
- `chisq`/`chisqPerm` - Chi-square tests for binary outcomes

**Multivariate (all voxels simultaneously):**
- `sccan` (default) - Sparse Canonical Correlations with automatic sparseness optimization via cross-validation
- `svr` - Support Vector Regression (experimental, slow)

### Multiple Comparison Correction

**Standard methods:** fdr, BH, BY, bonferroni, holm, hochberg, hommel

**Permutation methods:**
- `FWERperm` - Family-wise error rate via permutations (works with BMfast, regresfast)
- `clusterPerm` - Cluster-based permutation thresholding
- `none` - No correction

### C++ Performance Modules

The `src/` directory contains compiled implementations for performance:
- `BMfast.cpp` / `BMfast2.cpp` - Brunner-Munzel test (~60x faster than pure R)
- `TTfast.cpp` - Fast t-test implementation
- `regresfast.cpp` - Fast regression with covariate support
- `BMperm.cpp` - Permutation-based p-values
- `RcppExports.cpp` - Rcpp bindings

The `src/Makevars` links against Rcpp, LAPACK, and BLAS.

### Key Functions

**Entry point:** `lesymap()` - Main user-facing function with 200+ lines of documentation

**Core analysis:** `lsm_sccan()`, `lsm_BMfast()`, `lsm_ttestFast()`, `lsm_regresfast()`, `lsm_svr()`

**Data processing:** `getUniqueLesionPatches()`, `registerLesionToTemplate()`, `getLesionLoad()`, `simulateBehavior()`

**Utilities:** `checkAntsInput()`, `optimize_SCCANsparseness()`, `createFolds()`, `save.lesymap()`, `lesymap.predict()`

### Dependencies

**Required:** R (>= 3.0), ANTsR

**Imports:** graphics, lmPerm, Rcpp, stats, utils

**Suggested:** nparcomp, e1071 (for SVR)

**LinkingTo:** Rcpp, RcppArmadillo (linear algebra)

**NeedsCompilation:** yes

### Lesion Size Correction

Options via `correctByLesSize` parameter:
- `none` - No correction (default)
- `voxel` - Divide voxel values by 1/sqrt(lesionsize) - Mirman (2015), Zhang (2014) method
- `behavior` - Residualize behavioral scores by lesion size
- `both` - Both voxel and behavior residualization

### Important Implementation Notes

- Lesion maps must be registered to template space before analysis (use `registerLesionToTemplate()`)
- Patch-based analysis significantly reduces computation (e.g., 1.7x fewer voxels)
- SCCAN scales and centers both lesion and behavior data before running (hardcoded in `lsm_sccan`)
- Permutation testing is computationally expensive but provides stronger FWER control
- The package logs all operations with timestamps via `printInfo()` function
- Version checking happens on package load (via `.onLoad()` in `zzz.R`)

### Output Structure

Lesymap returns a list containing:
- Statistical maps (z-scores, p-values, raw statistics)
- Analysis parameters and metadata
- For SCCAN: raw weights, eig2, ccasummary
- Printed output captured in `printedOutput` variable

## Detailed Documentation

### SCCAN Method Reference

For an in-depth understanding of the SCCAN method, see **`SCCAN原理详解.md`** which covers:

1. **SCCAN Correlation Calculation** - How cross-validation correlation is computed
2. **Why Not AUC** - Why SCCAN uses correlation instead of AUC for regression
3. **SCCAN Prediction Principle** - How SCCAN performs predictions via weighted lesion scores
4. **Linear Calibration** - Why linear calibration is needed after SCCAN
5. **sparseDecom2 Function** - Core ANTsR function that executes SCCAN
6. **Binary Classification & AUC** - Using SCCAN for binary outcomes, high AUC/low F1 phenomenon
7. **Overfitting in LSM** - Why tree-based methods (xgboost, random forest) severely overfit
8. **Sparse Constraints** - Difference between input data sparsity vs model weight sparsity
9. **lesmat vs mask** - The distinction and purpose of each parameter

Key insights from the detailed documentation:
- **SCCAN optimizes correlation, not prediction accuracy** - Linear calibration (`lm(true ~ pred)`) corrects for this
- **AUC only measures ranking** - High AUC with low F1 is common in imbalanced data; use PR AUC, F1, MCC together
- **Tree methods severely overfit** - In high-dimensional small-sample data (p >> n), xgboost/random forest overfit badly; SVR/SCCAN preferred
- **Sparse constraint controls model weights** - The `sparseness` parameter (e.g., 0.045) controls what percentage of voxels get non-zero weights, not the input lesion mask sparsity
- **lesmat stores data, mask provides space** - `lesmat` is the analysis matrix (patients × voxels), `mask` defines the spatial framework and ROI

**Quick Reference for Common Issues:**

| Issue | Solution |
|-------|----------|
| High AUC but low F1 | Check prediction value separation (Cohen's d); use PR AUC instead of ROC AUC |
| Tree methods overfitting | Use SCCAN or SVR instead; or apply extreme regularization and dimensionality reduction |
| Understanding sparseness | See section 9: distinguishes input sparsity (lesion mask) from model weight sparsity |
| lesmat vs mask confusion | See section 10: lesmat = data matrix, mask = spatial template/ROI |

---

# Python Package (lesymap-python/)

A modern Python port of LESYMAP with enhanced features including model checkpointing, prediction API, and improved performance through JIT compilation.

## Installation (Python Version)

### From Source
```bash
cd lesymap-python
pip install -e .
```

### With SCCAN Support
```bash
pip install -e .[sccan]  # Installs ANTsPy for SCCAN method
```

### Development Installation
```bash
pip install -e .[dev]  # Includes pytest, black, ruff, mypy
```

## Architecture (Python Version)

### Directory Structure
```
lesymap-python/
├── lesymap/              # Main package (18 Python modules, ~4,733 lines)
│   ├── core/             # Core pipeline and I/O (6 modules)
│   │   ├── pipeline.py   # Main run_lesymap() function (471 lines)
│   │   ├── result.py     # LesymapResult class with save/load/predict
│   │   ├── io.py         # NIfTI loading and validation
│   │   ├── image_utils.py # Image operations and masking
│   │   ├── patch.py      # Patch computation
│   │   └── __init__.py
│   ├── methods/          # Statistical methods (4 modules)
│   │   ├── univariate.py # BMfast, t-test, regression, chi-square
│   │   ├── multivariate.py # SCCAN, SVR
│   │   ├── correction.py # Multiple comparison correction
│   │   └── __init__.py
│   ├── stats_compiled/   # Performance-critical code (4 modules)
│   │   ├── bm.py         # Brunner-Munzel (Numba JIT)
│   │   ├── ttest.py      # Fast t-test (Numba JIT)
│   │   ├── regression.py # Fast regression (Numba JIT)
│   │   └── __init__.py
│   ├── utils/            # Utilities (3 modules)
│   │   ├── validation.py # Input validation
│   │   ├── masking.py    # Mask operations
│   │   └── __init__.py
│   └── __init__.py       # Main lesymap() entry point
├── tests/                # Unit tests
├── examples/             # Example scripts
├── setup.py              # Package configuration
├── pyproject.toml        # Modern Python packaging
└── README.md             # User documentation
```

### Core Pipeline (Python)

The Python implementation follows a similar pipeline to the R version:

1. **Input Validation & Loading** (`core/io.py`)
   - `check_input_type()` - Detect input format (file paths, nibabel images, 4D arrays)
   - `load_lesions()` - Load NIfTI files with header validation
   - `load_behavior()` - Load behavioral scores from array or CSV
   - `check_binary_values()` - Validate binary lesion maps (0/1 or 0/255)

2. **Mask Preparation** (`core/image_utils.py`)
   - `mask_from_average()` - Auto-generate mask from average lesion map
   - `images_to_matrix()` - Convert images to subjects × voxels matrix
   - `get_lesion_load()` - Calculate lesion volumes per subject

3. **Patch Computation** (`core/patch.py`)
   - `get_unique_lesion_patches()` - Group identical lesion patterns
   - `filter_patches_by_prevalence()` - Filter by min subjects per voxel
   - `patches_to_voxels()` - Convert patch statistics back to voxel space

4. **Statistical Analysis** (`methods/`)
   - Univariate: `lsm_bmfast()`, `lsm_ttest()`, `lsm_welch()`, `lsm_regresfast()`, `lsm_chisq()`
   - Multivariate: `lsm_sccan()`, `lsm_svr()`

5. **Multiple Comparison Correction** (`methods/correction.py`)
   - `correct_pvalues()` - FDR, Bonferroni, Holm, etc.
   - `fwer_permutation_threshold()` - Family-wise error rate via permutation
   - `cluster_permutation_threshold()` - Cluster-based permutation

6. **Result Generation** (`core/result.py`)
   - `LesymapResult` class with statistical maps
   - `save()` - Save NIfTI maps and analysis info
   - `save_checkpoint()` / `load_checkpoint()` - Model serialization
   - `predict()` - Inference on new lesion maps

### Key Differences from R Version

**Python-specific enhancements:**

1. **Model Checkpointing & Prediction API**
   - Save trained models for later inference
   - `result.predict(new_lesions)` for SCCAN, SVR, regression
   - Linear calibration for SCCAN predictions

2. **Performance Optimization**
   - Numba JIT compilation for critical statistics (BMfast, t-test, regression)
   - No C++ compilation required (pure Python + Numba)
   - NumPy vectorization throughout

3. **Modern Python Packaging**
   - `pyproject.toml` for build configuration
   - Type hints throughout codebase
   - scikit-learn-style API

4. **Improved Validation**
   - Comprehensive parameter validation with helpful error messages
   - Automatic binary format detection (0/1 vs 0/255)
   - Input type auto-detection

5. **Dependencies**
   ```python
   # Core (required)
   numpy >= 1.21
   nibabel >= 5.0
   scipy >= 1.9
   statsmodels >= 0.13
   numba >= 0.56        # JIT compilation
   scikit-learn >= 1.0
   pandas >= 1.4
   joblib >= 1.2
   tqdm >= 4.64

   # Optional
   antspyx >= 0.3.6     # For SCCAN method
   ```

### Statistical Methods (Python)

**Univariate (patches or voxels):**
- `lsm_bmfast()`     - Brunner-Munzel test (Numba-accelerated)
- `lsm_ttest()`      - Student's t-test (equal variance)
- `lsm_welch()`      - Welch's t-test (unequal variance)
- `lsm_regresfast()` - Linear regression with covariates
- `lsm_chisq()`      - Chi-square test (binary outcomes)

**Multivariate (all voxels):**
- `lsm_sccan()`      - Sparse Canonical Correlation Analysis
  - Uses ANTsPy's `sparseDecom2` function
  - Auto-optimizes sparseness via cross-validation
  - Includes prediction with linear calibration
- `lsm_svr()`        - Support Vector Regression
  - Uses scikit-learn's SVR
  - Model checkpointing supported

### Multiple Comparison Correction (Python)

**Standard methods:**
- `'fdr'` - False discovery rate (Benjamini-Hochberg)
- `'bonferroni'` - Bonferroni correction
- `'holm'` - Holm-Bonferroni
- `'BH'`, `'BY'`, `'hochberg'`, `'hommel'` - Other standard methods

**Permutation methods:**
- `'FWERperm'` - Family-wise error rate via permutation
- `'clusterPerm'` - Cluster-based permutation
- `'none'` - No correction

### Result Object (Python)

The `LesymapResult` class provides:

**Attributes:**
- `stat_img` - Statistical map (z-scores, t-statistics)
- `pval_img` - P-value map
- `zmap_img` - Z-score map (after correction)
- `raw_weights_img` - Raw model weights (SCCAN, SVR)
- `mask_img` - Analysis mask
- `method` - Method name
- `model_params` - Method-specific parameters
- `callinfo` - Analysis metadata

**Methods:**
- `save(output_dir, save_model=True)` - Save all results and model
- `save_checkpoint(filename)` - Save model for inference only
- `load_checkpoint(filename)` - Load saved model (class method)
- `predict(new_lesions)` - Predict on new data (SCCAN, SVR, regression)

**Example workflow:**
```python
# Train and save
result = lesymap.lesymap(lesions, behavior, method='sccan')
result.save('output/', save_model=True)

# Later: load and predict
model = lesymap.LesymapResult.load_checkpoint('output/model_checkpoint.pkl')
predictions = model.predict(['new_subject.nii.gz'])
```

### Lesion Size Correction (Python)

Same options as R version via `correct_by_les_size` parameter:
- `'none'` - No correction (default)
- `'voxel'` - Divide voxel values by 1/√(lesion_size)
- `'behavior'` - Residualize behavioral scores by lesion size
- `'both'` - Both corrections

### Testing (Python)

```bash
# Run unit tests
cd lesymap-python
pytest tests/

# With coverage
pytest --cov=lesymap tests/

# Run specific test
pytest tests/test_stats.py::test_bmfast
```

### Code Quality Tools

```bash
# Format code
black lesymap/

# Lint
ruff check lesymap/

# Type checking
mypy lesymap/
```

## Comparison: R vs Python

| Feature | R Version | Python Version |
|---------|-----------|----------------|
| **Code size** | ~4,181 lines (31 files) | ~4,733 lines (23 files) |
| **Performance** | C++/Rcpp (BMfast, regression) | Numba JIT (pure Python) |
| **Compilation** | Requires C++ toolchain | No compilation needed |
| **SCCAN** | ANTsR required | ANTsPy optional |
| **Model saving** | save.lesymap() (R format) | Checkpoint API (pickle/joblib) |
| **Prediction** | lesymap.predict() (R function) | result.predict() (method) |
| **Validation** | Basic checks | Comprehensive with error messages |
| **Package format** | R package (CRAN-style) | Python package (PyPI-style) |
| **Dependencies** | ANTsR, Rcpp, RcppArmadillo | NumPy, SciPy, Numba, scikit-learn |
| **Testing** | Travis CI | pytest with coverage |

## Quick Start (Python)

```python
import lesymap

# Basic SCCAN analysis
result = lesymap.lesymap(
    lesions=['sub1.nii.gz', 'sub2.nii.gz', ...],
    behavior=[1.2, 3.4, 2.1, ...],
    method='sccan',
    optimize_sparseness=True,  # Auto-optimize via CV
    multiple_comparison='fdr',
    p_threshold=0.05
)

# Save results
result.save('output_dir/', save_model=True)

# Make predictions
new_predictions = result.predict(['new_subject.nii.gz'])
print(f"Predicted score: {new_predictions[0]:.2f}")
```

## Important Implementation Notes (Python)

1. **Input formats supported:**
   - List of file paths: `['sub1.nii.gz', 'sub2.nii.gz']`
   - List of nibabel images: `[img1, img2]`
   - 4D nibabel image: `nib.load('all_subjects.nii.gz')`

2. **Binary format auto-detection:**
   - Automatically detects 0/255 format and converts to 0/1
   - Set `binary_check=False` to skip validation

3. **SCCAN implementation:**
   - Requires ANTsPy: `pip install antspyx`
   - Uses `sparseDecom2` from ANTsPy
   - Includes linear calibration for predictions
   - See `SCCAN原理详解.md` for detailed explanation

4. **Performance:**
   - Numba JIT compilation on first run (may be slow)
   - Subsequent runs are fast (~60x speedup for BMfast)
   - Use `no_patch=True` to skip patch computation if needed

5. **Prediction workflow:**
   - Only available for SCCAN, SVR, and regression methods
   - Requires `save_model=True` when saving
   - New lesions must be in same space as training data

6. **Progress tracking:**
   - Set `show_info=True` for detailed progress (default)
   - Uses timestamped logging like R version
   - Shows compression ratio, voxel counts, etc.

## Development Status (Python)

**Implemented:**
- ✅ Core pipeline (input validation, masking, patches)
- ✅ Univariate methods (BMfast, t-test, Welch, regression, chi-square)
- ✅ Multivariate methods (SCCAN, SVR)
- ✅ Multiple comparison correction (FDR, Bonferroni, permutation)
- ✅ Lesion size correction (voxel, behavior, both)
- ✅ Model checkpointing and prediction API
- ✅ Comprehensive validation and error messages
- ✅ Unit tests for statistical methods

**In Progress / Future:**
- 🚧 Registration pipeline (ANTsPy integration)
- 🚧 Full test coverage (currently ~60%)
- 🚧 Performance benchmarking vs R version
- 🚧 Documentation website (Sphinx)
- 🚧 Example datasets
- 🚧 Jupyter notebook tutorials

**Not Yet Implemented:**
- ❌ Some advanced R features (exact equivalence TBD)
- ❌ Visualization utilities (overlay generation)
- ❌ Batch processing CLI tool
