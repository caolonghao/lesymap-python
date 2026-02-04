# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LESYMAP is an R package for lesion-symptom mapping analysis in neuroimaging research. It maps brain areas responsible for cognitive deficits by analyzing lesion maps (NIfTI images) and behavioral scores from stroke patients. The package supports both univariate (voxel-wise) and multivariate statistical methods.

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
