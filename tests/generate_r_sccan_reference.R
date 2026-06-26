#!/usr/bin/env Rscript
# Generate SCCAN reference data from R LESYMAP for Python validation
#
# This script runs lsm_sccan() with fixed parameters and saves results
# that can be compared with the Python implementation.
#
# Usage (Docker):
# docker run --rm \
#   -v /Users/tonycao/mycode/lesymap-python/tests/fixtures/r_reference_results:/data/r_reference_results \
#   -v /Users/tonycao/mycode/lesymap-python/tests:/scripts \
#   -v /Users/tonycao/mycode/lesymap-python/LESYMAP:/lesymap \
#   dorianps/lesymap Rscript /scripts/generate_r_sccan_reference.R

library(LESYMAP)
library(ANTsR)

cat("=== SCCAN Reference Data Generation ===\n")
cat("R version:", R.version.string, "\n")
cat("LESYMAP version:", as.character(packageVersion("LESYMAP")), "\n")
cat("ANTsR version:", as.character(packageVersion("ANTsR")), "\n\n")

# Output directory
output_dir <- "/data/r_reference_results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

run_sccan_cv <- identical(Sys.getenv("RUN_SCCAN_CV"), "1")

# Load example data from LESYMAP
lesydata <- file.path(find.package('LESYMAP'), 'extdata')
filenames <- Sys.glob(file.path(lesydata, 'lesions', '*.nii.gz'))
behavior_file <- file.path(lesydata, 'behavior', 'behavior.txt')
behavior <- read.table(behavior_file, header = FALSE)[, 1]

cat("Number of subjects:", length(filenames), "\n")
cat("Behavior range:", range(behavior), "\n\n")

# Create mask from average lesion image
cat("Creating mask from average lesion...\n")
avg <- antsAverageImages(filenames)
mask <- thresholdImage(avg, 0.1, Inf)
n_voxels <- sum(as.array(mask) > 0)
cat("Mask voxels:", n_voxels, "\n")

# Convert lesions to matrix
cat("Creating lesion matrix...\n")
lesmat <- imagesToMatrix(filenames, mask)
cat("Lesion matrix shape:", dim(lesmat), "\n\n")

# ============================================================
# TEST 1: Fixed sparseness without optimization
# ============================================================
cat("=== Test 1: Fixed sparseness (0.1) without optimization ===\n")

fixed_sparseness <- 0.1
result1 <- lsm_sccan(
  lesmat = lesmat,
  behavior = behavior,
  mask = mask,
  optimizeSparseness = FALSE,
  validateSparseness = FALSE,
  sparseness = fixed_sparseness,
  showInfo = TRUE,
  cthresh = 150,
  its = 20,
  smooth = 0.4,
  robust = 1,
  mycoption = 1,
  maxBased = FALSE,
  directionalSCCAN = TRUE
)

cat("\n--- Results ---\n")
cat("Non-zero weights:", sum(result1$statistic != 0), "\n")
cat("Weight range:", range(result1$statistic), "\n")
cat("eig2 value:", result1$sccan.eig2[1,1], "\n")

# Save raw weights as NIfTI
antsImageWrite(result1$rawWeights.img, file.path(output_dir, "test1_raw_weights.nii.gz"))

# Save normalized statistics
stat_img <- makeImage(mask, result1$statistic)
antsImageWrite(stat_img, file.path(output_dir, "test1_statistic.nii.gz"))

# Save scaling parameters and other metadata
metadata1 <- list(
  sparseness = fixed_sparseness,
  behavior_scaleval = result1$sccan.behavior.scaleval,
  behavior_centerval = result1$sccan.behavior.centerval,
  lesmat_scaleval = result1$sccan.lesmat.scaleval,
  lesmat_centerval = result1$sccan.lesmat.centerval,
  eig2 = as.numeric(result1$sccan.eig2),
  ccasummary_corr = result1$sccan.ccasummary$corrs[1],
  nonzero_weights = sum(result1$statistic != 0),
  weight_range = range(result1$statistic),
  predictlm_intercept = coef(result1$sccan.predictlm)[1],
  predictlm_slope = coef(result1$sccan.predictlm)[2]
)
saveRDS(metadata1, file.path(output_dir, "test1_metadata.rds"))

# Save as CSV for easier Python reading
write.csv(
  data.frame(
    sparseness = fixed_sparseness,
    behavior_scaleval = result1$sccan.behavior.scaleval,
    behavior_centerval = result1$sccan.behavior.centerval,
    eig2 = result1$sccan.eig2[1,1],
    ccasummary_corr = result1$sccan.ccasummary$corrs[1],
    nonzero_weights = sum(result1$statistic != 0),
    predictlm_intercept = coef(result1$sccan.predictlm)[1],
    predictlm_slope = coef(result1$sccan.predictlm)[2]
  ),
  file.path(output_dir, "test1_metadata.csv"),
  row.names = FALSE
)

# Save raw eig1 weights (before normalization/thresholding)
eig1_raw <- as.array(result1$rawWeights.img)[as.array(mask) > 0]
write.csv(data.frame(eig1 = eig1_raw), file.path(output_dir, "test1_eig1_raw.csv"), row.names = FALSE)

# Save statistic vector
write.csv(data.frame(statistic = result1$statistic), file.path(output_dir, "test1_statistic.csv"), row.names = FALSE)

if (run_sccan_cv) {
  # ============================================================
  # TEST 2: With sparseness validation (CV)
  # ============================================================
  cat("\n=== Test 2: Fixed sparseness (0.1) with CV validation ===\n")

  result2 <- lsm_sccan(
    lesmat = lesmat,
    behavior = behavior,
    mask = mask,
    optimizeSparseness = FALSE,
    validateSparseness = TRUE,
    sparseness = fixed_sparseness,
    showInfo = TRUE,
    cthresh = 150,
    its = 20,
    smooth = 0.4,
    robust = 1,
    mycoption = 1,
    maxBased = FALSE,
    directionalSCCAN = TRUE
  )

  cat("\n--- Results ---\n")
  cat("CV correlation:", result2$CVcorrelation.stat, "\n")
  cat("CV p-value:", result2$CVcorrelation.pval, "\n")

  # Save CV validation results
  metadata2 <- list(
    sparseness = fixed_sparseness,
    cv_correlation = result2$CVcorrelation.stat,
    cv_pvalue = result2$CVcorrelation.pval
  )
  saveRDS(metadata2, file.path(output_dir, "test2_metadata.rds"))

  write.csv(
    data.frame(
      sparseness = fixed_sparseness,
      cv_correlation = result2$CVcorrelation.stat,
      cv_pvalue = result2$CVcorrelation.pval
    ),
    file.path(output_dir, "test2_metadata.csv"),
    row.names = FALSE
  )
} else {
  cat("\n=== Test 2: CV validation skipped. Set RUN_SCCAN_CV=1 to enable. ===\n")
}

# ============================================================
# TEST 3: Data scaling verification
# ============================================================
cat("\n=== Test 3: Data scaling verification ===\n")

# Scale data manually as R does
behavior_scaled <- scale(behavior, scale = TRUE, center = TRUE)
lesmat_scaled <- scale(lesmat, scale = TRUE, center = TRUE)

cat("Behavior original mean:", mean(behavior), "\n")
cat("Behavior original sd:", sd(behavior), "\n")
cat("Behavior scaled mean:", mean(behavior_scaled), "\n")
cat("Behavior scaled sd:", sd(behavior_scaled), "\n")

# Save scaled data info
scaling_info <- data.frame(
  behavior_mean = mean(behavior),
  behavior_sd = sd(behavior),
  behavior_scale_attr = attr(behavior_scaled, "scaled:scale"),
  behavior_center_attr = attr(behavior_scaled, "scaled:center"),
  lesmat_col1_mean = mean(lesmat[,1]),
  lesmat_col1_sd = sd(lesmat[,1])
)
write.csv(scaling_info, file.path(output_dir, "test3_scaling_info.csv"), row.names = FALSE)

# Save first 10 rows of scaled lesmat for comparison
write.csv(
  data.frame(lesmat_scaled[1:10, 1:min(20, ncol(lesmat_scaled))]),
  file.path(output_dir, "test3_lesmat_scaled_sample.csv"),
  row.names = FALSE
)

# Save behavior data
write.csv(
  data.frame(
    behavior_orig = behavior,
    behavior_scaled = as.numeric(behavior_scaled)
  ),
  file.path(output_dir, "test3_behavior.csv"),
  row.names = FALSE
)

# ============================================================
# TEST 4: Prediction workflow
# ============================================================
cat("\n=== Test 4: Prediction workflow ===\n")

# Compute predictions as R does
# R: predbehav = lesmat %*% t(sccan$eig1) %*% sccan$eig2
eig1_matrix <- t(as.matrix(as.array(result1$rawWeights.img)[as.array(mask) > 0]))
predbehav_scaled <- lesmat_scaled %*% t(eig1_matrix) %*% result1$sccan.eig2

# Unscale predictions
predbehav_raw <- predbehav_scaled * result1$sccan.behavior.scaleval + result1$sccan.behavior.centerval

# Apply linear calibration
predbehav_calibrated <- predict(result1$sccan.predictlm, data.frame(predbehav.raw = predbehav_raw))

# Compute correlations
raw_corr <- cor(behavior, predbehav_raw)
calibrated_corr <- cor(behavior, predbehav_calibrated)

cat("Raw prediction correlation:", raw_corr, "\n")
cat("Calibrated prediction correlation:", calibrated_corr, "\n")

# Save prediction data
write.csv(
  data.frame(
    behavior_orig = behavior,
    pred_scaled = as.numeric(predbehav_scaled),
    pred_raw = as.numeric(predbehav_raw),
    pred_calibrated = as.numeric(predbehav_calibrated)
  ),
  file.path(output_dir, "test4_predictions.csv"),
  row.names = FALSE
)

prediction_metrics <- data.frame(
  raw_correlation = raw_corr,
  calibrated_correlation = calibrated_corr,
  lm_intercept = coef(result1$sccan.predictlm)[1],
  lm_slope = coef(result1$sccan.predictlm)[2]
)
write.csv(prediction_metrics, file.path(output_dir, "test4_metrics.csv"), row.names = FALSE)

# ============================================================
# Save mask for Python reference
# ============================================================
cat("\n=== Saving mask and lesion matrix ===\n")
antsImageWrite(mask, file.path(output_dir, "mask.nii.gz"))

# Full analysis inputs for Python-vs-R reruns. These are intentionally saved
# after the R outputs so tests can call Python lsm_sccan() on exactly the same
# matrix/behavior/mask that produced the reference maps.
write.csv(lesmat, gzfile(file.path(output_dir, "sccan_lesmat.csv.gz")), row.names = FALSE)
write.csv(
  data.frame(behavior = behavior),
  file.path(output_dir, "sccan_behavior.csv"),
  row.names = FALSE
)

# Save lesion matrix dimensions
write.csv(
  data.frame(
    n_subjects = nrow(lesmat),
    n_voxels = ncol(lesmat)
  ),
  file.path(output_dir, "lesmat_dims.csv"),
  row.names = FALSE
)

cat("\n=== All reference data saved to:", output_dir, "===\n")
cat("Files created:\n")
cat(paste(" -", list.files(output_dir), collapse = "\n"), "\n")
