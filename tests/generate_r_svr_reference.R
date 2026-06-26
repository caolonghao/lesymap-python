# Generate R SVR reference data for Python comparison testing.
#
# This script saves two kinds of reference data:
# 1. Method-level LESYMAP lsm_svr() outputs for end-to-end method comparison.
# 2. Direct e1071::svm diagnostics with predictions/support-vector weights for
#    debugging numerical differences between R and scikit-learn/libsvm.

library(LESYMAP)
library(ANTsR)

# Set seed for reproducibility
set.seed(42)
run_true_lsm_only <- identical(Sys.getenv("RUN_TRUE_LSM_ONLY"), "1")

# Output directory
output_dir <- "/data/r_reference_results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Loading example lesion data from LESYMAP package...\n")

# Load example lesion files from LESYMAP package
lesion_dir <- system.file("extdata", package = "LESYMAP")
lesion_files <- list.files(file.path(lesion_dir, "lesions"),
                            pattern = "\\.nii\\.gz$",
                            full.names = TRUE)

cat(sprintf("Found %d lesion files\n", length(lesion_files)))

# Load first 50 subjects for faster testing
n_subjects <- min(50, length(lesion_files))
lesion_files <- lesion_files[1:n_subjects]

# Load lesions as images
cat("Loading lesion images...\n")
lesions <- lapply(lesion_files, antsImageRead)

# Load behavior scores
behavior_file <- file.path(lesion_dir, "behavior", "behavior.txt")
behavior_data <- read.table(behavior_file, header = FALSE)
behavior <- behavior_data$V1[1:n_subjects]

cat(sprintf("Loaded %d subjects with behavior scores\n", n_subjects))

# Create mask from average lesion (threshold 0.1)
cat("Creating mask from average lesion...\n")
avg_lesion <- lesions[[1]] * 0
for (i in 1:length(lesions)) {
  avg_lesion <- avg_lesion + lesions[[i]]
}
avg_lesion <- avg_lesion / length(lesions)
mask <- thresholdImage(avg_lesion, 0.1, Inf)

mask_sum <- sum(as.array(mask))
cat(sprintf("Mask contains %d voxels\n", mask_sum))

# Convert lesions to matrix
cat("Converting lesions to matrix...\n")
lesmat <- imageListToMatrix(lesions, mask)
cat(sprintf("Lesion matrix dimensions: %d subjects x %d voxels\n",
            nrow(lesmat), ncol(lesmat)))

# Scale and center data (as R's lsm_svr does)
behavior_scaled <- scale(behavior, scale = TRUE, center = TRUE)
lesmat_scaled <- scale(lesmat, scale = TRUE, center = TRUE)

# Get scaling parameters for Python comparison
behavior_center <- attr(behavior_scaled, "scaled:center")
behavior_scale <- attr(behavior_scaled, "scaled:scale")
lesmat_center <- attr(lesmat_scaled, "scaled:center")
lesmat_scale <- attr(lesmat_scaled, "scaled:scale")

# Install e1071 if not available
if (!require("e1071", quietly = TRUE)) {
  install.packages("e1071", repos = "https://cloud.r-project.org")
  library(e1071)
}

if (!run_true_lsm_only) {
  # Run SVR with LINEAR kernel for coefficient comparison
  # Note: R defaults are radial kernel with cost=30, gamma=5
  # For comparison with Python sklearn, we use linear kernel
  cat("\n=== Running SVR with LINEAR kernel (for Python comparison) ===\n")
  cat("Parameters: kernel=linear, cost=1, epsilon=0.1\n")

  svr_linear <- svm(x = lesmat_scaled,
                    y = as.vector(behavior_scaled),
                    scale = FALSE,  # Already scaled
                    type = 'eps-regression',
                    kernel = 'linear',
                    cost = 1,  # Match sklearn default
                    epsilon = 0.1)

  # Get predictions
  pred_linear <- predict(svr_linear, lesmat_scaled)

  # Correlation with scaled behavior
  corr_linear <- cor(pred_linear, behavior_scaled)
  cat(sprintf("Linear SVR correlation (scaled): %.6f\n", corr_linear))

  # Get weights for linear kernel
  # For linear kernel: w = t(coefs) %*% SV
  w_linear <- t(svr_linear$coefs) %*% svr_linear$SV
  w_linear <- as.vector(w_linear)

  cat(sprintf("Linear kernel weights: min=%.6f, max=%.6f\n",
              min(w_linear), max(w_linear)))

  # Also run with RADIAL kernel (R default) for comparison
  cat("\n=== Running SVR with RADIAL kernel (R default) ===\n")
  cat("Parameters: kernel=radial, cost=30, gamma=5, epsilon=0.1\n")

  svr_radial <- svm(x = lesmat_scaled,
                    y = as.vector(behavior_scaled),
                    scale = FALSE,
                    type = 'eps-regression',
                    kernel = 'radial',
                    gamma = 5,
                    cost = 30,
                    epsilon = 0.1)

  pred_radial <- predict(svr_radial, lesmat_scaled)
  corr_radial <- cor(pred_radial, behavior_scaled)
  cat(sprintf("Radial SVR correlation (scaled): %.6f\n", corr_radial))

  # Get weights for radial kernel (note: not directly interpretable)
  w_radial <- t(svr_radial$coefs) %*% svr_radial$SV
  w_radial <- as.vector(w_radial)

  # Scale weights as in lsm_svr.R (betaScale = 10/max(abs(w)))
  betaScale_radial <- 10 / max(abs(w_radial))
  statistic_radial <- w_radial * betaScale_radial
} else {
  cat("\n=== Skipping direct e1071 diagnostics; RUN_TRUE_LSM_ONLY=1 ===\n")
}

# Run true LESYMAP lsm_svr() method-level references. Use SVR.nperm=1 so the
# permutation p-value path is deterministic and cheap. Do not use 0: in R,
# 1:0 evaluates to c(1, 0), so the original loop would still run.
cat("\n=== Running LESYMAP lsm_svr method-level references ===\n")
set.seed(42)
lsm_svr_linear <- lsm_svr(
  lesmat = lesmat,
  behavior = behavior,
  SVR.nperm = 1,
  SVR.kernel = "linear",
  SVR.gamma = 5,
  SVR.cost = 1,
  SVR.epsilon = 0.1,
  showInfo = FALSE
)

set.seed(42)
lsm_svr_radial <- lsm_svr(
  lesmat = lesmat,
  behavior = behavior,
  SVR.nperm = 1,
  SVR.kernel = "radial",
  SVR.gamma = 5,
  SVR.cost = 30,
  SVR.epsilon = 0.1,
  showInfo = FALSE
)

# Save reference data
cat("\nSaving reference data...\n")

if (!run_true_lsm_only) {
  # Save linear kernel results
  saveRDS(list(
    predictions = as.vector(pred_linear),
    correlation = as.numeric(corr_linear),
    weights = w_linear,
    n_support_vectors = svr_linear$tot.nSV,
    kernel = "linear",
    cost = 1,
    epsilon = 0.1
  ), file.path(output_dir, "svr_linear_results.rds"))

  # Save radial kernel results
  saveRDS(list(
    predictions = as.vector(pred_radial),
    correlation = as.numeric(corr_radial),
    weights = w_radial,
    statistic = statistic_radial,
    n_support_vectors = svr_radial$tot.nSV,
    kernel = "radial",
    cost = 30,
    gamma = 5,
    epsilon = 0.1
  ), file.path(output_dir, "svr_radial_results.rds"))

  # Save input data for Python to use
  saveRDS(list(
    lesmat = lesmat,
    lesmat_scaled = lesmat_scaled,
    behavior = behavior,
    behavior_scaled = as.vector(behavior_scaled),
    behavior_center = behavior_center,
    behavior_scale = behavior_scale,
    lesmat_center = lesmat_center,
    lesmat_scale = lesmat_scale,
    n_subjects = n_subjects,
    n_voxels = ncol(lesmat)
  ), file.path(output_dir, "svr_input_data.rds"))

  # Also save as CSV for easier Python loading
  write.csv(lesmat, gzfile(file.path(output_dir, "svr_lesmat.csv.gz")), row.names = FALSE)
  write.csv(lesmat_scaled, gzfile(file.path(output_dir, "svr_lesmat_scaled.csv.gz")), row.names = FALSE)
  write.csv(data.frame(behavior = behavior),
            file.path(output_dir, "svr_behavior.csv"), row.names = FALSE)
  write.csv(data.frame(behavior_scaled = as.vector(behavior_scaled)),
            file.path(output_dir, "svr_behavior_scaled.csv"), row.names = FALSE)

  # Save linear kernel results as CSV
  write.csv(data.frame(
    predictions = as.vector(pred_linear),
    correlation = as.numeric(corr_linear)
  ), file.path(output_dir, "svr_linear_predictions.csv"), row.names = FALSE)
  write.csv(data.frame(weights = w_linear),
            file.path(output_dir, "svr_linear_weights.csv"), row.names = FALSE)

  # Save radial kernel results as CSV
  write.csv(data.frame(
    predictions = as.vector(pred_radial),
    correlation = as.numeric(corr_radial)
  ), file.path(output_dir, "svr_radial_predictions.csv"), row.names = FALSE)
  write.csv(data.frame(weights = w_radial),
            file.path(output_dir, "svr_radial_weights.csv"), row.names = FALSE)
}

# Save true lsm_svr method-level outputs.
write.csv(data.frame(
  statistic = as.vector(lsm_svr_linear$statistic),
  pvalue = as.vector(lsm_svr_linear$pvalue)
), gzfile(file.path(output_dir, "svr_lsm_linear_results.csv.gz")), row.names = FALSE)

write.csv(data.frame(
  statistic = as.vector(lsm_svr_radial$statistic),
  pvalue = as.vector(lsm_svr_radial$pvalue)
), gzfile(file.path(output_dir, "svr_lsm_radial_results.csv.gz")), row.names = FALSE)

# Save metadata
metadata <- list(
  r_version = as.character(getRversion()),
  lesymap_version = as.character(packageVersion("LESYMAP")),
  e1071_version = as.character(packageVersion("e1071")),
  seed = 42,
  n_subjects = n_subjects,
  n_voxels = ncol(lesmat),
  mask_threshold = 0.1,
  generation_date = Sys.time()
)
saveRDS(metadata, file.path(output_dir, "svr_metadata.rds"))
write.csv(
  data.frame(
    r_version = metadata$r_version,
    lesymap_version = metadata$lesymap_version,
    e1071_version = metadata$e1071_version,
    seed = metadata$seed,
    n_subjects = metadata$n_subjects,
    n_voxels = metadata$n_voxels,
    mask_threshold = metadata$mask_threshold,
    generation_mode = ifelse(run_true_lsm_only, "true_lsm_only", "full"),
    generation_date = as.character(metadata$generation_date)
  ),
  file.path(output_dir, "svr_manifest.csv"),
  row.names = FALSE
)

cat("\n=== Summary ===\n")
cat(sprintf("Input matrix: %d subjects x %d voxels\n", n_subjects, ncol(lesmat)))
if (!run_true_lsm_only) {
  cat(sprintf("Linear kernel correlation: %.6f\n", corr_linear))
  cat(sprintf("Radial kernel correlation: %.6f\n", corr_radial))
  cat(sprintf("Linear kernel support vectors: %d\n", svr_linear$tot.nSV))
  cat(sprintf("Radial kernel support vectors: %d\n", svr_radial$tot.nSV))
}
cat(sprintf("\nReference data saved to: %s\n", output_dir))
cat("Files created:\n")
if (!run_true_lsm_only) {
  cat("  - svr_linear_results.rds\n")
  cat("  - svr_radial_results.rds\n")
  cat("  - svr_input_data.rds\n")
  cat("  - svr_lesmat.csv.gz\n")
  cat("  - svr_lesmat_scaled.csv.gz\n")
  cat("  - svr_behavior.csv\n")
  cat("  - svr_behavior_scaled.csv\n")
  cat("  - svr_linear_predictions.csv\n")
  cat("  - svr_linear_weights.csv\n")
  cat("  - svr_radial_predictions.csv\n")
  cat("  - svr_radial_weights.csv\n")
}
cat("  - svr_lsm_linear_results.csv.gz\n")
cat("  - svr_lsm_radial_results.csv.gz\n")
cat("  - svr_metadata.rds\n")
