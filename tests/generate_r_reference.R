# Generate reference results from R LESYMAP for Python validation
# Run this script inside the dorianps/lesymap Docker container

library(LESYMAP)

set.seed(42)

# Test case 1: Small synthetic data
n_subjects <- 20
n_voxels <- 50

# Create binary lesion matrix
lesmat <- matrix(rbinom(n_subjects * n_voxels, 1, 0.4), nrow = n_subjects, ncol = n_voxels)

# Create behavioral scores (continuous)
behavior <- rnorm(n_subjects, mean = 50, sd = 10)

# Ensure some voxels have enough subjects in each group
for (v in 1:n_voxels) {
  n_lesioned <- sum(lesmat[, v])
  if (n_lesioned < 3) {
    # Add more lesions
    zero_idx <- which(lesmat[, v] == 0)
    lesmat[sample(zero_idx, 3 - n_lesioned), v] <- 1
  } else if (n_lesioned > n_subjects - 3) {
    # Remove some lesions
    one_idx <- which(lesmat[, v] == 1)
    lesmat[sample(one_idx, n_lesioned - (n_subjects - 3)), v] <- 0
  }
}

# Run BMfast2 (the version actually used in lsm_BMfast)
cat("Running BMfast2...\n")
bm2_result <- BMfast2(lesmat, behavior, computeDOF = TRUE)

# Compute p-values (two-sided, matching Python implementation)
pvals_twosided <- 2 * pt(abs(bm2_result$statistic), bm2_result$dfbm, lower.tail = FALSE)

# Save results
output_dir <- "/data/r_reference_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Save as CSV for easy reading in Python
write.csv(data.frame(
  statistic = bm2_result$statistic,
  dfbm = bm2_result$dfbm,
  pvalue_twosided = pvals_twosided
), file.path(output_dir, "bmfast_results.csv"), row.names = FALSE)

# Save input data
write.csv(lesmat, file.path(output_dir, "bmfast_lesmat.csv"), row.names = FALSE)
write.csv(data.frame(behavior = behavior), file.path(output_dir, "bmfast_behavior.csv"), row.names = FALSE)

cat("\nResults saved to:", output_dir, "\n")
cat("Statistic range:", range(bm2_result$statistic), "\n")
cat("DOF range:", range(bm2_result$dfbm), "\n")
cat("P-value range:", range(pvals_twosided), "\n")
