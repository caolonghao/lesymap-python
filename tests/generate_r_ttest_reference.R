# Generate T-test reference results from R LESYMAP
library(LESYMAP)

set.seed(42)
n_subjects <- 20
n_voxels <- 50

# Create binary lesion matrix
lesmat <- matrix(rbinom(n_subjects * n_voxels, 1, 0.4), nrow = n_subjects, ncol = n_voxels)

# Create behavioral scores
behavior <- rnorm(n_subjects, mean = 50, sd = 10)

# Ensure each voxel has at least 3 subjects in each group
for (v in 1:n_voxels) {
  n_lesioned <- sum(lesmat[, v])
  if (n_lesioned < 3) {
    zero_idx <- which(lesmat[, v] == 0)
    lesmat[sample(zero_idx, 3 - n_lesioned), v] <- 1
  } else if (n_lesioned > n_subjects - 3) {
    one_idx <- which(lesmat[, v] == 1)
    lesmat[sample(one_idx, n_lesioned - (n_subjects - 3)), v] <- 0
  }
}

# Run TTfast with equal variance (Student's t-test)
cat("Running TTfast (Student's t-test)...\n")
ttest_result <- TTfast(lesmat, as.matrix(behavior), computeDOF = TRUE, varEqual = TRUE)

# Run TTfast with unequal variance (Welch's t-test)
cat("Running TTfast (Welch's t-test)...\n")
welch_result <- TTfast(lesmat, as.matrix(behavior), computeDOF = TRUE, varEqual = FALSE)

# Compute p-values (two-sided)
ttest_pvals <- 2 * pt(abs(ttest_result$statistic), ttest_result$df, lower.tail = FALSE)
welch_pvals <- 2 * pt(abs(welch_result$statistic), welch_result$df, lower.tail = FALSE)

# Save results
output_dir <- "/data/r_reference_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(data.frame(
  statistic = ttest_result$statistic,
  df = ttest_result$df,
  pvalue_twosided = ttest_pvals
), file.path(output_dir, "ttest_results.csv"), row.names = FALSE)

write.csv(data.frame(
  statistic = welch_result$statistic,
  df = welch_result$df,
  pvalue_twosided = welch_pvals
), file.path(output_dir, "welch_results.csv"), row.names = FALSE)

# Save input data (same as BMfast for consistency)
write.csv(lesmat, file.path(output_dir, "ttest_lesmat.csv"), row.names = FALSE)
write.csv(data.frame(behavior = behavior), file.path(output_dir, "ttest_behavior.csv"), row.names = FALSE)

cat("\nT-test results saved\n")
cat("Student t statistic range:", range(ttest_result$statistic), "\n")
cat("Welch t statistic range:", range(welch_result$statistic), "\n")
