# Generate R reference data for regression validation
# This script creates test data and runs regresfast() to generate reference results

library(LESYMAP)

# Set seed for reproducibility
set.seed(42)

# Create test data: 20 subjects x 50 voxels binary lesion matrix
n_subjects <- 20
n_voxels <- 50

# Generate binary lesion matrix (0/1)
lesmat <- matrix(rbinom(n_subjects * n_voxels, 1, 0.4), nrow = n_subjects, ncol = n_voxels)

# Generate behavioral scores (continuous)
behavior <- rnorm(n_subjects, mean = 50, sd = 10)

# Run regresfast to get t-statistics
cat("Running regresfast...\n")
result <- lsm_regresfast(lesmat, behavior, showInfo = FALSE)

# Extract results
t_statistics <- result$statistic
p_values <- result$pvalue
z_scores <- result$zscore

# Create output directory if it doesn't exist
output_dir <- "/data/r_reference_results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save results to CSV
results_df <- data.frame(
  voxel = 1:n_voxels,
  t_statistic = t_statistics,
  p_value = p_values,
  z_score = z_scores
)
write.csv(results_df, file.path(output_dir, "regression_results.csv"), row.names = FALSE)

# Save input data
write.csv(as.data.frame(lesmat), file.path(output_dir, "regression_lesmat.csv"), row.names = FALSE)
write.csv(data.frame(behavior = behavior), file.path(output_dir, "regression_behavior.csv"), row.names = FALSE)

# Print summary
cat("\nResults saved to:", output_dir, "\n")
cat("Number of subjects:", n_subjects, "\n")
cat("Number of voxels:", n_voxels, "\n")
cat("T-statistic range:", range(t_statistics), "\n")
cat("P-value range:", range(p_values), "\n")

# Print first few t-statistics for verification
cat("\nFirst 10 t-statistics:\n")
print(head(t_statistics, 10))
