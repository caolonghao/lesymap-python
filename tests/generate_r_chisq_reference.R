# Generate R reference data for Chi-square test validation
# This script creates test data and computes chi-square statistics using R's chisq.test

# Set seed for reproducibility
set.seed(42)

# Create test data: 20 subjects x 50 voxels
n_subjects <- 20
n_voxels <- 50

# Binary lesion matrix
lesmat <- matrix(rbinom(n_subjects * n_voxels, 1, 0.4), nrow = n_subjects, ncol = n_voxels)

# Binary behavioral scores (0/1)
behavior <- rbinom(n_subjects, 1, 0.5)

# Compute chi-square statistics for each voxel
# Following the same approach as lsm_chisq in R

# Pre-compute counts
behavOn <- sum(behavior == 1)
behavOff <- length(behavior) - behavOn
lesVox <- colSums(lesmat)
lesOnBehavOn <- colSums(apply(lesmat, 2, function(x) x * behavior))
lesOnBehavOff <- lesVox - lesOnBehavOn
lesOffBehavOn <- behavOn - lesOnBehavOn
lesOffBehavOff <- behavOff - lesOnBehavOff

# Build chi matrix (same structure as R implementation)
chimatrix <- rbind(lesOnBehavOff, lesOnBehavOn, lesOffBehavOff, lesOffBehavOn)

# Run chi-square test for each voxel (without Yates correction to match Python)
output <- apply(chimatrix, 2, function(x) {
  temp <- chisq.test(matrix(x, ncol = 2), correct = FALSE)
  return(list(
    stat = temp$statistic,
    pval = temp$p.value
  ))
})

# Extract statistics and p-values
temp <- unlist(output)
statistic <- unname(temp[seq(1, length(temp), by = 2)])
pvalue <- unname(temp[seq(2, length(temp), by = 2)])

# Save results
output_dir <- "/data/r_reference_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Save as CSV files
write.csv(lesmat, file.path(output_dir, "chisq_lesmat.csv"), row.names = FALSE)
write.csv(behavior, file.path(output_dir, "chisq_behavior.csv"), row.names = FALSE)
write.csv(statistic, file.path(output_dir, "chisq_statistic.csv"), row.names = FALSE)
write.csv(pvalue, file.path(output_dir, "chisq_pvalue.csv"), row.names = FALSE)

# Also save contingency tables for debugging
contingency_tables <- list()
for (vox in 1:n_voxels) {
  table_data <- matrix(chimatrix[, vox], ncol = 2)
  contingency_tables[[vox]] <- table_data
}

# Save first few contingency tables for verification
cat("First 5 contingency tables:\n")
for (i in 1:5) {
  cat(sprintf("\nVoxel %d:\n", i))
  print(matrix(chimatrix[, i], ncol = 2,
               dimnames = list(c("les_on", "les_off"), c("behav_off", "behav_on"))))
  cat(sprintf("Chi-sq stat: %.6f, p-value: %.6f\n", statistic[i], pvalue[i]))
}

# Save summary info
info <- data.frame(
  n_subjects = n_subjects,
  n_voxels = n_voxels,
  seed = 42,
  yates_correction = FALSE
)
write.csv(info, file.path(output_dir, "chisq_info.csv"), row.names = FALSE)

cat("\n\nReference data saved to:", output_dir, "\n")
cat("Files created:\n")
cat("  - chisq_lesmat.csv\n")
cat("  - chisq_behavior.csv\n")
cat("  - chisq_statistic.csv\n")
cat("  - chisq_pvalue.csv\n")
cat("  - chisq_info.csv\n")
