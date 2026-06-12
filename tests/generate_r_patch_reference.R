# Generate R reference data for patch computation validation
# Run with: Rscript generate_r_patch_reference.R

library(LESYMAP)

# Set seed for reproducibility
set.seed(42)

# Create a 20 subjects x 50 voxels binary lesion matrix with known patterns
n_subjects <- 20
n_voxels <- 50

# Create binary lesion matrix with some known patterns
# We want some voxels to have identical patterns across subjects
lesmat <- matrix(0, nrow = n_subjects, ncol = n_voxels)

# Create specific patterns:
# Pattern 1: voxels 1-5 have same pattern (first 10 subjects lesioned)
lesmat[1:10, 1:5] <- 1

# Pattern 2: voxels 6-10 have same pattern (subjects 5-15 lesioned)
lesmat[5:15, 6:10] <- 1

# Pattern 3: voxels 11-15 have same pattern (subjects 10-20 lesioned)
lesmat[10:20, 11:15] <- 1

# Pattern 4: unique patterns for remaining voxels (random)
for (i in 16:50) {
  n_lesioned <- sample(3:15, 1)
  lesmat[sample(1:n_subjects, n_lesioned), i] <- 1
}

cat("Input lesion matrix dimensions:", dim(lesmat), "\n")
cat("Total lesioned voxels:", sum(lesmat), "\n")

# Run the core patch computation algorithm directly
# (extracted from getUniqueLesionPatches to work with matrix directly)
add <- 1
summed <- rep(0, ncol(lesmat))
for (i in 1:nrow(lesmat)) {
  summed <- summed + (lesmat[i, ] * add)
  summed <- match(summed, unique(summed))
  add <- max(summed) + 1
}

# Compute patch matrix (extract first voxel of each patch)
patindx_bool <- as.numeric(!duplicated(summed))
patchmatrix <- lesmat[, patindx_bool == 1]

# Results
npatches <- length(unique(summed))
nvoxels <- ncol(lesmat)

cat("Number of unique patches:", npatches, "\n")
cat("Number of voxels:", nvoxels, "\n")
cat("Compression ratio:", nvoxels / npatches, "\n")
cat("Patch matrix dimensions:", dim(patchmatrix), "\n")

# Save results
output_dir <- "/data/r_reference_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Save as RDS files
saveRDS(lesmat, file.path(output_dir, "patch_lesmat.rds"))
saveRDS(summed, file.path(output_dir, "patch_patchindx.rds"))
saveRDS(patchmatrix, file.path(output_dir, "patch_patchmatrix.rds"))
saveRDS(npatches, file.path(output_dir, "patch_npatches.rds"))

# Also save as CSV for easier Python loading
write.csv(lesmat, file.path(output_dir, "patch_lesmat.csv"), row.names = FALSE)
write.csv(summed, file.path(output_dir, "patch_patchindx.csv"), row.names = FALSE)
write.csv(patchmatrix, file.path(output_dir, "patch_patchmatrix.csv"), row.names = FALSE)
write.csv(data.frame(npatches = npatches, nvoxels = nvoxels),
          file.path(output_dir, "patch_stats.csv"), row.names = FALSE)

cat("\nReference data saved to:", output_dir, "\n")
cat("Files created:\n")
cat("  - patch_lesmat.csv\n")
cat("  - patch_patchindx.csv\n")
cat("  - patch_patchmatrix.csv\n")
cat("  - patch_stats.csv\n")
