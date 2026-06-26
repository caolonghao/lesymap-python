#!/usr/bin/env Rscript
# Generate a tiny R LESYMAP SCCAN sparseness-optimization fixture.

library(LESYMAP)
library(ANTsR)

output_dir <- "/data/r_reference_results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

set.seed(7)
lesmat <- matrix(rbinom(18 * 24, 1, 0.25), nrow = 18, ncol = 24)
behavior <- scale(
  rowSums(lesmat[, 1:4]) - rowSums(lesmat[, 5:8]) + rnorm(18, 0, 0.2)
)[, 1]
mask <- makeImage(c(24, 1, 1), rep(1, 24))

result <- LESYMAP:::optimize_SCCANsparseness(
  lesmat = lesmat,
  behavior = behavior,
  mask = mask,
  nFolds = 3,
  cvRepetitions = 1,
  lowerSparseness = -0.2,
  upperSparseness = -0.01,
  tol = 0.08,
  robust = 0,
  its = 5,
  cthresh = 0,
  smooth = 0,
  showInfo = TRUE,
  directionalSCCAN = TRUE,
  maxBased = FALSE
)

write.csv(lesmat, file.path(output_dir, "sccan_cv_tiny_lesmat.csv"), row.names = FALSE)
write.csv(data.frame(behavior = behavior), file.path(output_dir, "sccan_cv_tiny_behavior.csv"), row.names = FALSE)
write.csv(
  data.frame(
    seed = 7,
    n_subjects = nrow(lesmat),
    n_voxels = ncol(lesmat),
    n_folds = 3,
    cv_repetitions = 1,
    lower_sparseness = -0.2,
    upper_sparseness = -0.01,
    tolerance = 0.08,
    robust = 0,
    its = 5,
    cthresh = 0,
    smooth = 0,
    sparseness_penalty = 0.03,
    r_minimum = result$minimum,
    r_optimal_sparseness = abs(result$minimum),
    r_objective = result$objective,
    r_cv_correlation = result$CVcorrelation.stat,
    lesymap_version = as.character(packageVersion("LESYMAP")),
    antsr_version = as.character(packageVersion("ANTsR"))
  ),
  file.path(output_dir, "sccan_cv_tiny_results.csv"),
  row.names = FALSE
)

cat("Tiny SCCAN CV fixture written to", output_dir, "\n")
