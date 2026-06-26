#!/usr/bin/env Rscript
# Generate tiny true R lsm_svr fixtures for fast Python end-to-end tests.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[startsWith(args, file_arg)][1])
if (is.na(script_path)) {
  script_path <- file.path(getwd(), "tests", "generate_r_svr_tiny_reference.R")
}
repo_root <- normalizePath(file.path(dirname(normalizePath(script_path)), ".."))
output_dir <- file.path(repo_root, "tests", "fixtures", "r_reference_results")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source(file.path(repo_root, "LESYMAP_R_repo", "R", "lsm_svr.R"))

if (!requireNamespace("e1071", quietly = TRUE)) {
  stop("e1071 is required. Install with install.packages('e1071').")
}

set.seed(123)

lesmat <- matrix(
  c(
    1, 0, 1, 0, 0, 1,
    1, 1, 1, 0, 0, 1,
    0, 1, 0, 1, 1, 0,
    0, 0, 0, 1, 1, 0,
    1, 0, 0, 1, 0, 1,
    0, 1, 1, 0, 1, 0,
    1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 0, 0
  ),
  nrow = 8,
  byrow = TRUE
)
behavior <- c(0.10, 0.35, 0.80, 0.95, 0.55, 0.70, 0.40, 1.05)

linear <- lsm_svr(
  lesmat = lesmat,
  behavior = behavior,
  SVR.nperm = 1,
  SVR.kernel = "linear",
  SVR.gamma = 5,
  SVR.cost = 1,
  SVR.epsilon = 0.1,
  showInfo = FALSE
)

radial <- lsm_svr(
  lesmat = lesmat,
  behavior = behavior,
  SVR.nperm = 1,
  SVR.kernel = "radial",
  SVR.gamma = 5,
  SVR.cost = 30,
  SVR.epsilon = 0.1,
  showInfo = FALSE
)

write.csv(lesmat, file.path(output_dir, "svr_tiny_lesmat.csv"), row.names = FALSE)
write.csv(data.frame(behavior = behavior), file.path(output_dir, "svr_tiny_behavior.csv"), row.names = FALSE)
write.csv(
  data.frame(statistic = linear$statistic, pvalue = linear$pvalue),
  file.path(output_dir, "svr_tiny_lsm_linear_results.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(statistic = radial$statistic, pvalue = radial$pvalue),
  file.path(output_dir, "svr_tiny_lsm_radial_results.csv"),
  row.names = FALSE
)

metadata <- data.frame(
  seed = 123,
  n_subjects = nrow(lesmat),
  n_voxels = ncol(lesmat),
  SVR_nperm = 1,
  linear_cost = 1,
  radial_cost = 30,
  gamma = 5,
  epsilon = 0.1,
  e1071_version = as.character(utils::packageVersion("e1071"))
)
write.csv(metadata, file.path(output_dir, "svr_tiny_metadata.csv"), row.names = FALSE)

cat("Wrote tiny R lsm_svr fixtures to ", output_dir, "\n", sep = "")
