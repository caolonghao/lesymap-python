# Generate R SVR reference data for Python comparison
# This script creates reference data using R's e1071::svm function
# to validate the Python SVR implementation

library(e1071)

# Set seed for reproducibility
set.seed(42)

# Create synthetic data: 20 subjects x 100 voxels
n_subjects <- 20
n_voxels <- 100

# Generate lesion matrix (binary lesions with some structure)
lesmat <- matrix(0, nrow = n_subjects, ncol = n_voxels)
for (i in 1:n_subjects) {
  # Each subject has ~30% lesion coverage with some spatial structure
  n_lesioned <- sample(20:40, 1)
  lesioned_voxels <- sample(1:n_voxels, n_lesioned)
  lesmat[i, lesioned_voxels] <- 1
}

# Create behavioral scores correlated with specific voxels
# Voxels 10-20 are "critical" - lesions here cause deficits
critical_voxels <- 10:20
lesion_load_critical <- rowSums(lesmat[, critical_voxels])
behavior <- 100 - 5 * lesion_load_critical + rnorm(n_subjects, 0, 5)

# Scale and center data (as done in R lsm_svr)
behavior_scaled <- scale(behavior, scale = TRUE, center = TRUE)
lesmat_scaled <- scale(lesmat, scale = TRUE, center = TRUE)

# Store scaling parameters
behavior_mean <- attr(behavior_scaled, "scaled:center")
behavior_sd <- attr(behavior_scaled, "scaled:scale")
lesmat_mean <- attr(lesmat_scaled, "scaled:center")
lesmat_sd <- attr(lesmat_scaled, "scaled:scale")

# Convert to plain matrices/vectors
behavior_scaled <- as.vector(behavior_scaled)
lesmat_scaled <- as.matrix(lesmat_scaled)

# Run SVR with specific parameters (matching R defaults)
# Using radial kernel as in R default
svr_radial <- svm(
  x = lesmat_scaled,
  y = behavior_scaled,
  scale = FALSE,  # Already scaled
  type = "eps-regression",
  kernel = "radial",
  gamma = 5,
  cost = 30,
  epsilon = 0.1
)

# Extract weights for radial kernel: w = t(coefs) %*% SV
w_radial <- t(svr_radial$coefs) %*% svr_radial$SV
w_radial <- as.vector(w_radial)

# Scale weights (as in R lsm_svr)
betaScale_radial <- 10 / max(abs(w_radial))
statistic_radial <- w_radial * betaScale_radial

# Get predictions
pred_radial <- predict(svr_radial, lesmat_scaled)

# Also run with linear kernel for direct weight comparison
svr_linear <- svm(
  x = lesmat_scaled,
  y = behavior_scaled,
  scale = FALSE,
  type = "eps-regression",
  kernel = "linear",
  cost = 30,
  epsilon = 0.1
)

# For linear kernel, weights can be extracted directly
w_linear <- t(svr_linear$coefs) %*% svr_linear$SV
w_linear <- as.vector(w_linear)
betaScale_linear <- 10 / max(abs(w_linear))
statistic_linear <- w_linear * betaScale_linear

pred_linear <- predict(svr_linear, lesmat_scaled)

# Also test with sklearn-like defaults (C=1, linear kernel)
svr_sklearn_defaults <- svm(
  x = lesmat_scaled,
  y = behavior_scaled,
  scale = FALSE,
  type = "eps-regression",
  kernel = "linear",
  cost = 1,
  epsilon = 0.1
)

w_sklearn <- t(svr_sklearn_defaults$coefs) %*% svr_sklearn_defaults$SV
w_sklearn <- as.vector(w_sklearn)
pred_sklearn <- predict(svr_sklearn_defaults, lesmat_scaled)

# Save all reference data
reference_data <- list(
  # Input data
  lesmat = lesmat,
  behavior = behavior,
  lesmat_scaled = lesmat_scaled,
  behavior_scaled = behavior_scaled,

  # Scaling parameters
  behavior_mean = behavior_mean,
  behavior_sd = behavior_sd,
  lesmat_mean = lesmat_mean,
  lesmat_sd = lesmat_sd,

  # Radial kernel results (R defaults)
  weights_radial = w_radial,
  statistic_radial = statistic_radial,
  predictions_radial = as.vector(pred_radial),
  sv_indices_radial = svr_radial$index,
  n_sv_radial = length(svr_radial$index),

  # Linear kernel results
  weights_linear = w_linear,
  statistic_linear = statistic_linear,
  predictions_linear = as.vector(pred_linear),
  sv_indices_linear = svr_linear$index,
  n_sv_linear = length(svr_linear$index),

  # sklearn-like defaults (C=1, linear)
  weights_sklearn_defaults = w_sklearn,
  predictions_sklearn_defaults = as.vector(pred_sklearn),

  # Parameters used
  params = list(
    n_subjects = n_subjects,
    n_voxels = n_voxels,
    seed = 42,
    radial = list(kernel = "radial", gamma = 5, cost = 30, epsilon = 0.1),
    linear = list(kernel = "linear", cost = 30, epsilon = 0.1),
    sklearn_defaults = list(kernel = "linear", cost = 1, epsilon = 0.1)
  )
)

# Save as RDS
saveRDS(reference_data, "fixtures/svr_reference_data.rds")

# Also save as CSV files for easy loading in Python
write.csv(lesmat, "fixtures/svr_lesmat.csv", row.names = FALSE)
write.csv(behavior, "fixtures/svr_behavior.csv", row.names = FALSE)
write.csv(w_linear, "fixtures/svr_weights_linear.csv", row.names = FALSE)
write.csv(pred_linear, "fixtures/svr_predictions_linear.csv", row.names = FALSE)
write.csv(w_sklearn, "fixtures/svr_weights_sklearn_defaults.csv", row.names = FALSE)
write.csv(pred_sklearn, "fixtures/svr_predictions_sklearn_defaults.csv", row.names = FALSE)

# Save scaling parameters
scaling_params <- data.frame(
  behavior_mean = behavior_mean,
  behavior_sd = behavior_sd
)
write.csv(scaling_params, "fixtures/svr_scaling_params.csv", row.names = FALSE)

cat("Reference data saved to fixtures/\n")
cat(sprintf("Number of support vectors (radial): %d\n", length(svr_radial$index)))
cat(sprintf("Number of support vectors (linear): %d\n", length(svr_linear$index)))
cat(sprintf("Correlation of predictions with behavior (radial): %.4f\n", cor(pred_radial, behavior_scaled)))
cat(sprintf("Correlation of predictions with behavior (linear): %.4f\n", cor(pred_linear, behavior_scaled)))
cat(sprintf("Correlation of predictions with behavior (sklearn defaults): %.4f\n", cor(pred_sklearn, behavior_scaled)))
