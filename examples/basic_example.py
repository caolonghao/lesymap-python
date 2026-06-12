"""
Basic example of using LESYMAP-Python.

This example demonstrates how to:
1. Load lesion maps and behavioral data
2. Run SCCAN analysis
3. Save results and make predictions
"""

import numpy as np
import nibabel as nib
import lesymap


def create_example_data(n_subjects=50, img_shape=(91, 109, 91)):
    """
    Create synthetic example data for demonstration.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    img_shape : tuple
        Image dimensions

    Returns
    -------
    lesions : list of nibabel images
        Synthetic lesion maps
    behavior : ndarray
        Synthetic behavioral scores
    """
    # Create affine (MNI-style)
    affine = np.array([
        [-2, 0, 0, 90],
        [0, 2, 0, -126],
        [0, 0, 2, -72],
        [0, 0, 0, 1]
    ])

    # Define a "true" region of interest
    # Lesions in frontal lobe affect behavior
    roi_center = (30, 40, 50)
    roi_radius = 10

    # Create coordinate grid
    x, y, z = np.mgrid[:img_shape[0], :img_shape[1], :img_shape[2]]
    dist_from_roi = np.sqrt(
        (x - roi_center[0])**2 +
        (y - roi_center[1])**2 +
        (z - roi_center[2])**2
    )
    roi_mask = dist_from_roi < roi_radius

    lesions = []
    behavior = []

    for i in range(n_subjects):
        # Generate random lesion
        lesion_prob = np.random.uniform(0, 0.1, img_shape)

        # Higher probability in ROI for some subjects
        if np.random.rand() > 0.5:
            lesion_prob[roi_mask] *= 3

        lesion_data = (np.random.rand(*img_shape) < lesion_prob).astype(np.float64)

        # Create lesion image
        lesion_img = nib.Nifti1Image(lesion_data, affine)
        lesions.append(lesion_img)

        # Behavior is affected by lesion load in ROI
        lesion_load_in_roi = np.sum(lesion_data[roi_mask])
        behav = 10 - lesion_load_in_roi * 2 + np.random.normal(0, 0.5)
        behavior.append(behav)

    return lesions, np.array(behavior)


def main():
    """Run example analysis."""
    print("LESYMAP-Python Example")
    print("=" * 50)

    # Create example data
    print("\n1. Creating synthetic data...")
    lesions, behavior = create_example_data(n_subjects=50)
    print(f"   Created {len(lesions)} lesion maps")
    print(f"   Behavior: mean={behavior.mean():.2f}, std={behavior.std():.2f}")

    # Run SCCAN analysis
    print("\n2. Running SCCAN analysis...")
    result = lesymap.lesymap(
        lesions=lesions,
        behavior=behavior,
        method='sccan',
        optimize_sparseness=False,  # Use default for speed
        sparseness=0.1,
        multiple_comparison='fdr',
        show_info=True
    )

    print(f"\n   SCCAN correlation: {result.model_params.get('correlation', 'N/A')}")

    # Run BMfast for comparison
    print("\n3. Running Brunner-Munzel test...")
    result_bm = lesymap.lesymap(
        lesions=lesions,
        behavior=behavior,
        method='BMfast',
        multiple_comparison='fdr',
        show_info=True
    )

    # Save results
    print("\n4. Saving results...")
    result.save('example_output/', save_model=True)
    print("   Results saved to example_output/")

    # Make predictions
    print("\n5. Testing prediction...")
    # Create a new test subject
    test_lesions, _ = create_example_data(n_subjects=1)
    predictions = result.predict(test_lesions)
    print(f"   Predicted behavior: {predictions[0]:.2f}")

    # Load saved model and predict again
    print("\n6. Loading saved model...")
    loaded_model = lesymap.LesymapResult.load_checkpoint(
        'example_output/model_checkpoint.pkl'
    )
    predictions_loaded = loaded_model.predict(test_lesions)
    print(f"   Predicted behavior (loaded): {predictions_loaded[0]:.2f}")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nOutput files:")
    print("  - example_output/stat.nii.gz     (statistical map)")
    print("  - example_output/weights.nii.gz  (model weights)")
    print("  - example_output/model_checkpoint.pkl  (for inference)")


if __name__ == '__main__':
    main()
