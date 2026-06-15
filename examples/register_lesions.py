"""
Example: Register lesion maps to MNI152 template space.

This is the preprocessing step required before running LESYMAP analysis.
All lesion maps must be in the same template space (MNI152) before calling
lesymap.lesymap().

Requirements:
    pip install antspyx

Template:
    Download MNI152 2009c from:
    https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip

    Place the three files in:
    lesymap/data/templates/MNI152_2009c/
        mni_icbm152_t1_tal_nlin_sym_09c.nii.gz
        mni_icbm152_t1_tal_nlin_sym_09c_mask.nii.gz
        mni_icbm152_t1_tal_nlin_sym_09c_mask_skullnoface.nii.gz

    Alternatively, pass template= explicitly to register_lesion_to_template().
"""

from pathlib import Path
from lesymap.core.registration import register_lesion_to_template, register_batch

# ── Single subject ────────────────────────────────────────────────────────────

result = register_lesion_to_template(
    subject_anatomical='data/sub-001/anat/sub-001_T1w.nii.gz',
    subject_lesion='data/sub-001/lesion/sub-001_lesion.nii.gz',
    # template defaults to bundled MNI152_2009c if available
    skull_strip=True,
    type_of_transform='SyN',          # 'SyNCC' for higher accuracy (~2h)
    output_prefix='output/sub-001',   # saves transforms + warped images
    show_info=True,
)

# lesion in template space (ANTs image)
lesion_mni = result['lesion_template']
print(f"Lesion voxels in MNI space: {lesion_mni.numpy().sum():.0f}")

# ── Batch registration ────────────────────────────────────────────────────────

subject_ids = ['sub-001', 'sub-002', 'sub-003']
anatomicals  = [f'data/{s}/anat/{s}_T1w.nii.gz'   for s in subject_ids]
lesions      = [f'data/{s}/lesion/{s}_lesion.nii.gz' for s in subject_ids]

registered_lesions, results = register_batch(
    subject_anatomicals=anatomicals,
    subject_lesions=lesions,
    skull_strip=True,
    type_of_transform='SyN',
    output_dir='output/registered',   # per-subject sub-dirs created here
    subject_ids=subject_ids,
    n_jobs=1,                         # increase for parallel processing
    show_info=True,
)

print(f"Registered {len(registered_lesions)} subjects.")

# ── Pass directly to lesymap ──────────────────────────────────────────────────

import lesymap

behavior_scores = [2.1, 3.5, 1.8]

result = lesymap.lesymap(
    lesions=registered_lesions,       # ANTs images returned by register_batch
    behavior=behavior_scores,
    method='BMfast',
    multiple_comparison='fdr',
)
result.save('output/lesymap_results/')
