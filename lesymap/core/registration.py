"""
Lesion-to-template registration using ANTsPy.

Python port of LESYMAP R's registerLesionToTemplate().
Registers a subject's anatomical image to a template and applies
the same transform to the lesion map.
"""

import warnings
from pathlib import Path
from typing import Optional, Union, Tuple

import nibabel as nib
import numpy as np


__all__ = [
    'register_lesion_to_template',
    'register_batch',
    'get_mni152_template_path',
]

# Default MNI152 2009c template filenames (must be downloaded separately)
_MNI152_FILES = {
    'template': 'mni_icbm152_t1_tal_nlin_sym_09c.nii.gz',
    'brain_mask': 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii.gz',
    'reg_mask': 'mni_icbm152_t1_tal_nlin_sym_09c_mask_skullnoface.nii.gz',
}

_MNI152_URL = (
    'https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/'
    'mni_icbm152_nlin_sym_09c_nifti.zip'
)


def get_mni152_template_path() -> Optional[Path]:
    """
    Return path to bundled MNI152 templates if present, else None.

    Templates are expected at:
        <package_root>/data/templates/MNI152_2009c/
    """
    pkg_root = Path(__file__).parent.parent
    template_dir = pkg_root / 'data' / 'templates' / 'MNI152_2009c'
    if template_dir.exists() and (template_dir / _MNI152_FILES['template']).exists():
        return template_dir
    return None


def _require_ants():
    try:
        import ants
        return ants
    except ImportError:
        raise ImportError(
            "ANTsPy is required for registration. Install with:\n"
            "  pip install antspyx\n"
            "or:\n"
            "  conda install -c conda-forge antspy"
        )


def register_lesion_to_template(
    subject_anatomical: Union[str, Path, 'nib.Nifti1Image'],
    subject_lesion: Union[str, Path, 'nib.Nifti1Image'],
    template: Union[str, Path, 'nib.Nifti1Image', None] = None,
    template_brain_mask: Union[str, Path, 'nib.Nifti1Image', None] = None,
    template_reg_mask: Union[str, Path, 'nib.Nifti1Image', None] = None,
    skull_strip: bool = True,
    type_of_transform: str = 'SyN',
    output_prefix: Optional[str] = None,
    show_info: bool = True,
) -> dict:
    """
    Register a subject's lesion map to template (MNI) space.

    Mirrors R's registerLesionToTemplate(). The registration is run
    with the subject as fixed and the template as moving, then the
    inverse transform is applied to bring the lesion into template space.

    Parameters
    ----------
    subject_anatomical : str, Path, or nibabel image
        Subject's T1-weighted anatomical image (with skull).
    subject_lesion : str, Path, or nibabel image
        Subject's lesion map (binary, in native space).
    template : str, Path, nibabel image, or None
        Template anatomical image. Defaults to MNI152 2009c if bundled
        templates are present, otherwise raises an error.
    template_brain_mask : str, Path, nibabel image, or None
        Brain mask for the template. Required when skull_strip=True.
    template_reg_mask : str, Path, nibabel image, or None
        Template mask (skull, no face) for skull-stripping step.
    skull_strip : bool
        Whether to skull-strip before registration (recommended).
        Falls back to False if template_brain_mask is not provided.
    type_of_transform : str
        ANTs registration type. 'SyN' is fast (~5-15 min); 'SyNCC' is
        more accurate but slow (~1-2 hours). Default: 'SyN'.
    output_prefix : str, optional
        If provided, save intermediate images and transforms to this prefix.
        The parent directory must already exist.
    show_info : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        subject_img          -- bias-corrected (and skull-stripped) anatomical
        subject_lesion       -- lesion in native space (binarized)
        subject_img_template -- anatomical warped to template space
        lesion_template      -- lesion warped to template space
        template_img         -- template image used
        transforms           -- dict with 'fwd' and 'inv' transform lists
        (subject_brain_mask) -- brain mask, only when skull_strip=True

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    FileNotFoundError
        If template is not provided and bundled templates are missing.
    """
    ants = _require_ants()

    def _load(x, name):
        if x is None:
            return None
        if isinstance(x, (str, Path)):
            if show_info:
                _info(f'Loading {name}...')
            return ants.image_read(str(x))
        # nibabel → ANTs
        if isinstance(x, nib.Nifti1Image):
            return ants.from_nibabel(x)
        # already an ANTs image
        return x

    def _info(msg):
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{ts}] {msg}')

    # ── resolve template ─────────────────────────────────────────────────────
    if template is None:
        template_dir = get_mni152_template_path()
        if template_dir is None:
            raise FileNotFoundError(
                "No template provided and bundled MNI152 templates not found.\n"
                "Either:\n"
                "  1. Pass template= explicitly, or\n"
                f"  2. Download MNI152 2009c from {_MNI152_URL}\n"
                "     and place the files in:\n"
                "     <lesymap package>/data/templates/MNI152_2009c/"
            )
        if show_info:
            _info(f'Using bundled MNI152_2009c template from {template_dir}')
        template = template_dir / _MNI152_FILES['template']
        if template_brain_mask is None:
            template_brain_mask = template_dir / _MNI152_FILES['brain_mask']
        if template_reg_mask is None:
            template_reg_mask = template_dir / _MNI152_FILES['reg_mask']

    # ── load images ──────────────────────────────────────────────────────────
    sub_img = _load(subject_anatomical, "subject anatomical")
    sub_les = _load(subject_lesion, "subject lesion")
    tmpl_img = _load(template, "template")
    tmpl_mask = _load(template_brain_mask, "template brain mask")
    tmpl_reg_mask = _load(template_reg_mask, "template registration mask")

    if tmpl_mask is None and skull_strip:
        warnings.warn(
            "template_brain_mask not provided — skull_strip forced to False."
        )
        skull_strip = False

    # binarize lesion and masks
    sub_les = ants.threshold_image(sub_les, 0.1, float('inf'))
    if tmpl_mask is not None:
        tmpl_mask = ants.threshold_image(tmpl_mask, 0.1, float('inf'))
    if tmpl_reg_mask is not None:
        tmpl_reg_mask = ants.threshold_image(tmpl_reg_mask, 0.1, float('inf'))

    # ── bias correction ──────────────────────────────────────────────────────
    if show_info:
        _info('Running N4 bias correction on anatomical...')
    sub_img = ants.n4_bias_field_correction(sub_img)

    # ── skull stripping ──────────────────────────────────────────────────────
    sub_brain_mask = None
    if skull_strip:
        if show_info:
            _info('Skull-stripping subject image...')
        # quick BET-style extraction via ANTs brain extraction
        bet = ants.brain_extraction(sub_img, modality='t1')
        sub_brain_mask = ants.threshold_image(bet, 0.5, float('inf'))
        # include lesion in brain mask, then dilate by 2 voxels
        sub_brain_mask = ants.threshold_image(sub_brain_mask + sub_les, 0.5, float('inf'))
        sub_brain_mask = ants.morphology(sub_brain_mask, operation='dilate', radius=2)
        sub_reg_mask = sub_brain_mask - sub_les
        # mask images
        sub_img = sub_img * sub_brain_mask
        tmpl_img = tmpl_img * tmpl_mask
    else:
        # registration mask = whole image minus lesion
        sub_reg_mask = ants.threshold_image(sub_img, 0.0001, float('inf')) - sub_les

    if output_prefix:
        if show_info:
            _info(f'Saving intermediates to {output_prefix}*')
        if sub_brain_mask is not None:
            ants.image_write(sub_brain_mask, f'{output_prefix}_subBrainMask.nii.gz')
        ants.image_write(sub_img, f'{output_prefix}_subBrainOnly.nii.gz')
        ants.image_write(sub_reg_mask, f'{output_prefix}_subRegMask.nii.gz')

    # ── registration ─────────────────────────────────────────────────────────
    if show_info:
        est = '(~5-15 min)' if type_of_transform == 'SyN' else '(~1-2 hours)'
        _info(f'Running {type_of_transform} registration {est}...')

    reg = ants.registration(
        fixed=sub_img,
        moving=tmpl_img,
        type_of_transform=type_of_transform,
        mask=sub_reg_mask,
        outprefix=output_prefix or '',
    )

    if output_prefix:
        ants.image_write(reg['warpedmovout'], f'{output_prefix}_templateInSubjectSpace.nii.gz')

    # ── apply inverse transform to lesion ────────────────────────────────────
    if show_info:
        _info('Applying inverse transform to lesion map...')

    lesion_template = ants.apply_transforms(
        fixed=tmpl_img,
        moving=sub_les,
        transformlist=reg['invtransforms'],
        whichtoinvert=[True, False],
        interpolator='nearestNeighbor',
    )

    if output_prefix:
        ants.image_write(lesion_template, f'{output_prefix}_lesionTemplate.nii.gz')

    # ── report sizes ─────────────────────────────────────────────────────────
    if show_info:
        spacing_native = np.prod(sub_les.spacing) / 1000
        spacing_template = np.prod(lesion_template.spacing) / 1000
        vol_native = float(sub_les.numpy().sum()) * spacing_native
        vol_template = float(lesion_template.numpy().sum()) * spacing_template
        _info(f'Lesion volume native:   {vol_native:.2f} ml')
        _info(f'Lesion volume template: {vol_template:.2f} ml')
        _info('Registration complete.')

    output = {
        'subject_img': reg['warpedfixout'],
        'subject_lesion': sub_les,
        'subject_img_template': reg['warpedmovout'],
        'lesion_template': lesion_template,
        'template_img': tmpl_img,
        'transforms': {
            'fwd': reg['fwdtransforms'],
            'inv': reg['invtransforms'],
        },
    }
    if sub_brain_mask is not None:
        output['subject_brain_mask'] = sub_brain_mask

    return output


def register_batch(
    subject_anatomicals: list,
    subject_lesions: list,
    template: Union[str, Path, None] = None,
    template_brain_mask: Union[str, Path, None] = None,
    template_reg_mask: Union[str, Path, None] = None,
    skull_strip: bool = True,
    type_of_transform: str = 'SyN',
    output_dir: Optional[Union[str, Path]] = None,
    subject_ids: Optional[list] = None,
    n_jobs: int = 1,
    show_info: bool = True,
) -> Tuple[list, list]:
    """
    Register a batch of subjects' lesions to template space.

    Parameters
    ----------
    subject_anatomicals : list of str/Path/nibabel
        One anatomical image per subject.
    subject_lesions : list of str/Path/nibabel
        One lesion map per subject (same order).
    template, template_brain_mask, template_reg_mask
        Passed through to register_lesion_to_template().
    skull_strip : bool
        See register_lesion_to_template().
    type_of_transform : str
        See register_lesion_to_template().
    output_dir : str or Path, optional
        Directory to save per-subject outputs. Created if it doesn't exist.
    subject_ids : list of str, optional
        Subject identifiers used for output file naming.
        Defaults to ['sub-001', 'sub-002', ...].
    n_jobs : int
        Number of parallel jobs. Requires joblib. Default 1 (sequential).
    show_info : bool
        Print per-subject progress.

    Returns
    -------
    registered_lesions : list of ANTs images
        Lesion maps in template space (one per subject).
    results : list of dict
        Full result dicts from register_lesion_to_template().
    """
    if len(subject_anatomicals) != len(subject_lesions):
        raise ValueError(
            f"subject_anatomicals ({len(subject_anatomicals)}) and "
            f"subject_lesions ({len(subject_lesions)}) must have the same length."
        )

    n = len(subject_anatomicals)
    if subject_ids is None:
        subject_ids = [f'sub-{i+1:03d}' for i in range(n)]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def _run_one(i):
        sid = subject_ids[i]
        prefix = str(output_dir / sid) if output_dir else None
        if show_info:
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{ts}] Processing subject {i+1}/{n}: {sid}')
        return register_lesion_to_template(
            subject_anatomical=subject_anatomicals[i],
            subject_lesion=subject_lesions[i],
            template=template,
            template_brain_mask=template_brain_mask,
            template_reg_mask=template_reg_mask,
            skull_strip=skull_strip,
            type_of_transform=type_of_transform,
            output_prefix=prefix,
            show_info=False,  # suppress per-step noise in batch mode
        )

    if n_jobs == 1:
        results = [_run_one(i) for i in range(n)]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn("joblib not available; falling back to n_jobs=1.")
            results = [_run_one(i) for i in range(n)]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(_run_one)(i) for i in range(n))

    registered_lesions = [r['lesion_template'] for r in results]
    return registered_lesions, results
