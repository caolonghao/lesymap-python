"""
Core module for LESYMAP-Python.

Provides I/O, image processing, patch computation, and pipeline functions.
"""

from .io import (
    check_input_type,
    load_lesions,
    load_behavior,
    check_headers_match,
    InputType,
)
from .image_utils import (
    mask_from_average,
    threshold_image,
    average_images,
    images_to_matrix,
    matrix_to_image,
    label_clusters,
)
from .patch import get_unique_lesion_patches
from .result import LesymapResult
from .pipeline import run_lesymap
from .registration import register_lesion_to_template, register_batch, get_mni152_template_path

__all__ = [
    'check_input_type',
    'load_lesions',
    'load_behavior',
    'check_headers_match',
    'InputType',
    'mask_from_average',
    'threshold_image',
    'average_images',
    'images_to_matrix',
    'matrix_to_image',
    'label_clusters',
    'get_unique_lesion_patches',
    'LesymapResult',
    'run_lesymap',
    'register_lesion_to_template',
    'register_batch',
    'get_mni152_template_path',
]
