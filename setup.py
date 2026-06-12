"""
LESYMAP-Python: Lesion-Symptom Mapping in Python

A Python implementation of LESYMAP for lesion-symptom mapping analysis
in neuroimaging research.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_file):
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = __doc__

setup(
    name='lesymap',
    version='0.1.0',
    description='Lesion-Symptom Mapping for Neuroimaging Research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='LESYMAP Python Contributors',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'nibabel>=5.0',
        'scipy>=1.9',
        'statsmodels>=0.13',
        'numba>=0.56',
        'pandas>=1.4',
        'scikit-learn>=1.0',
        'joblib>=1.2',
        'tqdm>=4.64',
    ],
    extras_require={
        'sccan': [
            # ANTsPy for SCCAN method and registration
            # Note: The package name may vary by installation method
            # For conda: conda install -c conda-forge antspy
            # For pip: pip install antspyx (recommended)
            'antspyx>=0.3.6',
        ],
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=3.0',
            'black>=22.0',
            'ruff>=0.0.200',
            'mypy>=0.950',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
