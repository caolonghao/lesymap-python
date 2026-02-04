"""
Result class for LESYMAP-Python.

Provides result storage, model checkpointing, and inference support.
Follows scikit-learn's API pattern for save/load/predict.
"""

import os
import json
import pickle
from pathlib import Path


__all__ = [
    'LesymapResult',
]
from typing import List, Union, Optional, Dict, Any

import numpy as np
import nibabel as nib
import joblib
from sklearn.linear_model import LinearRegression


class LesymapResult:
    """
    Result class with model checkpointing and inference support.

    This class stores statistical maps from lesion-symptom mapping
    and provides methods for saving results and making predictions
    on new lesion maps.

    Attributes
    ----------
    stat_img : nibabel image
        Statistical map (z-scores, t-statistics, etc.)
    mask_img : nibabel image
        Analysis mask
    pval_img : nibabel image, optional
        P-value map
    zmap_img : nibabel image, optional
        Z-score map
    raw_weights_img : nibabel image, optional
        Raw model weights (for SCCAN, SVR)
    method : str
        Analysis method used
    model_params : dict
        Model-specific parameters
    callinfo : dict
        Analysis call information
    patchinfo : dict, optional
        Patch information if patches were used

    Examples
    --------
    >>> result = lesymap.lesymap(lesions, behavior, method='sccan')
    >>> result.save('output_dir/', save_model=True)
    >>>
    >>> # Later, load and predict
    >>> loaded = LesymapResult.load_checkpoint('output_dir/model_checkpoint.pkl')
    >>> predictions = loaded.predict(new_lesions)
    """

    def __init__(self,
                 stat_img: nib.Nifti1Image,
                 mask_img: nib.Nifti1Image,
                 method: str,
                 **kwargs):
        # Statistical maps
        self.stat_img = stat_img
        self.mask_img = mask_img
        self.pval_img = kwargs.get('pval_img')
        self.zmap_img = kwargs.get('zmap_img')
        self.raw_weights_img = kwargs.get('raw_weights_img')

        # Model-specific parameters
        self.method = method
        self.model_params = kwargs.get('model_params', {})

        # SCCAN-specific
        self.sccan_weights = kwargs.get('sccan_weights')  # Voxel weights
        self.sccan_behavior_scale = kwargs.get('sccan_behavior_scale')
        self.sccan_behavior_center = kwargs.get('sccan_behavior_center')
        self.sccan_lesmat_scale = kwargs.get('sccan_lesmat_scale')
        self.sccan_lesmat_center = kwargs.get('sccan_lesmat_center')
        self.sccan_predict_lm = kwargs.get('sccan_predict_lm')  # Linear calibration

        # SVR-specific
        self.svr_model = kwargs.get('svr_model')  # sklearn SVR model

        # Regression-specific
        self.regression_coef = kwargs.get('regression_coef')
        self.regression_intercept = kwargs.get('regression_intercept')

        # Metadata
        self.callinfo = kwargs.get('callinfo', {})
        self.patchinfo = kwargs.get('patchinfo')

    def save(self,
             output_dir: Union[str, Path],
             save_model: bool = True,
             format: str = 'nifti') -> None:
        """
        Save results to directory.

        Parameters
        ----------
        output_dir : str or Path
            Output directory
        save_model : bool
            Whether to save model checkpoint for inference
        format : str
            Output format ('nifti', 'nifti_gz', or 'nrrd')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine extension
        if format == 'nifti_gz':
            ext = '.nii.gz'
        elif format == 'nrrd':
            ext = '.nrrd'
        else:
            ext = '.nii'

        # Save statistical maps
        nib.save(self.stat_img, output_dir / f'stat{ext}')

        if self.pval_img is not None:
            nib.save(self.pval_img, output_dir / f'pval{ext}')
        if self.zmap_img is not None:
            nib.save(self.zmap_img, output_dir / f'zmap{ext}')
        if self.raw_weights_img is not None:
            nib.save(self.raw_weights_img, output_dir / f'weights{ext}')

        # Save mask
        nib.save(self.mask_img, output_dir / f'mask{ext}')

        # Save metadata as JSON
        metadata = {
            'method': self.method,
            'callinfo': self._serialize_callinfo(self.callinfo),
            'model_params': self.model_params,
        }
        if self.patchinfo is not None:
            metadata['patchinfo'] = {
                k: str(v) if isinstance(v, np.ndarray) else v
                for k, v in self.patchinfo.items()
            }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save model checkpoint
        if save_model:
            self.save_checkpoint(output_dir / 'model_checkpoint.pkl')

    def _serialize_callinfo(self, callinfo: Dict) -> Dict:
        """Serialize callinfo for JSON."""
        serialized = {}
        for k, v in callinfo.items():
            if isinstance(v, np.ndarray):
                serialized[k] = v.tolist()
            elif isinstance(v, nib.Nifti1Image):
                serialized[k] = f"<NiftiImage: shape={v.shape}>"
            else:
                try:
                    json.dumps(v)
                    serialized[k] = v
                except (TypeError, ValueError):
                    serialized[k] = str(v)
        return serialized

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """
        Save model checkpoint for later inference.

        Similar to sklearn's joblib.dump(), but saves all necessary
        components for prediction including weights, scaling parameters,
        and calibration models.

        Parameters
        ----------
        filepath : str or Path
            Path to save checkpoint file
        """
        # Extract mask data for voxel indexing
        mask_data = self.mask_img.get_fdata() if self.mask_img is not None else None

        checkpoint_data = {
            'method': self.method,
            # SCCAN components
            'sccan_weights': self.sccan_weights,
            'sccan_behavior_scale': self.sccan_behavior_scale,
            'sccan_behavior_center': self.sccan_behavior_center,
            'sccan_lesmat_scale': self.sccan_lesmat_scale,
            'sccan_lesmat_center': self.sccan_lesmat_center,
            'sccan_predict_lm': self.sccan_predict_lm,
            # SVR components
            'svr_model': self.svr_model,
            # Regression components
            'regression_coef': self.regression_coef,
            'regression_intercept': self.regression_intercept,
            # Metadata
            'model_params': self.model_params,
            'callinfo': self.callinfo,
            'patchinfo': self.patchinfo,
            'mask': mask_data,
            'mask_affine': self.mask_img.affine if self.mask_img is not None else None,
            'mask_header': self.mask_img.header if self.mask_img is not None else None,
        }

        joblib.dump(checkpoint_data, filepath)

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path]) -> 'LesymapResult':
        """
        Load a saved model checkpoint.

        Parameters
        ----------
        filepath : str or Path
            Path to saved checkpoint file

        Returns
        -------
        LesymapResult
            Loaded result object with predict() capability
        """
        checkpoint_data = joblib.load(filepath)

        # Reconstruct mask image
        mask_img = None
        if checkpoint_data['mask'] is not None:
            mask_img = nib.Nifti1Image(
                checkpoint_data['mask'],
                checkpoint_data['mask_affine'],
                checkpoint_data['mask_header']
            )

        # Create result object from checkpoint
        result = cls(
            stat_img=None,  # Not in checkpoint
            mask_img=mask_img,
            method=checkpoint_data['method'],
            sccan_weights=checkpoint_data.get('sccan_weights'),
            sccan_behavior_scale=checkpoint_data.get('sccan_behavior_scale'),
            sccan_behavior_center=checkpoint_data.get('sccan_behavior_center'),
            sccan_lesmat_scale=checkpoint_data.get('sccan_lesmat_scale'),
            sccan_lesmat_center=checkpoint_data.get('sccan_lesmat_center'),
            sccan_predict_lm=checkpoint_data.get('sccan_predict_lm'),
            svr_model=checkpoint_data.get('svr_model'),
            regression_coef=checkpoint_data.get('regression_coef'),
            regression_intercept=checkpoint_data.get('regression_intercept'),
            model_params=checkpoint_data.get('model_params', {}),
            callinfo=checkpoint_data.get('callinfo', {}),
            patchinfo=checkpoint_data.get('patchinfo'),
        )

        return result

    def predict(self, new_lesions: Union[List[str], List[nib.Nifti1Image]]) -> np.ndarray:
        """
        Predict behavioral scores for new lesion maps.

        Parameters
        ----------
        new_lesions : list of str or nibabel images
            New lesion maps (must be in same template space as training data)

        Returns
        -------
        np.ndarray
            Predicted behavioral scores

        Raises
        ------
        ValueError
            If prediction is not supported for the method used

        Examples
        --------
        >>> result = lesymap.lesymap(lesions, behavior, method='sccan')
        >>> result.save('output/', save_model=True)
        >>>
        >>> # Later
        >>> loaded = LesymapResult.load_checkpoint('output/model_checkpoint.pkl')
        >>> predictions = loaded.predict(['new_sub1.nii.gz', 'new_sub2.nii.gz'])
        """
        if self.method == 'sccan':
            return self._predict_sccan(new_lesions)
        elif self.method == 'svr':
            return self._predict_svr(new_lesions)
        elif self.method in ['regres', 'regresfast']:
            return self._predict_regression(new_lesions)
        else:
            raise ValueError(
                f"Prediction not supported for method: {self.method}. "
                "Supported methods are: sccan, svr, regres, regresfast"
            )

    def _predict_sccan(self, new_lesions: Union[List[str], List[nib.Nifti1Image]]) -> np.ndarray:
        """SCCAN prediction with linear calibration."""
        if self.sccan_weights is None:
            raise ValueError("SCCAN weights not available for prediction")

        # Load lesions and convert to matrix
        mask_data = self.mask_img.get_fdata()
        voxel_indices = np.where(mask_data > 0)

        # Build lesion matrix for new subjects
        lesmat = np.array([
            self._load_and_extract(img, voxel_indices)
            for img in new_lesions
        ])

        # Scale lesion matrix using training parameters
        lesmat_scaled = (lesmat - self.sccan_lesmat_center) / self.sccan_lesmat_scale

        # Compute weighted scores
        weighted_scores = lesmat_scaled @ self.sccan_weights

        # Apply linear calibration if available
        if self.sccan_predict_lm is not None:
            predictions = self.sccan_predict_lm.predict(weighted_scores.reshape(-1, 1))
        else:
            predictions = weighted_scores

        return predictions

    def _predict_svr(self, new_lesions: Union[List[str], List[nib.Nifti1Image]]) -> np.ndarray:
        """SVR prediction using sklearn model."""
        if self.svr_model is None:
            raise ValueError("SVR model not available for prediction")

        mask_data = self.mask_img.get_fdata()
        voxel_indices = np.where(mask_data > 0)

        lesmat = np.array([
            self._load_and_extract(img, voxel_indices)
            for img in new_lesions
        ])

        predictions = self.svr_model.predict(lesmat)
        return predictions

    def _predict_regression(self, new_lesions: Union[List[str], List[nib.Nifti1Image]]) -> np.ndarray:
        """Linear regression prediction."""
        if self.regression_coef is None:
            raise ValueError("Regression coefficients not available for prediction")

        mask_data = self.mask_img.get_fdata()
        voxel_indices = np.where(mask_data > 0)

        lesmat = np.array([
            self._load_and_extract(img, voxel_indices)
            for img in new_lesions
        ])

        predictions = lesmat @ self.regression_coef
        if self.regression_intercept is not None:
            predictions += self.regression_intercept

        return predictions

    def _load_and_extract(self, img: Union[str, nib.Nifti1Image],
                          voxel_indices: tuple) -> np.ndarray:
        """Load image and extract voxel values."""
        if isinstance(img, str):
            img = nib.load(img)
        data = img.get_fdata()
        return data[voxel_indices]

    def __repr__(self) -> str:
        """String representation of the result."""
        info = [
            f"LesymapResult(method='{self.method}')",
        ]

        if self.stat_img is not None:
            info.append(f"  Statistical map: shape={self.stat_img.shape}")

        if self.pval_img is not None:
            info.append(f"  P-value map: shape={self.pval_img.shape}")

        if self.method in ['sccan', 'svr', 'regres', 'regresfast']:
            info.append(f"  Prediction: supported")

        return '\n'.join(info)
