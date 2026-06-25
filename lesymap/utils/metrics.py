"""
Prediction evaluation utilities for LESYMAP-Python.
"""

from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Union[str, float] = "youden",
) -> Dict[str, float]:
    """
    Evaluate continuous prediction scores against binary labels.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Binary labels encoded as 0/1.
    scores : array-like, shape (n_samples,)
        Continuous prediction scores. These are treated as risk scores, not
        calibrated probabilities.
    threshold : {'youden', 'mcc'} or float
        Threshold used to binarize scores. String thresholds are selected from
        the given scores and labels, so use them inside training folds only.

    Returns
    -------
    dict
        ROC-AUC, PR-AUC, selected threshold, confusion matrix, and common
        binary classification metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain binary labels encoded as 0/1")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Both binary classes must be present")

    selected_threshold = _select_threshold(y_true, scores, threshold)
    y_pred = (scores >= selected_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(selected_threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Union[str, float],
) -> float:
    if isinstance(threshold, (int, float)):
        return float(threshold)

    if threshold == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        finite = np.isfinite(thresholds)
        youden = tpr - fpr
        best = np.argmax(np.where(finite, youden, -np.inf))
        return float(thresholds[best])

    if threshold == "mcc":
        candidates = np.unique(scores)
        mcc_values = [
            matthews_corrcoef(y_true, (scores >= candidate).astype(int))
            for candidate in candidates
        ]
        return float(candidates[int(np.argmax(mcc_values))])

    raise ValueError("threshold must be 'youden', 'mcc', or a numeric value")
