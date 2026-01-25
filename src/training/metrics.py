"""Evaluation metrics for EEG classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_accuracy(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy score between 0 and 1.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_kappa(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """Compute Cohen's Kappa score.

    Kappa is a measure of agreement between predictions and ground truth,
    accounting for chance agreement. Commonly used in BCI evaluation.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Kappa score between -1 and 1.
    """
    return float(cohen_kappa_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        normalize: If True, normalize by row (true labels).

    Returns:
        Confusion matrix of shape (n_classes, n_classes).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    return cm


def compute_all_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy and kappa scores.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "kappa": compute_kappa(y_true, y_pred),
    }
