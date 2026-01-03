"""Metrics for legal quality classifier evaluation."""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    preds: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> dict[str, float]:
    """Compute comprehensive metrics for classifier evaluation.
    
    Args:
        preds: Predicted class labels (0-5)
        labels: True class labels (0-5)
        
    Returns:
        Dict with accuracy, macro_f1, per_class_f1, and binary_f1_3
    """
    preds = np.array(preds)
    labels = np.array(labels)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "binary_f1_3": binary_f1_at_threshold_3(preds, labels),
    }


def binary_f1_at_threshold_3(
    preds: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> float:
    """Calculate binary F1 treating scores ≥3 as positive.
    
    This is the "Money Metric" used by FineWeb-Edu:
    - Positive: Scores 3, 4, 5 (useful legal content)
    - Negative: Scores 0, 1, 2 (noise/low value)
    
    Args:
        preds: Predicted class labels (0-5)
        labels: True class labels (0-5)
        
    Returns:
        Binary F1 score
    """
    preds = np.array(preds)
    labels = np.array(labels)
    
    binary_preds = (preds >= 3).astype(int)
    binary_labels = (labels >= 3).astype(int)
    
    return f1_score(binary_labels, binary_preds, zero_division=0)


def per_class_f1(
    preds: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> dict[int, float]:
    """Compute F1 score for each class.
    
    Args:
        preds: Predicted labels
        labels: True labels
        
    Returns:
        Dict mapping class index to F1 score
    """
    preds = np.array(preds)
    labels = np.array(labels)
    
    scores = f1_score(labels, preds, average=None, labels=range(6), zero_division=0)
    return {i: float(scores[i]) for i in range(6)}


def get_confusion_matrix(
    preds: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        preds: Predicted labels
        labels: True labels
        
    Returns:
        6x6 confusion matrix
    """
    return confusion_matrix(labels, preds, labels=range(6))


def get_classification_report(
    preds: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> str:
    """Get formatted classification report.
    
    Args:
        preds: Predicted labels
        labels: True labels
        
    Returns:
        Formatted report string
    """
    target_names = [
        "Score 0 (Noise)",
        "Score 1 (Marketing)",
        "Score 2 (Basic)",
        "Score 3 (Useful)",
        "Score 4 (High Value)",
        "Score 5 (Gold)",
    ]
    return classification_report(
        labels,
        preds,
        labels=range(6),
        target_names=target_names,
        zero_division=0,
    )


def format_metrics(metrics: dict[str, float], epoch: Optional[int] = None) -> str:
    """Format metrics for logging.
    
    Args:
        metrics: Dict of metric name to value
        epoch: Optional epoch number
        
    Returns:
        Formatted string for logging
    """
    parts = []
    if epoch is not None:
        parts.append(f"Epoch {epoch}")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{name}: {value:.4f}")
        else:
            parts.append(f"{name}: {value}")
    
    return " | ".join(parts)
