"""Legal Quality Classifier module - Phase 2 of FineWeb-Legal."""

from fineweb_legal.classifier.model import LegalClassifier
from fineweb_legal.classifier.dataset import LegalDataset
from fineweb_legal.classifier.trainer import ClassifierTrainer
from fineweb_legal.classifier.metrics import compute_metrics, binary_f1_at_threshold_3
from fineweb_legal.classifier.inference import predict_score, batch_predict

__all__ = [
    "LegalClassifier",
    "LegalDataset", 
    "ClassifierTrainer",
    "compute_metrics",
    "binary_f1_at_threshold_3",
    "predict_score",
    "batch_predict",
]
