"""Training loop for legal quality classifier with comprehensive metrics tracking."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fineweb_legal.classifier.metrics import (
    compute_metrics,
    format_metrics,
    get_classification_report,
    per_class_f1,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for classifier training."""
    
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    save_dir: str = "models"
    log_dir: str = "logs"
    use_class_weights: bool = True


class ClassifierTrainer:
    """Trainer for LegalClassifier with metrics tracking and early stopping.
    
    Features:
    - Per-epoch metrics: loss, accuracy, macro F1, binary F1@3
    - TensorBoard logging
    - Early stopping based on validation F1
    - Best model checkpointing
    - Gradient accumulation for large contexts
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[str] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: LegalClassifier instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            device: Device to use (auto-detected if None)
            class_weights: Optional class weights for imbalanced data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Loss function with optional class weights
        if class_weights is not None and config.use_class_weights:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(config.log_dir) / f"run_{timestamp}"
        self.writer = SummaryWriter(log_dir=str(log_path))
        logger.info(f"TensorBoard logs: {log_path}")
        
        # Tracking
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        self.training_history: list[dict] = []
    
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.float(), labels)
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_scores = []
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.float(), labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            scores = self.model.predict_scores(logits).cpu().tolist()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(scores)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss
        
        # Add continuous score stats
        import numpy as np
        scores_arr = np.array(all_scores)
        metrics["score_mean"] = float(np.mean(scores_arr))
        metrics["score_std"] = float(np.std(scores_arr))
        
        return metrics
    
    def train(self) -> dict:
        """Run full training loop.
        
        Returns:
            Training history with all epoch metrics
        """
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.train_loader.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Check for improvement
            if val_metrics["binary_f1_3"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["binary_f1_3"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Store history
            self.training_history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            })
        
        # Final report
        self._final_report()
        self.writer.close()
        
        return {
            "best_val_f1": self.best_val_f1,
            "history": self.training_history,
        }
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Log metrics for an epoch."""
        # Console logging
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{self.config.epochs}")
        logger.info(f"  Train: loss={train_metrics['loss']:.4f} "
                   f"acc={train_metrics['accuracy']:.4f} "
                   f"f1={train_metrics['macro_f1']:.4f}")
        logger.info(f"  Val:   loss={val_metrics['loss']:.4f} "
                   f"acc={val_metrics['accuracy']:.4f} "
                   f"f1={val_metrics['macro_f1']:.4f} "
                   f"binary_f1@3={val_metrics['binary_f1_3']:.4f}")
        
        # TensorBoard logging
        self.writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)
        
        self.writer.add_scalars("Accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"],
        }, epoch)
        
        self.writer.add_scalars("Macro_F1", {
            "train": train_metrics["macro_f1"],
            "val": val_metrics["macro_f1"],
        }, epoch)
        
        self.writer.add_scalar("Binary_F1_at_3", val_metrics["binary_f1_3"], epoch)
        self.writer.add_scalar("Learning_Rate", self.scheduler.get_last_lr()[0], epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save best model checkpoint."""
        save_path = Path(self.config.save_dir) / "best"
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_path))
        
        # Save metrics
        import json
        with open(save_path / "metrics.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "metrics": metrics,
            }, f, indent=2)
        
        logger.info(f"  ✓ New best model saved (binary_f1@3={metrics['binary_f1_3']:.4f})")
    
    def _final_report(self) -> None:
        """Print final training report."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best Binary F1@3: {self.best_val_f1:.4f}")
        logger.info(f"Model saved to: {self.config.save_dir}/best")
        
        # Run final validation with detailed report
        val_metrics = self.validate()
        logger.info("\nFinal Validation Metrics:")
        for k, v in val_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
