"""Inference utilities for legal quality classifier."""

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def predict_score(logits: torch.Tensor) -> float:
    """Convert logits to continuous score via weighted probability average.
    
    Formula: score = sum(P(class=i) * i) for i in 0..5
    
    Example:
        If P(class=4) = 0.5 and P(class=5) = 0.5, then score = 4.5
    
    Args:
        logits: Raw logits [6] or [batch, 6]
        
    Returns:
        Continuous score in range [0.0, 5.0]
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    probs = torch.softmax(logits, dim=-1)
    weights = torch.arange(6, device=logits.device, dtype=logits.dtype)
    scores = (probs * weights).sum(dim=-1)
    
    if scores.numel() == 1:
        return scores.item()
    return scores


def batch_predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    show_progress: bool = True,
) -> tuple[list[int], list[float]]:
    """Run batch prediction on a dataset.
    
    Args:
        model: LegalClassifier model
        dataloader: DataLoader with samples
        device: Device to use
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (class_predictions, continuous_scores)
    """
    model.eval()
    model = model.to(device)
    
    all_classes = []
    all_scores = []
    
    iterator = tqdm(dataloader, desc="Predicting") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            
            # Class predictions
            classes = torch.argmax(logits, dim=-1).cpu().tolist()
            all_classes.extend(classes)
            
            # Continuous scores via weighted average
            scores = model.predict_scores(logits).cpu().tolist()
            all_scores.extend(scores)
    
    return all_classes, all_scores


class InferenceDataset(Dataset):
    """Simple dataset for inference on raw texts."""
    
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 2048,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def predict_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> list[float]:
    """Predict scores for a list of texts.
    
    Args:
        model: LegalClassifier model
        tokenizer: Model tokenizer
        texts: List of text strings
        batch_size: Batch size for inference
        device: Device to use
        max_length: Max sequence length
        
    Returns:
        List of continuous scores [0.0, 5.0]
    """
    dataset = InferenceDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    _, scores = batch_predict(model, dataloader, device=device)
    return scores


def stream_predict(
    model: torch.nn.Module,
    tokenizer,
    text_iterator: Iterator[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Iterator[tuple[str, float]]:
    """Stream predictions for an iterator of texts.
    
    Yields:
        Tuples of (text, score)
    """
    model.eval()
    model = model.to(device)
    
    batch_texts = []
    
    with torch.no_grad():
        for text in text_iterator:
            batch_texts.append(text)
            
            if len(batch_texts) >= batch_size:
                scores = predict_texts(
                    model, tokenizer, batch_texts,
                    batch_size=batch_size, device=device, max_length=max_length,
                )
                for t, s in zip(batch_texts, scores):
                    yield t, s
                batch_texts = []
        
        # Handle remaining
        if batch_texts:
            scores = predict_texts(
                model, tokenizer, batch_texts,
                batch_size=batch_size, device=device, max_length=max_length,
            )
            for t, s in zip(batch_texts, scores):
                yield t, s
