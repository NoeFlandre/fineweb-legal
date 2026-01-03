"""PyTorch Dataset for legal quality annotations."""

import logging
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class LegalDataset(Dataset):
    """PyTorch Dataset for legal annotation data.
    
    Loads annotations from parquet files and tokenizes text for the classifier.
    """
    
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to parquet file or directory of batch files
            tokenizer: HuggingFace tokenizer for the model
            max_length: Maximum sequence length (default 2048 for Gemma)
            split: "train" or "val" for automatic splitting, None for all data
            train_ratio: Ratio of data for training (default 0.8)
            seed: Random seed for reproducible splits
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        data_path = Path(data_path)
        if data_path.is_dir():
            # Load all batch files from directory
            logger.info(f"Loading batches from {data_path}")
            self.data = self._load_batches(data_path)
        else:
            # Load single parquet file
            logger.info(f"Loading {data_path}")
            table = pq.read_table(data_path)
            self.data = table.to_pydict()
        
        # Validate required columns
        required_cols = {"text", "score"}
        available_cols = set(self.data.keys())
        if not required_cols.issubset(available_cols):
            raise ValueError(f"Missing columns: {required_cols - available_cols}")
        
        # Create index mapping
        n_samples = len(self.data["text"])
        logger.info(f"Loaded {n_samples} samples")
        
        # Handle train/val split
        if split is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(n_samples).tolist()
            split_idx = int(n_samples * train_ratio)
            
            if split == "train":
                self.indices = indices[:split_idx]
            elif split == "val":
                self.indices = indices[split_idx:]
            else:
                raise ValueError(f"split must be 'train' or 'val', got {split}")
            
            logger.info(f"Split '{split}': {len(self.indices)} samples")
        else:
            self.indices = list(range(n_samples))
        
        # Log score distribution
        self._log_score_distribution()
    
    def _load_batches(self, batch_dir: Path) -> dict:
        """Load and merge all batch parquet files."""
        import pyarrow as pa
        
        batch_files = sorted(batch_dir.glob("batch_*.parquet"))
        if not batch_files:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        logger.info(f"Found {len(batch_files)} batch files")
        
        tables = []
        for bf in batch_files:
            tables.append(pq.read_table(bf))
        
        merged = pa.concat_tables(tables)
        return merged.to_pydict()
    
    def _log_score_distribution(self) -> None:
        """Log the distribution of scores in the dataset."""
        scores = [self.data["score"][i] for i in self.indices]
        distribution = {}
        for s in range(6):
            count = scores.count(s)
            pct = 100 * count / len(scores) if scores else 0
            distribution[s] = f"{count} ({pct:.1f}%)"
        logger.info(f"Score distribution: {distribution}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single tokenized sample.
        
        Returns:
            Dict with input_ids, attention_mask, and label tensors
        """
        real_idx = self.indices[idx]
        
        text = self.data["text"][real_idx]
        score = self.data["score"][real_idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(score, dtype=torch.long),
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data.
        
        Returns:
            Tensor of shape [6] with class weights
        """
        scores = [self.data["score"][i] for i in self.indices]
        counts = torch.zeros(6)
        for s in scores:
            counts[s] += 1
        
        # Inverse frequency weighting
        weights = len(scores) / (6 * counts + 1e-6)
        weights = weights / weights.sum() * 6  # Normalize to sum to num_classes
        
        return weights


def create_dataloaders(
    data_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    train_ratio: float = 0.8,
    num_workers: int = 4,
) -> tuple:
    """Create train and validation DataLoaders.
    
    Args:
        data_path: Path to annotations
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        train_ratio: Train/val split ratio
        num_workers: DataLoader workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = LegalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        train_ratio=train_ratio,
    )
    
    val_dataset = LegalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        train_ratio=train_ratio,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
