#!/usr/bin/env python3
"""
Phase 3 Beta: Stage 2 Model Inference (CUDA GPU Optimized)

Optimized for RTX 3090 (24GB VRAM) with 125GB system RAM.
Uses bfloat16 for best performance on Ampere GPUs.

Hardware Target: RTX 3090, 24GB VRAM, 125GB RAM
Input: data/stage1/part_*.parquet
Output: data/stage2/scored_part_*.parquet (only rows with score >= 3.0)

Usage:
    python model_inference_stage2_cuda.py
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Configuration - Optimized for RTX 3090 (24GB) + 125GB RAM
# =============================================================================
BASE_MODEL_ID = "google/embeddinggemma-300m"
ADAPTER_PATH = "models/best"
CLASSIFIER_HEAD_PATH = "models/best/classifier_head.pt"

INPUT_DIR = Path("data/stage1")
OUTPUT_DIR = Path("data/stage2")

MAX_LENGTH = 2048
BATCH_SIZE = 64  # Can go higher with 24GB VRAM
SCORE_THRESHOLD = 3.0
NUM_CLASSES = 6
NUM_WORKERS = 4  # Use multiple workers for data loading

# =============================================================================
# Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage info."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
        }
    return {}


def log_memory_usage(prefix: str = "") -> None:
    """Log current GPU and system memory usage."""
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        logger.info(
            f"{prefix}GPU Memory: {gpu_info['allocated_gb']:.2f}GB allocated, "
            f"{gpu_info['reserved_gb']:.2f}GB reserved, "
            f"{gpu_info['free_gb']:.2f}GB free / {gpu_info['total_gb']:.2f}GB total"
        )


# =============================================================================
# Model Loading (CUDA Optimized)
# =============================================================================

class LegalClassifierCUDA(torch.nn.Module):
    """Classifier wrapper optimized for CUDA with bfloat16."""
    
    EMBEDDING_DIM = 768
    NUM_CLASSES = 6
    
    def __init__(
        self,
        encoder: torch.nn.Module,
        tokenizer,
        classifier_head: torch.nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = classifier_head
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: encode text and classify."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Mean pooling over non-padded tokens
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Ensure consistent dtype with classifier
        pooled = pooled.to(self.classifier[1].weight.dtype)
        
        logits = self.classifier(pooled)
        return logits
    
    def predict_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to continuous scores via weighted probability average."""
        probs = torch.softmax(logits.float(), dim=-1)
        weights = torch.arange(
            self.NUM_CLASSES,
            device=logits.device,
            dtype=torch.float32,
        )
        scores = (probs * weights).sum(dim=-1)
        return scores


def load_model_for_cuda(
    base_model_id: str,
    adapter_path: str,
    classifier_head_path: str,
    device: str = "cuda",
) -> LegalClassifierCUDA:
    """Load model optimized for CUDA with bfloat16."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    logger.info("Loading base model in bfloat16...")
    base_model = AutoModel.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,  # Best for Ampere GPUs (RTX 3090)
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    encoder = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
    
    logger.info(f"Loading classifier head from {classifier_head_path}...")
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.0),  # Disable dropout for inference
        torch.nn.Linear(768, NUM_CLASSES),
    )
    
    state_dict = torch.load(classifier_head_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(torch.bfloat16)
    
    # Create wrapper model
    model = LegalClassifierCUDA(encoder, tokenizer, classifier)
    model = model.to(device)
    model.eval()
    
    # Enable torch compile for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled with torch.compile()")
    except Exception as e:
        logger.warning(f"torch.compile() not available: {e}")
    
    log_memory_usage("After model load: ")
    logger.info(f"Model loaded on {device}")
    return model


# =============================================================================
# Dataset for Batch Processing
# =============================================================================

class ParquetTextDataset(Dataset):
    """Dataset for reading texts from a parquet file."""
    
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


# =============================================================================
# Inference Logic
# =============================================================================

def score_parquet_file(
    model: LegalClassifierCUDA,
    input_path: Path,
    output_path: Path,
    batch_size: int = 64,
    score_threshold: float = 3.0,
    device: str = "cuda",
) -> tuple[int, int, int]:
    """Score all rows in a Parquet file and save passing rows."""
    # Read input file
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    
    if total_rows == 0:
        return 0, 0, 0
    
    texts = df["text"].tolist()
    
    # Create dataset and dataloader
    dataset = ParquetTextDataset(texts, model.tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,
    )
    
    all_scores = []
    
    # Run inference with mixed precision
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in tqdm(dataloader, desc=f"  Scoring {input_path.name}", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            logits = model(input_ids, attention_mask)
            scores = model.predict_scores(logits)
            
            all_scores.extend(scores.cpu().tolist())
    
    # Add scores to dataframe
    df["score"] = all_scores
    
    # Filter by threshold
    df_passed = df[df["score"] >= score_threshold].copy()
    passed_rows = len(df_passed)
    
    # Select output columns
    output_columns = ["id", "text", "score"]
    if "url" in df_passed.columns:
        output_columns.insert(2, "url")
    if "date" in df_passed.columns:
        output_columns.append("date")
    
    output_columns = [c for c in output_columns if c in df_passed.columns]
    df_output = df_passed[output_columns]
    
    # Write output
    if len(df_output) > 0:
        df_output.to_parquet(output_path, compression="snappy", index=False)
        written_rows = len(df_output)
    else:
        written_rows = 0
    
    # Clean up
    del df, df_passed, texts, dataset, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_rows, passed_rows, written_rows


def get_pending_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """Get list of input files that haven't been processed yet."""
    input_files = sorted(input_dir.glob("part_*.parquet"))
    pending = []
    
    for input_path in input_files:
        output_name = f"scored_{input_path.name}"
        output_path = output_dir / output_name
        
        if output_path.exists():
            logger.info(f"Skipping {input_path.name} (already scored)")
        else:
            pending.append((input_path, output_path))
    
    return pending


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    score_threshold: float = SCORE_THRESHOLD,
) -> None:
    """Main inference pipeline."""
    
    # Verify paths
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not Path(ADAPTER_PATH).exists():
        logger.error(f"Adapter path not found: {ADAPTER_PATH}")
        sys.exit(1)
    
    if not Path(CLASSIFIER_HEAD_PATH).exists():
        logger.error(f"Classifier head not found: {CLASSIFIER_HEAD_PATH}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        sys.exit(1)
    
    device = "cuda"
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    log_memory_usage("Initial: ")
    
    # Get pending files
    pending_files = get_pending_files(input_dir, output_dir)
    
    if not pending_files:
        logger.info("All files already processed!")
        return
    
    logger.info(f"Files to process: {len(pending_files)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Score threshold: >= {score_threshold}")
    
    # Load model
    model = load_model_for_cuda(
        BASE_MODEL_ID,
        ADAPTER_PATH,
        CLASSIFIER_HEAD_PATH,
        device=device,
    )
    
    # Process files
    total_processed = 0
    total_passed = 0
    total_written = 0
    start_time = time.time()
    
    for i, (input_path, output_path) in enumerate(pending_files, 1):
        logger.info(f"[{i}/{len(pending_files)}] Processing {input_path.name}")
        
        try:
            processed, passed, written = score_parquet_file(
                model,
                input_path,
                output_path,
                batch_size=batch_size,
                score_threshold=score_threshold,
                device=device,
            )
            
            total_processed += processed
            total_passed += passed
            total_written += written
            
            pass_rate = (passed / processed * 100) if processed > 0 else 0
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            
            logger.info(
                f"  {processed} rows → {passed} passed ({pass_rate:.1f}%) | "
                f"Total: {total_processed:,} @ {rate:.1f} docs/sec"
            )
            log_memory_usage("  ")
            
        except Exception as e:
            logger.exception(f"Error processing {input_path.name}: {e}")
            continue
    
    # Final summary
    elapsed = time.time() - start_time
    overall_rate = (total_passed / total_processed * 100) if total_processed > 0 else 0
    
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"  Processed: {total_processed:,} rows")
    logger.info(f"  Passed (>={score_threshold}): {total_passed:,} ({overall_rate:.1f}%)")
    logger.info(f"  Written: {total_written:,} rows")
    logger.info(f"  Time: {elapsed:.1f}s ({total_processed/elapsed:.1f} docs/sec)")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2 Model Inference (CUDA)")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        score_threshold=args.threshold,
    )
