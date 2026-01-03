#!/usr/bin/env python3
"""
Phase 3 Beta: Stage 2 Model Inference on MPS (Apple Silicon)

A resumable script to run inference on Stage 1 Parquet files using the trained
LoRA classifier on Mac M2 with Metal Performance Shaders (MPS).

Hardware Target: Mac M2 (8GB RAM), MPS acceleration
Input: data/stage1/part_*.parquet
Output: data/stage2/scored_part_*.parquet (only rows with score >= 3.0)

Usage:
    uv run python scripts/model_inference_stage2.py
    uv run python scripts/model_inference_stage2.py --batch-size 16  # Lower if OOM
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Configuration (Hardcoded as requested)
# =============================================================================
# User must verify this path exists before running
BASE_MODEL_ID = "google/embeddinggemma-300m"
ADAPTER_PATH = "models/best"  # User must verify this path exists
CLASSIFIER_HEAD_PATH = "models/best/classifier_head.pt"

INPUT_DIR = Path("data/stage1")
OUTPUT_DIR = Path("data/stage2")

MAX_LENGTH = 2048
BATCH_SIZE = 32
SCORE_THRESHOLD = 3.0  # Only save rows with score >= this value
NUM_CLASSES = 6

# =============================================================================
# Setup
# =============================================================================
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading (MPS Optimized)
# =============================================================================

class LegalClassifierMPS(torch.nn.Module):
    """Lightweight classifier wrapper optimized for MPS inference.
    
    Uses float16 instead of bfloat16 for MPS compatibility.
    """
    
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
        """Convert logits to continuous scores via weighted probability average.
        
        Formula: score = sum(P(class=i) * i) for i in 0..5
        """
        # Use float32 for softmax to avoid MPS precision issues
        probs = torch.softmax(logits.float(), dim=-1)
        weights = torch.arange(
            self.NUM_CLASSES,
            device=logits.device,
            dtype=torch.float32,
        )
        scores = (probs * weights).sum(dim=-1)
        return scores


def load_model_for_mps(
    base_model_id: str,
    adapter_path: str,
    classifier_head_path: str,
    device: str = "mps",
) -> LegalClassifierMPS:
    """Load model optimized for Apple Silicon MPS.
    
    Uses float16 instead of bfloat16 since MPS doesn't fully support bfloat16.
    """
    console.print("[dim]Loading tokenizer...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    console.print("[dim]Loading base model in float16...[/dim]")
    base_model = AutoModel.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,  # Use float16 for MPS
        trust_remote_code=True,
    )
    
    console.print(f"[dim]Loading LoRA adapter from {adapter_path}...[/dim]")
    encoder = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
    )
    
    console.print(f"[dim]Loading classifier head from {classifier_head_path}...[/dim]")
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(768, NUM_CLASSES),
    )
    
    # Load state dict and convert to float16
    state_dict = torch.load(classifier_head_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(torch.float16)
    
    # Create wrapper model
    model = LegalClassifierMPS(encoder, tokenizer, classifier)
    model = model.to(device)
    model.eval()
    
    console.print(f"[green]✓ Model loaded on {device}[/green]")
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
    model: LegalClassifierMPS,
    input_path: Path,
    output_path: Path,
    batch_size: int = 32,
    score_threshold: float = 3.0,
    device: str = "mps",
) -> tuple[int, int, int]:
    """Score all rows in a Parquet file and save passing rows.
    
    Returns:
        Tuple of (total_rows, passed_rows, written_rows)
    """
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
        num_workers=0,  # Single threaded for MPS stability
        pin_memory=False,
    )
    
    all_scores = []
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Scoring {input_path.name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            scores = model.predict_scores(logits)
            
            all_scores.extend(scores.cpu().tolist())
    
    # Add scores to dataframe
    df["score"] = all_scores
    
    # Filter by threshold
    df_passed = df[df["score"] >= score_threshold].copy()
    passed_rows = len(df_passed)
    
    # Select output columns (preserve what exists)
    output_columns = ["id", "text", "score"]
    if "url" in df_passed.columns:
        output_columns.insert(2, "url")
    if "date" in df_passed.columns:
        output_columns.append("date")
    
    # Ensure only existing columns are selected
    output_columns = [c for c in output_columns if c in df_passed.columns]
    df_output = df_passed[output_columns]
    
    # Write output
    if len(df_output) > 0:
        df_output.to_parquet(output_path, compression="snappy", index=False)
        written_rows = len(df_output)
    else:
        written_rows = 0
    
    return total_rows, passed_rows, written_rows


def get_pending_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """Get list of input files that haven't been processed yet.
    
    Returns list of (input_path, output_path) tuples for files that need processing.
    """
    input_files = sorted(input_dir.glob("part_*.parquet"))
    pending = []
    
    for input_path in input_files:
        # Convert part_00001.parquet -> scored_part_00001.parquet
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
        console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
        sys.exit(1)
    
    if not Path(ADAPTER_PATH).exists():
        console.print(f"[red]Error: Adapter path not found: {ADAPTER_PATH}[/red]")
        console.print("[yellow]Please verify the model checkpoint exists at this path.[/yellow]")
        sys.exit(1)
    
    if not Path(CLASSIFIER_HEAD_PATH).exists():
        console.print(f"[red]Error: Classifier head not found: {CLASSIFIER_HEAD_PATH}[/red]")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if not torch.backends.mps.is_available():
        console.print("[red]Error: MPS is not available on this system[/red]")
        console.print("[yellow]This script requires Apple Silicon (M1/M2/M3)[/yellow]")
        sys.exit(1)
    
    device = "mps"
    
    # Get pending files
    pending_files = get_pending_files(input_dir, output_dir)
    
    if not pending_files:
        console.print("[green]✓ All files already processed![/green]")
        return
    
    console.print(Panel(
        f"[bold]Stage 2: Model Inference[/bold]\n\n"
        f"Input: {input_dir}\n"
        f"Output: {output_dir}\n"
        f"Device: {device}\n"
        f"Batch size: {batch_size}\n"
        f"Score threshold: ≥{score_threshold}\n"
        f"Pending files: {len(pending_files)}",
        title="🧠 Legal Classifier Inference",
        border_style="blue",
    ))
    
    # Load model
    model = load_model_for_mps(
        BASE_MODEL_ID,
        ADAPTER_PATH,
        CLASSIFIER_HEAD_PATH,
        device=device,
    )
    
    # Process files
    total_processed = 0
    total_passed = 0
    total_written = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files", total=len(pending_files))
        
        for input_path, output_path in pending_files:
            progress.update(task, description=f"Processing {input_path.name}")
            
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
                console.print(
                    f"  [dim]{input_path.name}: {processed} → {passed} passed "
                    f"({pass_rate:.1f}%) → {output_path.name}[/dim]"
                )
                
            except Exception as e:
                console.print(f"[red]Error processing {input_path.name}: {e}[/red]")
                logger.exception(f"Error processing {input_path}")
                # Continue to next file rather than crashing
                continue
            
            progress.advance(task)
    
    # Final summary
    overall_rate = (total_passed / total_processed * 100) if total_processed > 0 else 0
    console.print(Panel(
        f"[bold]Processed:[/bold] {total_processed:,} rows\n"
        f"[bold]Passed (≥{score_threshold}):[/bold] {total_passed:,} ({overall_rate:.1f}%)\n"
        f"[bold]Written:[/bold] {total_written:,} rows\n"
        f"[bold]Output:[/bold] {output_dir}",
        title="📊 Inference Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    import typer
    
    app = typer.Typer(
        name="model-inference-stage2",
        help="Run model inference on Stage 1 filtered documents.",
        add_completion=False,
    )
    
    @app.command()
    def run(
        input_dir: Path = typer.Option(
            INPUT_DIR,
            "--input-dir", "-i",
            help="Directory containing Stage 1 parquet files",
        ),
        output_dir: Path = typer.Option(
            OUTPUT_DIR,
            "--output-dir", "-o",
            help="Directory to save scored parquet files",
        ),
        batch_size: int = typer.Option(
            BATCH_SIZE,
            "--batch-size", "-b",
            help="Batch size for inference (reduce if OOM)",
        ),
        score_threshold: float = typer.Option(
            SCORE_THRESHOLD,
            "--threshold", "-t",
            help="Minimum score to include in output",
        ),
    ) -> None:
        """Run Stage 2 model inference on filtered documents."""
        main(
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            score_threshold=score_threshold,
        )
    
    app()
