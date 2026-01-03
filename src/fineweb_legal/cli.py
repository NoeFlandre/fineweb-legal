"""FineWeb-Legal CLI - Annotation pipeline for legal value classification."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from fineweb_legal import __version__
from fineweb_legal.config import AnnotationConfig, load_config
from fineweb_legal.pipeline import run_pipeline
from fineweb_legal.stats import display_batch_inventory, display_stats
from fineweb_legal.storage import AnnotationStorage
from fineweb_legal.validation import display_validation_result, show_samples, validate_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("fineweb_legal")

console = Console()
app = typer.Typer(
    name="fineweb-legal",
    help="FineWeb-Legal annotation pipeline - creating legal value scores for web documents.",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"fineweb-legal v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """FineWeb-Legal: Create annotated legal datasets from FineWeb."""
    pass


@app.command()
def annotate(
    samples: int = typer.Option(
        100_000,
        "--samples",
        "-n",
        help="Target number of samples to annotate.",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Documents per batch file.",
    ),
    output_dir: Path = typer.Option(
        Path("./data"),
        "--output-dir",
        "-o",
        help="Output directory for batch files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate without making API calls.",
    ),
    request_delay: float = typer.Option(
        0.5,
        "--delay",
        "-d",
        help="Delay between API requests (seconds).",
    ),
    truncation: int = typer.Option(
        3000,
        "--truncation",
        "-t",
        help="Max characters sent to API for scoring.",
    ),
) -> None:
    """Run the annotation pipeline.
    
    Streams documents from FineWeb, annotates them using Mistral-Medium,
    and saves results to partitioned Parquet files.
    """
    try:
        config = load_config(
            output_dir=output_dir,
            target_samples=samples,
            batch_size=batch_size,
            request_delay=request_delay,
            inference_truncation_chars=truncation,
        )
        config.ensure_directories()
        
        # Check for API key if not dry-run
        if not dry_run and config.mistral_api_key is None:
            console.print("[red]Error: MISTRAL_API_KEY is required for live runs.[/red]")
            console.print("[dim]Set it in .env file or as environment variable.[/dim]")
            console.print("[dim]Use --dry-run for testing without API calls.[/dim]")
            sys.exit(1)
        
        console.print(Panel(
            f"[bold]Configuration[/bold]\n\n"
            f"Model: {config.mistral_model}\n"
            f"Target: {samples:,} samples\n"
            f"Batch size: {batch_size}\n"
            f"Request delay: {request_delay}s\n"
            f"Truncation: {truncation} chars\n"
            f"Output: {output_dir}",
            title="⚙️ Settings",
            border_style="blue",
        ))
        
        # Run the async pipeline with explicit overrides
        asyncio.run(run_pipeline(
            config, 
            target_samples=samples, 
            batch_size=batch_size,
            dry_run=dry_run
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Pipeline failed")
        sys.exit(1)


@app.command()
def resume(
    output_dir: Path = typer.Option(
        Path("./data"),
        "--output-dir",
        "-o",
        help="Output directory containing batch files.",
    ),
    samples: int = typer.Option(
        100_000,
        "--samples",
        "-n",
        help="Target number of samples.",
    ),
    request_delay: float = typer.Option(
        0.5,
        "--delay",
        "-d",
        help="Delay between API requests (seconds).",
    ),
) -> None:
    """Resume annotation from last completed batch.
    
    Counts existing batch files and continues from where it left off.
    """
    storage = AnnotationStorage(output_dir)
    next_batch = storage.get_next_batch_number()
    total = storage.get_total_annotated()
    
    console.print(Panel(
        f"[bold]Resume Point[/bold]\n\n"
        f"Existing batches: {next_batch - 1}\n"
        f"Total annotated: {total:,}\n"
        f"Target: {samples:,}\n"
        f"Remaining: {max(0, samples - total):,}",
        title="📂 State",
        border_style="blue",
    ))
    
    if total >= samples:
        console.print("[green]✓ Target already reached! Nothing to resume.[/green]")
        return
    
    # Run annotate with the same settings
    try:
        config = load_config(
            output_dir=output_dir,
            target_samples=samples,
            request_delay=request_delay,
        )
        
        # Check for API key
        if config.mistral_api_key is None:
            console.print("[red]Error: MISTRAL_API_KEY is required.[/red]")
            console.print("[dim]Set it in .env file or as environment variable.[/dim]")
            sys.exit(1)
        
        asyncio.run(run_pipeline(config, target_samples=samples))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Resume failed")
        sys.exit(1)


@app.command()
def merge(
    output_dir: Path = typer.Option(
        Path("./data"),
        "--output-dir",
        "-o",
        help="Output directory containing batch files.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-f",
        help="Output file path (default: output_dir/fineweb_legal_annotations.parquet).",
    ),
) -> None:
    """Merge all batch files into a single Parquet file.
    
    Combines all batch_XXXXXX.parquet files into one file for easier
    distribution and loading.
    """
    storage = AnnotationStorage(output_dir)
    batch_files = storage.get_batch_files()
    
    if not batch_files:
        console.print("[red]No batch files found to merge.[/red]")
        sys.exit(1)
    
    if output_file is None:
        output_file = output_dir / "fineweb_legal_annotations.parquet"
    
    console.print(f"Merging {len(batch_files)} batch files...")
    
    try:
        count = storage.merge_all_batches(output_file)
        console.print(f"[green]✓ Merged {count:,} samples to {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]Merge failed: {e}[/red]")
        sys.exit(1)


@app.command()
def stats(
    output_dir: Path = typer.Option(
        Path("./data"),
        "--output-dir",
        "-o",
        help="Output directory containing batch files.",
    ),
    target: int = typer.Option(
        100_000,
        "--target",
        "-t",
        help="Target samples for progress calculation.",
    ),
    inventory: bool = typer.Option(
        False,
        "--inventory",
        "-i",
        help="Show batch file inventory.",
    ),
) -> None:
    """Display annotation statistics.
    
    Shows score distribution, token usage, and cost estimates.
    """
    if inventory:
        display_batch_inventory(output_dir)
    else:
        display_stats(output_dir, target_samples=target)


@app.command()
def validate(
    output_dir: Path = typer.Option(
        Path("./data"),
        "--output-dir",
        "-o",
        help="Output directory containing batch files.",
    ),
) -> None:
    """Run quality checks on annotation data.
    
    Checks for file integrity, valid scores, duplicates, and more.
    """
    is_valid, issues = validate_data(output_dir)
    display_validation_result(is_valid, issues)
    
    if not is_valid:
        sys.exit(1)


@app.command()
def samples(
    output_dir: Path = typer.Option(
        Path("./data"),
        "--output-dir",
        "-o",
        help="Output directory containing batch files.",
    ),
    score: Optional[int] = typer.Option(
        None,
        "--score",
        "-s",
        min=0,
        max=5,
        help="Filter by score (0-5).",
    ),
    count: int = typer.Option(
        5,
        "--count",
        "-n",
        help="Number of samples to show.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible sampling.",
    ),
) -> None:
    """Show sample annotations for review.
    
    Display randomly selected samples, optionally filtered by score.
    """
    show_samples(output_dir, score=score, count=count, random_seed=seed)


@app.command()
def train(
    data_path: Path = typer.Option(
        Path("./data/batches"),
        "--data",
        "-d",
        help="Path to annotation data (parquet file or batches directory).",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Number of training epochs.",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Training batch size.",
    ),
    learning_rate: float = typer.Option(
        3e-4,
        "--lr",
        help="Learning rate.",
    ),
    max_length: int = typer.Option(
        2048,
        "--max-length",
        "-m",
        help="Maximum sequence length.",
    ),
    lora_rank: int = typer.Option(
        16,
        "--lora-r",
        help="LoRA rank.",
    ),
    save_dir: Path = typer.Option(
        Path("./models"),
        "--save-dir",
        "-s",
        help="Directory to save model checkpoints.",
    ),
    log_dir: Path = typer.Option(
        Path("./logs"),
        "--log-dir",
        help="Directory for TensorBoard logs.",
    ),
) -> None:
    """Train the legal quality classifier.
    
    Uses LoRA fine-tuning on Gemma Embedding 300M with the annotated data.
    Tracks per-epoch metrics and saves best model by validation F1.
    """
    from fineweb_legal.classifier.dataset import create_dataloaders
    from fineweb_legal.classifier.model import LegalClassifier
    from fineweb_legal.classifier.trainer import ClassifierTrainer, TrainingConfig
    
    console.print(Panel(
        f"[bold]Training Configuration[/bold]\n\n"
        f"Data: {data_path}\n"
        f"Epochs: {epochs}\n"
        f"Batch size: {batch_size}\n"
        f"Max length: {max_length}\n"
        f"Learning rate: {learning_rate}\n"
        f"LoRA rank: {lora_rank}\n"
        f"Save dir: {save_dir}",
        title="🎯 Phase 2: Classifier Training",
        border_style="green",
    ))
    
    try:
        # Initialize model
        console.print("[dim]Loading model...[/dim]")
        model = LegalClassifier(lora_r=lora_rank)
        
        # Create dataloaders
        console.print("[dim]Loading data...[/dim]")
        train_loader, val_loader = create_dataloaders(
            data_path=data_path,
            tokenizer=model.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
        )
        
        # Get class weights from training set
        class_weights = train_loader.dataset.get_class_weights()
        
        # Configure training
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir=str(save_dir),
            log_dir=str(log_dir),
        )
        
        # Train
        trainer = ClassifierTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            class_weights=class_weights,
        )
        
        results = trainer.train()
        
        console.print(Panel(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Best Binary F1@3: {results['best_val_f1']:.4f}\n"
            f"Model saved to: {save_dir}/best",
            title="✅ Success",
            border_style="green",
        ))
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        logger.exception("Training error")
        sys.exit(1)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint directory.",
    ),
    data_path: Path = typer.Option(
        Path("./data/batches"),
        "--data",
        "-d",
        help="Path to evaluation data.",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Evaluation batch size.",
    ),
    max_length: int = typer.Option(
        2048,
        "--max-length",
        "-m",
        help="Maximum sequence length.",
    ),
) -> None:
    """Evaluate a trained classifier.
    
    Loads a checkpoint and computes metrics on the validation set.
    """
    from fineweb_legal.classifier.dataset import LegalDataset
    from fineweb_legal.classifier.inference import batch_predict
    from fineweb_legal.classifier.metrics import (
        compute_metrics,
        get_classification_report,
        per_class_f1,
    )
    from fineweb_legal.classifier.model import LegalClassifier
    from torch.utils.data import DataLoader
    
    console.print(f"[dim]Loading model from {checkpoint}...[/dim]")
    
    try:
        model = LegalClassifier.from_pretrained(str(checkpoint))
        
        # Load validation data
        val_dataset = LegalDataset(
            data_path=data_path,
            tokenizer=model.tokenizer,
            max_length=max_length,
            split="val",
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Get labels
        labels = [val_dataset.data["score"][i] for i in val_dataset.indices]
        
        # Predict
        console.print("[dim]Running evaluation...[/dim]")
        preds, scores = batch_predict(model, val_loader)
        
        # Compute metrics
        metrics = compute_metrics(preds, labels)
        report = get_classification_report(preds, labels)
        class_f1 = per_class_f1(preds, labels)
        
        # Display results
        console.print("\n[bold]Classification Report:[/bold]")
        console.print(report)
        
        console.print(Panel(
            f"[bold]Evaluation Metrics[/bold]\n\n"
            f"Accuracy: {metrics['accuracy']:.4f}\n"
            f"Macro F1: {metrics['macro_f1']:.4f}\n"
            f"[bold]Binary F1@3: {metrics['binary_f1_3']:.4f}[/bold]\n\n"
            f"[dim]Per-class F1:[/dim]\n"
            + "\n".join(f"  Score {i}: {class_f1[i]:.4f}" for i in range(6)),
            title="📊 Evaluation Results",
            border_style="blue",
        ))
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        logger.exception("Evaluation error")
        sys.exit(1)


if __name__ == "__main__":
    app()
