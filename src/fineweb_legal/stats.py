"""Statistics tracking and reporting."""

from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fineweb_legal.storage import AnnotationStorage

console = Console()


def calculate_stats(storage: AnnotationStorage) -> dict:
    """Calculate comprehensive statistics from batch files.
    
    Args:
        storage: Storage instance to analyze.
        
    Returns:
        Dictionary with statistics.
    """
    batch_files = storage.get_batch_files()
    
    total_samples = 0
    total_tokens = 0
    total_text_length = 0
    score_distribution = {i: 0 for i in range(6)}
    
    for f in batch_files:
        try:
            table = pq.read_table(f)
            total_samples += len(table)
            
            if "input_tokens" in table.column_names:
                tokens = table["input_tokens"].to_pylist()
                total_tokens += sum(t for t in tokens if t is not None)
            
            if "text_length" in table.column_names:
                lengths = table["text_length"].to_pylist()
                total_text_length += sum(l for l in lengths if l is not None)
            
            if "score" in table.column_names:
                scores = table["score"].to_pylist()
                for score in scores:
                    if score is not None and 0 <= score <= 5:
                        score_distribution[score] += 1
        except Exception:
            continue
    
    # Calculate cost estimate (Mistral Medium pricing: ~$2.7/1M input tokens)
    estimated_cost = (total_tokens / 1_000_000) * 2.7
    
    return {
        "total_batches": len(batch_files),
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_text_length": total_text_length,
        "avg_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        "avg_text_length": total_text_length / total_samples if total_samples > 0 else 0,
        "score_distribution": score_distribution,
        "estimated_cost_usd": estimated_cost,
    }


def display_stats(output_dir: Path, target_samples: Optional[int] = None) -> None:
    """Display formatted statistics to console.
    
    Args:
        output_dir: Output directory containing batch files.
        target_samples: Optional target sample count for progress display.
    """
    storage = AnnotationStorage(output_dir)
    stats = calculate_stats(storage)
    
    # Overview panel
    overview_text = f"""[bold]Total Batches:[/bold] {stats['total_batches']}
[bold]Total Samples:[/bold] {stats['total_samples']:,}
[bold]Total Tokens:[/bold] {stats['total_tokens']:,}
[bold]Avg Tokens/Sample:[/bold] {stats['avg_tokens_per_sample']:.1f}
[bold]Avg Text Length:[/bold] {stats['avg_text_length']:.0f} chars
[bold]Estimated Cost:[/bold] ${stats['estimated_cost_usd']:.2f} USD"""
    
    if target_samples:
        progress = (stats["total_samples"] / target_samples) * 100
        overview_text += f"\n[bold]Progress:[/bold] {progress:.1f}% ({stats['total_samples']:,}/{target_samples:,})"
    
    console.print(Panel(overview_text, title="📊 Annotation Statistics", border_style="blue"))
    
    # Score distribution table
    table = Table(title="Score Distribution", show_header=True)
    table.add_column("Score", style="cyan", justify="center")
    table.add_column("Description", style="white")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")
    
    descriptions = {
        0: "Noise/Spam",
        1: "General/Marketing",
        2: "Basic Info",
        3: "Useful",
        4: "High Value",
        5: "Gold Standard",
    }
    
    total = stats["total_samples"]
    for score in range(6):
        count = stats["score_distribution"][score]
        pct = (count / total * 100) if total > 0 else 0
        
        # Add bar visualization
        bar_width = int(pct / 2)  # Scale to max 50 chars
        bar = "█" * bar_width
        
        table.add_row(
            str(score),
            descriptions[score],
            f"{count:,}",
            f"{pct:.1f}% {bar}",
        )
    
    console.print(table)


def display_batch_inventory(output_dir: Path) -> None:
    """Display inventory of batch files.
    
    Args:
        output_dir: Output directory containing batch files.
    """
    storage = AnnotationStorage(output_dir)
    batch_files = storage.get_batch_files()
    
    if not batch_files:
        console.print("[yellow]No batch files found.[/yellow]")
        return
    
    table = Table(title="Batch File Inventory", show_header=True)
    table.add_column("Batch", style="cyan", justify="right")
    table.add_column("Filename", style="white")
    table.add_column("Rows", style="green", justify="right")
    table.add_column("Size", style="yellow", justify="right")
    
    total_rows = 0
    total_size = 0
    
    for f in batch_files:
        try:
            metadata = pq.read_metadata(f)
            rows = metadata.num_rows
            size = f.stat().st_size
            
            batch_num = int(f.stem.split("_")[1])
            
            table.add_row(
                str(batch_num),
                f.name,
                f"{rows:,}",
                _format_size(size),
            )
            
            total_rows += rows
            total_size += size
        except Exception:
            continue
    
    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(batch_files)} files[/bold]",
        f"[bold]{total_rows:,}[/bold]",
        f"[bold]{_format_size(total_size)}[/bold]",
    )
    
    console.print(table)


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
