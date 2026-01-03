"""Data validation and quality checks."""

import random
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fineweb_legal.storage import AnnotationStorage

console = Console()


def validate_data(output_dir: Path) -> tuple[bool, list[str]]:
    """Run quality checks on annotation data.
    
    Args:
        output_dir: Output directory containing batch files.
        
    Returns:
        Tuple of (is_valid, list of issues).
    """
    storage = AnnotationStorage(output_dir)
    batch_files = storage.get_batch_files()
    issues: list[str] = []
    
    if not batch_files:
        issues.append("No batch files found")
        return False, issues
    
    console.print("[bold]Running validation checks...[/bold]\n")
    
    # Check 1: File integrity
    console.print("  [cyan]1/5[/cyan] Checking file integrity...")
    corrupt_files = []
    for f in batch_files:
        try:
            pq.read_metadata(f)
        except Exception as e:
            corrupt_files.append(f.name)
            issues.append(f"Corrupt file {f.name}: {e}")
    
    if corrupt_files:
        console.print(f"    [red]✗ {len(corrupt_files)} corrupt files[/red]")
    else:
        console.print(f"    [green]✓ All {len(batch_files)} files readable[/green]")
    
    # Check 2: Score range validation
    console.print("  [cyan]2/5[/cyan] Validating score ranges...")
    invalid_scores = 0
    for f in batch_files:
        try:
            table = pq.read_table(f, columns=["score"])
            scores = table["score"].to_pylist()
            for score in scores:
                if score is None or not (0 <= score <= 5):
                    invalid_scores += 1
        except Exception:
            continue
    
    if invalid_scores > 0:
        console.print(f"    [red]✗ {invalid_scores} invalid scores found[/red]")
        issues.append(f"{invalid_scores} samples have invalid scores (not 0-5)")
    else:
        console.print("    [green]✓ All scores in valid range (0-5)[/green]")
    
    # Check 3: Empty text detection
    console.print("  [cyan]3/5[/cyan] Checking for empty texts...")
    empty_texts = 0
    for f in batch_files:
        try:
            table = pq.read_table(f, columns=["text"])
            texts = table["text"].to_pylist()
            for text in texts:
                if not text or len(text.strip()) == 0:
                    empty_texts += 1
        except Exception:
            continue
    
    if empty_texts > 0:
        console.print(f"    [yellow]⚠ {empty_texts} empty texts found[/yellow]")
        issues.append(f"{empty_texts} samples have empty text")
    else:
        console.print("    [green]✓ No empty texts[/green]")
    
    # Check 4: Duplicate ID detection
    console.print("  [cyan]4/5[/cyan] Checking for duplicate IDs...")
    all_ids: set[str] = set()
    duplicate_count = 0
    for f in batch_files:
        try:
            table = pq.read_table(f, columns=["id"])
            ids = table["id"].to_pylist()
            for doc_id in ids:
                if doc_id in all_ids:
                    duplicate_count += 1
                else:
                    all_ids.add(doc_id)
        except Exception:
            continue
    
    if duplicate_count > 0:
        console.print(f"    [yellow]⚠ {duplicate_count} duplicate IDs found[/yellow]")
        issues.append(f"{duplicate_count} duplicate document IDs")
    else:
        console.print(f"    [green]✓ {len(all_ids):,} unique IDs[/green]")
    
    # Check 5: Score distribution sanity
    console.print("  [cyan]5/5[/cyan] Analyzing score distribution...")
    distribution = storage.get_score_distribution()
    total = sum(distribution.values())
    
    # Check for extreme distributions
    warnings = []
    for score, count in distribution.items():
        pct = (count / total * 100) if total > 0 else 0
        if pct > 80:
            warnings.append(f"Score {score} is {pct:.1f}% of data (possible annotation drift)")
    
    if warnings:
        for w in warnings:
            console.print(f"    [yellow]⚠ {w}[/yellow]")
            issues.append(w)
    else:
        console.print("    [green]✓ Score distribution looks healthy[/green]")
    
    console.print()
    
    is_valid = len([i for i in issues if "Corrupt" in i or "invalid scores" in i]) == 0
    return is_valid, issues


def display_validation_result(is_valid: bool, issues: list[str]) -> None:
    """Display validation results.
    
    Args:
        is_valid: Whether data passed validation.
        issues: List of issues found.
    """
    if is_valid:
        console.print(Panel(
            "[bold green]✓ Validation Passed[/bold green]\n\n"
            "Data is ready for use. Minor warnings may exist but won't affect training.",
            title="Validation Result",
            border_style="green",
        ))
    else:
        issue_text = "\n".join(f"• {issue}" for issue in issues)
        console.print(Panel(
            f"[bold red]✗ Validation Failed[/bold red]\n\n{issue_text}",
            title="Validation Result",
            border_style="red",
        ))
    
    if issues:
        console.print(f"\n[dim]Found {len(issues)} issue(s)[/dim]")


def show_samples(
    output_dir: Path,
    score: Optional[int] = None,
    count: int = 5,
    random_seed: Optional[int] = None,
) -> None:
    """Display sample annotations for review.
    
    Args:
        output_dir: Output directory containing batch files.
        score: Optional score to filter by (0-5).
        count: Number of samples to display.
        random_seed: Optional seed for reproducible sampling.
    """
    storage = AnnotationStorage(output_dir)
    batch_files = storage.get_batch_files()
    
    if not batch_files:
        console.print("[yellow]No batch files found.[/yellow]")
        return
    
    # Collect samples
    samples: list[dict] = []
    for f in batch_files:
        try:
            table = pq.read_table(f)
            for i in range(min(len(table), 100)):  # Sample up to 100 per file
                row = {col: table[col][i].as_py() for col in table.column_names}
                if score is None or row.get("score") == score:
                    samples.append(row)
        except Exception:
            continue
    
    if not samples:
        console.print(f"[yellow]No samples found{' with score ' + str(score) if score is not None else ''}.[/yellow]")
        return
    
    # Random sample
    if random_seed is not None:
        random.seed(random_seed)
    selected = random.sample(samples, min(count, len(samples)))
    
    # Display
    title = f"Sample Annotations" + (f" (Score {score})" if score is not None else "")
    console.print(f"\n[bold]{title}[/bold]\n")
    
    for i, sample in enumerate(selected, 1):
        score_val = sample.get("score", "?")
        reasoning = sample.get("reasoning", "N/A")
        text = sample.get("text", "")[:500]  # Truncate for display
        
        # Color based on score
        score_color = {
            0: "red", 1: "red", 2: "yellow",
            3: "blue", 4: "green", 5: "bright_green"
        }.get(score_val, "white")
        
        panel_content = Text()
        panel_content.append(f"Score: ", style="bold")
        panel_content.append(f"{score_val}", style=f"bold {score_color}")
        panel_content.append(f"\nReasoning: ", style="bold")
        panel_content.append(f"{reasoning}\n\n", style="italic")
        panel_content.append(f"{text}...", style="dim")
        
        console.print(Panel(
            panel_content,
            title=f"Sample {i}/{len(selected)} — ID: {sample.get('id', '?')[:20]}...",
            border_style=score_color,
        ))
        console.print()
