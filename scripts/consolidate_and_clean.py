#!/usr/bin/env python3
"""Consolidate curated + new annotations into a single deduplicated dataset.

This script:
1. Loads curated data (2,800 samples) and new annotations (54K+ samples)
2. Merges and deduplicates by text hash (SHA-256)
3. Creates stratified train/val/test splits (70/15/15)
4. Validates zero overlap between all splits

Usage:
    python scripts/consolidate_and_clean.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.model_selection import train_test_split

console = Console()

# Paths
CURATED_PATH = Path("data/curated/annotations_curated.parquet")
NEW_ANNOTATIONS_DIR = Path("data/new_annotations/batches")
OUTPUT_DIR = Path("data/v2")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_curated_data() -> pd.DataFrame:
    """Load curated annotations."""
    console.print("[cyan]Loading curated data...[/]")
    df = pd.read_parquet(CURATED_PATH)
    console.print(f"  → Loaded {len(df):,} samples from curated")
    return df


def load_new_annotations() -> pd.DataFrame:
    """Load all new annotation batches."""
    console.print("[cyan]Loading new annotations...[/]")
    batch_files = sorted(NEW_ANNOTATIONS_DIR.glob("*.parquet"))
    console.print(f"  → Found {len(batch_files)} batch files")
    
    dfs = []
    for f in batch_files:
        dfs.append(pd.read_parquet(f))
    
    df = pd.concat(dfs, ignore_index=True)
    console.print(f"  → Loaded {len(df):,} samples from new annotations")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by text hash, keeping first occurrence."""
    console.print("[cyan]Deduplicating by text hash...[/]")
    
    original_count = len(df)
    df["text_hash"] = df["text"].apply(compute_hash)
    df = df.drop_duplicates(subset=["text_hash"], keep="first")
    dedup_count = len(df)
    
    removed = original_count - dedup_count
    pct = (removed / original_count) * 100 if original_count > 0 else 0
    
    console.print(f"  → Removed {removed:,} duplicates ({pct:.1f}%)")
    console.print(f"  → Final unique samples: {dedup_count:,}")
    
    return df


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits."""
    console.print("[cyan]Creating stratified splits...[/]")
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["score"],
        random_state=RANDOM_SEED,
    )
    
    # Second split: val vs test (split the remaining 30% into 15%/15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 50% of 30% = 15%
        stratify=temp_df["score"],
        random_state=RANDOM_SEED,
    )
    
    console.print(f"  → Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    console.print(f"  → Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    console.print(f"  → Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def verify_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Verify zero overlap between all splits."""
    console.print("[cyan]Verifying zero overlap...[/]")
    
    train_hashes = set(train_df["text_hash"])
    val_hashes = set(val_df["text_hash"])
    test_hashes = set(test_df["text_hash"])
    
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    all_clean = True
    
    if train_val_overlap:
        console.print(f"  [red]✗ Train/Val overlap: {len(train_val_overlap)} samples[/]")
        all_clean = False
    else:
        console.print("  [green]✓ Train/Val: No overlap[/]")
    
    if train_test_overlap:
        console.print(f"  [red]✗ Train/Test overlap: {len(train_test_overlap)} samples[/]")
        all_clean = False
    else:
        console.print("  [green]✓ Train/Test: No overlap[/]")
    
    if val_test_overlap:
        console.print(f"  [red]✗ Val/Test overlap: {len(val_test_overlap)} samples[/]")
        all_clean = False
    else:
        console.print("  [green]✓ Val/Test: No overlap[/]")
    
    return all_clean


def show_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Show score distribution across splits."""
    table = Table(title="Score Distribution Across Splits")
    table.add_column("Score", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Total", justify="right", style="bold")
    
    for score in range(6):
        train_count = len(train_df[train_df["score"] == score])
        val_count = len(val_df[val_df["score"] == score])
        test_count = len(test_df[test_df["score"] == score])
        total = train_count + val_count + test_count
        
        table.add_row(
            str(score),
            f"{train_count:,}",
            f"{val_count:,}",
            f"{test_count:,}",
            f"{total:,}",
        )
    
    # Totals row
    table.add_row(
        "Total",
        f"{len(train_df):,}",
        f"{len(val_df):,}",
        f"{len(test_df):,}",
        f"{len(train_df) + len(val_df) + len(test_df):,}",
        style="bold green",
    )
    
    console.print(table)


def save_outputs(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Save all outputs."""
    console.print("[cyan]Saving outputs...[/]")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select columns to keep (clean schema)
    keep_cols = ["id", "text", "score", "reasoning", "url", "text_hash"]
    
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in keep_cols if c in df.columns]
        return df[available].copy()
    
    # Save full dataset
    full_clean = clean_df(full_df)
    full_clean.to_parquet(OUTPUT_DIR / "annotations.parquet", index=False)
    console.print(f"  → Saved annotations.parquet ({len(full_clean):,} rows)")
    
    # Save splits
    train_clean = clean_df(train_df)
    train_clean.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    console.print(f"  → Saved train.parquet ({len(train_clean):,} rows)")
    
    val_clean = clean_df(val_df)
    val_clean.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    console.print(f"  → Saved val.parquet ({len(val_clean):,} rows)")
    
    test_clean = clean_df(test_df)
    test_clean.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    console.print(f"  → Saved test.parquet ({len(test_clean):,} rows)")
    
    # Save metadata
    metadata = {
        "total_samples": len(full_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "score_distribution": full_df["score"].value_counts().sort_index().to_dict(),
        "created_at": pd.Timestamp.now().isoformat(),
        "source_files": {
            "curated": str(CURATED_PATH),
            "new_annotations": str(NEW_ANNOTATIONS_DIR),
        },
        "random_seed": RANDOM_SEED,
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    console.print("  → Saved metadata.json")


def main():
    console.print(Panel("[bold cyan]Consolidating Annotations for Classifier V2[/]"))
    
    # Load data
    curated_df = load_curated_data()
    new_df = load_new_annotations()
    
    # Merge
    console.print("\n[cyan]Merging datasets...[/]")
    
    # Normalize columns (curated has fewer columns)
    common_cols = ["id", "text", "score", "reasoning"]
    if "url" in new_df.columns:
        common_cols.append("url")
    
    # Ensure both have the same columns
    for col in common_cols:
        if col not in curated_df.columns:
            curated_df[col] = None
        if col not in new_df.columns:
            new_df[col] = None
    
    combined_df = pd.concat([curated_df[common_cols], new_df[common_cols]], ignore_index=True)
    console.print(f"  → Combined: {len(combined_df):,} samples")
    
    # Deduplicate
    dedup_df = deduplicate(combined_df)
    
    # Create splits
    console.print()
    train_df, val_df, test_df = create_splits(dedup_df)
    
    # Verify no overlap
    console.print()
    if not verify_no_overlap(train_df, val_df, test_df):
        console.print("[red bold]ERROR: Overlap detected! Aborting.[/]")
        return
    
    # Show distribution
    console.print()
    show_distribution(train_df, val_df, test_df)
    
    # Save
    console.print()
    save_outputs(dedup_df, train_df, val_df, test_df)
    
    console.print()
    console.print(Panel(
        f"[bold green]✓ Consolidation Complete![/]\n\n"
        f"Total unique samples: {len(dedup_df):,}\n"
        f"Output directory: {OUTPUT_DIR}",
        title="Success",
    ))


if __name__ == "__main__":
    main()
