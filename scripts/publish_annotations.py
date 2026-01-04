#!/usr/bin/env python3
"""
Publish the V2 Annotations Dataset to Hugging Face Hub.
Repository: NoeFlandre/fineweb-legal-annotations
"""

import os
from pathlib import Path
from rich.console import Console
from huggingface_hub import HfApi, create_repo

console = Console()

# Configuration
DATA_DIR = Path("data/v2")
REPO_ID = "NoeFlandre/fineweb-legal-annotations"
DATASET_CARD_PATH = Path("docs/DATASET_CARD_ANNOTATIONS.md")

def main():
    console.print(f"[bold cyan]Publishing to {REPO_ID}...[/]")
    
    # 1. Check HF Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]Error:[/] HF_TOKEN environment variable not set.")
        return

    api = HfApi(token=token)

    # 2. Verify files exist
    required = ["train.parquet", "val.parquet", "test.parquet"]
    for f in required:
        if not (DATA_DIR / f).exists():
            console.print(f"[red]Error:[/] Missing {f} in {DATA_DIR}")
            return
            
    if not DATASET_CARD_PATH.exists():
        console.print(f"[red]Error:[/] Missing Dataset Card at {DATASET_CARD_PATH}")
        return

    # 3. Create Repo (if not exists)
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=token)
        console.print("[green]✓ Repo exists/created[/]")
    except Exception as e:
        console.print(f"[red]Error creating repo:[/] {e}")
        return

    # 4. Upload Files
    console.print("[cyan]Uploading parquet files...[/]")
    try:
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=str(DATA_DIR),
            path_in_repo="data",  # Upload to 'data' folder for neatness
            allow_patterns=["*.parquet", "*.json"],
        )
        console.print("[green]✓ Data uploaded[/]")
    except Exception as e:
        console.print(f"[red]Error uploading data:[/] {e}")
        return

    # 5. Upload Dataset Card (README.md)
    console.print("[cyan]Uploading Dataset Card (README.md)...[/]")
    try:
        api.upload_file(
            repo_id=REPO_ID,
            repo_type="dataset",
            path_or_fileobj=DATASET_CARD_PATH,
            path_in_repo="README.md",
        )
        console.print("[green]✓ README.md uploaded[/]")
    except Exception as e:
        console.print(f"[red]Error uploading README:[/] {e}")
        return

    console.print(f"\n[bold green]Success! View at: https://huggingface.co/datasets/{REPO_ID}[/]")

if __name__ == "__main__":
    main()
