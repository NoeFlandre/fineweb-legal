
import os
import glob
from huggingface_hub import HfApi, create_repo
import pandas as pd
import shutil

def publish_dataset(
    data_dir="data/stage2",
    repo_id="fineweb-legal-pilot",
    username=None,  # Will try to infer or ask user 
    token=None
):
    print("Preparing to publish dataset to Hugging Face...")
    
    # 1. Collect Files
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        print("No parquet files found to upload.")
        return

    # 2. Setup API
    api = HfApi(token=token)
    
    if not username:
        try:
            whoami = api.whoami()
            username = whoami['name']
            print(f"Authenticated as: {username}")
        except Exception as e:
            print("Error: Not authenticated. Please run 'huggingface-cli login' or provide a token.")
            return

    full_repo_id = f"{username}/{repo_id}"
    
    # 3. Create Repo
    print(f"Creating/Checking repository: {full_repo_id}")
    try:
        create_repo(full_repo_id, repo_type="dataset", exist_ok=True, token=token)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # 4. Upload Files
    print(f"Uploading {len(files)} parquet files...")
    
    # Upload README (Dataset Card)
    if os.path.exists("DATASET_CARD.md"):
        print("Uploading Dataset Card...")
        api.upload_file(
            path_or_fileobj="DATASET_CARD.md",
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="dataset"
        )

    # Upload Data
    # Only upload from stage3_splits if it exists, otherwise fall back to stage2 or custom
    
    upload_path = "data/stage3_splits" if os.path.exists("data/stage3_splits") else data_dir
    print(f"Uploading data from: {upload_path}")

    # Check if this is a split directory (contains subfolders for configs)
    subdirs = [d for d in os.listdir(upload_path) if os.path.isdir(os.path.join(upload_path, d))]
    
    if subdirs and "train.parquet" not in os.listdir(upload_path):
        # Multi-config mode
        print(f"Detected configurations: {subdirs}")
        for config in subdirs:
            print(f"Uploading config: {config}")
            api.upload_folder(
                folder_path=os.path.join(upload_path, config),
                path_in_repo=config,  # Files go to root/{config}/
                repo_id=full_repo_id,
                repo_type="dataset",
                allow_patterns="*.parquet"
            )
    else:
        # Single config mode (flat folder)
        api.upload_folder(
            folder_path=upload_path,
            path_in_repo="data",
            repo_id=full_repo_id,
            repo_type="dataset",
            allow_patterns="*.parquet"
        )
    
    # Upload Analysis if exists
    if os.path.exists("logs/dataset_analysis.json"):
        api.upload_file(
            path_or_fileobj="logs/dataset_analysis.json",
            path_in_repo="analysis_metrics.json",
            repo_id=full_repo_id,
            repo_type="dataset"
        )
    
    # Upload Split Stats if exists
    if os.path.exists(os.path.join(upload_path, "split_stats.json")):
        api.upload_file(
            path_or_fileobj=os.path.join(upload_path, "split_stats.json"),
            path_in_repo="split_stats.json",
            repo_id=full_repo_id,
            repo_type="dataset"
        )

    print(f"\n✅ Successfully published dataset to https://huggingface.co/datasets/{full_repo_id}")

def publish_model(
    model_dir="models/best",
    repo_id="fineweb-legal-classifier",
    username=None,
    token=None
):
    print("\nPreparing to publish model to Hugging Face...")
    
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Skipping model upload.")
        return

    api = HfApi(token=token)
    
    # Infer username if needed
    if not username:
        try:
            username = api.whoami()['name']
        except:
            print("Error: Not authenticated.")
            return

    full_repo_id = f"{username}/{repo_id}"
    print(f"Creating/Checking model repository: {full_repo_id}")
    
    try:
        create_repo(full_repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Error creating model repo: {e}")
        return

    print("Uploading model files...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=full_repo_id,
        repo_type="model"
    )

    # Upload Model Card
    if os.path.exists("docs/MODEL_CARD.md"):
        print("Uploading Model Card...")
        api.upload_file(
            path_or_fileobj="docs/MODEL_CARD.md",
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="model"
        )
        
    print(f"✅ Successfully published model to https://huggingface.co/{full_repo_id}")

if __name__ == "__main__":
    # Publish Dataset
    publish_dataset()
    
    # Publish Model
    publish_model()
