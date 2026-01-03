
import pandas as pd
import glob
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def create_splits(
    input_dir="data/stage2",
    output_dir="data/stage3_splits",
    test_size=0.1,
    seed=42
):
    print(f"Reading data from {input_dir}...")
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        print("No files found!")
        return

    # Load all data
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Total documents: {len(df)}")

    # Define Configurations (Subsets)
    configs = {
        "default": df,                        # Score >= 3.0 (Implicit)
        "high_quality": df[df['score'] >= 4.0],
        "supreme": df[df['score'] >= 4.8]
    }

    # Clean output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("\nGenerating splits...")
    stats = {}

    for config_name, dataframe in configs.items():
        if len(dataframe) == 0:
            print(f"Skipping {config_name} (empty)")
            continue
            
        print(f"Processing config: {config_name} ({len(dataframe)} docs)")
        
        # Split Train/Test
        train_df, test_df = train_test_split(
            dataframe, 
            test_size=test_size, 
            random_state=seed,
            shuffle=True
        )

        # Create Config Directory
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Save Splits
        # HuggingFace expects 'train.parquet', 'test.parquet', etc.
        train_path = os.path.join(config_dir, "train.parquet")
        test_path = os.path.join(config_dir, "test.parquet")

        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        
        stats[config_name] = {
            "train": len(train_df),
            "test": len(test_df),
            "total": len(dataframe),
            "min_score": float(dataframe['score'].min()),
            "avg_score": float(dataframe['score'].mean())
        }

    # Save stats summary
    import json
    with open(os.path.join(output_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Split Statistics ---")
    for name, s in stats.items():
        print(f"{name:<15} | Total: {s['total']:<6} | Train: {s['train']:<6} | Test: {s['test']:<6} | Avg Score: {s['avg_score']:.2f}")
    
    print(f"\nSplits created in {output_dir}")

if __name__ == "__main__":
    create_splits()
