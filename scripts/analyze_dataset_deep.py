
import pandas as pd
import glob
import os
import json
from urllib.parse import urlparse
from collections import Counter
import numpy as np

def analyze_dataset(data_dir="data/stage2"):
    print(f"Analyzing parquet files in {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    
    if not files:
        print("No parquet files found!")
        return

    # Load all data
    df_list = []
    for f in files:
        try:
            df_batch = pd.read_parquet(f)
            df_list.append(df_batch)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        return

    df = pd.concat(df_list, ignore_index=True)
    
    # Basic Stats
    total_docs = len(df)
    print(f"\nTotal Documents: {total_docs}")
    
    # Score Stats
    print("\n--- Score Statistics ---")
    print(df['score'].describe())
    
    # Length Stats (Text Length)
    df['char_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print("\n--- Content Statistics ---")
    print(f"Total Characters: {df['char_length'].sum():,}")
    print(f"Total Words: {df['word_count'].sum():,}")
    print(f"Avg Document Length (words): {df['word_count'].mean():.2f}")
    
    # Domain Analysis
    if 'url' in df.columns:
        print("\n--- Top Domains ---")
        def get_domain(url):
            try:
                return urlparse(url).netloc
            except:
                return "unknown"
        
        df['domain'] = df['url'].apply(get_domain)
        print(df['domain'].value_counts().head(20))
        
        # Diversity check
        unique_domains = df['domain'].nunique()
        print(f"\nUnique Domains: {unique_domains}")

    # Percentile Analysis
    print("\n--- Score Percentiles ---")
    for p in [50, 75, 90, 95, 99]:
        score_at_p = np.percentile(df['score'], p)
        print(f"{p}th Percentile: {score_at_p:.4f}")

    # Output samples for manual review
    print("\n--- Saving Samples for Review ---")
    high_quality = df[df['score'] >= 4.5].head(3)
    mid_quality = df[(df['score'] >= 3.0) & (df['score'] < 3.5)].head(3)
    
    samples = {
        "high_quality": high_quality[['text', 'score', 'url']].to_dict(orient='records'),
        "mid_quality": mid_quality[['text', 'score', 'url']].to_dict(orient='records')
    }
    
    with open("dataset_analysis.json", "w") as f:
        json.dump({
            "total_docs": int(total_docs),
            "total_words": int(df['word_count'].sum()),
            "avg_score": float(df['score'].mean()),
            "unique_domains": int(unique_domains) if 'url' in df.columns else 0,
            "samples": samples
        }, f, indent=2)
        
    print("Analysis complete. Saved summary to dataset_analysis.json")

if __name__ == "__main__":
    analyze_dataset()
