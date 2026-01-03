#!/usr/bin/env python3
"""
Test inference script for FineWeb-Legal classifier on Mac M2.

Uses LOCAL annotated data from data/batches to verify model predictions.
"""

import csv
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def setup_device():
    """Set up the best available device for Mac."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Metal Performance Shaders) on Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (MPS not available)")
    return device


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained classifier with LoRA adapter."""
    print("\n📦 Loading model...")
    
    # Load tokenizer
    model_name = "google/embeddinggemma-300m"
    print(f"   Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model in float32 for MPS compatibility
    print(f"   Loading base model in float32 (MPS compatible)...")
    base_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    print(f"   Loading LoRA adapter from {checkpoint_path}")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.float32,
    )
    
    # Load classification head
    head_path = Path(checkpoint_path) / "classifier_head.pt"
    print(f"   Loading classification head from {head_path}")
    classifier_state = torch.load(head_path, map_location="cpu", weights_only=True)
    
    # Recreate classifier head in float32
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(768, 6),
    ).to(torch.float32)
    
    # Convert state dict from bfloat16 to float32
    classifier_state_f32 = {k: v.float() for k, v in classifier_state.items()}
    classifier.load_state_dict(classifier_state_f32)
    
    # Move to device
    model = model.to(device)
    classifier = classifier.to(device)
    model.eval()
    classifier.eval()
    
    print("✅ Model loaded successfully!")
    return model, classifier, tokenizer


def stream_fineweb_samples(num_samples: int = 50, skip_samples: int = 200000):
    """
    Stream samples from FineWeb dataset.
    
    Uses skip_samples to ensure we get data distinct from training.
    Training used samples 0-6500, so we skip far ahead.
    """
    import time
    from datasets import load_dataset
    
    print(f"\n🌐 Streaming FineWeb dataset (skipping first {skip_samples} samples)...")
    print("   (This ensures we're NOT using training data)")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=True,
            )
            
            # Skip ahead to ensure distinct data from training
            samples = []
            for i, sample in enumerate(dataset):
                if i < skip_samples:
                    if i % 50000 == 0:
                        print(f"   Skipping... {i}/{skip_samples}")
                    continue
                if len(samples) >= num_samples:
                    break
                samples.append(sample)
            
            print(f"✅ Collected {len(samples)} samples (from position {skip_samples}+)")
            return samples
            
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise


def run_inference(
    model,
    classifier,
    tokenizer,
    samples: list,
    device: torch.device,
    max_length: int = 512,  # Reduced for faster inference on M2
):
    """Run inference on samples and return predictions."""
    print(f"\n🔮 Running inference on {len(samples)} samples...")
    
    results = []
    start_time = time.time()
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            text = sample.get("text", "")
            
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Forward pass through encoder
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            # Classification
            logits = classifier(pooled)
            
            # Weighted probability score (0.0 - 5.0)
            probs = torch.softmax(logits, dim=-1)
            weights = torch.arange(6, device=device, dtype=torch.float32)
            raw_score = (probs * weights).sum(dim=-1).item()
            
            # Argmax prediction
            pred_class = logits.argmax(dim=-1).item()
            
            # Threshold at 3
            label = "LEGAL" if raw_score >= 3.0 else "NON-LEGAL"
            
            results.append({
                "text_snippet": text[:200],
                "full_text": text,
                "raw_score": round(raw_score, 3),
                "pred_class": pred_class,
                "prediction_label": label,
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(samples)} samples...")
    
    elapsed = time.time() - start_time
    speed = len(samples) / elapsed
    
    print(f"✅ Inference complete!")
    print(f"   ⏱️  Time: {elapsed:.2f}s")
    print(f"   🚀 Speed: {speed:.2f} samples/second")
    
    return results, speed


def save_results(results: list, output_path: str = "inference_audit.csv"):
    """Save results to CSV file."""
    print(f"\n💾 Saving results to {output_path}...")
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["text_snippet", "full_text", "raw_score", "pred_class", "prediction_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} results to {output_path}")


def display_examples(results: list, num_examples: int = 3):
    """Display example predictions."""
    legal = [r for r in results if r["prediction_label"] == "LEGAL"]
    non_legal = [r for r in results if r["prediction_label"] == "NON-LEGAL"]
    
    print("\n" + "=" * 70)
    print("📊 INFERENCE RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n   LEGAL detections:     {len(legal)}")
    print(f"   NON-LEGAL detections: {len(non_legal)}")
    
    print("\n" + "-" * 70)
    print("✅ LEGAL EXAMPLES (Score >= 3.0)")
    print("-" * 70)
    
    for i, r in enumerate(legal[:num_examples]):
        print(f"\n[{i+1}] Score: {r['raw_score']:.2f} | Class: {r['pred_class']}")
        print(f"    {r['text_snippet'][:120]}...")
    
    if not legal:
        print("   (No LEGAL samples found)")
    
    print("\n" + "-" * 70)
    print("❌ NON-LEGAL EXAMPLES (Score < 3.0)")
    print("-" * 70)
    
    for i, r in enumerate(non_legal[:num_examples]):
        print(f"\n[{i+1}] Score: {r['raw_score']:.2f} | Class: {r['pred_class']}")
        print(f"    {r['text_snippet'][:120]}...")
    
    if not non_legal:
        print("   (No NON-LEGAL samples found)")
    
    print("\n" + "=" * 70)


def main():
    """Main inference pipeline."""
    print("🧪 FineWeb-Legal Inference Test (Local Data)")
    print("=" * 50)
    
    # Configuration
    checkpoint_path = "models/best"
    num_samples = 50
    output_file = "inference_audit.csv"
    
    # Setup
    device = setup_device()
    
    # Load model
    model, classifier, tokenizer = load_model(checkpoint_path, device)
    
    # Stream FineWeb data (skip 200K to ensure distinct from training)
    samples = stream_fineweb_samples(num_samples=num_samples, skip_samples=200000)
    
    # Run inference
    results, speed = run_inference(model, classifier, tokenizer, samples, device)
    
    # Save results
    save_results(results, output_file)
    
    # Display examples
    display_examples(results)
    
    print(f"\n✅ All done! Results saved to: {output_file}")
    print(f"   Speed: {speed:.2f} samples/second")


if __name__ == "__main__":
    main()
