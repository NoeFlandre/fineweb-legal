#!/usr/bin/env python3
"""Rapid Ablation Study Framework for Legal Classifier.

Features:
- Saves model, results, and full config after EACH experiment
- Resumable: skips already completed experiments
- All outputs in ablation_results/{experiment_name}/
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fineweb_legal.classifier.model import LegalClassifier


OUTPUT_DIR = Path("ablation_results")


@dataclass
class AblationConfig:
    """Single ablation experiment configuration."""
    name: str = "baseline"
    # Data
    train_samples: int = 5000  # Small subset for speed
    val_samples: int = 1000
    max_length: int = 1024  # Shorter seq for faster iteration
    # Model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_class_weights: bool = True
    # Training
    epochs: int = 2
    batch_size: int = 64
    lr: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Monitoring
    eval_every_steps: int = 50  # Frequent eval for signal
    seed: int = 42


class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0), 
            "attention_mask": enc["attention_mask"].squeeze(0), 
            "label": torch.tensor(self.labels[i], dtype=torch.long)
        }


def class_weights(labels, dev, use_weights=True):
    if not use_weights:
        return torch.ones(6, device=dev)
    c = Counter(labels)
    t = len(labels)
    w = torch.tensor([t / (6 * (c.get(i, 1))) for i in range(6)])
    return (w / w.sum() * 6).to(dev)


def evaluate(model, loader, crit, dev):
    """Quick evaluation returning key metrics."""
    model.eval()
    loss_sum, preds, labs = 0, [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(dev)
            mask = b["attention_mask"].to(dev)
            lab = b["label"].to(dev)
            with autocast("cuda", torch.bfloat16):
                logits = model(ids, mask)
                loss_sum += crit(logits, lab).item()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(lab.cpu().tolist())
    
    # Binary F1 for legal relevance (score >= 3 is "relevant")
    bin_p = [1 if p >= 3 else 0 for p in preds]
    bin_l = [1 if l >= 3 else 0 for l in labs]
    
    return {
        "loss": loss_sum / len(loader),
        "acc": accuracy_score(labs, preds),
        "f1_macro": f1_score(labs, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labs, preds, average="weighted", zero_division=0),
        "f1_binary": f1_score(bin_l, bin_p, zero_division=0),
        "f1_per_class": f1_score(labs, preds, average=None, zero_division=0).tolist(),
        "preds": preds,
        "labs": labs,
    }


def is_experiment_completed(cfg: AblationConfig) -> bool:
    """Check if experiment already completed (for resumability)."""
    exp_dir = OUTPUT_DIR / cfg.name
    results_file = exp_dir / "results.json"
    return results_file.exists()


def save_experiment(cfg: AblationConfig, model, results: Dict, train_stats: Dict, val_stats: Dict):
    """Save complete experiment: model, results, config, and data stats."""
    exp_dir = OUTPUT_DIR / cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model weights (LoRA adapters + classifier head)
    model_dir = exp_dir / "model"
    model.save_pretrained(str(model_dir))
    print(f"  → Model saved to {model_dir}")
    
    # 2. Save full config
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"  → Config saved to {config_path}")
    
    # 3. Save training results (metrics over time)
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        # Strip large arrays for cleaner output
        results_clean = {
            "config": asdict(cfg),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "training_history": results.get("steps", []),
            "final_metrics": {k: v for k, v in results["final"]["final_metrics"].items() if k not in ["preds", "labs"]},
            "best_f1_macro": results["final"]["best_f1_macro"],
            "elapsed_seconds": results["final"]["elapsed_seconds"],
            "samples_per_second": results["final"]["samples_per_second"],
            "timestamp": datetime.now().isoformat(),
        }
        json.dump(results_clean, f, indent=2)
    print(f"  → Results saved to {results_path}")
    
    # 4. Save confusion matrix
    if "preds" in results["final"]["final_metrics"] and "labs" in results["final"]["final_metrics"]:
        cm = confusion_matrix(
            results["final"]["final_metrics"]["labs"],
            results["final"]["final_metrics"]["preds"]
        )
        np.save(exp_dir / "confusion_matrix.npy", cm)
        print(f"  → Confusion matrix saved")
    
    # 5. Save classification report
    if "preds" in results["final"]["final_metrics"] and "labs" in results["final"]["final_metrics"]:
        report = classification_report(
            results["final"]["final_metrics"]["labs"],
            results["final"]["final_metrics"]["preds"],
            target_names=[f"Score{i}" for i in range(6)],
            zero_division=0
        )
        with open(exp_dir / "classification_report.txt", "w") as f:
            f.write(report)
        print(f"  → Classification report saved")


def run_ablation(cfg: AblationConfig, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Run a single ablation experiment with full saving."""
    
    # Check if already completed (resumability)
    if is_experiment_completed(cfg):
        print(f"\n⏭ SKIPPING {cfg.name} (already completed)")
        with open(OUTPUT_DIR / cfg.name / "results.json") as f:
            return json.load(f)
    
    print(f"\n{'='*70}")
    print(f"ABLATION: {cfg.name}")
    print(f"{'='*70}")
    print(f"Config: samples={cfg.train_samples}, seq_len={cfg.max_length}, "
          f"lora_r={cfg.lora_r}, lr={cfg.lr}, class_weights={cfg.use_class_weights}")
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device("cuda")
    
    # Subsample data (with fixed seed for reproducibility)
    train_sub = train_df.sample(n=min(cfg.train_samples, len(train_df)), random_state=cfg.seed)
    val_sub = val_df.sample(n=min(cfg.val_samples, len(val_df)), random_state=cfg.seed)
    
    # Data stats for logging
    train_stats = {
        "n_samples": len(train_sub),
        "label_distribution": train_sub["score"].value_counts().to_dict(),
    }
    val_stats = {
        "n_samples": len(val_sub),
        "label_distribution": val_sub["score"].value_counts().to_dict(),
    }
    print(f"Train: {train_stats['n_samples']} samples, Val: {val_stats['n_samples']} samples")
    
    # Initialize model
    model = LegalClassifier(
        use_lora=True,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        load_in_4bit=False,
        use_gradient_checkpointing=True,
        attn_implementation="sdpa"
    ).to(dev)
    model.encoder.config.use_cache = False
    tok = model.tokenizer
    
    # Data loaders
    train_ld = DataLoader(
        LegalDataset(train_sub["text"].tolist(), train_sub["score"].tolist(), tok, cfg.max_length),
        batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_ld = DataLoader(
        LegalDataset(val_sub["text"].tolist(), val_sub["score"].tolist(), tok, cfg.max_length),
        batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Loss and optimizer
    cw = class_weights(train_sub["score"].tolist(), dev, cfg.use_class_weights)
    crit = nn.CrossEntropyLoss(weight=cw.to(torch.bfloat16))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    total_steps = len(train_ld) * cfg.epochs
    sched = OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.warmup_ratio)
    
    # Training loop with frequent eval
    results = {"config": asdict(cfg), "steps": [], "final": None}
    global_step = 0
    best_f1 = 0
    t0 = time.time()
    
    for ep in range(cfg.epochs):
        model.train()
        ep_loss = 0
        
        pbar = tqdm(train_ld, desc=f"Ep{ep+1}")
        for step, b in enumerate(pbar):
            ids = b["input_ids"].to(dev)
            mask = b["attention_mask"].to(dev)
            lab = b["label"].to(dev)
            
            with autocast("cuda", torch.bfloat16):
                loss = crit(model(ids, mask), lab)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad()
            
            ep_loss += loss.item()
            global_step += 1
            
            # Frequent evaluation for signal
            if global_step % cfg.eval_every_steps == 0:
                val_metrics = evaluate(model, val_ld, crit, dev)
                results["steps"].append({
                    "step": global_step,
                    "epoch": ep + 1,
                    "train_loss": ep_loss / (step + 1),
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_f1_binary": val_metrics["f1_binary"],
                })
                pbar.set_postfix({
                    "loss": f"{ep_loss/(step+1):.3f}",
                    "val_f1": f"{val_metrics['f1_macro']:.3f}"
                })
                
                if val_metrics["f1_macro"] > best_f1:
                    best_f1 = val_metrics["f1_macro"]
                
                model.train()
        
        # End of epoch eval
        val_metrics = evaluate(model, val_ld, crit, dev)
        print(f"\nEp{ep+1} Final: Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f} "
              f"F1={val_metrics['f1_macro']:.4f} BinaryF1={val_metrics['f1_binary']:.4f}")
    
    # Final results
    elapsed = time.time() - t0
    results["final"] = {
        "best_f1_macro": best_f1,
        "final_metrics": val_metrics,
        "elapsed_seconds": elapsed,
        "samples_per_second": (cfg.train_samples * cfg.epochs) / elapsed
    }
    
    print(f"\n★ ABLATION COMPLETE: Best F1={best_f1:.4f} in {elapsed:.0f}s")
    
    # SAVE EVERYTHING
    save_experiment(cfg, model, results, train_stats, val_stats)
    
    # Cleanup
    del model, opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all", 
                        help="Which experiment to run: all, lr, lora, seq_len, weights")
    args = parser.parse_args()
    
    print("="*70)
    print("LEGAL CLASSIFIER - ABLATION STUDY FRAMEWORK")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data once
    train_df = pd.read_parquet("data/v2/train.parquet")
    val_df = pd.read_parquet("data/v2/val.parquet")
    print(f"Data loaded: Train={len(train_df):,} Val={len(val_df):,}")
    
    # Define ablation experiments
    experiments = {}
    
    # 1. Learning Rate Sweep
    if args.experiment in ["all", "lr"]:
        experiments["lr_sweep"] = [
            AblationConfig(name="lr_1e-4", lr=1e-4),
            AblationConfig(name="lr_2e-4", lr=2e-4),
            AblationConfig(name="lr_5e-4", lr=5e-4),
            AblationConfig(name="lr_1e-3", lr=1e-3),
        ]
    
    # 2. LoRA Rank Sweep
    if args.experiment in ["all", "lora"]:
        experiments["lora_sweep"] = [
            AblationConfig(name="lora_r8", lora_r=8, lora_alpha=16),
            AblationConfig(name="lora_r16", lora_r=16, lora_alpha=32),
            AblationConfig(name="lora_r32", lora_r=32, lora_alpha=64),
        ]
    
    # 3. Sequence Length Impact
    if args.experiment in ["all", "seq_len"]:
        experiments["seq_len_sweep"] = [
            AblationConfig(name="seq_512", max_length=512, batch_size=128),
            AblationConfig(name="seq_1024", max_length=1024, batch_size=64),
            AblationConfig(name="seq_2048", max_length=2048, batch_size=32),
        ]
    
    # 4. Class Weights Impact
    if args.experiment in ["all", "weights"]:
        experiments["weights_sweep"] = [
            AblationConfig(name="with_weights", use_class_weights=True),
            AblationConfig(name="no_weights", use_class_weights=False),
        ]
    
    # Run all experiments
    all_results = {}
    for exp_name, configs in experiments.items():
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT GROUP: {exp_name}")
        print(f"{'#'*70}")
        
        exp_results = []
        for cfg in configs:
            result = run_ablation(cfg, train_df, val_df)
            exp_results.append(result)
        
        all_results[exp_name] = exp_results
        
        # Summary table for this group
        print(f"\n{'='*70}")
        print(f"SUMMARY: {exp_name}")
        print(f"{'='*70}")
        print(f"{'Config':<20} {'Best F1':>10} {'Final F1':>10} {'Time':>10}")
        print("-" * 50)
        for r in exp_results:
            best_f1 = r.get("best_f1_macro", r.get("final", {}).get("best_f1_macro", 0))
            final_f1 = r.get("final_metrics", r.get("final", {}).get("final_metrics", {})).get("f1_macro", 0)
            elapsed = r.get("elapsed_seconds", r.get("final", {}).get("elapsed_seconds", 0))
            name = r.get("config", {}).get("name", "unknown")
            print(f"{name:<20} {best_f1:>10.4f} {final_f1:>10.4f} {elapsed:>8.0f}s")
    
    # Save master summary
    summary_path = OUTPUT_DIR / "master_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiments": list(experiments.keys()),
            "results": all_results
        }, f, indent=2, default=str)
    print(f"\n★ Master summary saved to {summary_path}")
    
    # Final recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    for exp_name, exp_results in all_results.items():
        best = max(exp_results, key=lambda x: x.get("best_f1_macro", x.get("final", {}).get("best_f1_macro", 0)))
        best_f1 = best.get("best_f1_macro", best.get("final", {}).get("best_f1_macro", 0))
        name = best.get("config", {}).get("name", "unknown")
        print(f"• {exp_name}: Best config = {name} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
