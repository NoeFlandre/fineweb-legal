# FineWeb-Legal-Pilot

![FineWeb-Legal Logo](assets/logo.png)

**A pilot dataset of legal domains text extracted from FineWeb 10BT.**

This project represents the **first version** of the FineWeb-Legal initiative. It validates our methodology by filtering the 10-billion-token (`sample-10BT`) subset of FineWeb, producing a highly curated dataset of legal documents.

**[📄 Technical Report](docs/TECHNICAL_REPORT.md)** • **[Blog](https://noeflandre.bearblog.dev/i-made-a-legal-dataset-and-put-it-on-the-internet/)** • **[🤗 Dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-legal-pilot)**

---

## 🎯 Project Overview

**FineWeb-Legal-Pilot** implements a three-phase pipeline on the 10BT subset drawing inspiration from the FineWeb-Edu filtering:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Pre-filter with heuristics & annotate a subsample using Mistral-Medium LLM (0-5 legal value scores) | ✅ Complete |
| **Phase 2** | Train a LoRA classifier on Gemma Embedding 300M | ✅ Complete |
| **Phase 3** | Score the subset 10BT of the FineWeb dataset with the fast classifier | ✅ Complete |

### Key Results

| Metric | Value | Target |
|--------|-------|--------|
| **Binary F1@3** | **97.99%** | >82% ✅ |
| Validation Accuracy | 88.8% | >60% ✅ |
| Macro F1 | 0.857 | >0.50 ✅ |
| **Total Documents** | **52,132** | - |
| **Total Words** | **66.8M** | - |
| Inference Speed (RTX 3090) | 41.3 samples/sec | - |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/NoeFlandre/fineweb-legal.git
cd fineweb-legal

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Copy environment template
cp .env.example .env

# Add your Mistral API key to .env
echo "MISTRAL_API_KEY=your_key_here" >> .env
```

### Requirements

- Python 3.12+
- CUDA GPU for training and inference of the embedding model (tested on RTX 3090)
- Heuristics filtering : Apple Silicon for M1/M2 inference (MPS backend)

---

## 🚀 Quick Start

### Phase 1: Annotate Samples with Mistral

```bash
# Test run (100 samples)
fineweb-legal annotate --samples 100

# Production run (50K samples)
fineweb-legal annotate --samples 50000 --batch-size 100 --delay 0

# View statistics
fineweb-legal stats

# Review high-value samples
fineweb-legal samples --score 5 --count 10
```

### Phase 2: Train Classifier

```bash
# Train on annotated data
fineweb-legal train --epochs 10 --batch-size 8

# Full training with all options
fineweb-legal train \
    --data data/batches \
    --epochs 20 \
    --batch-size 8 \
    --max-length 2048 \
    --lora-r 16 \
    --learning-rate 3e-4

# Evaluate checkpoint
fineweb-legal evaluate --checkpoint models/best
```

### Phase 3: Run Inference

#### Stage 1: CPU Heuristic Filter (Phase 3 Alpha)

```bash
# Stream FineWeb and filter with 4-stage heuristics
uv run python scripts/stream_filter_stage1.py

# Test run (limited docs)
uv run python scripts/stream_filter_stage1.py --max-docs 10000

# Output: data/stage1/part_*.parquet
# State:  data/stage1/processing_state.json (auto-resume)
```

#### Stage 2: Model Inference (GPU)

```bash
# Run inference on Stage 1 files using trained LoRA model
# Local (Mac M2 - MPS):
uv run python scripts/model_inference_stage2.py --batch-size 16

# Remote GPU (RTX 3090 - CUDA):
python3 scripts/model_inference_stage2_cuda.py --batch-size 384

# Output: data/stage2/scored_part_*.parquet (only score >= 3.0)
```

#### Stage 3: Quality Splits

The final dataset is split into 3 quality tiers:

| Split | Min Score | Documents | Avg Score |
|-------|-----------|-----------|----------|
| `default` | ≥ 3.0 | 52,132 | 4.21 |
| `high_quality` | ≥ 4.0 | 32,335 | 4.60 |
| `supreme` | ≥ 4.8 | 16,635 | 4.98 |

Each split has `train.parquet` and `test.parquet` (90/10 split).

---

## 📁 Project Structure

```
fineweb-legal/
├── assets/                  # Images and logos
├── docs/                    # Documentation & Reports
│   ├── DATASET_CARD.md
│   └── TECHNICAL_REPORT.md
│
├── src/fineweb_legal/       # Main package
│   ├── cli.py               # Typer CLI application
│   ├── classifier/          # Model & Training components
│   └── ...
│
├── data/
│   ├── batches/             # Annotated subsets
│   ├── stage2/              # Scored parquet files
│   └── stage3_splits/       # Final train/test splits
│       ├── default/         # Score ≥ 3.0 (52K docs)
│       ├── high_quality/    # Score ≥ 4.0 (32K docs)
│       └── supreme/         # Score ≥ 4.8 (16K docs)
│
├── models/                  # Trained checkpoints
├── scripts/                 # Utility scripts
│   ├── sh/                  # Operations (deploy, train)
│   ├── analyze_dataset_deep.py
│   └── publish_to_hf.py
│
└── pyproject.toml
```

---

## 🏗️ Architecture

### Phase 1: Annotation Pipeline

```
FineWeb Stream → Elite Filter → Mistral API → Parquet Storage
                     ↓
            4-Stage Filtering:
            1. Reject boilerplate (ToS, Privacy...)
            2. Require legal keywords (plaintiff, defendant...)
            3. Reject news URLs (nytimes, cnn...)
            4. Require citations (§, v., U.S.C.)
```

### Phase 2: Classifier Architecture

```
Text Input
    ↓
┌─────────────────────────────────────────┐
│  Gemma Embedding 300M (frozen)          │
│  + LoRA Adapters (q,k,v,o projections)  │
│  → 768-dim embeddings                   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Mean Pooling over sequence            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Classification Head                    │
│  Dropout(0.1) → Linear(768, 6)         │
└─────────────────────────────────────────┘
    ↓
6-class logits → Weighted Probability → Score 0.0-5.0
```

### Scoring Formula

```
Final Score = Σ(P(class=i) × i) for i ∈ {0,1,2,3,4,5}
```

This produces a continuous score from 0.0 to 5.0 based on weighted class probabilities.

---

## 📊 Scoring Rubric

| Score | Label | Description | Examples |
|-------|-------|-------------|----------|
| **0** | Noise/Spam | Navigation, ads, gibberish | Cookie notices, site menus |
| **1** | General/Marketing | Law firm ads, generic news | "Call our lawyers today!" |
| **2** | Basic Info | Wikipedia summaries, Reddit questions | ELI5 legal questions |
| **3** | Useful | Detailed legal news, government guides | IRS guidelines, legal blogs |
| **4** | High Value | Case text, statutes, contracts | Court filings, legislation |
| **5** | Gold Standard | Supreme Court opinions, law journals | Academic legal research |

### Binary Classification Threshold

For filtering purposes, we use **Score ≥ 3** as the threshold:
- **LEGAL** (3-5): Useful legal content for training
- **NON-LEGAL** (0-2): Low-quality or non-legal content

---

## 🔬 Training Details

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `google/embeddinggemma-300m` |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj` |
| Max Sequence Length | 2048 tokens |
| Classification Head | `Dropout(0.1) → Linear(768, 6)` |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Gradient Accumulation | 8 (effective batch = 32) |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Epochs | 10 (early stopped at 4) |
| Early Stopping Patience | 5 |
| Class Weights | Inverse frequency weighting |

### Hardware Used

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3090 (24GB VRAM) |
| VRAM Used | ~10-12 GB |
| Training Time | 1h 50m |
| Provider | vast.ai |

---

## 📈 Training Results

### Epoch-by-Epoch Progress

| Epoch | Train Loss | Val Acc | Macro F1 | Binary F1@3 |
|-------|------------|---------|----------|-------------|
| 1 | 0.8976 | 77.5% | 0.737 | 95.3% |
| 2 | 0.4620 | 81.2% | 0.772 | 91.1% |
| 3 | 0.3000 | 89.5% | 0.872 | 96.3% |
| **4** | **0.2287** | **88.8%** | **0.857** | **97.99%** ✅ |

### Final Model Metrics

```json
{
  "epoch": 4,
  "accuracy": 0.888,
  "macro_f1": 0.857,
  "weighted_f1": 0.891,
  "macro_precision": 0.850,
  "macro_recall": 0.884,
  "binary_f1_3": 0.980,
  "loss": 0.477
}
```

---

## 💻 Inference

### Loading the Trained Model

```python
from fineweb_legal.classifier.model import LegalClassifier

# Load from checkpoint
model = LegalClassifier.from_pretrained("models/best")

# Move to device
model.to("cuda")  # or "mps" for Mac
model.eval()
```

### Scoring Text

```python
import torch

def score_text(model, tokenizer, text: str) -> float:
    """Score a single text and return 0.0-5.0 legal quality score."""
    encoding = tokenizer(
        text,
        max_length=2048,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        
        logits = model(input_ids, attention_mask)
        score = model.predict_scores(logits)
    
    return score.item()

# Example usage
score = score_text(model, model.tokenizer, "The Supreme Court held that...")
print(f"Legal quality score: {score:.2f}")  # e.g., 4.85
```

### Mac M2 Inference

For Apple Silicon Macs, use the provided test script:

```bash
python test_inference_mac.py
```

This script:
- Uses MPS (Metal Performance Shaders) acceleration
- Loads model in float32 for MPS compatibility
- Streams FineWeb samples (skipping training data)
- Outputs results to `inference_audit.csv`

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
MISTRAL_API_KEY=your_key_here
MISTRAL_MODEL=mistral-medium-latest
TARGET_SAMPLES=50000
BATCH_SIZE=100
REQUEST_DELAY=0
```

### HuggingFace Authentication

For Gemma model access (gated model):

```bash
huggingface-cli login
# Enter your HuggingFace token
```

---

## 📋 CLI Reference

| Command | Description |
|---------|-------------|
| `fineweb-legal annotate` | Run annotation pipeline |
| `fineweb-legal resume` | Resume from last batch |
| `fineweb-legal stats` | Show score distribution |
| `fineweb-legal validate` | Run quality checks |
| `fineweb-legal samples` | Review sample annotations |
| `fineweb-legal merge` | Combine batches to single file |
| `fineweb-legal train` | Train LoRA classifier |
| `fineweb-legal evaluate` | Evaluate model checkpoint |

### Annotation Options

```bash
fineweb-legal annotate \
    --samples 50000 \     # Number of samples to annotate
    --batch-size 100 \    # Samples per batch file
    --delay 0 \           # Delay between API calls (seconds)
    --seed 42             # Random seed for reproducibility
```

### Training Options

```bash
fineweb-legal train \
    --data data/batches \     # Path to annotation data
    --epochs 20 \             # Maximum epochs
    --batch-size 8 \          # Batch size per GPU
    --max-length 2048 \       # Maximum sequence length
    --lora-r 16 \             # LoRA rank
    --learning-rate 3e-4 \    # Learning rate
    --save-dir models/        # Checkpoint directory
```

---

## 🗂️ Data Format

### Annotation Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique document ID |
| `text` | string | Full document text |
| `score` | int8 | Legal quality score (0-5) |
| `reasoning` | string | LLM annotation reasoning |
| `annotated_at` | timestamp | Annotation timestamp |
| `truncated_len` | int32 | Chars sent to API |

### Example Record

```python
{
    "id": "CC-MAIN-2024-10/000_00001",
    "text": "The Supreme Court of the United States...",
    "score": 5,
    "reasoning": "Primary source legal document - Supreme Court opinion",
    "annotated_at": "2026-01-02T14:30:00Z",
    "truncated_len": 3000
}
```

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) - Source dataset
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Methodology inspiration
- [Gemma](https://huggingface.co/google/embeddinggemma-300m) - Base embedding model
- [PEFT](https://huggingface.co/docs/peft) - LoRA implementation
- [Mistral AI](https://mistral.ai/) - Annotation LLM
