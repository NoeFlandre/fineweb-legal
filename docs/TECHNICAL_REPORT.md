# FineWeb-Legal-Pilot Technical Report

![FineWeb-Legal Logo](../assets/logo.png)

**Date:** January 2, 2026  
**Author:** Noé Flandre  
**Version:** 1.0.0 (Pilot Release)

---

## Executive Summary

This report documents the development and training of **FineWeb-Legal-Pilot**, a proof-of-concept dataset identifying high-quality legal content within the FineWeb `sample-10BT` subset. The project successfully achieved **97.99% Binary F1@3** and produced a pilot dataset of **52,132 documents**, paving the way for scaling to the full 44TB FineWeb corpus.

---

## 1. Introduction

### 1.1 Background

FineWeb is a 44TB dataset of web-crawled text from Common Crawl. While it contains diverse content, legal professionals and researchers require filtered, high-quality legal documents for training language models. FineWeb-Legal addresses this need by implementing a classifier to score documents on their legal value.

### 1.2 Objectives

1. **Annotation Pipeline**: Create a scalable system to annotate web documents with legal quality scores using a state-of-the-art LLM
2. **Classifier Training**: Train an efficient classifier that can rapidly score millions of documents
3. **Production Deployment**: Enable full-corpus scoring for creating legal domain datasets

### 1.3 Methodology Inspiration

This project follows the methodology established by [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), which successfully filtered FineWeb for educational content using a similar multi-phase approach.

---

## 2. Phase 1: Annotation Pipeline

### 2.1 Data Streaming

Documents are streamed directly from the HuggingFace Hub without downloading the full 44TB dataset:

```python
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True,
)
```

### 2.2 Elite Filtering

Before annotation, documents pass through a 4-stage heuristic filter:

| Stage | Filter | Rationale |
|-------|--------|-----------|
| A | Reject boilerplate | Remove ToS, Privacy Policy, cookie notices |
| B | Require 2+ legal keywords | Ensure legal domain relevance |
| C | Reject news URLs | Filter out news aggregation sites |
| D | Require citation patterns | Prioritize formal legal documents |

**Legal Keywords Used:**
- Primary: plaintiff, defendant, court, statute, jurisdiction...
- Secondary: appellant, appellee, injunction, habeas corpus...
- Citation patterns: §, v., U.S.C., F.2d, F.3d...

### 2.3 Annotation Model

**Model:** `mistral-medium-latest`  
**Provider:** Mistral AI  

### 2.4 Scoring Rubric

| Score | Label | Description |
|-------|-------|-------------|
| 0 | Noise/Spam | Navigation, ads, gibberish |
| 1 | General/Marketing | Law firm ads, generic content |
| 2 | Basic Info | Wikipedia summaries, simple Q&A |
| 3 | Useful | Detailed guides, legal news analysis |
| 4 | High Value | Case text, statutes, contracts |
| 5 | Gold Standard | Supreme Court opinions, law journals |

### 2.5 Annotation Results

**Total Annotated:** 6,500 samples

**Score Distribution:**

| Score | Count | Percentage |
|-------|-------|------------|
| 0 | 2,494 | 38.4% |
| 1 | 780 | 12.0% |
| 2 | 574 | 8.8% |
| 3 | 793 | 12.2% |
| 4 | 1,372 | 21.1% |
| 5 | 487 | 7.5% |

**Observations:**
- Heavy skew toward score 0 (spam/noise) - expected from web crawl
- Score 4 is second most common - indicates good elite filtering
- Score 5 (gold standard) is rare - appropriate for Supreme Court level content

---

## 3. Phase 2: Classifier Training

### 3.1 Model Architecture

**Base Model:** `google/embeddinggemma-300m`
- 308M parameters
- 768-dimensional embeddings
- 2048 token context window

**LoRA Configuration:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.1
- Target modules: `q_proj, k_proj, v_proj, o_proj`
- Trainable parameters: 1,966,080 (0.65% of total)

**Classification Head:**
```
Dropout(0.1) → Linear(768, 6)
```

### 3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 32 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Max Epochs | 10 |
| Early Stopping Patience | 5 |
| Max Sequence Length | 2048 |

**Class Weights (Inverse Frequency):**
| Class | Weight |
|-------|--------|
| 0 | 0.322 |
| 1 | 1.062 |
| 2 | 1.387 |
| 3 | 1.035 |
| 4 | 0.585 |
| 5 | 1.608 |

### 3.3 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3090 |
| VRAM | 24 GB (used ~10-12 GB) |
| GPU Utilization | 80-85% |
| Training Time | 1h 50m |
| Provider | vast.ai |
| Cost | ~$0.3 |

### 3.4 Training Progress

| Epoch | Train Loss | Train Acc | Val Acc | Macro F1 | Binary F1@3 |
|-------|------------|-----------|---------|----------|-------------|
| 1 | 0.8976 | 68.1% | 77.5% | 0.737 | 95.32% |
| 2 | 0.4620 | 85.0% | 81.2% | 0.772 | 91.08% |
| 3 | 0.3000 | 91.0% | 89.5% | 0.872 | 96.27% |
| **4** | **0.2287** | **93.5%** | **88.8%** | **0.857** | **97.99%** |

Training was stopped after epoch 4 as it achieved the best Binary F1@3 score.

### 3.5 Final Model Metrics

```json
{
  "epoch": 4,
  "accuracy": 0.8877,
  "macro_f1": 0.8570,
  "weighted_f1": 0.8910,
  "macro_precision": 0.8501,
  "macro_recall": 0.8838,
  "binary_f1_3": 0.9799,
  "loss": 0.4769
}
```

---

## 4. Inference

### 4.1 Scoring Formula

For inference, we use weighted probability averaging:

```
Score = Σ(P(class=i) × i) for i ∈ {0,1,2,3,4,5}
```

This produces a continuous score from 0.0 to 5.0, allowing fine-grained filtering.

### 4.2 Binary Threshold

For practical filtering, we use **Score ≥ 3.0** as the threshold:
- **LEGAL**: Scores 3.0-5.0 (useful for training)
- **NON-LEGAL**: Scores 0.0-2.9 (to be filtered out)

### 4.3 Inference Performance

| Platform | Device | Speed |
|----------|--------|-------|
| RTX 3090 | CUDA | ~41 docs/sec|

### 4.4 Validation on Wild Data

Tested on 50 FineWeb samples (position 200,000+, never seen during training):

| Prediction | Count | Percentage |
|------------|-------|------------|
| LEGAL | 2 | 4% |
| NON-LEGAL | 48 | 96% |

**LEGAL Examples Detected:**
1. "Delta State High Court... sentenced two kidnappers to 113 years" → Score 3.98
2. "Dept of Ed: Some bullying violates federal law" → Score 3.77

**NON-LEGAL Examples (Correctly Rejected):**
1. "Zoo Breeding" article → Score 0.01
2. "Public Profile Info" → Score 0.00

The classifier correctly identifies legal content while rejecting non-legal material.

---

## 5. Saved Artifacts

### 5.1 Model Checkpoint

```
models/best/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights (7.9 MB)
├── classifier_head.pt        # Classification head (11 KB)
├── metrics.json              # Final evaluation metrics
└── README.md                 # Model card
```

### 5.2 Training Logs

```
logs/run_20260102_142030/
├── events.out.tfevents.*     # TensorBoard scalars
├── Accuracy_train/
├── Accuracy_val/
├── Loss_train/
├── Loss_val/
├── Macro_F1_train/
└── Macro_F1_val/
```

View with: `tensorboard --logdir logs/run_20260102_142030`

### 5.3 Annotation Data

```
data/batches/
├── batch_000001.parquet
├── batch_000002.parquet
├── ... (65 total)
└── batch_000065.parquet

Total: 6,500 annotated samples
```

---

## 5. Phase 3 Alpha: CPU Heuristic Pre-Filter (MPS)

### 5.1 Overview

Phase 3 involves scoring the full FineWeb corpus. Given the ~10B documents in `sample-10BT`, we implement a two-stage approach:

| Stage | Filter | Expected Pass Rate | Output |
|-------|--------|-------------------|--------|
| **Stage 1 (Alpha)** | CPU Heuristic (4-stage filter) | ~1% | `data/stage1/*.parquet` |
| **Stage 2** | Model Inference | ~5-10% of Stage 1 | `data/stage2/*.parquet` |

Stage 1 reduces the dataset from billions to millions of candidate documents, making Stage 2 (expensive GPU inference) tractable.

### 5.2 Implementation

**Script:** `scripts/stream_filter_stage1.py`

#### Features

| Feature | Implementation |
|---------|----------------|
| **Streaming** | `datasets.load_dataset(..., streaming=True)` |
| **Pre-compiled Regex** | 14 citation patterns compiled once at module load |
| **4-Stage Filter** | News URL → Boilerplate → Keywords (≥2) → Citation |
| **Resumability** | JSON state file + `dataset.skip()` |
| **Batched Output** | 1,000 docs per Parquet partition (Snappy compressed) |
| **Progress** | tqdm with docs/sec, pass rate, batch count |
| **Graceful Shutdown** | SIGINT/SIGTERM handlers flush buffer before exit |

#### Filter Stages

```
Stage C: Reject news URLs (fastest check)
   ↓
Stage A: Reject boilerplate (ToS, Privacy, cookies)
   ↓
Stage B: Require ≥2 strict legal keywords
   ↓
Stage D: Require at least one formal citation pattern (§, v., U.S.C.)
```

### 5.3 Performance

| Metric | Value |
|--------|-------|
| Throughput | ~2200 docs/sec (mostly network bottlenecked)|
| Pass Rate | ~1.06% |
| Memory | Constant (streaming + 1K buffer) |
| Storage per batch | ~4 KB per document (Snappy) |

### 5.4 State File Format

```json
{
  "total_processed_count": 14868862,
  "total_passed_count": 149432,
  "batch_index": 151,
  "last_updated": "2026-01-03T06:12:14.210721+00:00"
}
```

### 5.5 Usage

```bash
# Full run (unlimited, auto-resumes)
uv run python scripts/stream_filter_stage1.py

# Test run with limit
uv run python scripts/stream_filter_stage1.py --max-docs 10000

# Custom output directory
uv run python scripts/stream_filter_stage1.py --output-dir data/stage1_custom
```

---

## 5B. Phase 3 Beta: CUDA Inference (RTX 3090)

### 5B.1 Overview

Stage 2 runs the trained LoRA classifier on Stage 1 output, scoring documents and filtering to only keep high-quality legal content (score ≥ 3.0).

**Script:** `scripts/model_inference_stage2_cuda.py`

### 5B.2 Implementation

| Feature | Implementation |
|---------|----------------|
| **Device** | NVIDIA RTX 3090 (24GB VRAM) |
| **Precision** | bfloat16 (optimized for Ampere GPUs) |
| **Max Sequence Length** | 2048 tokens |
| **Batch Size** | default 64 (configurable via `--batch-size`, used 384 for training) |
| **Data Workers** | 4 (parallel data loading with pin_memory) |
| **Optimization** | `torch.compile(mode="reduce-overhead")` |
| **Score Calculation** | Weighted probability: `Σ(P(i) × i)` for i=0..5 |
| **Threshold** | ≥ 3.0 (discards low-quality noise) |
| **Resumability** | File-level (skips existing `scored_part_*.parquet`) |

### 5B.3 Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Original document ID |
| `text` | string | Document text |
| `url` | string | Source URL (if present) |
| `score` | float | Model score [0.0, 5.0] |

### 5B.4 Final Results

Inference was run on a Remote RTX 3090 (24GB VRAM) via vast.ai:

| Metric | Value |
|--------|-------|
| **Input** | 143,379 documents (Stage 1) |
| **Output** | 52,132 documents (36.4% pass rate) |
| **Total Content** | 66.8M words |
| **Avg Document Length** | 1,282 words |
| **Unique Domains** | 18,851 |
| **Throughput** | 41.3 docs/sec |

### 5B.5 Dataset Composition
Top domains in the produced dataset:
1. `openjurist.org` (2,185)
2. `il.findacase.com` (872)
3. `ny.findacase.com` (803)
4. `caselaw.findlaw.com` (797)
5. `pa.findacase.com` (770)

The high prevalence of case law repositories confirms the classifier's ability to identify primary legal sources.

### 5B.6 Pipeline Summary

```
FineWeb sample-10BT (10B tokens)
        │
        ▼ Stage 1: CPU Heuristic Filter (~2200 docs/sec)
   143,379 documents (1.06% pass rate)
        │
        ▼ Stage 2: LoRA Classifier (RTX 3090, bfloat16, 41.3 docs/sec)
    52,132 documents (score ≥ 3.0)
        │
        ▼ Output: data/stage2/scored_part_*.parquet
```

### 5B.7 Quality Splits (Stage 3)

The final dataset is split into 3 quality tiers for flexible usage:

| Split | Min Score | Train | Test | Total | Avg Score |
|-------|-----------|-------|------|-------|-----------|
| `default` | ≥ 3.0 | 46,918 | 5,214 | 52,132 | 4.21 |
| `high_quality` | ≥ 4.0 | 29,101 | 3,234 | 32,335 | 4.60 |
| `supreme` | ≥ 4.8 | 14,971 | 1,664 | 16,635 | 4.98 |

**Output:** `data/stage3_splits/{default,high_quality,supreme}/{train,test}.parquet`

---

## 6. Lessons Learned

### 6.1 Technical Challenges

1. **GPU Memory Management:** Initial OOM errors with batch_size=8 and max_length=2048. Resolved by reducing to batch_size=4 with gradient_accumulation=8.

2. **Dtype Compatibility:** MPS (Apple Silicon) doesn't handle bfloat16 well in softmax operations. Resolved by using float32 for inference.

3. **HuggingFace Gated Models:** Gemma Embedding requires authentication. Added HF token handling to training scripts.

4. **API Rate Limiting:** Mistral API occasionally returns 429 errors. Implemented exponential backoff with tenacity.

### 6.2 Recommendations for Future Work

1. **Scale Annotation:** Increase to 50K+ samples for more robust training
2. **Ensemble Methods:** Train multiple classifiers and ensemble predictions
3. **Active Learning:** Use model uncertainty to select samples for annotation
4. **Multi-GPU Training:** Implement distributed training for faster iteration
5. Scale the filtering to a bigger subset

---

## 7. Conclusion

FineWeb-Legal successfully demonstrates that the FineWeb-Edu methodology can be adapted for legal domain classification. The achieved **97.99% Binary F1@3** significantly exceeds the 82% baseline, indicating strong potential for creating high-quality legal training datasets.

---

## Appendix A: Environment Setup

```bash
# Python version
Python 3.12+

# Key dependencies
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.19.0
scikit-learn>=1.4.0
tensorboard>=2.16.0
typer>=0.12.0
rich>=13.7.0
```

## Appendix B: Reproducibility

**Random Seeds:**
- Annotation: 42
- Train/Val Split: 42
- PyTorch: Set via `torch.manual_seed(42)`

**Model Versions:**
- Base model: `google/embeddinggemma-300m` (commit: main)
- Annotation model: `mistral-medium-latest` (January 2026)

**Data Version:**
- FineWeb: `sample-10BT` split
- HuggingFace commit: `9bb295ddab0e05d785b879661af7260fed5140fc`
