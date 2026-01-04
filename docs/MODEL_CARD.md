---
language:
- en
license: mit
library_name: peft
tags:
- legal
- classification
- gemma
- lora
- fineweb
- ablation-study
base_model: google/embeddinggemma-300m
---

# ⚖️ FineWeb-Legal Ablation Studies

<center>
    <img src="https://raw.githubusercontent.com/NoeFlandre/fineweb-legal/main/assets/logo.png" alt="FineWeb-Legal Logo" width="400"/>
</center>

This repository contains **ablation study results** for the FineWeb-Legal classifier project. We systematically tested hyperparameters to identify the optimal configuration for legal document classification.

## 📊 Ablation Results

### Sequence Length Impact
Legal documents are long. We tested context windows from 512 to 2048 tokens.

| Context Window | Best Macro F1 | Impact |
|:---|:---|:---|
| 512 | 0.5797 | ❌ Too short (loss of context) |
| 1024 | 0.6645 | Baseline |
| **2048** | **0.6715** | **✅ Winner (+1.05%)** |

**Conclusion**: Increasing to **2048 tokens** provides the most significant boost.

### Learning Rate Sweep
Tested from 1e-4 to 1e-3.

| LR | Best Macro F1 | Notes |
|:---|:---|:---|
| 1e-4 | 0.6548 | Underfitting |
| 2e-4 | 0.6645 | Stable |
| **5e-4** | **0.6655** | **Best** |
| 1e-3 | 0.6644 | Diminishing returns |

**Conclusion**: **3e-4 to 5e-4** is optimal.

### LoRA Rank Analysis
Testing adapter capacity.

| Rank | Best Macro F1 | Notes |
|:---|:---|:---|
| 8 | 0.6406 | Underfitting |
| **16** | **0.6645** | **Optimal** |
| 32 | 0.6645 | No gain, higher VRAM |

**Conclusion**: **Rank 16** is the sweet spot.

### Class Weights
| Configuration | Best Macro F1 |
|:---|:---|
| No Weights | 0.6635 |
| **With Weights** | **0.6645** |

**Conclusion**: Class weights improve performance on imbalanced data.

## 🎯 Optimal Configuration

Based on these studies, the recommended V2 configuration is:
- **Sequence Length**: 2048 tokens
- **Learning Rate**: 3e-4
- **LoRA Rank**: 16
- **Class Weights**: Enabled
- **Base Model**: `google/embeddinggemma-300m`

## 📁 Repository Structure

```
ablation_results/
├── lr_1e-4/          # Learning rate experiments
├── lr_2e-4/
├── lr_5e-4/
├── lr_1e-3/
├── lora_r8/          # LoRA rank experiments
├── lora_r16/
├── lora_r32/
├── seq_512/          # Sequence length experiments
├── seq_1024/
├── seq_2048/
├── with_weights/     # Class weight experiments
├── no_weights/
└── master_summary.json
```

Each experiment folder contains:
- `results.json` - Performance metrics
- `config.json` - Hyperparameters used
- `model/` - Trained LoRA adapters
- `confusion_matrix.npy` - Confusion matrix
- `classification_report.txt` - Detailed metrics

## 🔗 Related Artifacts

- **Filtered Dataset**: [FineWeb-Legal-Pilot](https://huggingface.co/datasets/NoeFlandre/fineweb-legal-pilot)
- **Raw Annotations**: [FineWeb-Legal-Annotations](https://huggingface.co/datasets/NoeFlandre/fineweb-legal-annotations)
- **Code & Documentation**: [GitHub Repository](https://github.com/NoeFlandre/fineweb-legal)

## 📝 Methodology

All experiments used:
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **Training samples**: 5,000 (stratified)
- **Validation samples**: 1,000 (stratified)
- **Base model**: `google/embeddinggemma-300m`
- **Task**: 6-class legal quality classification (0-5)
- **Metric**: Macro F1 Score

## License

MIT License.
