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
base_model: google/embeddinggemma-300m
pipeline_tag: text-classification
widget:
- text: "The Supreme Court of the United States held that the First Amendment protects..."
- text: "Cookie Policy: We use cookies to improve your experience."
---

# ⚖️ FineWeb-Legal-Classifier

<center>
    <img src="https://raw.githubusercontent.com/NoeFlandre/fineweb-legal/main/assets/logo.png" alt="FineWeb-Legal Logo" width="400"/>
</center>

**FineWeb-Legal-Classifier** is a lightweight (300M parameter) model designed to identify high-quality legal content from web crawls. It was trained to filter the [FineWeb-Legal-Pilot](https://huggingface.co/datasets/NoeFlandre/fineweb-legal-pilot) dataset.

## Model Description

- **Base Model**: `google/embeddinggemma-300m`
- **Architecture**: PEFT LoRA Adapter + Linear Classification Head
- **Task**: Regression (Score 0.0 - 5.0) converted to classes
- **Training Data**: 6,500 FineWeb samples annotated by Mistral-Medium

## Performance

| Metric | Score |
|--------|-------|
| **Binary F1 (@3.0)** | **97.99%** |
| Validation Acc | 88.8% |
| Macro F1 | 0.857 |

## Usage

```python
import torch
from components.model import LegalClassifier # From our repo

# Load Model
model = LegalClassifier.from_pretrained("NoeFlandre/fineweb-legal-classifier")
model.eval()

# Score Text
text = "The plaintiff filed a motion for summary judgment..."
score = model.predict(text)
print(f"Legal Quality Score: {score:.2f} / 5.0")
```

## Training Details

- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Epochs**: 4 (Early Stopped)
- **Batch Size**: 32 (Effective)
- **Learning Rate**: 3e-4

## Ablation Studies

We conducted extensive ablation studies to identify optimal hyperparameters for V2 training:

### Sequence Length Impact
| Context Window | Best Macro F1 | Notes |
|:---|:---|:---|
| 512 | 0.5797 | ❌ Too short |
| 1024 | 0.6645 | Baseline |
| **2048** | **0.6715** | **✅ Winner (+1.05%)** |

### Learning Rate Sweep
| LR | Best Macro F1 |
|:---|:---|
| 1e-4 | 0.6548 |
| 2e-4 | 0.6645 |
| **5e-4** | **0.6655** |
| 1e-3 | 0.6644 |

### LoRA Rank Analysis
| Rank | Best Macro F1 |
|:---|:---|
| 8 | 0.6406 |
| **16** | **0.6645** |
| 32 | 0.6645 |

**Conclusion**: Optimal config for V2 is **Seq=2048, LR=3e-4, Rank=16, Class Weights=Enabled**.

## Related Artifacts

- **Filtered Dataset**: [FineWeb-Legal-Pilot](https://huggingface.co/datasets/NoeFlandre/fineweb-legal-pilot)
- **Raw Annotations**: [FineWeb-Legal-Annotations](https://huggingface.co/datasets/NoeFlandre/fineweb-legal-annotations)
- **Code & Ablations**: [GitHub Repository](https://github.com/NoeFlandre/fineweb-legal)

## License

MIT License.
