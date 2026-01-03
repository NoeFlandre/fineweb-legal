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

## License

MIT License.
