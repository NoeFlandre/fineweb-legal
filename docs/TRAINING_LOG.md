# Training Log - FineWeb-Legal Classifier

## Run Information

| Field | Value |
|-------|-------|
| **Run ID** | `run_20260102_142030` |
| **Date** | January 2, 2026 |
| **Duration** | 1h 50m |
| **Status** | ✅ Completed Successfully |

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3090 |
| VRAM | 24 GB |
| VRAM Used | ~10-12 GB |
| Provider | vast.ai |
| SSH Host | `root@remote-gpu` |


---

## Configuration

```python
TrainingConfig(
    epochs=10,
    learning_rate=3e-4,
    batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    max_length=2048,
    save_dir="models",
    log_dir="logs",
    early_stopping_patience=5,
)

LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

---

## Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | 5,200 | data/batches (80%) |
| Validation | 1,300 | data/batches (20%) |

### Class Distribution (Training)

| Score | Count | Percentage | Weight |
|-------|-------|------------|--------|
| 0 | 2,002 | 38.5% | 0.322 |
| 1 | 607 | 11.7% | 1.062 |
| 2 | 465 | 8.9% | 1.387 |
| 3 | 623 | 12.0% | 1.035 |
| 4 | 1,102 | 21.2% | 0.585 |
| 5 | 401 | 7.7% | 1.608 |

---

## Epoch-by-Epoch Results

### Epoch 1
```
Time: 14:08:21 - 14:34:31
Train: loss=0.8976 acc=0.6813 f1=0.6047
Val:   loss=0.5923 acc=0.7746 f1=0.7370 binary_f1@3=0.9532
✓ New best model saved
```

### Epoch 2
```
Time: 14:34:31 - 14:48:33
Train: loss=0.4620 acc=0.8504 f1=0.8125
Val:   loss=0.5946 acc=0.8123 f1=0.7721 binary_f1@3=0.9108
```

### Epoch 3
```
Time: 14:48:33 - 15:02:34
Train: loss=0.3000 acc=0.9102 f1=0.8848
Val:   loss=0.4036 acc=0.8954 f1=0.8721 binary_f1@3=0.9627
✓ New best model saved
```

### Epoch 4 (Final)
```
Time: 15:02:34 - 15:16:35 (estimated)
Train: loss=0.2287 acc=0.9346 f1=0.9XX
Val:   loss=0.4769 acc=0.8877 f1=0.8570 binary_f1@3=0.9799
✓ New best model saved
```

---

## Best Model Checkpoint

**Saved at:** `models/best/`

```json
{
  "epoch": 4,
  "accuracy": 0.8876923076923077,
  "macro_f1": 0.8569523339367481,
  "weighted_f1": 0.890974297168496,
  "macro_precision": 0.8501100785783818,
  "macro_recall": 0.8838008163270122,
  "binary_f1_3": 0.9799426934097422,
  "loss": 0.47692759129576956
}
```

---

## Files Generated

### Model Artifacts
- `models/best/adapter_config.json` (1.0 KB)
- `models/best/adapter_model.safetensors` (7.9 MB)
- `models/best/classifier_head.pt` (11 KB)
- `models/best/metrics.json` (375 B)
- `models/best/README.md` (5.2 KB)

### TensorBoard Logs
- `logs/run_20260102_142030/` (main run)
- `logs/run_20260102_140820/` (failed run 1)
- `logs/run_20260102_141033/` (failed run 2)
- `logs/run_20260102_141155/` (batch_size=2 test)
- `logs/run_20260102_141914/` (OOM run)

---

## Issues Encountered

### Issue 1: HuggingFace Authentication
- **Problem:** `google/embeddinggemma-300m` is a gated model
- **Error:** `401 Unauthorized`
- **Solution:** Added `huggingface-cli login` with token

### Issue 2: CUDA Out of Memory
- **Problem:** batch_size=16 with max_length=2048 exceeded 24GB
- **Error:** `torch.OutOfMemoryError`
- **Solution:** Reduced to batch_size=4 with gradient_accumulation=8

### Issue 3: Dtype Mismatch
- **Problem:** Encoder outputs bfloat16, classifier expects float32
- **Error:** `RuntimeError: mat1 and mat2 must have the same dtype`
- **Solution:** Added explicit cast to bfloat16 in forward pass

### Issue 4: Stale GPU Processes
- **Problem:** Previous training run left GPU memory allocated
- **Solution:** `pkill -9 -f python3` before new run

---

## Commands Used

### Training Script
```bash
cd /root/fineweb-legal
export HF_TOKEN="hf_xxx"
python3 << 'EOF'
from huggingface_hub import login
login(token="hf_xxx")

from fineweb_legal.classifier.dataset import create_dataloaders
from fineweb_legal.classifier.model import LegalClassifier
from fineweb_legal.classifier.trainer import ClassifierTrainer, TrainingConfig

model = LegalClassifier(lora_r=16)
train_loader, val_loader = create_dataloaders(
    data_path='data/batches',
    tokenizer=model.tokenizer,
    batch_size=4,
    max_length=2048,
)

config = TrainingConfig(
    epochs=10,
    learning_rate=3e-4,
    save_dir='models',
    log_dir='logs',
    gradient_accumulation_steps=8,
)

trainer = ClassifierTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    class_weights=train_loader.dataset.get_class_weights(),
)

results = trainer.train()
print(f'Best F1@3: {results["best_val_f1"]:.4f}')
EOF
```

### Download Results
```bash
rsync -avz -e "ssh -p 22" root@remote-gpu:/root/fineweb-legal/models/ models/
rsync -avz -e "ssh -p 22" root@remote-gpu:/root/fineweb-legal/logs/ logs/
```

---

## Verification

### Mac M2 Inference Test
```bash
python test_inference_mac.py
```

**Results:**
- Speed: 2.94 samples/second
- LEGAL detections: 2/50 (4%)
- NON-LEGAL detections: 48/50 (96%)
- Model correctly identifies legal content ✅
