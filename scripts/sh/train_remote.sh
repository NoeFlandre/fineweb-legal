#!/bin/bash
# Full automated remote training pipeline
# Run from: /Volumes/Seagate M3/fineweb-legal/

set -e

REMOTE="root@your-remote-ip"
PORT="22"
REMOTE_DIR="/root/fineweb-legal"
LOCAL_DIR="."

echo "🚀 FineWeb-Legal Automated Remote Training"
echo "==========================================="
echo "Remote: $REMOTE:$PORT"
echo "Local: $LOCAL_DIR"
echo ""

# Step 1: Transfer source code
echo "📦 [1/6] Transferring source code..."
rsync -avz --progress -e "ssh -p $PORT" \
    "$LOCAL_DIR/src/" "$REMOTE:$REMOTE_DIR/src/"

rsync -avz -e "ssh -p $PORT" \
    "$LOCAL_DIR/pyproject.toml" "$REMOTE:$REMOTE_DIR/"

rsync -avz -e "ssh -p $PORT" \
    "$LOCAL_DIR/README.md" "$REMOTE:$REMOTE_DIR/"

# Step 2: Transfer data
echo "📊 [2/6] Transferring annotation data..."
rsync -avz --progress -e "ssh -p $PORT" \
    "$LOCAL_DIR/data/batches/" "$REMOTE:$REMOTE_DIR/data/batches/"

# Step 3: Install dependencies on remote (without editable install)
echo "🔧 [3/6] Installing dependencies..."
ssh -p $PORT $REMOTE << 'REMOTE_INSTALL'
cd /root/fineweb-legal
pip3 install --quiet torch transformers peft scikit-learn tensorboard accelerate pyarrow huggingface_hub
# Login to HuggingFace for Gemma access
# NOTE: Replace with your actual token or set HF_TOKEN env var
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
echo "Dependencies installed!"
REMOTE_INSTALL

# Step 4: Run training
echo "🎯 [4/6] Starting training (this may take 30-60 minutes)..."
ssh -p $PORT $REMOTE << 'REMOTE_TRAIN'
cd /root/fineweb-legal
export PYTHONPATH="/root/fineweb-legal/src:$PYTHONPATH"
# Ensure HF_TOKEN is set in your local environment or replace here
export HF_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 << 'EOF'
import os
# Token is inherited from env
# os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE" 

import logging
import sys
import torch
sys.path.insert(0, '/root/fineweb-legal/src')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {tag}: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total")

from huggingface_hub import login
# login(token=os.environ.get("HF_TOKEN")) # Optional if env var is set correctly, but handled by hf-cli usually

from fineweb_legal.classifier.dataset import create_dataloaders
from fineweb_legal.classifier.model import LegalClassifier
from fineweb_legal.classifier.trainer import ClassifierTrainer, TrainingConfig

log_gpu_memory("Before model load")

print('Loading model...')
model = LegalClassifier(lora_r=16)
log_gpu_memory("After model load")

print('Loading data...')
train_loader, val_loader = create_dataloaders(
    data_path='data/batches',
    tokenizer=model.tokenizer,
    batch_size=4,  # ~10-12GB on RTX 3090
    max_length=2048,  # Full context for legal docs
)
log_gpu_memory("After data load")

class_weights = train_loader.dataset.get_class_weights()

config = TrainingConfig(
    epochs=10,
    learning_rate=3e-4,
    save_dir='models',
    log_dir='logs',
    gradient_accumulation_steps=8,  # Effective batch = 32
)

trainer = ClassifierTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    class_weights=class_weights,
)

log_gpu_memory("Before training")
results = trainer.train()
log_gpu_memory("After training")
print(f'Training complete! Best F1@3: {results["best_val_f1"]:.4f}')
EOF
REMOTE_TRAIN


# Step 5: Download trained model
echo "📥 [5/6] Downloading trained model..."
rsync -avz --progress -e "ssh -p $PORT" \
    "$REMOTE:$REMOTE_DIR/models/" "$LOCAL_DIR/models/"

# Step 6: Download logs
echo "📈 [6/6] Downloading training logs..."
rsync -avz -e "ssh -p $PORT" \
    "$REMOTE:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"

echo ""
echo "✅ COMPLETE! Trained model saved to: $LOCAL_DIR/models/"
echo "📊 TensorBoard logs saved to: $LOCAL_DIR/logs/"
echo ""
echo "You can now delete the remote machine."
