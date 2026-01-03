#!/bin/bash
# Remote Training Deployment Script for FineWeb-Legal
# Transfers code and data to remote GPU machine, runs training, retrieves results

set -e

# Remote configuration
REMOTE_HOST="root@your-remote-ip"
REMOTE_PORT="22"
REMOTE_DIR="/root/fineweb-legal"
LOCAL_DIR="."

echo "🚀 FineWeb-Legal Remote Training Deployment"
echo "============================================"

# Step 1: Create remote directory structure
echo "📁 Setting up remote directories..."
ssh -p $REMOTE_PORT $REMOTE_HOST "mkdir -p $REMOTE_DIR/data/batches $REMOTE_DIR/src $REMOTE_DIR/models $REMOTE_DIR/logs"

# Step 2: Transfer source code
echo "📦 Transferring source code..."
rsync -avz --progress -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_DIR/src/" "$REMOTE_HOST:$REMOTE_DIR/src/"

rsync -avz --progress -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_DIR/pyproject.toml" "$REMOTE_HOST:$REMOTE_DIR/"

# Step 3: Transfer annotation data (only parquet files)
echo "📊 Transferring annotation data..."
rsync -avz --progress -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_DIR/data/batches/" "$REMOTE_HOST:$REMOTE_DIR/data/batches/"

# Step 4: Install dependencies on remote
echo "🔧 Installing dependencies on remote..."
ssh -p $REMOTE_PORT $REMOTE_HOST "cd $REMOTE_DIR && pip install -e . torch transformers peft scikit-learn tensorboard accelerate"

# Step 5: Run training
echo "🎯 Starting training..."
ssh -p $REMOTE_PORT $REMOTE_HOST "cd $REMOTE_DIR && python -m fineweb_legal.cli train --epochs 10 --batch-size 8 --max-length 2048"

# Step 6: Retrieve trained model
echo "📥 Downloading trained model..."
rsync -avz --progress -e "ssh -p $REMOTE_PORT" \
    "$REMOTE_HOST:$REMOTE_DIR/models/" "$LOCAL_DIR/models/"

echo "✅ Training complete! Model saved to $LOCAL_DIR/models/"
