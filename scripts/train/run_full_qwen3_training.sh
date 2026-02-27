#!/usr/bin/env bash
set -euo pipefail

export HOME=/scratch/cvlab/home/shuli
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
cd /scratch/cvlab/home/shuli/agentic-research/Archer2.0

# Step 1: Ensure model is downloaded
bash scripts/train/setup_qwen3_0.6b.sh

# Step 2: Run full training
echo "[INFO] Starting full Qwen3-0.6B GRPO math training..."
bash scripts/train/run_archer2.0_qwen3_0.6b_math.sh \
    2>&1 | tee output/archer-qwen3-math-v3.log
