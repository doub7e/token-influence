#!/usr/bin/env bash
set -euo pipefail

export HOME=/scratch/cvlab/home/shuli
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
cd /scratch/cvlab/home/shuli/agentic-research/Archer2.0

# Step 1: Setup model
bash scripts/train/setup_qwen3_0.6b.sh

# Step 2: Run smoke test (vLLM Qwen3 patch is auto-applied via verl/__init__.py)
echo "[INFO] Starting smoke test..."
bash scripts/train/run_archer2.0_qwen3_0.6b_math_smoke_4gpu.sh \
    2>&1 | tee output/archer-qwen3-patched-smoke.log
