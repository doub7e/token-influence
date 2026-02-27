#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p output data/train data/test models
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface

if [ ! -f data/test/aime2025.json ]; then
  echo "[INFO] data/test/aime2025.json missing; creating from smoke file."
  cp data/test/smoke_math.json data/test/aime2025.json
fi

echo "[INFO] Downloading Qwen3-0.6B-Base model checkpoint..."
python scripts/train/download_hf_model.py \
  --repo-id Qwen/Qwen3-0.6B-Base \
  --local-dir models/Qwen3-0.6B-Base \
  2>&1 | tee output/download_qwen3_0p6b.log

echo "[INFO] Downloading Qwen3-0.6B (Instruct) model checkpoint..."
python scripts/train/download_hf_model.py \
  --repo-id Qwen/Qwen3-0.6B \
  --local-dir models/Qwen3-0.6B \
  2>&1 | tee output/download_qwen3_0p6b_instruct.log

echo "[OK] Qwen3-0.6B setup completed."
