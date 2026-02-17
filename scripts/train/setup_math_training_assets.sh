#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p output data/train data/test models
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface

echo "[INFO] Preparing DAPO-Math-17k training file..."
python scripts/train/prepare_dapo_math17k.py \
  --dataset BytedTsinghua-SIA/DAPO-Math-17k \
  --split train \
  --max-samples 17000 \
  --output data/train/archer2.0-math-1.5b-train.json \
  2>&1 | tee output/prepare_dapo_math17k.log

if [ ! -f data/test/aime2025.json ]; then
  echo "[INFO] data/test/aime2025.json missing; creating from smoke file."
  cp data/test/smoke_math.json data/test/aime2025.json
fi

echo "[INFO] Downloading model checkpoint..."
python scripts/train/download_hf_model.py \
  --repo-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B \
  2>&1 | tee output/download_deepseek_1p5b.log

echo "[OK] Setup completed."
