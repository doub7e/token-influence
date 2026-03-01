#!/bin/bash
# End-to-end evaluation pipeline for a single checkpoint on MATH-500
# Steps: 1) model_merge (FSDP -> HF)  2) run_eval_math500
#
# Usage: bash scripts/eval/eval_checkpoint_math500.sh <checkpoint_dir> [n_samples] [n_gpus]
#
# Example:
#   bash scripts/eval/eval_checkpoint_math500.sh output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-4H200-mb8-lp8/global_step_100 4 4

set -eo pipefail
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

checkpoint_dir=${1:?Usage: eval_checkpoint_math500.sh <checkpoint_dir> [n_samples] [n_gpus]}
n_samples=${2:-4}
n_gpus=${3:-4}

actor_dir="${checkpoint_dir}/actor"
hf_model_dir="${actor_dir}/hf_model"

# Step 0: Prepare MATH-500 dataset if not already present
if [ ! -f "data/test/math500.json" ]; then
    echo "=== Preparing MATH-500 dataset ==="
    python scripts/eval/prepare_math500.py
fi

# Step 1: Convert FSDP checkpoint to HF format (skip if already done)
if [ -f "${hf_model_dir}/config.json" ]; then
    echo "=== HF model already exists at ${hf_model_dir}, skipping merge ==="
else
    echo "=== Step 1: Converting FSDP checkpoint to HF format ==="
    python -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${hf_model_dir}"
    echo "=== Model merge complete ==="
fi

# Step 2: Run MATH-500 evaluation
echo "=== Step 2: Running MATH-500 evaluation (n_samples=${n_samples}, n_gpus=${n_gpus}) ==="
bash scripts/eval/run_eval_math500.sh "${hf_model_dir}" "${n_samples}" "${n_gpus}"

echo "=== Evaluation complete ==="
echo "Results: ${hf_model_dir}/output/math500.parquet.pass.csv"
