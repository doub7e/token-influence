#!/bin/bash
# Batch evaluate multiple checkpoints on MATH-500
# Also evaluates the base model for comparison

set -eo pipefail
cd /scratch/cvlab/home/shuli/agentic-research/Archer2.0
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

N_SAMPLES=4
N_GPUS=1

CKPT_ROOT=output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-4H200-mb8-lp8
BASE_MODEL=/scratch/cvlab/home/shuli/agentic-research/Archer2.0/models/DeepSeek-R1-Distill-Qwen-1.5B

# Step 0: Prepare MATH-500 dataset
if [ ! -f "data/test/math500.json" ]; then
    echo "=== Preparing MATH-500 dataset ==="
    python scripts/eval/prepare_math500.py
fi

echo "===== MATH-500 Evaluation Pipeline ====="
echo "n_samples=${N_SAMPLES}, n_gpus=${N_GPUS}"
echo ""

# Evaluate base model first
echo "=== Evaluating base model: ${BASE_MODEL} ==="
bash scripts/eval/run_eval_math500.sh "${BASE_MODEL}" "${N_SAMPLES}" "${N_GPUS}" || echo "Base model eval failed"
echo ""

# Evaluate checkpoints: 100, 150, 190
for step in 100 150 190; do
    ckpt_dir="${CKPT_ROOT}/global_step_${step}"
    if [ ! -d "${ckpt_dir}" ]; then
        echo "=== Skipping step ${step}: directory not found ==="
        continue
    fi

    echo "=== Evaluating step ${step} ==="
    bash scripts/eval/eval_checkpoint_math500.sh "${ckpt_dir}" "${N_SAMPLES}" "${N_GPUS}" || echo "Step ${step} eval failed"
    echo ""
done

echo "===== All evaluations complete ====="
echo ""
echo "=== Results Summary ==="
echo ""

# Print results
for csv in "${BASE_MODEL}/output/math500.parquet.pass.csv" \
           "${CKPT_ROOT}/global_step_100/actor/hf_model/output/math500.parquet.pass.csv" \
           "${CKPT_ROOT}/global_step_150/actor/hf_model/output/math500.parquet.pass.csv" \
           "${CKPT_ROOT}/global_step_190/actor/hf_model/output/math500.parquet.pass.csv"; do
    if [ -f "$csv" ]; then
        echo "--- $(dirname $(dirname $(dirname $csv))) ---"
        cat "$csv"
        echo ""
    fi
done
