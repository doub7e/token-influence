#!/bin/bash
# Batch evaluate InflWeight-PerPrompt checkpoints on MATH-500

set -eo pipefail
cd /scratch/cvlab/home/shuli/agentic-research/Archer2.0
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

N_SAMPLES=4
N_GPUS=1

CKPT_ROOT=output/Archer2.0/Archer2.0-R1-1.5B-InflWeight-PerPrompt

# Step 0: Prepare MATH-500 dataset
if [ ! -f "data/test/math500.json" ]; then
    echo "=== Preparing MATH-500 dataset ==="
    python scripts/eval/prepare_math500.py
fi

echo "===== MATH-500 Eval: InflWeight-PerPrompt ====="
echo "n_samples=${N_SAMPLES}, n_gpus=${N_GPUS}"
echo ""

for step in 100 140; do
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

for step in 100 140; do
    csv="${CKPT_ROOT}/global_step_${step}/actor/hf_model/output/math500.parquet.pass.csv"
    if [ -f "$csv" ]; then
        echo "--- Step ${step} ---"
        cat "$csv"
        echo ""
    fi
done
