#!/usr/bin/env bash
set -euo pipefail

export HOME=/scratch/cvlab/home/shuli
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
cd /scratch/cvlab/home/shuli/agentic-research/Archer2.0

echo "[INFO] Testing vLLM Qwen3 patch condition fix"

# Quick check: does the patch run?
python -c "
from verl.third_party.vllm import vllm_version
print(f'vllm_version={vllm_version}')
"

echo "[INFO] Starting smoke test..."
bash scripts/train/run_archer2.0_qwen3_0.6b_math_smoke_4gpu.sh \
    2>&1 | tee output/archer-qwen3-patch3.log
