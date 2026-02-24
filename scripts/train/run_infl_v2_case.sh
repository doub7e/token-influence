#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train/run_infl_v2_case.sh <exp_name> <scope>
# Example:
#   bash scripts/train/run_infl_v2_case.sh infl-v2-perprompt-0222 per_prompt

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <exp_name> <scope>" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

exp_name="$1"
scope="$2"

if [ ! -f /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key ]; then
  echo "[FATAL] wandb key file not found" >&2
  exit 2
fi

export WANDB_API_KEY="$(cat /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key)"
export WANDB_MODE=online

export VERL_PG_TIMEOUT_SECONDS=10800
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

export INFLUENCE_EXP_NAME="${exp_name}"
export INFLUENCE_TOTAL_STEPS=1
export INFLUENCE_PPO_EPOCHS=1
export INFLUENCE_TOTAL_EPOCHS=1
export INFLUENCE_HESSIAN_MODE=inverse
export INFLUENCE_OUTPUT_FUNCTION=log_prob_reward
export INFLUENCE_ACCEPTED_REJECTED_SCOPE="${scope}"
export INFLUENCE_PROJECTION_DIM_FACTOR=32
export INFLUENCE_MAX_HESSIAN_DIM=-1
export INFLUENCE_PROFILE_TIMING=True
export INFLUENCE_MAX_TOKENS_PER_RESPONSE=-1
export INFLUENCE_SKIP_OPTIMIZER_STEP=False
export INFLUENCE_EXCLUDE_SELF_RESPONSE=True
export INFLUENCE_CONTRASTIVE_AGG=mean
export INFLUENCE_HESSIAN_SOURCE=token
export INFLUENCE_GRAD_OFFLOAD_TO_CPU=True
export INFLUENCE_DEBUG_HESSIAN_SIMILARITY=True

echo "[RUN] exp=${exp_name} scope=${scope}"
echo "[CFG] output_function=log_prob_reward contrastive_agg=mean hessian_source=token"
echo "[CFG] projection_dim_factor=32 max_hessian_dim=-1 steps=1 skip_optimizer_step=False"

bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh
