#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train/run_single_influence_case_0220.sh <exp_base> <mode> <scope> <hessian_mode>
# Example:
#   bash scripts/train/run_single_influence_case_0220.sh archer-infl-abl-3-logpadv-prompt-f128-retry-0220 log_prob_advantage per_prompt inverse

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <exp_base> <mode> <scope> <hessian_mode>" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

exp_base="$1"
mode="$2"
scope="$3"
hessian_mode="$4"

if [ ! -f /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key ]; then
  echo "[FATAL] wandb key file not found: /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key" >&2
  exit 2
fi

export WANDB_API_KEY="$(cat /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key)"
export WANDB_MODE=online

# Prevent NCCL watchdog timeouts for long influence post-processing.
export VERL_PG_TIMEOUT_SECONDS=10800
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

summary_root="output/influence_ablation_0220_resume"
log_dir="${summary_root}/logs"
mkdir -p "${log_dir}"

summary_csv="${summary_root}/experiment_cases.csv"
if [ ! -f "${summary_csv}" ]; then
  printf "mode,scope,hessian_mode,exp_name,log_path,offload_to_cpu,status\n" > "${summary_csv}"
fi

# Start from CPU offload by default to avoid known slow/failing non-offload paths.
# Set INFLUENCE_START_WITH_CPU_OFFLOAD=False to force non-offload (no auto fallback).
offload="${INFLUENCE_START_WITH_CPU_OFFLOAD:-True}"
exp_name="${exp_base}"
if [ "${offload}" = "True" ]; then
  exp_name="${exp_base}-offcpu"
fi
log_path="${log_dir}/${exp_name}.log"

export INFLUENCE_EXP_NAME="${exp_name}"
export INFLUENCE_TOTAL_STEPS=1
export INFLUENCE_PPO_EPOCHS=1
export INFLUENCE_TOTAL_EPOCHS=1
export INFLUENCE_PROJECTION_DIM_FACTOR="${INFLUENCE_PROJECTION_DIM_FACTOR:-128}"
export INFLUENCE_OUTPUT_FUNCTION="${mode}"
export INFLUENCE_ACCEPTED_REJECTED_SCOPE="${scope}"
export INFLUENCE_HESSIAN_MODE="${hessian_mode}"
export INFLUENCE_GRAD_OFFLOAD_TO_CPU="${offload}"

: > "${log_path}"
echo "[RUN] mode=${mode} scope=${scope} hessian=${hessian_mode} offload=${offload} exp=${exp_name}" | tee -a "${log_path}"
echo "[CFG] steps=${INFLUENCE_TOTAL_STEPS} factor=${INFLUENCE_PROJECTION_DIM_FACTOR}" | tee -a "${log_path}"
echo "[CFG] VERL_PG_TIMEOUT_SECONDS=${VERL_PG_TIMEOUT_SECONDS}" | tee -a "${log_path}"
echo "[CFG] TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING} TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" | tee -a "${log_path}"

set +e
bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh 2>&1 | tee -a "${log_path}"
rc=${PIPESTATUS[0]}
set -e

if [ "${rc}" -eq 0 ]; then
  printf "%s,%s,%s,%s,%s,%s,%s\n" "${mode}" "${scope}" "${hessian_mode}" "${exp_name}" "${log_path}" "${offload}" "ok" >> "${summary_csv}"
  exit 0
fi

printf "%s,%s,%s,%s,%s,%s,%s\n" "${mode}" "${scope}" "${hessian_mode}" "${exp_name}" "${log_path}" "${offload}" "failed" >> "${summary_csv}"
exit "${rc}"
