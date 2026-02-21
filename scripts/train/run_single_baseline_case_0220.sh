#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train/run_single_baseline_case_0220.sh <exp_name>

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <exp_name>" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."
exp_name="$1"

if [ ! -f /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key ]; then
  echo "[FATAL] wandb key file not found: /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key" >&2
  exit 2
fi

export WANDB_API_KEY="$(cat /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key)"
export WANDB_MODE=online
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

log_path="${log_dir}/${exp_name}.log"
ckpt_dir="./output/Archer2.0/${exp_name}"
mkdir -p "${ckpt_dir}"

: > "${log_path}"
echo "[RUN] baseline exp=${exp_name}" | tee -a "${log_path}"
echo "[CFG] VERL_PG_TIMEOUT_SECONDS=${VERL_PG_TIMEOUT_SECONDS}" | tee -a "${log_path}"
echo "[CFG] TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING} TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" | tee -a "${log_path}"

set +e
bash scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh \
  trainer.experiment_name="${exp_name}" \
  trainer.default_local_dir="${ckpt_dir}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  +trainer.entropy_trace.enable=False \
  +trainer.influence_trace.enable=False \
  2>&1 | tee -a "${log_path}"
rc=${PIPESTATUS[0]}
set -e

if [ "${rc}" -eq 0 ]; then
  printf "%s,%s,%s,%s,%s,%s,%s\n" "baseline" "na" "na" "${exp_name}" "${log_path}" "False" "ok" >> "${summary_csv}"
  exit 0
fi

printf "%s,%s,%s,%s,%s,%s,%s\n" "baseline" "na" "na" "${exp_name}" "${log_path}" "False" "failed" >> "${summary_csv}"
exit "${rc}"
