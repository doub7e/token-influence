#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

project_name="Archer2.0"
exp_name="Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace"
ckpt_dir="./output/${project_name}/${exp_name}"
trace_dir="${ckpt_dir}/entropy_trace"

mkdir -p "${ckpt_dir}" "${trace_dir}"

bash scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh \
  trainer.experiment_name="${exp_name}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=100 \
  +trainer.entropy_trace.enable=True \
  +trainer.entropy_trace.output_dir="${trace_dir}" \
  "$@"
