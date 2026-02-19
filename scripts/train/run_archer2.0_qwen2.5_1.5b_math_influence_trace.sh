#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

project_name="Archer2.0"
exp_name="Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace"
ckpt_dir="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "./output/${project_name}/${exp_name}")"
entropy_trace_dir="${ckpt_dir}/entropy_trace"
influence_trace_dir="${ckpt_dir}/influence_trace"

mkdir -p "${ckpt_dir}" "${entropy_trace_dir}" "${influence_trace_dir}"

bash scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh \
  trainer.experiment_name="${exp_name}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  +trainer.entropy_trace.enable=True \
  +trainer.entropy_trace.output_dir="${entropy_trace_dir}" \
  +trainer.entropy_trace.write_every=1 \
  +trainer.entropy_trace.update_summary_every=1 \
  +trainer.entropy_trace.atomic_write=True \
  +trainer.entropy_trace.fsync=False \
  +trainer.influence_trace.enable=True \
  +trainer.influence_trace.output_dir="${influence_trace_dir}" \
  +trainer.influence_trace.write_every=1 \
  +trainer.influence_trace.atomic_write=True \
  +trainer.influence_trace.fsync=False \
  +trainer.influence_trace.max_prompts_per_step=2 \
  +trainer.influence_trace.reg_lambda=-1.0 \
  +trainer.influence_trace.module_name_filter='[self_attn.o_proj,mlp.down_proj]' \
  +trainer.influence_trace.max_modules=1 \
  +trainer.influence_trace.max_proj_vector_sum=64 \
  +trainer.influence_trace.max_hessian_dim=2500 \
  "$@"

