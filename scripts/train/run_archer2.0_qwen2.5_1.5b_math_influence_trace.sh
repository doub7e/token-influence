#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

project_name="Archer2.0"
exp_name="${INFLUENCE_EXP_NAME:-Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace}"
ckpt_dir="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "./output/${project_name}/${exp_name}")"
entropy_trace_dir="${ckpt_dir}/entropy_trace"
influence_trace_dir="${ckpt_dir}/influence_trace"
total_steps="${INFLUENCE_TOTAL_STEPS:-2}"
ppo_epochs="${INFLUENCE_PPO_EPOCHS:-1}"
total_epochs="${INFLUENCE_TOTAL_EPOCHS:-1}"
max_prompts_per_step="${INFLUENCE_MAX_PROMPTS_PER_STEP:-0}"
reg_lambda="${INFLUENCE_REG_LAMBDA:--1.0}"
hessian_mode="${INFLUENCE_HESSIAN_MODE:-inverse}"
output_function="${INFLUENCE_OUTPUT_FUNCTION:-training_loss}"
accepted_rejected_scope="${INFLUENCE_ACCEPTED_REJECTED_SCOPE:-per_prompt}"
module_name_filter="${INFLUENCE_MODULE_NAME_FILTER:-[self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj]}"
max_modules="${INFLUENCE_MAX_MODULES:--1}"
projection_dim_factor="${INFLUENCE_PROJECTION_DIM_FACTOR:-512}"
max_proj_vector_sum="${INFLUENCE_MAX_PROJ_VECTOR_SUM:--1}"
max_hessian_dim="${INFLUENCE_MAX_HESSIAN_DIM:-2500}"
max_tokens_per_response="${INFLUENCE_MAX_TOKENS_PER_RESPONSE:--1}"
skip_optimizer_step="${INFLUENCE_SKIP_OPTIMIZER_STEP:-False}"
grad_offload_to_cpu="${INFLUENCE_GRAD_OFFLOAD_TO_CPU:-False}"
force_gpu_compute="${INFLUENCE_FORCE_GPU_COMPUTE:-True}"
profile_timing="${INFLUENCE_PROFILE_TIMING:-False}"
exclude_self_response="${INFLUENCE_EXCLUDE_SELF_RESPONSE:-False}"
contrastive_agg="${INFLUENCE_CONTRASTIVE_AGG:-sum}"
hessian_source="${INFLUENCE_HESSIAN_SOURCE:-response}"
debug_hessian_similarity="${INFLUENCE_DEBUG_HESSIAN_SIMILARITY:-False}"

mkdir -p "${ckpt_dir}" "${entropy_trace_dir}" "${influence_trace_dir}"

bash scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh \
  trainer.experiment_name="${exp_name}" \
  trainer.default_local_dir="${ckpt_dir}" \
  actor_rollout_ref.actor.ppo_epochs="${ppo_epochs}" \
  trainer.total_epochs="${total_epochs}" \
  trainer.total_training_steps="${total_steps}" \
  +trainer.entropy_trace.enable=False \
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
  +trainer.influence_trace.max_prompts_per_step="${max_prompts_per_step}" \
  +trainer.influence_trace.reg_lambda="${reg_lambda}" \
  +trainer.influence_trace.hessian_mode="${hessian_mode}" \
  +trainer.influence_trace.output_function="${output_function}" \
  +trainer.influence_trace.accepted_rejected_scope="${accepted_rejected_scope}" \
  +trainer.influence_trace.module_name_filter="${module_name_filter}" \
  +trainer.influence_trace.max_modules="${max_modules}" \
  +trainer.influence_trace.projection_dim_factor="${projection_dim_factor}" \
  +trainer.influence_trace.max_proj_vector_sum="${max_proj_vector_sum}" \
  +trainer.influence_trace.max_hessian_dim="${max_hessian_dim}" \
  +trainer.influence_trace.max_tokens_per_response="${max_tokens_per_response}" \
  +trainer.influence_trace.skip_optimizer_step="${skip_optimizer_step}" \
  +trainer.influence_trace.grad_offload_to_cpu="${grad_offload_to_cpu}" \
  +trainer.influence_trace.force_gpu_compute="${force_gpu_compute}" \
  +trainer.influence_trace.profile_timing="${profile_timing}" \
  +trainer.influence_trace.exclude_self_response="${exclude_self_response}" \
  +trainer.influence_trace.contrastive_agg="${contrastive_agg}" \
  +trainer.influence_trace.hessian_source="${hessian_source}" \
  +trainer.influence_trace.debug_hessian_similarity="${debug_hessian_similarity}" \
  "$@"
