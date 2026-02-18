#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

project_name="Archer2.0"
exp_name="Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace-v9"
trace_dir="./output/${project_name}/${exp_name}/entropy_trace"
mkdir -p "${trace_dir}"

bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh \
  trainer.experiment_name="${exp_name}" \
  trainer.entropy_trace.output_dir="${trace_dir}" \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=100 \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  data.max_prompt_length=128 \
  data.max_response_length=64 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=256 \
  actor_rollout_ref.rollout.max_model_len=192 \
  actor_rollout_ref.rollout.val_kwargs.response_length=64 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=16
