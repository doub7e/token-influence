#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/train/run_archer2.0_qwen3_0.6b_math.sh \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  data.train_files=./data/train/archer2.0-math-1.5b-train.json \
  data.val_files=./data/test/aime2025.json \
  data.train_batch_size=8 \
  data.gen_batch_size=8 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1536 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1536 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1536 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
  actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.val_kwargs.response_length=512 \
  "$@"
