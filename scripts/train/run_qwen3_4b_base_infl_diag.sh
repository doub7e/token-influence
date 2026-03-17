#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_API_KEY="$(cat /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key)"

# Timeouts for distributed training
export VERL_PG_TIMEOUT_SECONDS=10800
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

# --- Download model if needed ---
python scripts/train/download_hf_model.py \
  --repo-id Qwen/Qwen3-4B-Base \
  --local-dir models/Qwen3-4B-Base

nnodes=1

project_name='Archer2.0'
exp_name='Archer2.0-Qwen3-4B-Base-InflDiag-MLP-f64'

adv_estimator=grpo

# kl config
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.0
kl_loss_type=low_var_kl

# clip
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode=token-mean

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
enable_overlong_buffer=True
overlong_buffer_len=256
overlong_penalty_factor=1.0
v_max_response_length=$((1024 * 4))

train_prompt_bsz=32
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=8

# Paths
MODEL_PATH=./models/Qwen3-4B-Base
CKPTS_DIR=./output/${project_name}/${exp_name} && mkdir -p $CKPTS_DIR
data_dir=./data
TRAIN_FILE=$data_dir/train/archer2.0-math-1.5b-train.json
TEST_FILE=$data_dir/test/math500.json

# Algorithm
n_resp_per_prompt=16
temperature=1.0
top_p=1.0
top_k=-1
v_n=4
v_temperature=0.6
v_top_p=0.95
v_top_k=-1

# Performance
sp_size=1
gen_tp=1
use_dynamic_bsz=False
micro_batch_size_per_gpu=2  # 4B model + 4k response → small micro-batch
actor_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
offload=False

# Actor
use_archer_policy_loss=False
use_dynamic_clip=False
token_entropy_quantile=0.8
high_entropy_kl_loss_scale_coef=0.0
low_entropy_clip_ratio_low=0.2
low_entropy_clip_ratio_high=0.2
high_entropy_clip_ratio_low=0.4
high_entropy_clip_ratio_high=0.4

# Trainer
use_overlong_filter=False


python -m dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    +actor_rollout_ref.actor.high_entropy_kl_loss_scale_coef=${high_entropy_kl_loss_scale_coef} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_low=${low_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_high=${low_entropy_clip_ratio_high} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_low=${high_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_high=${high_entropy_clip_ratio_high} \
    +actor_rollout_ref.actor.use_archer_policy_loss=${use_archer_policy_loss} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.82 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${v_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${v_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${v_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${v_n} \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${v_max_response_length} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=wizard \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${nnodes}" \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=999 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    +trainer.enable_overlong_filter=${use_overlong_filter} \
    +trainer.rejection_sample=True \
    +trainer.influence_trace.enable=True \
    +trainer.influence_trace.output_dir="${CKPTS_DIR}/influence_trace" \
    +trainer.influence_trace.write_every=1 \
    +trainer.influence_trace.atomic_write=True \
    +trainer.influence_trace.fsync=False \
    +trainer.influence_trace.max_prompts_per_step=0 \
    +trainer.influence_trace.reg_lambda=-1.0 \
    +trainer.influence_trace.hessian_mode=inverse \
    +trainer.influence_trace.output_function=log_prob \
    +trainer.influence_trace.accepted_rejected_scope=per_prompt \
    +trainer.influence_trace.module_name_filter='[mlp.gate_proj,mlp.up_proj,mlp.down_proj]' \
    +trainer.influence_trace.max_modules=-1 \
    +trainer.influence_trace.projection_dim_factor=64 \
    +trainer.influence_trace.max_proj_vector_sum=-1 \
    +trainer.influence_trace.max_hessian_dim=-1 \
    +trainer.influence_trace.max_tokens_per_response=-1 \
    +trainer.influence_trace.skip_optimizer_step=False \
    +trainer.influence_trace.grad_offload_to_cpu=False \
    +trainer.influence_trace.force_gpu_compute=True \
    +trainer.influence_trace.profile_timing=True \
    +trainer.influence_trace.exclude_self_response=True \
    +trainer.influence_trace.contrastive_agg=mean \
    +trainer.influence_trace.hessian_source=token \
    +trainer.influence_trace.debug_hessian_similarity=False \
    +trainer.influence_trace.score_normalization=none \
    +trainer.influence_trace.token_unit_norm=True \
    $@ 2>&1 | tee ${CKPTS_DIR}/${project_name}_${exp_name}_grpo.log
