#!/usr/bin/env bash
# Run 4 influence trace configs sequentially for quality analysis.
set -xeuo pipefail

export WANDB_API_KEY="$(cat /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key)"
export VERL_PG_TIMEOUT_SECONDS=10800
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

cd /scratch/cvlab/home/shuli/agentic-research/Archer2.0

python scripts/train/download_hf_model.py \
  --repo-id Qwen/Qwen3-4B-Base \
  --local-dir models/Qwen3-4B-Base

BASE_DIR="./output/Archer2.0"
SRC_CKPT="$BASE_DIR/Archer2.0-Qwen3-4B-Base-Baseline-ep1-clip02-028/global_step_300"

# Common Hydra args (array)
COMMON_ARGS=(
    data.train_files=./data/train/archer2.0-math-1.5b-train.json
    data.val_files=./data/test/math500.json
    data.prompt_key=prompt
    data.filter_overlong_prompts=True
    "data.truncation=error"
    data.max_prompt_length=2048
    data.max_response_length=4096
    data.gen_batch_size=32
    data.train_batch_size=32
    actor_rollout_ref.rollout.n=16
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=3.0
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    +actor_rollout_ref.actor.high_entropy_kl_loss_scale_coef=0.0
    +actor_rollout_ref.actor.low_entropy_clip_ratio_low=0.2
    +actor_rollout_ref.actor.low_entropy_clip_ratio_high=0.2
    +actor_rollout_ref.actor.high_entropy_clip_ratio_low=0.4
    +actor_rollout_ref.actor.high_entropy_clip_ratio_high=0.4
    +actor_rollout_ref.actor.use_archer_policy_loss=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6144
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144
    actor_rollout_ref.model.path=./models/Qwen3-4B-Base
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.82
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=6144
    actor_rollout_ref.rollout.max_model_len=6144
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    "actor_rollout_ref.rollout.top_k=-1"
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=4
    +actor_rollout_ref.rollout.val_kwargs.response_length=4096
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
    reward_model.reward_manager=wizard
    reward_model.overlong_buffer.enable=True
    reward_model.overlong_buffer.len=256
    reward_model.overlong_buffer.penalty_factor=1.0
    "trainer.logger=[console,wandb]"
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=999
    trainer.total_epochs=1
    trainer.total_training_steps=302
    trainer.resume_mode=auto
    +trainer.enable_overlong_filter=False
    +trainer.rejection_sample=True
    +trainer.influence_trace.enable=True
    +trainer.influence_trace.write_every=1
    +trainer.influence_trace.atomic_write=True
    +trainer.influence_trace.fsync=False
    +trainer.influence_trace.max_prompts_per_step=0
    +trainer.influence_trace.reg_lambda=-1.0
    +trainer.influence_trace.hessian_mode=inverse
    +trainer.influence_trace.output_function=log_prob
    +trainer.influence_trace.max_modules=-1
    +trainer.influence_trace.max_proj_vector_sum=-1
    +trainer.influence_trace.max_hessian_dim=-1
    +trainer.influence_trace.max_tokens_per_response=-1
    +trainer.influence_trace.skip_optimizer_step=False
    +trainer.influence_trace.grad_offload_to_cpu=False
    +trainer.influence_trace.force_gpu_compute=True
    +trainer.influence_trace.profile_timing=True
    +trainer.influence_trace.exclude_self_response=True
    +trainer.influence_trace.contrastive_agg=mean
    +trainer.influence_trace.hessian_source=token
    +trainer.influence_trace.debug_hessian_similarity=False
    +trainer.influence_trace.score_normalization=none
    +trainer.influence_trace.token_unit_norm=True
)

run_one() {
    local NAME="$1"
    shift
    local DST="$BASE_DIR/$NAME"
    mkdir -p "$DST/influence_trace"
    if [ ! -d "$DST/global_step_300" ]; then
        cp -r "$SRC_CKPT" "$DST/global_step_300"
    fi
    echo "300" > "$DST/latest_checkpointed_iteration.txt"
    echo "====== START: $NAME ======"
    python -m dapo.main_dapo \
        "${COMMON_ARGS[@]}" \
        trainer.project_name=Archer2.0 \
        trainer.experiment_name="$NAME" \
        trainer.default_local_dir="$DST" \
        +trainer.validation_data_dir="$DST/eval" \
        +trainer.influence_trace.output_dir="$DST/influence_trace" \
        "$@" \
        2>&1 | tee "$DST/${NAME}_grpo.log"
    echo "====== DONE: $NAME ======"
}

# Config 6: SKIPPED (lm_head hook crashes with inplace div_ in logit computation)

# Config 7: last MLP block only (max_modules=1 takes the last match per filter), factor=32, all_selected
run_one "infl-diag-lastmlp-allsel" \
    ++trainer.influence_trace.projection_dim_factor=32 \
    ++trainer.influence_trace.max_hessian_dim=-1 \
    ++trainer.influence_trace.max_modules=1 \
    ++trainer.influence_trace.accepted_rejected_scope=all_selected \
    "++trainer.influence_trace.module_name_filter=[mlp.gate_proj,mlp.up_proj,mlp.down_proj]"

echo "===== ALL CONFIGS COMPLETE ====="
