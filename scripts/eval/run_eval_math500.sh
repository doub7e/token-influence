#!/bin/bash
# Evaluate a checkpoint on MATH-500 with n_samples=4 (avg@4, pass@4)
# Usage: bash scripts/eval/run_eval_math500.sh <model_path> [n_samples] [n_gpus]
#
# Examples:
#   bash scripts/eval/run_eval_math500.sh /path/to/hf_model
#   bash scripts/eval/run_eval_math500.sh /path/to/hf_model 4 4

set -eo pipefail
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

model_path=${1:?Usage: run_eval_math500.sh <model_path> [n_samples] [n_gpus]}
n_samples=${2:-4}
n_gpus=${3:-4}

nnodes=1
tp_size=1

dataset=math500
data_dir=data/test
output_dir=${model_path}/output

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))

python -m verl.trainer.main_generation \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus} \
    +trainer.use_wandb=False \
    +trainer.project_name=ArcherEval \
    +trainer.experiment_name=math500_eval \
    +trainer.task_name=${dataset} \
    +trainer.global_step=0 \
    model.path=${model_path} \
    data.path=${data_dir}/${dataset}.json \
    data.output_path=${output_dir}/${dataset}.parquet \
    data.batch_size=512 \
    data.n_samples=${n_samples} \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.tensor_model_parallel_size=${tp_size} \
    rollout.temperature=0.8 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.prompt_length=$max_prompt_length \
    rollout.response_length=$max_response_length \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
