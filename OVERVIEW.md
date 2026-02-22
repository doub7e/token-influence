# Archer2.0 Overview

## Scope
Archer2.0 is a distributed RL post-training stack for LLMs built on Ray + verl.
The current active workflow is PPO-based math training with optional entropy and token-influence tracing.

Repository root for this project is `Archer2.0/`.

## Top-Level Layout
- `scripts/train/`: run wrappers, dataset prep, trace visualizer, and ablation utilities.
- `dapo/`: Archer trainer entry and orchestration (`main_dapo.py`, `dapo_ray_trainer.py`), trace writers.
- `verl/`: actor/rollout/ref/reward workers and core distributed RL logic.
- `rewards/`: reward parsing and reward manager adapters.
- `data/`: train/val JSON datasets.
- `models/`: local model checkpoints/snapshots.
- `output/`: experiment artifacts (checkpoints, trace files, logs).
- `wandb/`: local W&B run cache.
- `diary/`: per-commit engineering diary and experiment progress.

## Main Entrypoints
### Baseline / training
- `scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh`
- `scripts/train/run_archer2.0_qwen2.5_1.5b_code.sh`

### Trace-enabled runs
- `scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh`
- `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh`

### Influence ablation helpers
- `scripts/train/run_single_influence_case_0220.sh`
- `scripts/train/run_single_baseline_case_0220.sh`
- `scripts/train/run_influence_ablation_0220.sh`
- `scripts/train/run_influence_ablation_0220_extra_nohessian.sh`
- `scripts/train/collect_influence_ablation_results_0220.py`

### Visualization
- `scripts/train/visualize_rollout_entropy.py`
  - Supports side-by-side entropy and influence visualization from saved trace files.

## Execution Chain
1. A run script sets Hydra overrides and launches `python -m dapo.main_dapo`.
2. `dapo/main_dapo.py` loads config and starts the Ray trainer.
3. `dapo/dapo_ray_trainer.py` orchestrates rollout, reward, advantage, PPO update, and trace selection/write.
4. Worker execution is handled in `verl/workers/*`.
5. Trace artifacts are written by:
   - `dapo/entropy_trace.py`
   - `dapo/influence_trace.py`

## Influence Trace Pipeline (Current)
### Runtime capture
- `verl/workers/actor/dp_actor.py`
- `verl/workers/actor/influence_trace.py`

### Trainer-side selection/writing
- `dapo/dapo_ray_trainer.py`
- `dapo/influence_trace.py`

### Core behavior
- Capture projected per-token gradient factors from real training backprop.
- Support influence modes:
  - `hessian_mode=inverse`
  - `hessian_mode=identity`
- Support output objectives:
  - `output_function=training_loss`
  - `output_function=log_prob_advantage`
  - `output_function=log_prob_reward`
- Support accepted/rejected aggregation scopes:
  - `accepted_rejected_scope=per_prompt`
  - `accepted_rejected_scope=all_selected`
- Support contrastive aggregation modes:
  - `contrastive_agg=sum`
  - `contrastive_agg=mean`
- Support Hessian source modes:
  - `hessian_source=response`
  - `hessian_source=token`

## Important Influence Config Knobs
Configured through `trainer.influence_trace.*` (typically via env vars in run scripts):
- `enable`
- `hessian_mode`
- `output_function`
- `accepted_rejected_scope`
- `contrastive_agg`
- `hessian_source`
- `module_name_filter`
- `max_modules`
- `projection_dim_factor`
- `max_proj_vector_sum`
- `max_hessian_dim`
- `max_tokens_per_response`
- `grad_offload_to_cpu`
- `force_gpu_compute`
- `profile_timing`
- `max_prompts_per_step` (`0` means no cap)

## Trace Artifact Paths
For experiment `<exp>` under `output/Archer2.0/<exp>/`:
- Entropy trace: `entropy_trace/`
- Influence trace: `influence_trace/`
  - `summary.json`
  - `manifest.jsonl`
  - `steps/step_*.npz`

## Debugging Pointers
- If influence rows are unexpectedly empty, inspect metrics emitted by actor update:
  - `influence_trace/rows_emitted`
  - `influence_trace/debug_*`
  - `timing_s/influence_*`
- If distributed jobs timeout during long post-processing, check process-group timeout handling in:
  - `verl/workers/fsdp_workers.py`
  - env: `VERL_PG_TIMEOUT_SECONDS`

## Recommended Read Order For New Contributors
1. `OVERVIEW.md`
2. `diary/index.md` (newest first)
3. `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh`
4. `dapo/main_dapo.py`
5. `dapo/dapo_ray_trainer.py`
6. `verl/workers/actor/dp_actor.py`
7. `verl/workers/actor/influence_trace.py`
8. `dapo/influence_trace.py`
