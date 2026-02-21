# Archer2.0 Project Overview

## What
Distributed RL post-training framework for LLMs, built on Ray + verl. Supports PPO with entropy/influence trace capture for token-level analysis.

## Directory Layout
- `scripts/train/`: experiment entry scripts, dataset prep scripts, and trace visualizer.
- `dapo/`: Archer-specific trainer orchestration (`main_dapo.py`, `dapo_ray_trainer.py`) and trace writers.
- `verl/`: distributed RL framework components (trainer loop, actor/critic/rollout workers).
- `rewards/` and `verl/workers/reward_manager/`: reward parsing and reward-model interfaces.
- `data/`: train/validation JSON data consumed by training scripts.
- `output/` and `wandb/`: checkpoints, logs, and experiment telemetry.
- `models/`: local model snapshots used as training initialization.
- `diary/`: per-commit diary entries and experiment progress.

## Key Training Entrypoints
- `scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh`: baseline math training wrapper.
- `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh`: entropy + influence trace run.
- `scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh`: entropy trace run.

## Key Execution Chain
1. Run script sets overrides and calls `python -m dapo.main_dapo`.
2. `dapo/main_dapo.py` loads config and launches Ray-based trainer.
3. `dapo/dapo_ray_trainer.py` orchestrates rollout, reward, advantage, PPO updates, and trace writing.
4. `verl/workers/*` performs distributed actor/ref/rollout/reward computation.
5. Trace artifacts are persisted by `dapo/entropy_trace.py` and `dapo/influence_trace.py`.
6. Visualization is served by `scripts/train/visualize_rollout_entropy.py`.

## Influence-Trace Path (debugging reference)
- Runtime capture: `verl/workers/actor/influence_trace.py` + `verl/workers/actor/dp_actor.py`
- Trainer-side selection and write: `dapo/dapo_ray_trainer.py` + `dapo/influence_trace.py`
- Visual inspection: `scripts/train/visualize_rollout_entropy.py` with `--trace-dir .../influence_trace`

## Recommended Read Order
1. This file (`OVERVIEW.md`)
2. `diary/index.md` (latest entries first)
3. `scripts/train/*target_run*.sh` → `dapo/main_dapo.py` → `dapo/dapo_ray_trainer.py`
4. Relevant worker module in `verl/workers/` for the subsystem being modified.
