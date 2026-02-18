# Entropy Trace Workflow

This workflow captures token-level entropy for every rollout step and provides an interactive visualizer for manual inspection.

## Added Components

- Training entry script  
  - `scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh`
- Runtime trace writer  
  - `dapo/entropy_trace.py`
- Trainer integration point  
  - `dapo/dapo_ray_trainer.py` (`old_log_prob` stage in each rollout step)
- Visualizer  
  - `scripts/train/visualize_rollout_entropy.py`

## Training Behavior

`run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh` applies:

- `actor_rollout_ref.actor.ppo_epochs=1`
- `trainer.total_training_steps=100`
- `+trainer.entropy_trace.enable=True`
- `+trainer.entropy_trace.output_dir=.../entropy_trace`

## Run Training

```bash
cd Archer2.0
bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh
```

You can still append extra Hydra overrides at the end.

## Trace Output Format

Default output root:

- `output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace/entropy_trace`

Generated files:

- `summary.json`: run-level metadata
- `manifest.jsonl`: one line per rollout step
- `steps/step_XXXXXX.npz`: compressed arrays for each step

Per-step `.npz` fields include:

- `entropies`: float16, shape `[num_responses, response_len]`
- `response_mask`: bool, same shape
- `responses`: int32 token IDs, same shape
- `response_index`: int32, shape `[num_responses]`
- optional `prompt_length`, `uid`, `sample_index` when available

## Launch Visualizer

```bash
cd Archer2.0
python scripts/train/visualize_rollout_entropy.py \
  --trace-dir output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace/entropy_trace \
  --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B \
  --host 0.0.0.0 \
  --port 7862
```

## Visualizer Features

- Step-level manifest inspection
- Response-level entropy overview
- Token-level entropy heatmap (color-coded)
- Entropy range filtering and token substring search
- Sort by position / entropy ascending / entropy descending
- Top-K highest and lowest entropy tokens per step
- CSV export for the selected response view
