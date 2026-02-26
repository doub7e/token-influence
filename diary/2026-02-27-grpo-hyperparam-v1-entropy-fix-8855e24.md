# GRPO Hyperparameter Tuning v1: Fix Entropy Collapse

| Field | Value |
|-------|-------|
| Date | 2026-02-27 |
| Commit | `8855e24` |
| RunAI Job | `archer-qwen3-math-v1` (4x H100) |
| W&B Experiment | `Archer2.0-Qwen3-0.6B-Math-v1` |
| Status | Submitted, awaiting results |

## Problem

Previous run `archer-qwen3-math-0226c` (71 steps, ~8h) exhibited:
- **Entropy collapse**: 2.9 → 0.07 in 71 steps
- **Degenerate outputs**: repetitive garbage (`& & & &`, `ゆ ゆ ゆ`)
- **Extremely slow**: ~10 min/step due to `max_response_length=30720`
  - 8% of responses hit max length, contributing ~80% of total tokens

Root causes: `entropy_coeff=0`, `kl_loss_coef=0.001` (too weak), 30K response length.

## Changes (v1)

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `max_response_length` | 30720 | **4096** | pos/mean ~1-2K; cuts 80% garbage tokens; ~5x faster/step |
| `v_max_response_length` | 30720 | **4096** | Consistent with training |
| `entropy_coeff` | 0 | **0.01** | Entropy bonus prevents mode collapse |
| `kl_loss_coef` | 0.001 | **0.01** | 10x stronger anchor to reference policy |
| `overlong_buffer_len` | 16 | **256** | Proportional buffer for shorter responses |
| `exp_name` | `Archer2.0-Qwen3-0.6B-Math` | `Archer2.0-Qwen3-0.6B-Math-v1` | Separate W&B run |

Derived values auto-updated:
- `actor_ppo_max_token_len` = 2048 + 4096 = 6144 (was 32768)
- `max_num_batched_tokens` = 6144
- `max_model_len` = 6144

## Monitoring Criteria

| Metric | Good | Bad (needs v2) |
|--------|------|----------------|
| `actor/entropy` | Stays > 1.0 at step 100 | Drops below 0.5 |
| `critic/score/mean` | Upward trend | Flat or decreasing |
| `batch/solve_none` | Decreasing over time | Stuck or increasing |
| `timing_s/step` | < 120s | > 300s |
| `actor/grad_norm` | < 5.0 | > 10 (instability) |

## Contingency Plans

- **v2**: If entropy still collapses, increase `entropy_coeff` to 0.05 or add `repetition_penalty=1.2`
- **v3**: If base model fundamentally can't learn, switch to `Qwen/Qwen3-0.6B` (instruct)

## Files Changed

- `scripts/train/run_archer2.0_qwen3_0.6b_math.sh` — 6 hyperparameter changes
- `scripts/train/run_full_qwen3_training.sh` — wrapper script, log path updated
- `verl/third_party/vllm/__init__.py` — Qwen3 vLLM 0.7.x model patch
- `verl/third_party/vllm/qwen3.py` — Qwen3ForCausalLM implementation for vLLM
