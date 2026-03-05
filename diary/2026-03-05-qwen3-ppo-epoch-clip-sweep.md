# Qwen3-1.7B-Base GRPO: PPO Epoch & Clipping Sweep (Zero Regularization)

**Date:** 2026-03-05
**Commit:** `bd4d008`

## Summary

Following the entropy stabilization experiments (diary 2026-03-04), we tested whether **ppo_epochs** and **clipping** alone (without entropy/KL regularization) can prevent entropy collapse on Qwen3-1.7B-Base. We ran 5 experiments with entropy_coeff=0, kl_loss_coef=0, and varying ppo_epochs (1/2/3) and asymmetric clip ratios (DAPO-style, clip_high = 1.4x clip_low).

**Key finding:** All 5 experiments showed entropy collapse (to 0.09-0.34), but **the collapsed models still achieve strong MATH-500 scores** — up to **68.0% pass@1**, surpassing the previous best regularized config (61.7%).

## MATH-500 Evaluation Results (All Checkpoints)

| Experiment | epochs | clip_l/h | Step 100 | Step 200 | Step 300 | Step 400 | Peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ep2-clip01-014** | 2 | 0.1/0.14 | 63.7 / 76.8 | **68.0 / 80.2** | 66.6 / 77.4 | — | **68.0%** |
| ep1-noreg | 1 | 0.2/0.28 | 62.2 / 78.2 | 64.8 / 78.6 | 65.0 / 78.4 | — | 65.0% |
| ep3-clip01-014 | 3 | 0.1/0.14 | 64.7 / 78.6 | 63.4 / 76.8 | 60.0 / 73.8 | 59.1 / 73.0 | 64.7% |
| ep3-clip005-007 | 3 | 0.05/0.07 | 64.5 / 77.6 | 62.0 / 73.8 | 62.1 / 73.2 | — | 64.5% |
| ep2-clip028 | 2 | 0.2/0.28 | 63.4 / 78.4 | 63.5 / 75.6 | 61.4 / 75.0 | — | 63.5% |

*Values: pass@1 / pass@4*

### Comparison with Regularized Configs

| Config | Best pass@1 | Best pass@4 |
| --- | --- | --- |
| **ep2-clip01-014 (no reg, step 200)** | **68.0%** | **80.2%** |
| ep1-noreg (no reg, step 300) | 65.0% | 78.4% |
| ent01-kl05 (ent=0.01, kl=0.05, step 200) | 61.7% | 76.0% |
| ent01-kl05-clip028 (ent=0.01, kl=0.05, clip 0.28, step 100) | 60.1% | 78.6% |
| Qwen3-1.7B-Base (original) | 36.4% | 65.0% |

## Experiment Details

All experiments share:
- Model: Qwen3-1.7B-Base (no instruct FT)
- Algorithm: GRPO with token-mean loss, rejection sampling
- entropy_coeff=0, kl_loss_coef=0 (zero regularization)
- n_resp_per_prompt=16, response_length=4096, lr=1e-6, temperature=1.0
- Asymmetric clipping: clip_high = 1.4 * clip_low (DAPO style)
- save_freq=100, 4x H200 GPUs

### Entropy Trajectories

All experiments collapsed entropy within 50 steps:
- ep1-noreg: entropy ~0.25 by step 100, stable at 0.2-0.3
- ep2-clip028: entropy ~0.30 by step 100
- ep2-clip01-014: entropy ~0.15 by step 100
- ep3-clip01-014: entropy ~0.14 by step 100, then **U-shaped recovery** to 1.5-2.0 after step 250
- ep3-clip005-007: entropy ~0.09 by step 100 (tightest clip = lowest entropy)

### U-Shaped Recovery (ep3-clip01-014)

ep3-clip01-014 showed a unique spontaneous entropy recovery without any regularization:
- Steps 1-200: collapsed to 0.1-0.4
- Steps 200-290: gradual recovery 0.5-1.0
- Steps 290-400: stable at 1.5-2.0

However, **performance degraded during and after recovery** (64.7% → 59.1%), suggesting the recovery represents catastrophic forgetting rather than beneficial exploration.

## Key Insights

1. **Entropy collapse ≠ bad performance.** Models with entropy 0.1-0.3 achieve 63-68% pass@1, better than entropy-stable models (~2) that get 61.7%. Low entropy means the model is confident and consistent.

2. **Tight clip + 2 epochs is optimal.** ep2-clip01-014 (clip 0.1/0.14, 2 epochs) peaks at 68.0% — the tightest per-step constraint with enough update capacity.

3. **Over-training is the real problem, not entropy collapse.** All experiments show performance peaks then declines. The right strategy is early stopping, not entropy regularization.

4. **pass@1 and pass@4 trends are aligned.** No divergence observed — when pass@1 drops, pass@4 drops too. The model loses capability, not just consistency.

5. **ppo_epochs=1 trains slowest but most stably** — still improving at step 300 (65.0%), while epoch=2,3 configs already peaked.

## Response Quality Analysis

| Metric | ep2-clip01-014 s200 (68%) | ep1-noreg s300 (65%) | ep3-clip01-014 s400 (59%) |
| --- | --- | --- | --- |
| Median length | 1311 chars | 1551 chars | 1312 chars |
| Has \boxed | 497/500 | 499/500 | 489/500 |
| 20-gram repetition | 33/500 | 46/500 | 38/500 |
| >5000 chars | 1/500 | 4/500 | 13/500 |

The s400 checkpoint shows degraded outputs: repetitive loops (147K chars), semantic gibberish, and arithmetic errors.

## Next Steps

- Adopt **ep2-clip01-014** as the default Qwen3 training config going forward
- Investigate optimal early stopping criteria
- Test this config on larger models or harder datasets
