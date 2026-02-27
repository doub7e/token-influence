# GRPO Hyperparameter Tuning v5: Revert to V3 Config for Long Run

| Field | Value |
|-------|-------|
| Date | 2026-02-27 |
| Commit | `35e19f7` |
| RunAI Job | `archer-qwen3-math-v5` (4x H100) |
| W&B Experiment | `Archer2.0-Qwen3-0.6B-Math-v5` |
| Status | Submitted, monitoring |

## V4 Post-Mortem (17 steps, ~40 min)

V4 (kl_loss_coef=0.03) collapsed catastrophically by step 14:

| Metric | Step 1 | Step 9 | Step 14 | Step 17 |
|--------|--------|--------|---------|---------|
| Entropy | 3.31 | 5.16 | **7.63** | **7.23** |
| Score/mean | -0.163 | -0.022 | -0.062 | 0.062 |
| Valid | 62 | 22 | **1** | **1** |
| Solve_none | 2 | 42 | **63** | **63** |
| KL_loss | 0.001 | 0.054 | 0.039 | 0.224 |
| Resp_len/mean | 1539 | 482 | 380 | **15.6** |

Entropy oscillated wildly between 2.08 and 7.63. KL_loss spiked to 4.978 at step 16. Model lost virtually all ability to solve problems (valid=1/64). kl_loss_coef=0.03 is in a "dangerous middle ground" — too weak to anchor but enough to create instability.

## Complete Hyperparameter Search Summary (V0-V5)

| Version | entropy_coeff | kl_loss_coef | Score@peak | Entropy@~100 | Valid@peak | Outcome |
|---------|---------------|--------------|------------|--------------|------------|---------|
| V0 | 0 | 0.001 | — | 0.07 | — | Entropy collapsed |
| V1 | 0.01 | 0.01 | **0.43** | 8.4 | 15 | High score via shortcut guessing |
| V2 | 0.005 | 0.01 | 0.13 | 0.18 | 52→15 | Entropy collapsed at step 78 |
| V3 | 0.01 | 0.05 | 0.072 | 8.5 | **27** | Best quality, low score |
| V4 | 0.01 | 0.03 | — | 2-7 (oscillating) | 1 | Collapsed at step 14 |
| V5 | 0.01 | **0.05** | TBD | TBD | TBD | Long run of V3 config |

## V5 Strategy

V3 was the only configuration that produced genuine reasoning chains with good output quality (valid=27, logical_conn=0.28-0.44). Its low score (0.072 at step 114) may simply be due to insufficient training steps. The stronger KL anchor (0.05) makes learning slower but more stable.

V5 = exact V3 config, run for 300+ steps. Hypothesis: score will eventually rise above 0.15 given more training time.

## V5 Monitoring Criteria

| Metric | Target | Abandon if |
|--------|--------|------------|
| Score/mean @ step 200 | > 0.10 | < 0.05 (plateau) |
| Score/mean @ step 300 | > 0.15 | < 0.08 |
| Entropy | 5-9 | < 2 (collapse) |
| Valid | > 20 | < 10 |
| Logical_conn (pos) | > 5 | < 1 |

## Changes

| Path | Change | Why |
|------|--------|-----|
| `scripts/train/run_archer2.0_qwen3_0.6b_math.sh` | exp_name→v5, kl_loss_coef→0.05 | Revert to V3 config |
| `scripts/train/run_full_qwen3_training.sh` | Log→v5 | Track separately |

## Git

| Field | Value |
|-------|-------|
| Commit | `35e19f7` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | No |
