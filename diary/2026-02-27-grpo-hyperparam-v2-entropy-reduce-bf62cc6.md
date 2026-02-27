# GRPO Hyperparameter Tuning v2: Reduce Entropy Coefficient

| Field | Value |
|-------|-------|
| Date | 2026-02-27 |
| Commit | `bf62cc6` |
| RunAI Job | `archer-qwen3-math-v2` (4x H100) |
| W&B Experiment | `Archer2.0-Qwen3-0.6B-Math-v2` |
| Status | Submitted, monitoring |

## V1 Results Summary (150 steps, ~4.3 hours)

| Metric | Step 20 | Step 61 | Step 102 | Step 148 | Trend |
|--------|---------|---------|----------|----------|-------|
| Entropy | 5.93 | 7.35 | 7.17 | 8.01 | Rising too high |
| Score/mean | 0.047 | 0.217 | 0.233 | 0.433 | Improving |
| Valid | 4 | 23 | 16 | 14 | Stagnated |
| Solve_none | 60 | 41 | 46 | 50 | Stagnated |
| Solve_all | 0 | 0 | 2 | 0 | Appeared briefly |
| Logical_conn | 0.154 | 0.040 | 0.022 | 0.015 | Collapsed |
| Pos resp len | 192 | 14 | 16 | 22 | Model guessing |

### V1 Diagnosis

- `entropy_coeff=0.01` was too strong: entropy rose from 5.9→8.4 (target ~3-5)
- Model learned shortcut: output `คำตอบ: <number>` (6-20 token answer guesses) instead of reasoning
- Negative responses became random Unicode gibberish (maximizing entropy bonus)
- Score improved (0.047→0.43) but through pattern-matching, not reasoning
- logical_connectives collapsed from 0.154 to 0.015 (no structured output)

## V2 Change

| Parameter | V1 | V2 | Rationale |
|-----------|-----|-----|-----------|
| `entropy_coeff` | 0.01 | **0.005** | Half the bonus; still prevents collapse but allows more focused policy |
| `exp_name` | `...-v1` | `...-v2` | Clean W&B experiment |

All other parameters unchanged from v1 (max_response_length=4096, kl_loss_coef=0.01, etc.)

## V2 Monitoring Criteria

| Metric | Good | Bad (needs v3) |
|--------|------|----------------|
| `actor/entropy` | Stays 3.0-6.0 at step 100 | Rises above 7 or drops below 1 |
| `logical_connectives` | Stays > 0.05 | Drops below 0.02 |
| `response_length/pos/mean` | > 50 tokens (actual reasoning) | < 20 tokens (just guessing) |
| `critic/score/mean` | > 0.2 at step 100 | < 0.1 |

## Files Changed

- `scripts/train/run_archer2.0_qwen3_0.6b_math.sh` — entropy_coeff 0.01→0.005, exp_name v2
- `scripts/train/run_full_qwen3_training.sh` — log path updated to v2
