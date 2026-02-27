# GRPO Hyperparameter Tuning v3: Dual Regularization

| Field | Value |
|-------|-------|
| Date | 2026-02-27 |
| Commit | `fb7d945` |
| RunAI Job | `archer-qwen3-math-v3` (4x H100) |
| W&B Experiment | `Archer2.0-Qwen3-0.6B-Math-v3` |
| Status | Submitted, monitoring |

## Key Insight: Entropy Bonus Has a Collapse Feedback Loop

The entropy bonus `H_coeff * H(π)` becomes weaker as entropy drops, since the bonus is proportional to entropy itself. Once the model starts collapsing, the bonus shrinks, accelerating the collapse. This explains why:
- V1 (entropy_coeff=0.01): entropy rose monotonically to 8+ (too high, but no collapse)
- V2 (entropy_coeff=0.005): entropy rose to 5.3 then suddenly collapsed to 0.18 at step 78

The KL loss doesn't have this problem — it measures deviation from reference, which INCREASES as the model deviates. So KL loss is a more robust anchor.

## V2 Results Summary (78 steps, ~2.4 hours)

| Metric | Step 6 | Step 25 | Step 50 | Step 78 | Trend |
|--------|--------|---------|---------|---------|-------|
| Entropy | 3.90 | 3.45 | 5.34 | **0.18** | Rose then COLLAPSED |
| Score/mean | -0.11 | -0.02 | 0.09 | 0.13 | Slowly improving |
| Logical_conn | 7.88 | 1.30 | 0.35 | **0.001** | Collapsed with entropy |
| Pos resp len | 1128 | 171 | 93.5 | **6.3** | Collapsed to guessing |
| Valid | 52 | 10 | 21 | 15 | Degraded |

V2's early steps showed great promise (structured math reasoning), but the entropy collapse at step 78 destroyed output quality.

## V3 Changes: Dual Regularization

| Parameter | V2 | V3 | Rationale |
|-----------|-----|-----|-----------|
| `entropy_coeff` | 0.005 | **0.01** | Restore v1 level — proven to prevent collapse |
| `kl_loss_coef` | 0.01 | **0.05** | 5x stronger KL anchor to prevent over-randomness |

Strategy: entropy bonus prevents collapse (floor), KL loss prevents over-randomness (ceiling). The two forces should balance to keep entropy in the 3-6 range.

With `entropy_coeff=0.01` and entropy ~5: bonus ≈ 0.05.
With `kl_loss_coef=0.05` and KL ~1.5: penalty ≈ 0.075.
These are comparable magnitudes — should create a stable equilibrium.

## V3 Monitoring Criteria

| Metric | Good | Bad |
|--------|------|-----|
| `actor/entropy` | Stays 3.0-6.0 at step 100 | Rises > 7 OR drops < 1 |
| `logical_connectives` | Stays > 0.05 | Drops below 0.02 |
| `response_length/pos/mean` | > 50 tokens | < 20 tokens |
| `actor/kl_loss` | < 3.0 | > 5.0 (too constrained) |

## Hyperparameter Search Summary

| Version | entropy_coeff | kl_loss_coef | Entropy @ 100 steps | Outcome |
|---------|---------------|--------------|---------------------|---------|
| V0 | 0 | 0.001 | 0.07 | Collapsed |
| V1 | 0.01 | 0.01 | 8.0 | Too random |
| V2 | 0.005 | 0.01 | 0.18 (collapsed at 78) | Collapsed |
| V3 | 0.01 | **0.05** | TBD | Running |
