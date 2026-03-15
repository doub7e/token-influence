# Credit & Mask Mode Token Influence Experiments

**Date:** 2026-03-16
**Commit:** `d13eebf`
**Scope:** Token-level credit assignment in GRPO — additive credit, binary mask, and ablations

---

## Summary

Extensive experiments exploring how to use per-token influence scores for credit assignment in GRPO training. Tested multiplicative weighting (sign flip), additive credit, and binary mask approaches. Key finding: **influence-based masking provides early-stage acceleration but degrades later due to entropy stagnation; random masking provides competitive regularization benefits.**

## Experiment Results

### Multiplicative Weighting (v2-v4): Sign Flip Instability

| Experiment | Mode | Sign Flip | λ | Clamp | Result |
|---|---|---|---|---|---|
| AllSel-v2 | additive (mult) | Yes | 0.5* | [-1,3]* | Entropy explosion |
| AllSel-tightclamp-v2 | additive (mult) | Yes | 0.01 | [0.75,1.25] | Entropy explosion |
| AllSel-v4 | additive (mult) | Yes | 0.5* | [0,2]→[-1,3]* | Entropy explosion |
| NoFlipNeg-v4 | additive (mult) | No | 0.5* | [-1,3]* | Stable, ~64% |
| PosOnly-v4 | additive (mult) | Pos only | 0.5* | [-1,3]* | Entropy explosion |

\* Config whitelist bug: `additive_lambda`, `additive_clamp_min/max` were not passed to workers — all ran with defaults (λ=0.5, clamp=[-1,3]). **Fixed in this commit.**

**Conclusion:** Multiplicative sign flip is fundamentally unstable regardless of clamp range. The issue is that `A_t × w_t` with sign flip can reverse the gradient direction, creating positive feedback with entropy.

### Additive Credit Mode (v5)

| Experiment | λ | Clamp | Weight std | Peak Acc | Notes |
|---|---|---|---|---|---|
| Credit-v5 (pre-fix) | 0.5* | [-1,3]* | ~0.5 | 64.9% | Config bug, ran with defaults |
| Credit-v5b | 0.03 | [-0.5,0.5] | 0.028 | 64.9% | Too small correction (~3% of A) |
| Credit-narrow-v5b | 0.01 | [-0.1,0.1] | 0.009 | 63.3% | Negligible correction (~1% of A) |
| Credit-l01-v5c | 0.1 | [-0.5,0.5] | 0.09 | **64.9%** | ~10% correction, stable |
| Credit-l03-v5c | 0.3 | [-1.0,1.0] | 0.27 | 64.4% | ~30% correction, stable |
| Credit-wide-v5 | 0.3 | [-1.0,1.0] | 0.27 | Poor | Wide clamp degraded |

**Conclusion:** Additive credit (`A'_t = A_t + λz_t`) with sign flip is stable (no entropy explosion), but performance matches baseline (~64%). The z-scored correction is zero-mean per response, so it only redistributes learning within a response without changing the total gradient direction. Response-level signal dominates.

### Mask Mode (v6) — Key Findings

| Experiment | Mode | Scope | Peak Acc (early) | Final Acc | Entropy |
|---|---|---|---|---|---|
| mask-v6 | mask (influence) | AllSel | **66.8%** (step 90) | 62.5% (step 270) | Stuck ~0.5 |
| maskrandom-v6 | mask_random | AllSel | 64.3% (step 100) | **66.0%** (step 160) | Drops to ~0.1 |
| mask-pp-v6 | mask (influence) | PerPrompt | - | Poor | Stuck high |
| masksoft-v6b | mask_soft (0.5) | AllSel | 66.8% (step 120) | 59.7% (step 390) | - |
| maskthresh-v6b | mask_threshold (z>1) | AllSel | 66.6% (step 240) | 61.3% (step 320) | - |

**Early-stage comparison (steps 10-100, mask vs maskrandom):**

| Step | mask-v6 | maskrandom-v6 | Δ |
|---|---|---|---|
| 40 | 0.630 | 0.601 | +2.9% |
| 60 | 0.644 | 0.616 | +2.8% |
| 90 | **0.668** | 0.633 | **+3.5%** |
| 100 | 0.648 | 0.643 | +0.5% |

### Key Insights

1. **Influence sign provides real early-stage signal** — mask-v6 leads maskrandom by ~3% for the first 90 steps, indicating the influence score correctly identifies which tokens to mask in early training.

2. **Systematic masking causes entropy stagnation** — influence-based mask always excludes the same token types (score<0 in accepted, score>0 in rejected). These tokens never receive learning signal, preventing the distribution from sharpening. Entropy stays at ~0.5 vs baseline <0.1.

3. **Random mask provides dropout-like regularization** — maskrandom-v6 achieves comparable or better final performance with normal entropy dynamics, suggesting the primary benefit of masking is regularization, not credit assignment.

4. **Soft and threshold variants don't fix the degradation** — masksoft (weight=0.5) and maskthresh (|z|>1.0) resumed from step 100 both show declining accuracy, indicating the fundamental issue is systematic bias, not mask strength.

5. **Config whitelist bug** — `additive_lambda`, `additive_clamp_min/max` were missing from `dapo_ray_trainer.py` whitelist. All early additive/credit experiments ran with default values. Fixed in this commit.

## Normalization Analysis

- **Direction side (d):** Changed from full TrackStar (both sides normalized) to Option B (raw gradients for d, only scoring side normalized). Preserves natural gradient magnitude weighting in contrastive direction.
- **Scoring side normalization:** Mahalanobis norm `||H^{-1/2} g_t||` confirmed as most principled choice — avoids double-counting gradient magnitude with the training gradient, and prevents double-preconditioning with AdamW.

## Files Changed

- `verl/workers/actor/influence_token_weight.py` — Added credit, mask, mask_soft, mask_threshold, mask_random modes
- `verl/workers/actor/dp_actor.py` — Additive application for credit mode, multiplicative for mask modes
- `verl/workers/actor/influence_trace.py` — Direction-side normalization change (raw g_tok for g_resp)
- `dapo/dapo_ray_trainer.py` — Fixed config whitelist (additive_lambda, clamp_min, clamp_max)
- Training scripts for all experiment variants
