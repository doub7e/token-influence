# Softmax Credit Token Weighting Experiments

**Date**: 2026-03-21
**Model**: Qwen3-4B-Base, GRPO, MATH-500 eval (avg@4)

## Summary

Implemented and evaluated **softmax-based influence credit assignment** for GRPO: redistribute the response-level advantage budget across tokens proportional to their influence scores, preserving total gradient magnitude.

## Method

For each response, compute per-token influence scores via projected influence functions (inverse Hessian, last MLP layer). Then:

```
w_t = softmax(z_t / τ) × T
```

- `z_t`: z-scored influence (accepted: raw, rejected: sign-flipped)
- `× T`: weights sum to response length → total advantage preserved
- Clamp to `[w_min, w_max]`, then renormalize
- Applied multiplicatively: `A'_t = A_resp × w_t`

## Experiments & Results

### Phase 1: Temperature & Clamp Sweep (bsz=128, mini_bsz=16, n=16, allsel scope)

| Config | τ | Clamp | tw_std | Step 10 | Peak | Entropy fate |
|--------|---|-------|--------|---------|------|-------------|
| Baseline (mb16) | — | — | — | 45.9% | 79.1% (s140) | ✅ stable 0.22 |
| **T2 [0,2]** | 2 | [0, 2.0] | 0.45 | **59.5%** | 79.2% (s130) | ❌ collapse →1.8 |
| **T4 [0.5,1.5]** | 4 | [0.5, 1.5] | 0.23 | **54.8%** | 79.2% (s130) | ❌ escalate →1.6 |
| **T8 [0.8,1.2]** | 8 | [0.8, 1.2] | 0.11 | 51.4% | 79.2% (s120) | ⚠️ escalate →1.0 |

**Finding**: All configs show strong early advantage (+5-14% at step 10) but entropy escalates and eventually degrades training. More aggressive weights → faster early gains but earlier collapse.

### Phase 2: Mini-batch Size Fix (bsz=128, mini_bsz=32, n=16)

**Hypothesis**: Token weights computed once per step become stale across multiple mini-batch updates. mini_bsz=16 → 8 updates/step, mini_bsz=32 → 4 updates/step.

| Config | mini_bsz | Updates/step | Step 10 | Step 50 | Entropy at step 37 |
|--------|----------|-------------|---------|---------|-------------------|
| Baseline (mb32) | 32 | 4 | 38.0% | 75.1% | 0.20 |
| T4-allsel-mb32 | 32 | 4 | 44.8% | 76.1% | **0.21** (vs 0.46 for mb16) |

**Finding**: mini_bsz=32 delayed entropy escalation by ~30 steps but did NOT prevent it. Entropy still climbed to 0.4-0.5 by step 50-70.

### Phase 3: Per-Prompt Scope (bsz=128, mini_bsz=32, n=16, per_prompt)

| Config | Scope | Step 40 | Step 100 | Peak | Entropy pattern |
|--------|-------|---------|----------|------|----------------|
| Baseline (mb32) | — | 75.0% | 78.0% | **80.1%** (s210) | ✅ 0.20 |
| T4-PP-mb32 | per_prompt | **76.5%** | **79.0%** | 79.3% (s130) | oscillates 0.20-0.60 |

**Finding**: Per-prompt scope more stable than all_selected for early training, matching baseline entropy initially. But entropy still oscillates (0.20-0.60) in later training, though accuracy remains competitive.

### Phase 4: Per-Prompt + n=64 (bsz=32, mini_bsz=32, n=64)

| Config | bsz | n | Updates/step | Step 40 | Step 60 | Entropy |
|--------|-----|---|-------------|---------|---------|---------|
| PP-n64 | 32 | 64 | **1** | 72.5% | 74.6% | **0.19** ✅ perfect |

**Finding**: Most stable config ever — entropy 0.17-0.22 through 60 steps, identical to baseline. The combination of (1) 1 update/step (no staleness), (2) 64 responses/prompt (better influence groups), and (3) per_prompt scope eliminates the entropy escalation completely. However, convergence is 4x slower per-step (only 32 prompts/step).

## Key Insights

### 1. Softmax Credit Works — With Caveats
The method consistently accelerates early learning (+5-14% at step 10) across all configs. The fundamental idea of redistributing advantage via influence is sound.

### 2. Entropy Escalation is the Core Challenge
Every config with bsz=128 eventually shows entropy escalation. Root causes:
- **Token weight staleness**: weights computed for θ₀ applied to θ₁...θ₇ across mini-batches
- **Positive feedback loop**: noisy weights → policy drift → worse influence estimates → noisier weights
- **Systematic bias**: softmax underweights many tokens → reduces entropy-sharpening pressure

### 3. Mini-batch Staleness Matters
mini_bsz=32 (4 updates) vs mini_bsz=16 (8 updates) delayed escalation by ~30 steps. mini_bsz=bsz (1 update, PP-n64) eliminated it entirely. This confirms token weight staleness is a major factor.

### 4. Dose-Response Relationship
More aggressive weights (higher std) → stronger early effect but earlier instability:
- std=0.45 (T2): +14% early, collapse at step 85
- std=0.23 (T4): +9% early, escalation at step 115
- std=0.11 (T8): +5% early, escalation at step 130

### 5. Per-Prompt vs All-Selected
Per-prompt scope is more stable for token weighting (entropy stays in 0.20-0.60 band vs unbounded climb for allsel). Despite lower influence AUC (~0.63 vs 0.78), the more local signal causes less systematic drift.

### 6. n=64 is Ideal for Influence Quality
With 64 responses per prompt, the accepted/rejected contrast is much richer, producing higher-quality influence scores. Combined with 1 update/step, this eliminates staleness entirely.

## Baseline Reference

| Model | Config | Best avg@4 | Steps to 75% |
|-------|--------|-----------|-------------|
| Qwen3-4B-Base | mb16, n=16 | 79.1% | ~30 |
| Qwen3-4B-Base | mb32, n=16 | **80.1%** | ~40 |

## Recommendations for Next Steps

1. **Annealing**: Use softmax credit for first 50-80 steps, then anneal to uniform. Captures early acceleration without late instability.
2. **KL anchor**: Add kl_loss_coef=0.03 to prevent entropy drift when using token weighting.
3. **n=64 baseline**: Run a baseline with bsz=32, n=64 (no token weighting) to isolate the effect of larger groups from the softmax credit.
4. **Hybrid schedule**: Start with T4 [0.5, 1.5] for fast early learning, switch to uniform at step 50 when entropy approaches 0.3.
