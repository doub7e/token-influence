## 2026-03-09 - Additive Influence Token Weight Mode

| Item | Details |
| --- | --- |
| Request | Implement a new "additive" mode for influence token weighting that conserves total advantage budget |
| Delivery | New mode `w_t = 1 + λ × z_t` with z-score normalization; 3 experiment scripts (λ=0.3, 0.5, 1.0); all submitted |
| Scope | Archer2.0, Qwen3-1.7B-Base, 4×H100 per job |

### Motivation

Instead of multiplying two advantage signals (influence × A_resp, as in ratio mode), the additive mode uses influence as an additive correction:

```
z_t = (score_t - mean_score) / std_score
w_t = 1 + λ × z_t
A_t = A_resp × w_t
```

Since `mean(z_t) = 0`, `mean(w_t) = 1` before clamping — the total advantage budget is conserved. λ controls redistribution strength: λ=0 recovers uniform GRPO.

### Changes

| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_token_weight.py` | Added `additive` mode + config params (`additive_lambda`, `additive_clamp_min/max`) | New weighting mode |
| `verl/workers/actor/dp_actor.py` | Added `"additive"` to advantage-target mode list | Apply weights to advantage |
| `scripts/train/run_qwen3_1.7b_base_infl_additive_l03.sh` | New script (λ=0.3) | Experiment |
| `scripts/train/run_qwen3_1.7b_base_infl_additive_l05.sh` | New script (λ=0.5) | Experiment |
| `scripts/train/run_qwen3_1.7b_base_infl_additive_l10.sh` | New script (λ=1.0) | Experiment |

### Experiments

| Job | λ | Experiment Name | GPUs | Status |
| --- | --- | --- | --- | --- |
| `additive-l03-v1` | 0.3 | `Archer2.0-Qwen3-1.7B-Base-Additive-L03-v1` | 4×H100 | Running |
| `additive-l05-v1` | 0.5 | `Archer2.0-Qwen3-1.7B-Base-Additive-L05-v1` | 4×H100 | Running |
| `additive-l10-v1` | 1.0 | `Archer2.0-Qwen3-1.7B-Base-Additive-L10-v1` | 4×H100 | Running |

Common settings: all_selected scope, mean contrastive_agg, exclude_self_response=True, score_normalization=none, apply_epochs=[0,1], adv_target=advantage, clamp [-1, 3].

### Reference Results (AllSel MeanRatio)

| Experiment | Step | pass@1 | pass@4 |
| --- | --- | --- | --- |
| Baseline (ep2-clip01-014) | 100 | 68.0% | — |
| AllSel MeanRatio | 100 | 65.6% | 78.6% |
| AllSel MeanRatio | 200 | 62.65% | 74.6% |
| GlobalSel MeanRatio | 100 | 63.4% | 79.0% |

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| Syntax check | `py_compile` on both modified files | Pass |
| Job submission | `runai list` shows all 3 Running | Pass |

### Git

| Field | Value |
| --- | --- |
| Commit | `81d699c` |
| Branch | `main` |
| Remote | `doub7e` (git@github.com:doub7e/token-influence.git) |
| Push | No |

### Notes
- Monitor W&B for early signs: if λ=1.0 causes instability (loss spikes, reward collapse), stop early.
- Eval at step 100 on MATH-500 for all three.
- Key comparison: does additive mode outperform ratio mode (AllSel 65.6%) and/or baseline (68.0%)?
- If all λ values underperform, consider that the influence signal itself may need improvement before the weighting formulation matters.
