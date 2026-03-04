## 2026-03-04 - Influence Token Weight: Tanh Mode + Softmax Eval + Deep Analysis

| Item | Details |
| --- | --- |
| Request | 1) Stop softmax training & eval on MATH-500; 2) Improve influence→weight conversion (tanh mode); 3) Deep analysis of influence trace |
| Delivery | Softmax eval complete, tanh mode implemented & training at step 111, deep analysis done, progress.md reorganized |
| Scope | DeepSeek-R1-Distill-Qwen-1.5B, 4xH200, GRPO, MATH-500 eval |

### MATH-500 Results (all R1-1.5B, step 100)

| Model | avg@4 | pass@4 |
| --- | --- | --- |
| Baseline GRPO | 81.15% | 90.4% |
| Archer GRPO | 79.75% | 91.2% |
| InflWeight-PerPrompt (zero) | 78.05% | 90.8% |
| InflWeight-AllSel (zero) | 80.35% | 90.0% |
| **InflWeight-Softmax T=10** | **80.50%** | **91.6%** |
| InflWeight-Tanh a=0.5 | running | - |

Best results at later steps: Baseline 83.60%@190, Archer 83.85%@300.

### Deep Influence Analysis Key Findings

| Finding | Value |
| --- | --- |
| Length confound correlation | r = -0.449 (accepted), r = -0.486 (rejected) |
| Span autocorrelation | 0.098 (token-level noise dominates) |
| EOS/termination bias | Cohen's d ~ 0.001 (none) |
| Position bias (z-scored) | None |
| Influence noise growth | +29% std over training (0.59 → 0.76) |
| Quality signal (acc > rej) | 58.7% of prompt groups |

### Tanh Mode Design

Replaced competitive softmax with bounded non-competitive per-token weights:
- `w_t = 1 + alpha * tanh(z / tau)`, where z is z-score normalized influence
- Config: `alpha=0.5`, `tau=1.0`, `clamp=[0.5, 2.0]`
- If influence is noise (z ≈ 0), weights ≈ 1.0 (no effect) — unlike softmax which always redistributes

### Tanh Training Status (step 111)

| Metric | Step 1 | Step 111 |
| --- | --- | --- |
| score/mean | -0.603 | 0.466 |
| response_length | 3905 | 1886 |
| entropy | 0.788 | 0.888 |
| grad_norm | 0.075 | 0.109 |
| masked_frac | 0.688 | 0.952 |
| solve_none | 0 | 22 |

### Changes

| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_token_weight.py` | Added tanh mode + params | Non-competitive bounded weights |
| `verl/workers/actor/dp_actor.py` | Pass accepted labels + tanh weight logging | Support influence_trace_accepted for tanh |
| `verl/trainer/ppo/core_algos.py` | token_weights param in loss functions | Apply per-token weights to PG loss |
| `dapo/dapo_ray_trainer.py` | Pass tanh_alpha, tanh_tau to config | Config plumbing |
| `scripts/train/run_r1_15b_infl_weight_tanh.sh` | New | Tanh training script |
| `scripts/train/run_r1_15b_infl_weight_softmax.sh` | New | Softmax training script (reference) |
| `scripts/train/run_r1_15b_infl_weight_perprompt.sh` | New | PerPrompt zero-mode script |
| `scripts/train/run_r1_15b_infl_weight_allsel.sh` | New | AllSel zero-mode script |
| `scripts/train/run_r1_15b_infl_weight_flip.sh` | New | Flip-mode script |
| `scripts/eval/run_batch_eval_math500_inflw_*.sh` | New | Batch eval scripts for influence weight runs |
| `diary/progress.md` | Updated | Reorganized with full MATH-500 summary table |

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| Softmax MATH-500 eval | pass.csv: 80.50% avg@4, 91.6% pass@4 | Pass |
| Tanh training runs | Step 111, no crashes, grad_norm stable | Pass |
| Deep analysis complete | `/tmp/influence_analysis_results.txt` (597 lines) | Pass |

### Notes

- No influence weighting variant improves over baseline at step 100. Primary effect is faster response length decrease.
- Root cause: contrastive direction is length-confounded (r = -0.449). Token influence signal is mostly noise.
- Tanh mode's advantage over softmax: non-competitive, so noisy signal → weights ≈ 1.0 (no harm). Need eval results to confirm.
- Tanh training checkpoint at step 100 ready for MATH-500 evaluation.
