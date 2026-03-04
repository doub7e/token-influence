# Archer2.0 Progress

Latest experiment outcomes (newest-first).

---

## MATH-500 Evaluation Summary (DeepSeek-R1-Distill-Qwen-1.5B)

All models trained with GRPO on `archer2.0-math-1.5b-train.json`.
Eval: temperature=0.8, top_p=0.95, n_samples=4, max_response_length=16384.
avg@4 = mean accuracy across 4 samples (≈ pass@1). pass@4 = at least 1 of 4 correct.

| Model | Step | avg@4 | pass@4 | Notes |
| --- | --- | --- | --- | --- |
| Base (no training) | - | 82.45% | 90.8% | Pre-training checkpoint |
| **Baseline GRPO** | **190** | **83.60%** | **91.8%** | Best baseline; plateau after step 150 |
| Baseline GRPO | 150 | 81.90% | 90.4% | |
| Baseline GRPO | 100 | 81.15% | 90.4% | |
| **Archer GRPO** | **300** | **83.85%** | 91.4% | Best overall; no plateau, higher entropy |
| Archer GRPO | 250 | 83.60% | **92.6%** | Best pass@4 |
| Archer GRPO | 190 | 83.05% | 91.4% | |
| Archer GRPO | 150 | 81.80% | 90.4% | |
| Archer GRPO | 100 | 79.75% | 91.2% | Slower start than baseline |
| InflWeight-Softmax T=10 | 100 | 80.50% | 91.6% | `mode=softmax`, `T=10.0`, `clamp=[0.1,5.0]`, `per_prompt`, f=64 |
| InflWeight-AllSel (zero) | 100 | 80.35% | 90.0% | `mode=zero`, `threshold=1.0`, `all_selected`, f=64 |
| InflWeight-AllSel (zero) | 150 | 79.20% | 89.0% | Degraded at later steps |
| InflWeight-PerPrompt (zero) | 100 | 78.05% | 90.8% | `mode=zero`, `threshold=1.0`, `per_prompt`, f=64 |
| InflWeight-Tanh a=0.5 | - | - | - | Running (`archer-inflw-tanh`, 4xH200); `mode=tanh`, `alpha=0.5`, `tau=1.0`, `clamp=[0.5,2.0]` |

### Key Findings

1. **No influence weighting variant improves over baseline at step 100** (81.15%). Softmax is closest (80.50%).
2. **Softmax has best pass@4 at step 100** (91.6%), suggesting preserved diversity.
3. **Zero-mode hurts**: PerPrompt (78.05%) and AllSel (79.20%@150) degrade below even the untrained base model.
4. **Primary effect of influence weighting is faster response length decrease**, not accuracy improvement. Length-matched comparison shows negligible score difference (+0.009 at matched RespLen ~3000).
5. **Root cause**: contrastive direction `d = mean(g_acc) - mean(g_rej)` is length-confounded (r = -0.449). Token influence signal is mostly noise (span autocorrelation = 0.098).

---

## Experiment Log

| Date | Run | Config | Result | Key Metrics |
| --- | --- | --- | --- | --- |
| 2026-03-04 | `archer-inflw-tanh` | 4xH200, R1-1.5B, `mode=tanh`, `alpha=0.5`, `tau=1.0`, `clamp=[0.5,2.0]`, f=64 | Running | Step 1: `masked_frac=0.688`, `grad_norm=0.075`, `entropy=0.788` |
| 2026-03-03 | `InflWeight-Softmax` | 4xH200, R1-1.5B, `mode=softmax`, `T=10.0`, `clamp=[0.1,5.0]`, f=64 | Pass | MATH-500 step 100: 80.50% avg@4, 91.6% pass@4; stopped at step 100 |
| 2026-03-02 | `InflWeight-PerPrompt` | 4xH200, R1-1.5B, `mode=zero`, `threshold=1.0`, `per_prompt`, f=64 | Pass | MATH-500 step 100: 78.05% avg@4, 90.8% pass@4 |
| 2026-03-02 | `InflWeight-AllSel` | 4xH200, R1-1.5B, `mode=zero`, `threshold=1.0`, `all_selected`, f=64 | Pass | MATH-500 step 100: 80.35%/90.0%; step 150: 79.20%/89.0% |
| 2026-03-01 | `archer-r1-15b-lp8` (baseline) | 4xH200, DeepSeek-R1-1.5B, GRPO, `archer_loss=False` | Pass | MATH-500: best **83.60%** avg@4, 91.8% pass@4 @ step 190; plateau after step 150 |
| 2026-03-01 | `archer-r1-15b-archer` | 4xH200, DeepSeek-R1-1.5B, GRPO, `archer_loss=True` | Pass | MATH-500: best **83.85%** avg@4, 92.6% pass@4 @ step 250/300; no plateau, higher entropy |
| 2026-02-23 | `infl-v2-allsel-0223b` | 8xH100, `all_selected`, f128, v2 + **plain log_prob fix** | Pass | `rows=96/80`, `grad_norm=0.006/0.011`, rej_mean=-1.67e-3, acc_mean=+4.11e-3, rej 42:58, acc 61:39, per-response sign 100% correct |
| 2026-02-23 | `infl-v2-perprompt-0223b` | 8xH100, `per_prompt`, f128, v2 + **plain log_prob fix** | Pass | `rows=96/80`, `grad_norm=0.006/0.011`, rej_mean=-0.054, acc_mean=+0.109, rej 37:63, acc 69:31 (mixed-sign, no double-counting) |
| 2026-02-23 | `infl-v2-perprompt-0223` | 8xH100, `per_prompt`, f128, v2 + **signed reward fix** | Pass | `rows=96/80`, `grad_norm=0.006/0.011`, rejected `all_zero=False`, rej_mean=-0.055, acc_mean=+0.109 |
| 2026-02-22 | `infl-v2-perprompt-0222` | 8xH100, `per_prompt`, `inverse`, f128, `log_prob_reward`, `contrastive_agg=mean`, `hessian_source=token` | Pass | `rows=96/80`, `grad_norm=0.006/0.011`, `hessian_solve=10.9s/13.1s`, `pop_rows=29.7s/31.6s`, `groups=1158/965` |
| 2026-02-22 | `infl-v2-allsel-0222` | 8xH100, `all_selected`, `inverse`, f128, `log_prob_reward`, `contrastive_agg=mean`, `hessian_source=token` | Pass | `rows=96/80`, `grad_norm=0.006/0.011`, `hessian_solve=10.1s/12.5s`, `pop_rows=24.9s/31.4s`, `groups=193/193` |
| 2026-02-22 | `infl-excl-perprompt-0222` | 8xH200, `per_prompt`, `inverse`, f512, `exclude_self_response=True` | Pass | `rows=96/80`, `grad_norm=0.054/0.066`, `pop_rows=2.3s`, `score_std=0.139` |
| 2026-02-22 | `infl-excl-allsel-0222` | 8xH200, `all_selected`, `inverse`, f512, `exclude_self_response=True` | Pass | `rows=96/80`, `grad_norm=0.054/0.066`, `pop_rows=1.2s`, `score_std=0.088` |
| 2026-02-22 | `infl-excl-allsel-f128-0222` | 8xH200, `all_selected`, `inverse`, f128, `exclude_self_response=True` | Pass | `rows=96/80`, `pop_rows=8.5s`, `hessian_solve=6.6s`, `score_std=0.180` |
| 2026-02-22 | `infl-excl-allsel-f64-0222` | 8xH200, `all_selected`, `inverse`, f64, `exclude_self_response=True` | Pass | `rows=96/80`, `pop_rows=34.5s`, `hessian_solve=31.1s`, `score_std=0.207` (mem metric unreliable) |
| 2026-02-21 | `infl-alltoken-timing-0221` | 8xH200, same as fix28h + `max_tokens_per_response=-1`, `profile_timing=True` | Pass | `rows_emitted=96`, `capture_tokens=733870`, `NaN_valid=0%`, `pop_rows=3.432s` (hessian=1.47s, scoring=1.35s), `update_actor=230.5s`, `step=713.8s` |
| 2026-02-21 | `infl-fix28h-anchorparambw-realopt-0221b` | 8xH200, `log_prob_advantage`, `per_prompt`, `inverse`, `factor=256`, `skip_optimizer_step=False` | Pass | `rows_emitted=96`, `grad_norm=0.006`, `timing_s/update_actor=198.783`, `timing_s/step=698.964` |
| 2026-02-21 | `infl-fix27h-anchorparambw-skipopt-0221b` | 8xH200, same as above but `skip_optimizer_step=True` | Pass | `rows_emitted=96`, `debug_output_grad_hook_calls=18528`, `timing_s/update_actor=199.957`, `timing_s/step=688.775` |
| 2026-02-21 | `infl-fix26h-debughooks-skipopt-0221b` | 8xH200, pre-fix diagnostic | Fail (expected diagnostic) | `rows_emitted=0`, `debug_output_grad_hook_calls=0`, `timing_s/update_actor=135.963`, `timing_s/step=625.670` |
