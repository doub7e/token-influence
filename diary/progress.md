# Archer2.0 Progress

Latest experiment outcomes (newest-first).

| Date | Run | Config | Result | Key Metrics |
| --- | --- | --- | --- | --- |
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
