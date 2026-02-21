# Archer2.0 Progress

Latest experiment outcomes (newest-first).

| Date | Run | Config | Result | Key Metrics |
| --- | --- | --- | --- | --- |
| 2026-02-21 | `infl-alltoken-timing-0221` | 8xH200, same as fix28h + `max_tokens_per_response=-1`, `profile_timing=True` | Pass | `rows_emitted=96`, `capture_tokens=733870`, `NaN_valid=0%`, `pop_rows=3.432s` (hessian=1.47s, scoring=1.35s), `update_actor=230.5s`, `step=713.8s` |
| 2026-02-21 | `infl-fix28h-anchorparambw-realopt-0221b` | 8xH200, `log_prob_advantage`, `per_prompt`, `inverse`, `factor=256`, `skip_optimizer_step=False` | Pass | `rows_emitted=96`, `grad_norm=0.006`, `timing_s/update_actor=198.783`, `timing_s/step=698.964` |
| 2026-02-21 | `infl-fix27h-anchorparambw-skipopt-0221b` | 8xH200, same as above but `skip_optimizer_step=True` | Pass | `rows_emitted=96`, `debug_output_grad_hook_calls=18528`, `timing_s/update_actor=199.957`, `timing_s/step=688.775` |
| 2026-02-21 | `infl-fix26h-debughooks-skipopt-0221b` | 8xH200, pre-fix diagnostic | Fail (expected diagnostic) | `rows_emitted=0`, `debug_output_grad_hook_calls=0`, `timing_s/update_actor=135.963`, `timing_s/step=625.670` |
