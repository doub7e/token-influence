## 2026-02-21 - Full-Token Influence + Timing Breakdown

| Item | Details |
| --- | --- |
| Request | Compute influence for all tokens (`max_tokens_per_response=-1`) and add per-phase timing breakdown to `pop_rows`. |
| Delivery | Instrumented `_module_token_scores()` with 4 timing buckets, emitted new `timing_s/influence_*` metrics, fixed timing reset bug, validated on RunAI. |
| Scope | 8xH200, `log_prob_advantage`, `per_prompt`, `inverse`, `factor=256`, `max_tokens_per_response=-1`, `profile_timing=True` |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Added `_timing` dict, `_reset_timing()`, `pop_timing()`, instrumented `_module_token_scores()` with `torch.cuda.synchronize()+time.perf_counter()` | Per-phase timing breakdown for grad staging, hessian solve, token scoring, score aggregation |
| `verl/workers/actor/influence_trace.py` | Separated `_reset_timing()` from `_reset_debug()` | Fix bug where `debug_stats(reset=True)` zeroed timing before `pop_timing()` could read it |
| `verl/workers/actor/dp_actor.py` | Call `pop_timing()` after `pop_token_influence_rows()`, emit 4 new `timing_s/influence_*` metrics | Surface breakdown in W&B |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| Timing metrics non-zero | W&B `infl-alltoken-timing-0221` step 1 | Pass: `grad_staging=0.055s`, `hessian_solve=1.469s`, `token_scoring=1.348s`, `score_aggregation=0.016s` |
| Rows emitted | W&B `influence_trace/rows_emitted` | Pass: 96 rows |
| Full-token capture | `debug_capture_selected_tokens=733870` | Pass: all valid response tokens captured (no capping) |
| 0% NaN in valid positions | `step_000001.npz`: 6,129,414 valid positions, 0 NaN | Pass |
| NPZ file sizes | `step_000001.npz=29MB`, `step_000002.npz=27MB` | Pass (much larger than 512-capped version) |
| W&B online | `https://wandb.ai/doub7e/Archer2.0/runs/infl-alltoken-timing-0221` | Pass |

### Timing Breakdown (step 1)
| Phase | Time (s) | % of pop_rows |
| --- | --- | --- |
| `grad_staging` | 0.055 | 1.6% |
| `hessian_solve` | 1.469 | 42.8% |
| `token_scoring` | 1.348 | 39.3% |
| `score_aggregation` | 0.016 | 0.5% |
| Other (score_map loop) | ~0.544 | 15.9% |
| **pop_rows total** | **3.432** | **100%** |

### Overall Step Timing (step 1)
| Phase | Time (s) |
| --- | --- |
| `forward_1` (influence capture) | 38.900 |
| `logprob_backward` | 76.420 |
| `forward_2` (training graph) | 38.130 |
| `loss_backward` | 70.784 |
| `pop_rows` | 3.432 |
| **update_actor total** | **230.518** |
| **step total** | **713.768** |

### Git
| Field | Value |
| --- | --- |
| Commit | `bbd4d1a` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | No |

### Notes
- The hessian solve (42.8%) and token scoring (39.3%) dominate `pop_rows`, but `pop_rows` itself is only 1.5% of `update_actor`. The backward passes (logprob: 76s + loss: 71s = 147s) are the real bottleneck at 64% of actor update time.
- First run had timing all-zero due to `_reset_debug()` also resetting `_timing` before `pop_timing()` was called. Fixed by separating `_reset_timing()`.
- RunAI job `infl-alltoken-timing-0221` may auto-restart/retry; both steps already have valid NPZ output.
