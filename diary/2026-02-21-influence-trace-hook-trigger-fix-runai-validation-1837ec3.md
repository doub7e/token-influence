## 2026-02-21 - Influence Trace Hook Trigger Fix + RunAI Validation

| Item | Details |
| --- | --- |
| Request | Continue debugging why influence trace stayed empty on 8xH200 and document all concrete findings for handoff. |
| Delivery | Fixed gradient-capture trigger path for `log_prob_advantage`, verified non-zero influence rows on both skip-optimizer and real-optimizer runs, and recorded timing/regression trade-offs. |
| Scope | `Archer2.0`, 8xH200 RunAI jobs, 1 training step, `projection_dim_factor=256`, `output_function=log_prob_advantage`, `accepted_rejected_scope=per_prompt`. |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/dp_actor.py` | Updated influence backward from `autograd.grad(..., anchor_tensor)` to `torch.autograd.backward(..., inputs=[anchor_param])`; cleared anchor param grad after the call; added/kept detailed influence debug metrics emission. | `autograd.grad` on a detached anchor path did not activate module gradient-capture path, resulting in `rows_emitted=0`. |
| `verl/workers/actor/influence_trace.py` | Added `anchor_parameter()` and richer debug counters (`forward_capture_calls`, `forward_set_v_calls`, `backward_hook_calls`). | Needed explicit parameter-targeted backward trigger and precise instrumentation to isolate where hook activation failed. |
| `diary/progress.md` | Added experiment result rows for fix26/fix27/fix28. | Keep cross-run performance/correctness trend visible for future agents. |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| Pre-fix diagnostic (hook path still broken) | RunAI job `infl-fix26h-debughooks-skipopt-0221b` | `rows_emitted=0`, `debug_output_grad_hook_calls=0`, confirmed failure mode reproducible. |
| Post-fix, skip optimizer | RunAI job `infl-fix27h-anchorparambw-skipopt-0221b` | `rows_emitted=96`, `debug_output_grad_hook_calls=18528`; influence trace file written. |
| Post-fix, real optimizer | RunAI job `infl-fix28h-anchorparambw-realopt-0221b` | `rows_emitted=96`, `actor/grad_norm=0.006`; real training path also valid. |
| Trace artifacts | `output/Archer2.0/infl-fix27h-anchorparambw-skipopt-0221b/influence_trace/steps/step_000001.npz`, `output/Archer2.0/infl-fix28h-anchorparambw-realopt-0221b/influence_trace/steps/step_000001.npz` | Pass |

### Key Metrics Snapshot
| Run | timing_s/update_actor | timing_s/step | rows_emitted | Notes |
| --- | --- | --- | --- | --- |
| `infl-fix26h-debughooks-skipopt-0221b` | `135.963` | `625.670` | `0` | Broken capture path baseline. |
| `infl-fix27h-anchorparambw-skipopt-0221b` | `199.957` | `688.775` | `96` | Capture fixed; added backward cost visible. |
| `infl-fix28h-anchorparambw-realopt-0221b` | `198.783` | `698.964` | `96` | Real optimizer enabled; behavior consistent with fix27. |

### Git
| Field | Value |
| --- | --- |
| Commit | `TBD (this commit)` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | `Yes` |

### Notes
- Root cause: `autograd.grad` target selection bypassed the gradient-collection path used by influence tracer hooks in this graph.
- Effective fix: run a targeted `autograd.backward` on a real module parameter used as anchor input target, then clear temporary grad.
- Trade-off: influence capture now works but adds substantial backward cost (`~10%-12%` step-time increase in this setup).
- Next optimization candidates: reduce capture token count (`max_tokens_per_response`), reduce module scope, or compute influence less frequently than every update.
