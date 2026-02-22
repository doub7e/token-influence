## 2026-02-22 - Influence Scoring V2: log_prob_reward + mean agg + token-level Hessian

| Item | Details |
| --- | --- |
| Request | Add three new influence pipeline options: `log_prob_reward` output function, `mean` contrastive aggregation, and `token`-level Hessian source |
| Delivery | Implemented all three features with full config propagation, run script support, and backward-compatible defaults |
| Scope | Archer2.0 influence trace pipeline; no GPU validation yet (RunAI jobs to follow) |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Added `contrastive_agg` and `hessian_source` fields to `InfluenceTraceConfig` dataclass and `from_meta()` parser; extended `output_function` validation to accept `log_prob_reward` | Config layer for three new features |
| `verl/workers/actor/influence_trace.py` | Updated `_module_token_scores_identity()`: replaced `row_sign` with `row_weight`, added `mean` branch with `1/n_acc` and `-1/n_rej` weights | Mean contrastive aggregation in identity mode |
| `verl/workers/actor/influence_trace.py` | Updated `_module_token_scores()` inverse branch: added `mean` vs `sum` dispatch for `infl[row_acc]` aggregation; added LOO correction for `exclude_self_response + mean` | Mean contrastive aggregation in inverse mode |
| `verl/workers/actor/influence_trace.py` | In inverse branch, select `hess_source = g_tok if hessian_source == "token" else g_resp` before `_compute_kernel_solve` | Token-level Hessian raises effective rank from n_responses to n_tokens |
| `verl/workers/actor/dp_actor.py` | Added `_compute_log_prob_reward_objective()` method using `log_prob * reward.unsqueeze(-1)` | New output function weighting log-prob by raw reward (0/1) |
| `verl/workers/actor/dp_actor.py` | Unified `capture_with_logprob_backward` into `capture_with_separate_backward` covering both `log_prob_advantage` and `log_prob_reward`; dispatched backward by `output_function` | Single code path for all separate-backward influence objectives |
| `verl/workers/actor/dp_actor.py` | Added `contrastive_agg` and `hessian_source` to setup log message | Visibility |
| `dapo/dapo_ray_trainer.py` | Added `contrastive_agg` and `hessian_source` to influence config dict propagated to actor | Trainer → actor config plumbing |
| `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh` | Added `INFLUENCE_CONTRASTIVE_AGG` and `INFLUENCE_HESSIAN_SOURCE` env vars and Hydra overrides | Run script support |
| `OVERVIEW.md` | Documented new config knobs and output function | Discoverability |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| Syntax / import correctness | All edits preserve existing structure, no new imports needed | Pass (visual) |
| Default backward compatibility | All new fields default to previous behavior (`sum`, `response`, no `log_prob_reward`) | Pass (by design) |
| GPU validation | Pending RunAI submission | Pending |

### Git
| Field | Value |
| --- | --- |
| Commit | `9d1f864` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | No |

### Notes
- The `log_prob_reward` path reads `data["influence_trace_reward"]` which is already propagated from the trainer side (added in earlier commits).
- Token-level Hessian will significantly increase the Cholesky solve cost since the kernel matrix grows from `n_responses x n_responses` to `n_tokens x n_tokens`. The `max_hessian_dim` cap still applies.
- Next: submit two RunAI jobs (`infl-v2-perprompt-0222`, `infl-v2-allsel-0222`) with `log_prob_reward + mean + token` on 8xH200 to validate.
