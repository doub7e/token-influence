## 2026-02-23 - Fix log_prob_reward: Plain log_prob (No Reward Weighting)

| Item | Details |
| --- | --- |
| Request | Signed reward (2r-1) double-counted accepted/rejected sign: once in gradient, once in contrastive scoring |
| Delivery | Changed `log_prob_reward` to use plain `log_prob` (no weighting); contrastive scoring alone handles the sign |
| Scope | Archer2.0 influence trace, `log_prob_reward` output function |

### Root Cause
Previous fix used `signed_reward = 2*reward - 1` to map 0/1 → ±1. This baked the accepted/rejected sign into the gradient direction, but contrastive scoring (`mean_acc - mean_rej`) applied the sign again. Result: double-counted contrast (accepted tokens always large positive, rejected always large negative).

Correct approach: all responses capture the same type of gradient (∇log_prob). The contrastive scoring phase handles accepted vs rejected separation entirely.

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/dp_actor.py` | `_compute_log_prob_reward_objective`: removed `signed_reward`, use `agg_loss(loss_mat=log_prob, ...)` | Eliminate double-counted sign; let contrastive scoring handle directionality |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| `infl-v2-perprompt-0223b` 2 steps | RunAI logs | Pass — rows=96/80, grad_norm=0.006/0.011 |
| Rejected tokens all_zero? | NPZ: `all_zero=False`, `near_zero=92/3.9M` | Pass |
| Rejected mean influence | `-0.054` (negative, correct direction) | Pass |
| Accepted mean influence | `+0.109` (positive, correct direction) | Pass |
| Rejected pos:neg ratio | 37:63 (majority negative) | Pass |
| Accepted pos:neg ratio | 69:31 (majority positive) | Pass |
| No double-counting | Both classes have mixed-sign scores, not all-same-sign | Pass |

### Git
| Field | Value |
| --- | --- |
| Commit | `5d65757` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | Yes |

### Notes
- `log_prob_reward` is now effectively "plain log_prob" — the `reward` argument is accepted but unused. The name is kept for config backward compatibility.
- Compared to `log_prob_advantage`, this avoids GRPO advantage normalization scale variation while producing the same gradient direction for all responses.
