## 2026-02-23 - Fix log_prob_reward: Use Signed Reward (2r-1)

| Item | Details |
| --- | --- |
| Request | Rejected responses (reward=0) had all-zero token influence scores in the visualizer |
| Delivery | Changed `log_prob_reward` objective from `log_prob * reward` to `log_prob * (2*reward - 1)`, mapping 0/1 to ±1 |
| Scope | Archer2.0 influence trace, `log_prob_reward` output function |

### Root Cause
With raw binary reward (0/1), `log_prob * 0 = 0` for all rejected tokens. Zero gradients → zero influence scores and zero Hessian contribution for the entire rejected half of the data.

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/dp_actor.py` | `signed_reward = 2.0 * reward - 1.0` before multiplying with `log_prob` | Ensures both accepted (+1) and rejected (-1) produce equal-magnitude non-zero gradients |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| `infl-v2-perprompt-0223` step 1 rows_emitted | RunAI logs | Pass — 96 rows |
| `infl-v2-perprompt-0223` step 2 rows_emitted | RunAI logs | Pass — 80 rows |
| grad_norm finite | `0.006 / 0.011` | Pass |
| Rejected tokens all-zero? | NPZ analysis: `all_zero=False, near_zero=0/3883971` | Fixed |
| Rejected mean influence | `-0.055` (negative, expected) | Pass |
| Accepted mean influence | `+0.109` (positive, expected) | Pass |
| No CUDA errors / NaN | RunAI logs | Pass |

### Git
| Field | Value |
| --- | --- |
| Commit | `ce8d5ac` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | Yes |

### Notes
- The signed reward mapping `2r-1` is specific to binary 0/1 rewards (math tasks). For continuous rewards, a different centering strategy may be needed.
- Rejected influence range `[-121.25, 3.84]` has larger magnitude outliers than accepted `[-11.80, 27.72]` — likely because rejected responses tend to be longer and accumulate more gradient mass. Worth monitoring in longer runs.
