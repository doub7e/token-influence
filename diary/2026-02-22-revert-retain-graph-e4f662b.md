## 2026-02-22 - Revert retain_graph Code

| Item | Details |
| --- | --- |
| Request | Remove retain_graph code since it is FSDP-incompatible |
| Delivery | Reverted all 4 files; diary entry with finding preserved |
| Scope | Code-only revert, no experiment |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Removed `retain_graph` field from config + parser | Revert |
| `verl/workers/actor/dp_actor.py` | Removed `use_retain_graph` logic, restored unconditional Forward 2 | Revert |
| `dapo/dapo_ray_trainer.py` | Removed `retain_graph` propagation | Revert |
| `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh` | Removed env var + Hydra override | Revert |

### Git
| Field | Value |
| --- | --- |
| Commit | `e4f662b` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | No |

### Notes
- Finding documented in `diary/2026-02-22-retain-graph-config-fsdp-incompatible-4cf82f4.md`.
- Root cause: FSDP post-backward hook frees parameter storage, breaking retained graph tensor references.
