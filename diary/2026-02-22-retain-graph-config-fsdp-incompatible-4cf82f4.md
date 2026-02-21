## 2026-02-22 - retain_graph Config (FSDP-Incompatible)

| Item | Details |
| --- | --- |
| Request | Add `retain_graph=True` option to skip Forward 2 in `log_prob_advantage` mode, saving ~15% of `update_actor` |
| Delivery | Config field, conditional skip logic, FSDP incompatibility discovered and documented |
| Scope | 1xH200 smoke test, `projection_dim_factor=32`, `hessian_mode=identity` |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Added `retain_graph: bool = False` to config + parser | New toggle |
| `verl/workers/actor/dp_actor.py` | Conditionally skip Forward 2 when `retain_graph=True`; added FSDP warning | Core optimization |
| `dapo/dapo_ray_trainer.py` | Propagate `retain_graph` to batch meta_info | Config wiring |
| `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh` | Added `INFLUENCE_RETAIN_GRAPH` env var + Hydra override | Run script |

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| Baseline (`retain_graph=False`) 1xH200, 2 steps | W&B `infl-rg-smoke-off-0222c` | Pass |
| Baseline timing (step 2) | `forward_1=0.576s`, `forward_2=0.566s`, `logprob_bw=0.880s`, `loss_bw=1.033s` | Forward 2 is ~15% of `update_actor` |
| Optimized (`retain_graph=True`) 1xH200 | W&B `infl-rg-smoke-on-0222c` | **Fail** — FSDP crash |
| Error | `RuntimeError: setStorage... storage of size 0` at `loss.backward()` | FSDP frees param storage after Backward 1 |

### FSDP Incompatibility Analysis

| Phase | What Happens | Problem |
| --- | --- | --- |
| Forward 1 | FSDP unshards params → graph nodes reference unsharded tensors | OK |
| Backward 1 (`retain_graph=True`) | Backprop through graph; FSDP post-backward hook reshards (frees storage) | Graph nodes now reference freed tensors |
| Backward 2 (`loss.backward()`) | Tries to access retained graph nodes → storage of size 0 | **Crash** |

**Conclusion**: `retain_graph=True` is fundamentally incompatible with FSDP. All production runs use FSDP, so this optimization is not viable without significant FSDP surgery (e.g., preventing reshard after backward, which has memory implications).

### Git
| Field | Value |
| --- | --- |
| Commit | `4cf82f4` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | No |

### Notes
- The `retain_graph` config defaults to `False` and is safe to keep in code.
- A runtime `warnings.warn()` fires when `retain_graph=True` is configured, explaining the FSDP limitation.
- Forward 2 cost is ~0.57s on a tiny 1-GPU batch. On the 8xH200 production run, Forward 2 ≈ 38s/step (from previous timing data).
- Future alternatives to consider: (1) fuse both backward passes into one without `retain_graph`, (2) use `training_loss` output function (single pass), (3) accept Forward 2 overhead.
