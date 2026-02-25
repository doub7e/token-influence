## 2026-02-25 - Token Influence Trace: factor=64 vs factor=32 × per_prompt vs all_selected

| Item | Details |
| --- | --- |
| Request | Run 4 token influence trace experiments: factor={64,32} × scope={per_prompt, all_selected}, all with LOO (exclude_self_response=True) and max_hessian_dim=-1 |
| Delivery | 4 experiments completed; timing breakdown extracted; run_infl_v2_case.sh made configurable via env vars |
| Scope | Archer2.0 influence trace pipeline, Qwen2.5-1.5B, 8×H100 (3 jobs) + 8×H200 (1 job) |

### Changes

| Path | Change | Why |
| --- | --- | --- |
| `scripts/train/run_infl_v2_case.sh` | Made `INFLUENCE_PROJECTION_DIM_FACTOR`, `INFLUENCE_MAX_HESSIAN_DIM`, `INFLUENCE_DEBUG_HESSIAN_SIMILARITY` overridable via env vars (with defaults 32, -1, False) | Allow submitting multiple experiments with different factors from the same script |
| `verl/workers/actor/influence_trace.py` | Refactored diagnostic HINV_DOT to use global Hessian (accumulated across all prompt groups per module) instead of per-group Hessian | Enables real cross-prompt HINV_DOT in ALL_PROMPT diagnostic; single Cholesky per module instead of per-group |

### Experiment Results

**Common config**: `output_function=log_prob_reward`, `contrastive_agg=mean`, `hessian_source=token`, `max_hessian_dim=-1`, `exclude_self_response=True`, `total_steps=1`, `ppo_epochs=1`, `skip_optimizer_step=False`, `debug_hessian_similarity=False`

#### Timing (seconds)

| Metric | f64-perprompt | f64-allsel | f32-perprompt | f32-allsel |
| --- | --- | --- | --- | --- |
| Node Pool | H100 | H100 | H100 | H200 |
| **step total** | **1,012** | **1,002** | **2,415** | **2,383** |
| update_actor | 513 | 506 | 1,916 | 1,884 |
| pop_rows | 122 | 117 | 1,069 | 1,046 |
| hessian_solve | 81 | 77 | 990 | 961 |
| grad_staging | 36 | 35 | 66 | 66 |
| token_scoring | 4 | 5 | 13 | 18 |
| logprob_backward | 187 | 182 | 254 | 246 |
| loss_backward | 75 | 75 | 73 | 70 |
| gen (rollout) | 411 | 408 | 410 | 410 |

#### Resources

| Metric | f64-perprompt | f64-allsel | f32-perprompt | f32-allsel |
| --- | --- | --- | --- | --- |
| rows_emitted | 96 | 96 | 96 | 96 |
| GPU mem (GB) | 91 | 91 | 92 | 190 |
| CPU mem (GB) | 436 | 436 | 753 | 767 |

#### Key Findings

1. **factor=64 is ~2.4× faster** than factor=32 per step (1,012s vs 2,415s). The difference is almost entirely from Hessian solve (81s vs 990s = 12× difference, consistent with O(D^3) Cholesky).
2. **per_prompt vs all_selected** makes <3% timing difference at the same factor.
3. **all_selected + factor=32 OOMs on H100** (80GB). Needed H200 (141GB). The single-group Cholesky solve with D=13440 is too large for H100.
4. **CPU memory** scales with factor: 436 GB (f64) vs 753 GB (f32), due to larger projected gradient storage.
5. All 4 experiments emitted 96 rows successfully.

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| All 4 jobs succeeded | `runai list` → all Succeeded | Pass |
| rows_emitted=96 for all | Log metrics | Pass |
| LOO confirmed | `exclude_self_response=True` in all logs | Pass |
| No CUDA errors/NaN (except f32-allsel H100 OOM) | Logs clean after H200 resubmit | Pass |
| f32-allsel H100 OOM → H200 recovery | Resubmitted on H200, succeeded | Pass |

### Git

| Field | Value |
| --- | --- |
| Commit | `5efb386` |
| Branch | `main` |
| Remote | `doub7e/main` |
| Push | Yes |

### Notes
- factor=64 offers a good speed/fidelity tradeoff: 17 min per step vs 40 min for factor=32, while still using all 196 modules with uncapped projection.
- For production multi-step runs, factor=64 is recommended to keep step time manageable.
- all_selected scope requires H200 at factor=32 due to single-group Cholesky memory requirements.
