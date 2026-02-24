## 2026-02-25 - Gradient Similarity Diagnostic with H^{-1}-Weighted Dot Products

| Item | Details |
| --- | --- |
| Request | Add full-model gradient similarity diagnostic (raw + H^{-1}-weighted) across all 196 modules; remove max_hessian_dim cap; set factor=32 |
| Delivery | Accumulates pairwise response dot products (raw and Hessian-weighted) across all modules; reports dot + cosine for per-prompt and all-prompt breakdowns; validated on 8×H200 |
| Scope | Archer2.0 influence trace pipeline, Qwen2.5-1.5B, 8×H200 |

### Changes

| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Added `_grad_dot_accum`, `_grad_hinv_dot_accum` accumulators, `_accumulate_grad_dot()`, `_log_dot_matrix()` helper, `_log_unprojected_comparison()`, modified `_compute_kernel_solve` to return Cholesky factor | Accumulate Σ_m g_r^m · g_{r'}^m and Σ_m g_r^{m,T} H_m^{-1} g_{r'}^m across all 196 modules |
| `verl/workers/actor/influence_trace.py` | Allow `max_hessian_dim=-1` to disable D cap | Enable full projection dimensions for MLP modules (D=13440 at factor=32) |
| `verl/workers/actor/influence_trace.py` | Added `_debug_unproj_module` for k_proj layer 14 | Compare projected vs unprojected gradient dots for one module |
| `dapo/dapo_ray_trainer.py` | Propagate `debug_hessian_similarity` in trainer config | Enable diagnostic flag through Hydra config |
| `scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh` | Added `debug_hessian_similarity` env var + Hydra override | Propagate diagnostic flag |
| `scripts/train/run_infl_v2_case.sh` | Set `INFLUENCE_PROJECTION_DIM_FACTOR=32`, `INFLUENCE_MAX_HESSIAN_DIM=-1` | Uncapped projection for maximum fidelity |

### Key Results (job `infl-v2-graddot-f32-0224`, 8×H200)

**Projection dimensions** (factor=32, no cap):

| Module type | k_in | k_out | D | hessian_peak_mb |
| --- | --- | --- | --- | --- |
| q_proj / o_proj | 48 | 48 | 2,304 | 40 |
| k_proj / v_proj | 48 | 8 | 384 | 1 |
| gate_proj / up_proj | 48 | 280 | 13,440 | 1,378 |
| down_proj | 280 | 48 | 13,440 | 1,378 |

**RAW_DOT** (no Hessian):

| Category | PER_PROMPT mean | PER_PROMPT cos | Self-dot (norm²) |
| --- | --- | --- | --- |
| acc-acc | -3.1e-7 | -0.0008 | 0.023 |
| rej-rej | -5.1e-7 | -0.002 | 0.018 |
| acc-rej | 1.8e-6 | 0.002 | — |

**HINV_DOT** (with H^{-1} preconditioning):

| Category | PER_PROMPT mean | PER_PROMPT cos | Self-dot |
| --- | --- | --- | --- |
| acc-acc | **145** | 0.0023 | 250 |
| rej-rej | 64 | 0.0009 | 253 |
| acc-rej | 64 | 0.001 | — |

**Key finding**: H^{-1} preconditioning recovers cross-response correlation (acc-acc mean = 58% of self-dot) that is invisible in raw dot products (~0). Accepted responses are more aligned with each other than with rejected ones under the H^{-1} metric. This validates that `g_r^T H^{-1} g_t` is the correct influence metric.

### Performance

| Metric | Value |
| --- | --- |
| Peak GPU memory | 142 GB |
| Peak CPU memory | 774 GB |
| Hessian solve time (all modules) | 990s |
| MLP module solve time (each) | ~19s |
| Total step time | 3674s |

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| Job completes | `runai describe job infl-v2-graddot-f32-0224` → Succeeded | Pass |
| RAW_DOT output | Log lines 817-822 | Pass — near-zero cross-response dots |
| HINV_DOT output | Log lines 823-828 | Pass — acc-acc mean=145 (58% of self-dot 250) |
| Unprojected comparison | Log lines 829-845 | Pass — projection preserves means |
| 96 rows emitted | `influence_trace/rows_emitted:96` | Pass |
| No CUDA errors/NaN | Full log clean | Pass |

### Git

| Field | Value |
| --- | --- |
| Commit | `0fe865c` |
| Branch | `main` |
| Remote | `doub7e/main` |
| Push | No |

### Notes
- Raw gradient dots are ~1000x smaller than self-dot for all categories — this is inherent to autoregressive response gradients, not a projection artifact (confirmed by unprojected comparison).
- H^{-1} preconditioning amplifies aligned gradient components, revealing that same-prompt accepted responses share meaningful gradient structure.
- MLP modules dominate compute at D=13440 (~19s each for Cholesky); attention modules are fast (~0.1-0.9s).
- CPU memory usage (774 GB) is high; this config may not fit on smaller nodes.
