## 2026-02-19 - Token Influence Trace Pipeline + Entropy/Influence Visualizer

| Item | Details |
| --- | --- |
| Request | Implement a complex new feature to visualize token influence (not only entropy), compute token-to-response influence within each rollout prompt group, keep entropy and influence side-by-side in one visualizer, then run on RunAI and debug. |
| Delivery | Implemented end-to-end influence tracing on real PPO backward pass, per-step influence artifact writing, trainer/worker integration, and dual heatmap visualizer (entropy + influence). Fixed trace writing robustness and validated a 2-step 8xH200 run. |
| Scope | GPU execution only on RunAI; no local CUDA execution. Main run used baseline rollout shape (`train_batch_size=64`, `rollout.n=16`) with `trainer.total_training_steps=2` for fast validation. |

### Influence Design (Detailed)
| Topic | Decision | Implementation |
| --- | --- | --- |
| Target definition | Token influence for visualization is computed per token against all responses in the same prompt group. | In actor trace logic, influence score is aggregated at token level and later written as `influence` matrix aligned to response tokens. |
| Accept/Reject aggregation | Use sum, not mean: `sum_{accepted r} infl(t->r) - sum_{rejected r} infl(t->r)`. | `verl/workers/actor/influence_trace.py` computes `score = infl[row_acc].sum(dim=0) - infl[~row_acc].sum(dim=0)`. |
| Data source for gradients | Use real training gradients; no extra backward pass. | Hooks run during normal PPO `loss.backward()` path in `verl/workers/actor/dp_actor.py`. |
| Gradient representation | Use per-token rank-1 form (`u`, `v`) with two-sided random projection to avoid storing full gradients. | For each selected linear module, save projected `u = dL/dy @ P_out^T`, `v = x @ P_in^T` from backward/forward hooks. |
| Response gradient | Response-level gradient reconstructed as average over token gradients in that response. | Build `g_resp` by `index_add_` from token gradients then divide by token count per response. |
| Hessian/kernel size | Use projected dimension `D = k_in * k_out`; Hessian is `D x D`, independent of token count. | `_pick_projection_dims` enforces `k_in + k_out <= max_proj_vector_sum` and `k_in*k_out <= max_hessian_dim`. |
| Inverse-Hessian compute | Use dense Cholesky solve on projected Hessian; no low-rank approximation in this version. | `_compute_kernel_solve()` computes `H = g_resp^T g_resp + λI`, then `torch.linalg.cholesky` + `torch.cholesky_solve`. |
| Regularization | If `reg_lambda <= 0`, auto-scale by Hessian trace. | `reg = 0.1 * trace(H) / D` fallback. |
| Group filtering | Skip degenerate groups (all accepted or all rejected); skip UID groups crossing ranks for this version. | `_prepare_influence_trace_batch()` in trainer pre-selects valid groups and writes debug counters. |
| GPU/CPU transfer | Keep repeated heavy math on GPU; only move final artifacts to CPU for persistence. | Hook captures and influence compute are on GPU; writer converts final tensors to numpy once per step. |
| Memory guardrail | Track projected Hessian peak estimate using `2 * D * D * fp32`. | Printed once on rank0 from actor tracer via `estimate_hessian_memory_mb()`. |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `Archer2.0/verl/workers/actor/influence_trace.py` | Added | Core token-influence tracer: module selection, random projections, hook capture, per-group Hessian solve, token score aggregation, and memory estimation. |
| `Archer2.0/verl/workers/actor/dp_actor.py` | Updated | Integrated influence tracer lifecycle into PPO update, including setup from `meta_info`, rmpad index mapping, microbatch begin/end capture, and exporting influence rows. |
| `Archer2.0/verl/workers/fsdp_workers.py` | Updated | Extended actor update output to include `non_tensor_batch["influence_trace_rows"]` for trainer-side persistence. |
| `Archer2.0/dapo/dapo_ray_trainer.py` | Updated | Added influence pre-selection by UID/reward, accept/reject labels, per-step influence metrics, influence config forwarding, and writer invocation. |
| `Archer2.0/dapo/influence_trace.py` | Added | New writer for influence step artifacts (`entropies`, `influence`, `reward`, `accepted`, `group_id`, `prompt_ids`, etc.) with manifest/summary/latest step tracking. |
| `Archer2.0/dapo/entropy_trace.py` | Updated | Added robust atomic writing, fsync option, `write_every`, summary updates, and `latest_step.txt` to align behavior with influence trace IO robustness. |
| `Archer2.0/scripts/train/visualize_rollout_entropy.py` | Updated | Upgraded to dual visualization: entropy and influence heatmaps side-by-side, prompt decode, accept/reject/reward metadata display, influence summary stats, and improved colormap scaling. |
| `Archer2.0/scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh` | Added | Dedicated reproducible launcher enabling both entropy + influence traces, fixed output dirs, and initial projection/Hessian constraints for fast debug. |
| `Archer2.0/scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh` | Updated | Switched to absolute trace output dir and explicit entropy trace IO flags for consistency and safer artifact discovery. |
| `Archer2.0/scripts/train/prepare_dapo_math17k_openr1.sh` | Added | Utility wrapper to regenerate and overwrite math training dataset from `open-r1/DAPO-Math-17k-Processed` with logging and sanity checks. |

### Visualizer Behavior Notes
| Aspect | Entropy Trace (`entropy_trace`) | Influence Trace (`influence_trace`) |
| --- | --- | --- |
| Contains entropy | Yes | Yes |
| Contains influence | No | Yes |
| Contains reward/accept/group/prompt_ids | No | Yes |
| Typical row coverage | Full rollout batch | Selected subset used for influence compute |
| Recommended usage | Global entropy-only analysis | Entropy + influence comparative analysis |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| Python syntax checks | `python3 -m py_compile dapo/dapo_ray_trainer.py dapo/entropy_trace.py dapo/influence_trace.py scripts/train/visualize_rollout_entropy.py verl/workers/actor/dp_actor.py verl/workers/actor/influence_trace.py verl/workers/fsdp_workers.py` | Pass |
| Shell syntax checks | `bash -n scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh scripts/train/prepare_dapo_math17k_openr1.sh` | Pass |
| RunAI influence debug run | Job `archer-influence-h200x8-fix-02182328` (`Succeeded`) | Pass (reached `step:2`, wrote checkpoints and trace artifacts) |
| Runtime influence metrics present | Run log contains `influence_trace/selected_responses`, `influence_trace/selected_prompts`, UID debug counters | Pass |
| W&B online sync | `https://wandb.ai/doub7e/Archer2.0/runs/Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace-h200x8-fix-02182328` | Pass |
| Trace artifact persistence | `Archer2.0/output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace/influence_trace/steps/step_000001.npz`, `step_000002.npz`; same for `entropy_trace` | Pass |

### Repro Commands
| Purpose | Command |
| --- | --- |
| Launch influence trace run | `bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh` |
| Launch visualizer (influence + entropy) | `python scripts/train/visualize_rollout_entropy.py --trace-dir output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace/influence_trace --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B --host 0.0.0.0 --port 7862` |
| Launch entropy-only visualizer input | `python scripts/train/visualize_rollout_entropy.py --trace-dir output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace/entropy_trace --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B --host 0.0.0.0 --port 7862` |

### Git
| Field | Value |
| --- | --- |
| Commit | `1837ec3` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | Yes (`5abc299..1837ec3`) |

### Notes
- Trace output directory in the current influence launcher is fixed by `exp_name` (`Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace`) and is intentionally separate from training checkpoint run folders.
- Current implementation skips cross-rank UID groups for influence computation to avoid incorrect group-local Hessian construction under distributed sharding.
- Influence artifacts intentionally store selected rows only; this is expected and different from entropy trace coverage.
