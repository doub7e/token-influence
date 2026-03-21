## 2026-03-09 - Cross-GPU Contrastive Direction (`global_selected` Mode)

| Item | Details |
| --- | --- |
| Request | Implement cross-GPU all-reduce for contrastive direction so all ~640 responses (4 GPUs × 160) contribute to the signal |
| Delivery | New `global_selected` accepted_rejected_scope; all-reduce in both inverse and identity Hessian paths; training script; RunAI job submitted |
| Scope | Archer2.0, Qwen3-1.7B-Base, 4×H100, influence trace pipeline |

### Changes

| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Added `global_selected` to config validation | New scope option |
| `verl/workers/actor/influence_trace.py` | Grouping: treat `global_selected` same as `all_selected` | Single group with all local tokens |
| `verl/workers/actor/influence_trace.py` | Inverse-Hessian: fused all-reduce of `[sum_acc \| sum_rej \| n_acc \| n_rej]` | Form global `d = mean_acc - mean_rej`, local `H^{-1}` solve |
| `verl/workers/actor/influence_trace.py` | Identity-Hessian: all-reduce `[M_acc \| M_rej \| counts]` matrices | Global contrastive `M`, local `score_t = u_t^T M v_t` |
| `verl/workers/actor/influence_trace.py` | Skip-guard: bypass all-same check when `use_global=True` | Prevent deadlock — all GPUs must participate in all-reduce |
| `verl/workers/actor/influence_trace.py` | LOO correction uses global counts | Unbiased leave-one-out with cross-GPU statistics |
| `scripts/train/run_qwen3_1.7b_base_infl_meanratio_globalsel.sh` | New training script | Based on allsel; `accepted_rejected_scope=global_selected` |
| `scripts/train/run_qwen3_1.7b_base_infl_meanratio_allsel.sh` | New (committed with batch) | AllSel experiment script |
| `dapo/dapo_ray_trainer.py` | Prior uncommitted changes included | Influence token weight integration |
| `verl/workers/actor/dp_actor.py` | Prior uncommitted changes included | Score normalization, self_influence_scale support |
| `verl/workers/actor/influence_token_weight.py` | Prior uncommitted changes included | Ratio mode, SNR threshold |

### Design

| Aspect | Detail |
| --- | --- |
| Communication | One all-reduce of `[2D + 2]` floats per module per step |
| Worst-case payload | ~24 KB per module, ~2.6 MB total across 196 modules |
| Overhead vs step time | Negligible (~ms vs ~60s step time) |
| Fallback | `world_size == 1` → identical to `all_selected` (no all-reduce) |
| Deadlock safety | All-same local skip bypassed when `use_global=True` |

### Experiment

| Parameter | Value |
| --- | --- |
| Experiment name | `Archer2.0-Qwen3-1.7B-Base-MeanRatio-GlobalSel-v2` |
| RunAI job | `meanratio-globalsel-v2` |
| GPUs | 4×H100 |
| Scope | `global_selected` |
| Contrastive agg | `mean` |
| Exclude self | `True` |
| Score norm | `none` |
| Weight mode | `ratio`, clamp `[-1, 3]`, SNR=0.0 |

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| Python syntax check | `py_compile.compile(...)` | Pass |
| Codex review | Attempted but codex-cli unresponsive (gpt-5.4 timeout) | Skipped — manual self-review completed |
| RunAI job submission | `runai describe job meanratio-globalsel-v2` | Pending (awaiting scheduling) |

### Git

| Field | Value |
| --- | --- |
| Commit | `f4075f0` |
| Branch | `main` |
| Remote | `doub7e` (git@github.com:doub7e/token-influence.git) |
| Push | No |

### Notes
- Expected: `sign_guard_frac` should be similar or better than `all_selected` (~45%) since more responses → stronger contrastive signal.
- Verify at step 100 on MATH-500.
- Compare with `meanratio-allsel-resume` (currently at step 178).
- Also bundled prior uncommitted changes: score_normalization (`h_inv_norm`), self_influence_scale, contrastive_agg=mean/advantage, hessian_source=token, and multiple experiment scripts.
