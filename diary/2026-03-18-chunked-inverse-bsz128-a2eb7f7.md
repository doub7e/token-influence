## 2026-03-18 - Chunked Inverse Hessian for bsz=128 Influence Trace

| Item | Details |
| --- | --- |
| Request | Fix OOM at bsz=128 influence trace; compare hessian vs no-hessian; report timing overhead |
| Delivery | Chunked inverse mode implementation, successful bsz=128 run, comparison analysis |
| Scope | 4x H200 GPUs, Qwen3-4B-Base, step 300→301, 1088 responses (68 prompts × 16 responses) |

### Key Results

| Config | Responses | AUC | GapDir% | FracPos Gap | Step Time | Influence Overhead |
| --- | --- | --- | --- | --- | --- | --- |
| bsz=32, hessian, f16 (baseline) | 192 | 0.749 | 70.8% | 0.056 | ~100s | ~50% |
| **bsz=128, hessian, f16** | **1088** | **0.978** | **81.9%** | **0.099** | **337s** | **32% (+82s)** |
| bsz=128, identity, f16 | 1088 | 0.618 | 50.3% | 0.022 | 256s | 0.4% (+1s) |

### Timing Breakdown (bsz=128, hessian)

| Component | Seconds |
| --- | --- |
| hessian_solve (Cholesky + chunked solve) | 38.1 |
| token_scoring (chunked scoring loop) | 31.0 |
| pop_rows (score aggregation) | 69.2 |
| forward_1 + forward_2 | 25.1 |
| loss_backward | 32.8 |
| **Total influence overhead** | **~82s** |

### Changes

| Path | Change | Why |
| --- | --- | --- |
| `verl/workers/actor/influence_trace.py` | Replaced monolithic g_tok reconstruction with 3-phase chunked approach | OOM: [200k tokens, 49920 dim] = 40 GiB exceeds GPU memory |
| `scripts/train/run_4b_infl_bsz128.sh` | New script for bsz=128 influence comparison | Test influence at production batch size |
| `scripts/train/run_4b_infl_no_unitnorm.sh` | New script for unit-norm ablation | Compare with/without token_unit_norm |
| `scripts/deep_token_influence_analysis.py` | 10-part deep token-level analysis | Cross-config agreement, entropy, error localization, etc. |
| `docs/deep-token-influence-analysis.md` | Analysis documentation | Key findings: boxed tokens negative, 97% within-response variance, U-shaped AUC |

### Chunked Implementation Design

The OOM root cause: `g_tok = bmm(u_g, v_g).reshape(n_tokens, D)` with n_tokens=200k and D=49920 requires ~40 GiB in one allocation.

Solution — 3-phase chunked approach:
1. **Accumulate**: Stream u,v chunks → reconstruct g_chunk → `index_add` into g_resp + `addmm` into hessian
2. **Factorize**: Cholesky decomposition of hessian (once, O(D^3))
3. **Score**: Stream chunks again → `cholesky_solve` per chunk → compute scores

Chunk size bounded by `_MAX_CHUNK_BYTES = 4 GiB`. Peak memory: hessian [D,D] fp32 = 9.3 GB + one chunk ≈ 4 GB + g_resp + overhead.

### Validation

| Check | Evidence | Result |
| --- | --- | --- |
| No OOM at bsz=128 | RunAI job `infl-bsz128-v2`, peak 161 GB on H200 | Pass |
| Hessian AUC > bsz=32 | 0.978 vs 0.749 | Pass (scaling improves signal) |
| Identity mode weaker | AUC 0.618 vs 0.978 | Pass (Hessian correction essential) |
| Both configs complete | "ALL BSZ128 CONFIGS COMPLETE" in logs | Pass |

### Git

| Field | Value |
| --- | --- |
| Commit | `a2eb7f7` |
| Branch | `main` |
| Remote | `origin` |
| Push | No |

### Notes

- Scaling from bsz=32→128 massively improves hessian mode (0.749→0.978 AUC) — more responses per prompt gives the contrastive direction stronger signal
- Identity mode does NOT benefit from larger batch (0.618) — the Hessian correction is essential for quality at scale
- The 32% overhead is acceptable; main bottleneck is pop_rows (69s) which includes score aggregation and I/O
- Cross-config Pearson correlation is only 0.26 — hessian and identity produce fundamentally different signals
- Unit norm comparison (separate experiment): keeping unit norm improves token-level AUC at the cost of response-level AUC; recommended for per-token credit
- Remaining: analyze whether identity mode can skip random projection entirely (rank-1 dot product trick)
