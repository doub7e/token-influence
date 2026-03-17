# Per-Token Gradient Semantics in Influence Trace

## The Problem

Our influence trace captures "per-token gradients" via forward/backward hooks on linear layers. However, these are **not** the pure per-token gradients due to cross-position gradient flow in transformers.

## Mathematical Framework

For a linear layer $W$ at layer $l$, define:

$$c_{t,s} := \frac{\partial \ell_s}{\partial y_t^{(l)}} \cdot \frac{\partial y_t^{(l)}}{\partial W}$$

where $\ell_s = \log \pi(a_s | s_{<s})$ is token $s$'s log-probability loss, and $y_t^{(l)}$ is the layer output at position $t$. By causality, $c_{t,s} = 0$ for $s < t$.

### Two decompositions of the same total gradient

| Decomposition | Formula | Meaning | How to compute |
|---|---|---|---|
| **By position** (row sums) | $\tilde{g}_t = \sum_{s \geq t} c_{t,s}$ | Position $t$'s contribution to the total gradient (includes future token effects) | 1 backward pass (what our hooks capture) |
| **By loss term** (column sums) | $\nabla_W \ell_t = \sum_{t' \leq t} c_{t',t}$ | Pure gradient of token $t$'s log-prob w.r.t. $W$ | $T$ separate backward passes |

Both decompositions sum to the same total:

$$\sum_t \tilde{g}_t = \sum_t \nabla_W \ell_t = \nabla_W L$$

**Our tracer returns $\tilde{g}_t$ (row sums), not $\nabla_W \ell_t$ (column sums).**

## Why they differ

In a transformer, position $t$'s output $y_t^{(l)}$ influences future positions $t+1, t+2, \ldots$ through attention in subsequent layers. The backward hook at position $t$ captures:

$$\frac{\partial L}{\partial y_t^{(l)}} = \underbrace{\frac{\partial \ell_t}{\partial y_t^{(l)}}}_{\text{local: token } t \text{'s own loss}} + \underbrace{\sum_{s > t} \frac{\partial \ell_s}{\partial y_t^{(l)}}}_{\text{cross-position: future tokens' losses routed via attention}}$$

The cross-position term makes $\tilde{g}_t$ carry information about the **entire future trajectory**, not just token $t$.

### Layer dependence

- **Last layer MLP**: No subsequent attention layers → $\frac{\partial \ell_s}{\partial y_t} = 0$ for $s \neq t$ → $\tilde{g}_t = \nabla_W \ell_t$ (clean)
- **Early layer MLP**: Many subsequent attention layers → large cross-position terms (heavily contaminated)
- **Attention K/V projections**: Contaminated even in the last block (K/V directly affect future queries within the same attention layer)

## Consequences for influence scoring

### Contrastive direction $d$ — correct at response level

$$d = \text{mean}(\sum_t \tilde{g}_t \text{ for accepted}) - \text{mean}(\sum_t \tilde{g}_t \text{ for rejected})$$

Since $\sum_t \tilde{g}_t = \nabla_W L_{\text{resp}}$, $d$ is the correct contrastive direction in parameter space. **The direction is valid.**

### Per-token score $s_t = d^T H^{-1} \tilde{g}_t$ — semantically shifted

$$s_t = d^T H^{-1} \tilde{g}_t = \sum_{s \geq t} d^T H^{-1} c_{t,s}$$

This measures "how much does parameter usage at position $t$ align with the accepted-vs-rejected direction **for the entire continuation**" — a prefix/eligibility-style saliency signal.

It does **not** measure "how much does token $t$ itself contribute to correctness."

**Observable consequence**: Tokens at the start of a response (e.g., "To find the area") have identical context across all responses to the same prompt, yet receive very different influence scores between accepted and rejected responses — because their gradients carry the entire future trajectory's information.

## Why per-token advantage is still valid

The GRPO loss is:

$$L = \sum_t A_t \cdot f(\text{ratio}_t)$$

Its gradient is (at ratio $\approx 1$):

$$\frac{\partial L}{\partial W} = \sum_t A_t \cdot \nabla_W \log\pi(a_t)$$

This total gradient is **mathematically exact** regardless of how the backward pass decomposes it across positions. Training only needs the total gradient, not the per-position decomposition.

| | Training (loss → gradient) | Influence (gradient → analysis) |
|---|---|---|
| Needs | Total gradient $\sum_t A_t \nabla_W \ell_t$ | Each $\nabla_W \ell_t$ individually |
| Backward provides | Total sum correct ✓ | Per-position decomposition ≠ per-token ✗ |
| Conclusion | Per-token advantage is valid | Per-token influence is biased |

## Recommendations

1. **Keep $d$ as a response-level contrastive direction** — it is correct.
2. **Do not interpret $s_t$ as pure token-level credit** — it is a trajectory-level saliency signal.
3. **For cleaner token-local signals**: restrict to final-block modules (`mlp.{gate,up,down}`, `self_attn.o_proj`, `self_attn.q_proj`). Exclude `k_proj`/`v_proj`.
4. **For GRPO advantage modification**: prefer additive residual corrections (`A'_t = A_t + c_t`) over multiplicative sign-sensitive weighting (`A'_t = A_t \times w_t`). The contamination is less harmful for additive schemes because $A_t$ already carries the response-level credit; the token score only provides a noisy within-response residual.
5. **Pure per-token gradients** require $O(T)$ backward passes — computationally infeasible for $T \sim 1000\text{-}4000$. The only one-pass alternative is to restrict to modules after the last cross-position mixing point (e.g., final layer MLP only).

## Empirical Validation (Qwen3-4B-Base, step 300)

We tested the hypothesis that restricting to final-layer modules (minimal cross-position contamination) would give cleaner per-token credit signals.

| Config | Modules | Scope | Global AUC | Sign % |
|--------|---------|-------|-----------|--------|
| f64-mlp-pp | 108 MLP, all layers | per_prompt | 0.50 | 28.6% |
| f32-mlp-allsel | 108 MLP, all layers | all_selected | 0.72 | 71.4% |
| f64-full-allsel | 252 all, all layers | all_selected | 0.73 | 81.2% |
| **f32-full-allsel** | **252 all, all layers** | **all_selected** | **0.78** | **66.7%** |
| lastmlp-allsel | 3 MLP, last block only | all_selected | 0.64 | 35.7% |
| lm_head-only | 1 module (LM head) | all_selected | N/A | crash (inplace hook conflict) |

**Key finding**: The "cleanest" per-token gradient (last block MLP, minimal cross-position contamination) has the **weakest** signal (AUC 0.64). The all-layers configuration with maximum contamination has the **strongest** signal (AUC 0.78). This demonstrates that cross-position gradient flow carries useful trajectory-level information for response-level discrimination, even though it muddles per-token attribution.

**Practical implication**: For token-level credit assignment in GRPO, we should:
1. Use all layers + all modules for maximum signal strength
2. Accept that scores are "trajectory saliency" rather than pure token credit
3. Use additive residual corrections (not multiplicative sign-based weighting)
4. The `all_selected` scope is critical — `per_prompt` fails on unbalanced groups

## References

- Codex analysis (2026-03-17): mathematical framework for $c_{t,s}$ decomposition
- `verl/workers/actor/influence_trace.py`: hook-based gradient capture implementation
- `docs/compatible-token-advantage-residual.md`: additive residual advantage correction theory
