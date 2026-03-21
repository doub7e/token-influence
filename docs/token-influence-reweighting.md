# Token Influence Reweighting for GRPO

## Motivation

In GRPO (Group Relative Policy Optimization), a binary outcome reward (correct/incorrect) is broadcast uniformly to all tokens in a response. This is wasteful: not every token contributes equally to the outcome. We use **per-token influence scores** to reweight the policy gradient loss, giving more gradient mass to tokens that matter.

## Overview

The pipeline has two stages, both running online within each training step:

1. **Influence Scoring** — compute a per-token influence score $s_t$ measuring how much token $t$ moves the model toward generating accepted responses and away from rejected ones.
2. **Score-to-Weight Conversion** — convert $s_t$ into a per-token weight or correction applied to the PG objective in subsequent PPO epochs.

---

## Stage 1: Per-Token Influence Scoring

### Setup

For each prompt, the model generates $n$ responses (e.g., 16). A binary reward labels each response as accepted or rejected. In PPO epoch 0, we compute influence scores; in epochs 1+, we apply the resulting weights.

### Step 1: Projected Gradient Capture

For each linear layer $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, we define two fixed random Rademacher projection matrices:

$$P_\text{in} \in \{-1,+1\}^{k_\text{in} \times d_\text{in}}, \quad P_\text{out} \in \{-1,+1\}^{k_\text{out} \times d_\text{out}}$$

scaled by $1/\sqrt{k}$, where $k_\text{in} = d_\text{in}/f$, $k_\text{out} = d_\text{out}/f$, and $f$ is the projection factor (default 64).

During the forward and backward pass of a **plain log-probability objective** $L = \sum_t \log \pi_\theta(a_t | s_t)$ (no reward weighting — the accepted/rejected distinction is handled entirely by contrastive scoring), we capture per-token projected gradients:

$$v_t = x_t \cdot P_\text{in}^\top \in \mathbb{R}^{k_\text{in}}, \quad u_t = \frac{\partial L}{\partial y_t} \cdot P_\text{out}^\top \in \mathbb{R}^{k_\text{out}}$$

The projected per-token gradient is the rank-1 outer product:

$$g_t = \text{vec}(u_t \otimes v_t^\top) \in \mathbb{R}^D, \quad D = k_\text{out} \times k_\text{in}$$

### Step 2: TrackStar-Style Unit Normalization (Optional)

Following [TrackStar (2410.17413v2)](https://arxiv.org/abs/2410.17413), per-token gradient vectors can be unit-normalized before scoring:

$$\hat{g}_t = \frac{g_t}{\|g_t\| + \epsilon}$$

This converts dot-product influence into cosine similarity, preventing a handful of high-gradient-magnitude tokens from dominating the score distribution. **Raw** (unnormalized) gradients are still used for Hessian construction (which captures parameter-space curvature).

Config: `token_unit_norm=True`.

### Step 3: Response-Level Gradient Aggregation

Sum (normalized) token gradients within each response $r$:

$$g_r = \sum_{t \in r} \hat{g}_t$$

### Step 4: Projected Hessian Inverse

Build and invert the Gauss-Newton Hessian in the projected space:

$$H = G^\top G + \lambda I \in \mathbb{R}^{D \times D}$$

where $G$ is the matrix of response gradients $[g_{r_1}; \ldots; g_{r_n}]$, and $\lambda$ is auto-scaled as $0.1 \cdot \text{tr}(H)/D$. Solve via Cholesky:

$$S = H^{-1} G_\text{tok}^\top \in \mathbb{R}^{D \times T}$$

where $G_\text{tok}$ is the matrix of all **raw** token gradients (Hessian uses raw gradients even when `token_unit_norm=True`).

### Step 5: Contrastive Influence Score

Compute the influence of each token on each response, then contrast:

$$\text{infl}_{r,t} = g_r \cdot S_{:,t} = g_r^\top H^{-1} g_t$$

$$s_t = \frac{1}{|\mathcal{A}|}\sum_{r \in \mathcal{A}} \text{infl}_{r,t} - \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} \text{infl}_{r,t}$$

where $\mathcal{A}$, $\mathcal{R}$ are the accepted and rejected response sets. A high $s_t$ means token $t$ pushes the model toward accepted responses.

### Contrastive Scope

The accepted/rejected sets can be defined at different granularities:

| Scope | $\mathcal{A}$, $\mathcal{R}$ | Pros | Cons |
|-------|------|------|------|
| `per_prompt` | Same prompt only | Clean signal per prompt | Few accepted responses when solve rate is low → noisy $d$ |
| `all_selected` | All responses on local GPU | More responses → stable $d$ | Cross-prompt contamination |
| `global_selected` | All responses across all GPUs (all-reduce) | Maximum data | Communication cost; cross-prompt contamination |

### Multi-Module Aggregation

Scores from all linear modules (Q, K, V, O, gate, up, down projections) are summed per token.

---

## Stage 2: Score-to-Weight Conversion

### Score Semantics

The influence score $s_t$ encodes a **signed** signal:
- $s_t > 0$: increasing token $t$'s log-prob promotes accepted responses (beneficial token)
- $s_t < 0$: increasing token $t$'s log-prob promotes rejected responses (harmful token)

This sign is meaningful for both accepted and rejected responses.

### Weighting Modes

Raw scores are first **z-score normalized** per response: $z_t = (s_t - \mu) / \sigma$.

#### Additive Mode (Multiplicative Application)

$$w_t = 1 + \lambda \cdot z_t, \quad w_t \in [\text{clamp\_min}, \text{clamp\_max}]$$

Applied multiplicatively: $A'_t = A_t \times w_t$.

**Sign flip option**: For rejected responses, the score can be negated before z-scoring ($z_t$ computed from $-s_t$). This inverts the weighting: tokens that promote correct answers get $w < 1$ (less suppression) in rejected responses.

**`apply_to` variants**:
- `all` — sign flip for rejected (dense credit assignment)
- `noflip_neg` — no sign flip for rejected (saliency/importance weighting)
- `positive` — only weight accepted responses
- `negative` — only weight rejected responses

#### Credit Mode (Additive Application) — v5

$$\text{correction}_t = \lambda \cdot z_t, \quad \text{correction}_t \in [\text{clamp\_min}, \text{clamp\_max}]$$

Applied additively: $A'_t = A_t + \text{correction}_t$.

Score sign is flipped for rejected responses (same dense credit semantics as multiplicative sign-flip). The key difference is that the correction magnitude is **independent of $A_t$**, preventing the instability caused by multiplicative interaction with large $|A_t|$.

#### Legacy Modes

- **Tanh**: $w_t = 1 + \alpha \cdot \tanh(z_t^{\text{eff}} / \tau)$, clamped to $[0.5, 2.0]$.
- **Softmax**: $w_t \propto \exp(z_t^{\text{eff}} / T)$, normalized so $\sum w_t = n_\text{tokens}$.
- **Ratio**: $w_t = s_t / \mu_r$ (proposition corollary), with SNR threshold fallback.
- **Direct**: $w_t = 1 + \lambda \cdot s_t$ (unnormalized).

### Application

Depending on the mode, weights are applied to the clipped PG objective:

**Multiplicative** (additive, direct, ratio, tanh, softmax):

$$L_\text{weighted} = \sum_t w_t \cdot \min\left(\rho_t \hat{A}_t, \text{clip}(\rho_t) \hat{A}_t\right) \cdot m_t$$

**Additive** (credit):

$$L_\text{weighted} = \sum_t \min\left(\rho_t (\hat{A}_t + c_t), \text{clip}(\rho_t) (\hat{A}_t + c_t)\right) \cdot m_t$$

Weights are applied in PPO epochs 1+ (epoch 0 is used for influence capture).

---

## Diagnostic Metrics

To evaluate influence score quality **before** running full training, we use 1-step diagnostic runs that emit NPZ traces.

### GapDir% (Gap Correct Direction)

For each prompt $i$, compute the gap between accepted and rejected response-level mean influence:

$$\text{gap}_i = \frac{1}{|\mathcal{A}_i|}\sum_{r \in \mathcal{A}_i} \bar{s}_r - \frac{1}{|\mathcal{R}_i|}\sum_{r \in \mathcal{R}_i} \bar{s}_r$$

where $\bar{s}_r = \text{mean}_{t \in r}(s_t)$ is the mean token influence of response $r$.

$$\text{GapDir\%} = \frac{|\{i : \text{gap}_i > 0\}|}{|\text{prompts}|}$$

**Interpretation**: Fraction of prompts where accepted responses have higher mean influence than rejected ones. Random baseline = 50%.

### GapSNR (Gap Signal-to-Noise Ratio)

$$\text{GapSNR} = \frac{|\text{mean}_i(\text{gap}_i)|}{\text{std}_i(\text{gap}_i)}$$

**Interpretation**: Even if most prompts have correct gap direction, large variance means the signal is unreliable. Higher SNR = more consistent signal. Random baseline $\approx 0$.

### Corr (Response-Reward Correlation)

Pearson correlation between per-response mean influence $\bar{s}_r$ and binary reward $y_r \in \{0, 1\}$ across all responses.

**Interpretation**: Overall alignment between influence magnitude and correctness. Positive = influence correctly tracks reward.

### Reference Results (1-step diagnostic, Qwen3-1.7B-Base)

| Config | GapDir% | GapSNR | Corr |
|--------|---------|--------|------|
| AllSel-P64 (DSR1) | 70.7% | 0.389 | 0.119 |
| PP-P64 (DSR1) | 58.5% | 0.265 | 0.118 |
| AllSel-P64 + UnitNorm (DSR1) | — | — | — |

---

## Experimental Findings

### Multiplicative Sign Flip Causes Entropy Explosion

When using multiplicative weighting ($A'_t = A_t \times w_t$) with sign flip for rejected responses, training consistently diverges with entropy explosion, regardless of clamp range:

| Config | Clamp | $\lambda$ | Outcome |
|--------|-------|-----------|---------|
| Additive, sign flip (v2) | $[0, 2]$ | 0.03 | Entropy explosion |
| Additive, sign flip, tight (v2) | $[0.75, 1.25]$ | 0.01 | Entropy explosion |
| Additive, pos-only (v4) | $[0, 2]$ | 0.03 | Entropy explosion |

**Root cause**: In the multiplicative form $A_t \times w_t$, the sign flip creates an asymmetric bias. For rejected responses ($A_t < 0$), tokens with $w_t < 1$ have their suppression reduced — but this reduction is proportional to $|A_t|$, which amplifies noise. The asymmetry introduces a one-sided bias that increases token probabilities regardless of whether they should be suppressed, creating a positive feedback loop: entropy rises → solve rate drops → noisier scores → more entropy.

### No-Flip Multiplicative Weighting Is Stable

Without sign flip (`apply_to=noflip_neg`), both accepted and rejected responses get the same saliency-based weighting (high-influence tokens get higher weight). This is symmetric — the amplification on accepted and rejected sides cancels out — so it cannot create a one-sided entropy bias.

| Config | Clamp | $\lambda$ | Outcome |
|--------|-------|-----------|---------|
| Additive, no-flip (v1/v4) | $[0, 2]$ | 0.03 | Stable, outperforms baseline |

**Limitation**: No-flip is "discriminative importance weighting" (which tokens matter), not "credit assignment" (which tokens are good/bad). The sign of $s_t$ is unused.

### Credit Mode (Additive Application) Enables Stable Sign Flip

The additive form $A'_t = A_t + \lambda \cdot z_t$ decouples the correction from $A_t$ magnitude, preventing the multiplicative amplification that causes divergence:

| Config | Clamp | $\lambda$ | Outcome |
|--------|-------|-----------|---------|
| Credit, sign flip (v5) | $[-0.5, 0.5]$ | 0.03 | Stable |
| Credit, sign flip, wide (v5) | $[-1.0, 1.0]$ | 0.03 | In progress |
| Credit, sign flip, narrow (v5) | $[-0.1, 0.1]$ | 0.01 | In progress |

**Key insight**: The instability from sign flip is not about the *direction* of the credit signal, but about the *multiplicative interaction* with advantage magnitude. Additive credit avoids this by construction.

---

## Configuration Reference

### Influence Trace (`trainer.influence_trace.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable influence scoring |
| `hessian_mode` | `inverse` | `inverse` (Cholesky solve) or `identity` (no Hessian) |
| `accepted_rejected_scope` | `per_prompt` | `per_prompt`, `all_selected`, or `global_selected` |
| `projection_dim_factor` | 64 | Projection dimension = layer_dim / factor |
| `token_unit_norm` | `False` | TrackStar-style unit normalization |
| `contrastive_agg` | `mean` | `mean` or `sum` for contrastive direction |
| `hessian_source` | `token` | `token` or `response` for Hessian construction |
| `exclude_self_response` | `True` | LOO correction: exclude token's own response from scoring |
| `module_name_filter` | all | Which linear layers to include |
| `reg_lambda` | -1.0 | Hessian regularization (-1 = auto-scale) |

### Influence Token Weight (`trainer.influence_token_weight.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `False` | Enable token reweighting |
| `mode` | `zero` | `additive`, `credit`, `direct`, `ratio`, `tanh`, `softmax`, etc. |
| `apply_epochs` | `[1, 2]` | Which PPO epochs to apply weights |
| `additive_lambda` | 0.5 | Scaling factor for z-scored influence |
| `additive_clamp_min` | -1.0 | Minimum weight/correction |
| `additive_clamp_max` | 3.0 | Maximum weight/correction |
| `apply_to` | `all` | `all` (sign flip), `noflip_neg`, `positive`, `negative` |
| `adv_target` | `advantage` | `advantage` (modify A) or `loss` (modify loss directly) |

---

## Code Pointers

| Component | File | Key Lines |
|-----------|------|-----------|
| Influence scoring | `verl/workers/actor/influence_trace.py` | `_module_token_scores()`, `_module_token_scores_identity()` |
| Token weight computation | `verl/workers/actor/influence_token_weight.py` | `build_token_loss_weights()` |
| Weight application | `verl/workers/actor/dp_actor.py` | `advantages *= token_weights` / `advantages += token_weights` |
| Config passing | `dapo/dapo_ray_trainer.py` | influence config whitelist |
