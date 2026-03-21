# Compatible Token-Advantage Residual for GRPO

## Purpose

This note justifies a token-level credit design for GRPO that does **not** rely on the sign of the response-mean influence score. The key idea is to treat the response-level advantage as the trusted coarse signal and use token influence only for **within-response redistribution**.

The recommended form is:

$$
z_{it} = \frac{s_{it} - \mu_i}{\sigma_i + \varepsilon},
\qquad
r_{it} = \psi(z_{it}),
\qquad
\tilde r_{it} = r_{it} - \frac{1}{T_i}\sum_{u=1}^{T_i} r_{iu},
$$

$$
A'_{it} = A_i + \lambda_i \tilde r_{it}.
$$

Here:

- $i$ indexes a response.
- $t$ indexes a token inside response $i$.
- $s_{it}$ is the token influence score produced by the current tracer.
- $A_i$ is the response-level GRPO advantage already used by PPO.
- $\psi$ is a bounded monotone transform such as `tanh` or clipped identity.

The central claim is not that $s_{it}$ recovers the true token-level advantage exactly. The strongest defensible claim is:

> After prompt-local contrastive scoring and response-wise standardization, the transformed score behaves like a noisy estimator of the **compatible token-advantage residual**, and the additive residual rule is the minimum-change token-level refinement of the original GRPO advantage under a mean-preserving constraint.

## 1. Setup

Consider an autoregressive policy

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid h_t),
$$

with token score feature

$$
g_{it} := \nabla_\theta \log \pi_\theta(y_{it} \mid h_{it}).
$$

Define the response-level score feature

$$
G_i := \sum_{t=1}^{T_i} g_{it}.
$$

Let $A_{it}^\star$ denote the true causal token-level advantage. Then the exact policy-gradient target is

$$
\nabla J(\theta)
=
\mathbb{E}\left[\sum_{i,t} g_{it} A_{it}^\star \right].
$$

Decompose token advantage into a response-level mean plus a zero-mean residual:

$$
A_{it}^\star = \bar A_i^\star + \delta_{it}^\star,
\qquad
\frac{1}{T_i}\sum_{t=1}^{T_i} \delta_{it}^\star = 0.
$$

For GRPO/PPO, the available scalar advantage $A_i$ is a response-level signal. It is therefore natural to let the token-level module estimate only the redistribution term $\delta_{it}^\star$.

## 2. Compatible Token Advantage

As in compatible function approximation for actor-critic methods, the most natural object to recover is not the exact $A_{it}^\star$, but its projection onto the token score-feature span.

For a prompt-conditioned Fisher matrix

$$
F_x := \mathbb{E}[g_{it} g_{it}^\top \mid x],
$$

define the compatible token-advantage parameter

$$
w_x^\star := F_x^{-1}\,\mathbb{E}[g_{it} A_{it}^\star \mid x],
$$

and the compatible token advantage

$$
A_{it}^{\mathrm{comp}} := g_{it}^\top w_x^\star.
$$

This is the $L^2(F_x)$ projection of the true token advantage onto linear functions of the score feature. It is the strongest token-level target one can hope to justify from score-feature geometry alone.

Likewise, define the compatible residual

$$
\delta_{it}^{\mathrm{comp}}
:=
A_{it}^{\mathrm{comp}}
-
\frac{1}{T_i}\sum_{u=1}^{T_i} A_{iu}^{\mathrm{comp}}.
$$

This is the object the additive residual scheme tries to approximate.

## 3. What the Current Influence Score Estimates

The current tracer computes a token score of the form

$$
s_{it} = g_{it}^\top \hat H^{-1} d_{-i},
$$

where:

- $\hat H$ is the projected empirical curvature matrix used by the tracer.
- $d_{-i}$ is a leave-one-out accepted-vs-rejected contrastive direction.

In the cleanest setting (`per_prompt + mean + exclude_self_response=True`), assume a prompt-local single-index condition:

$$
\mathbb{E}[d_{-i} \mid x]
=
c_x F_x w_x^\star + \eta_x,
$$

where $c_x > 0$ is a scalar and $\eta_x$ is a nuisance term.

Then

$$
\mathbb{E}[s_{it} \mid x]
=
\mathbb{E}[g_{it}^\top F_x^{-1} d_{-i} \mid x]
=
c_x A_{it}^{\mathrm{comp}}
+
g_{it}^\top F_x^{-1}\eta_x.
$$

So the influence score is not arbitrary. Under prompt-local contrastive alignment, it is a plug-in estimator for the compatible token advantage up to prompt- or response-specific nuisance terms.

This is the correct theoretical interpretation:

> The current token influence score is best viewed as an estimator of a compatible token-advantage direction, not a direct estimator of the exact token-level advantage.

## 4. Why Response-Wise Centering and Scaling Are Justified

The failure mode observed in practice is that the response-level mean

$$
\mu_i := \frac{1}{T_i}\sum_t s_{it}
$$

often has an unreliable sign. This breaks multiplicative schemes that require accepted responses to have $\mu_i > 0$ and rejected responses to have $\mu_i < 0$.

That failure does **not** imply the token ordering inside the response is useless. A more realistic model is

$$
s_{it}
=
\alpha_i A_{it}^{\mathrm{comp}} + \beta_i + \xi_{it},
\qquad
\alpha_i > 0,
$$

where:

- $\alpha_i$ is a response-specific scale factor.
- $\beta_i$ is a response-level offset.
- $\xi_{it}$ is token noise.

The nuisance terms $\alpha_i$ and $\beta_i$ are exactly what multiplicative sign-dependent schemes are sensitive to.

Now define

$$
z_{it} := \frac{s_{it} - \mu_i}{\sigma_i + \varepsilon},
\qquad
\sigma_i^2 := \frac{1}{T_i}\sum_t (s_{it} - \mu_i)^2.
$$

Under the affine nuisance model above,

$$
z_{it}
\approx
\frac{A_{it}^{\mathrm{comp}} - \bar A_i^{\mathrm{comp}}}{\mathrm{std}(A_i^{\mathrm{comp}})}
+
\text{normalized noise}.
$$

So response-wise centering and scaling remove the two nuisance degrees of freedom that are most likely to be unstable:

- response-level offset $\beta_i$,
- response-level scale $\alpha_i$.

This is the main reason to prefer standardized within-response scores over raw scores when building token-level credit.

## 5. Why Additive Residual Is Better Than Multiplicative Weight

A multiplicative scheme uses

$$
A'_{it} = A_i\, w_{it}.
$$

This couples two conceptually different objects:

- the response-level sign and magnitude carried by $A_i$,
- the within-response token redistribution carried by $w_{it}$.

If $w_{it}$ depends on the sign of the raw response mean $\mu_i$, then any response-level sign error directly corrupts token credit.

The additive residual scheme instead uses

$$
A'_{it} = A_i + \lambda_i \tilde r_{it},
$$

which separates:

- coarse response-level credit: $A_i$,
- fine token-level redistribution: $\tilde r_{it}$.

This is structurally more aligned with the decomposition

$$
A_{it}^\star = \bar A_i^\star + \delta_{it}^\star.
$$

## 6. Constrained Optimization Derivation

The additive rule follows from a simple variational principle.

Suppose for a fixed response $i$ we want token-level advantages $a_{it}$ that:

1. stay close to the original response advantage $A_i$,
2. align with a token ranking signal $u_{it}$,
3. preserve the response mean.

Consider the problem

$$
\max_{a_i}
\sum_{t=1}^{T_i} a_{it} u_{it}
-
\frac{1}{2\lambda_i}
\sum_{t=1}^{T_i}(a_{it} - A_i)^2
$$

subject to

$$
\frac{1}{T_i}\sum_{t=1}^{T_i} a_{it} = A_i.
$$

If $u_{it}$ has zero mean within the response, the KKT solution is

$$
a_{it} = A_i + \lambda_i u_{it}.
$$

Therefore, once we construct a centered token signal $\tilde r_{it}$, the additive residual form

$$
A'_{it} = A_i + \lambda_i \tilde r_{it}
$$

is not heuristic. It is the minimum-change mean-preserving solution that pushes token-level credit in the direction preferred by the influence ranking.

## 7. Why We Need a Bounded Transform

A raw standardized score $z_{it}$ may still contain large outliers. To stabilize training, use a bounded monotone map

$$
r_{it} = \psi(z_{it}),
$$

with either:

$$
\psi(z) = \tanh(z / \tau)
$$

or

$$
\psi(z) = \mathrm{clip}(z, -c, c).
$$

Then recenter:

$$
\tilde r_{it}
=
r_{it}
-
\frac{1}{T_i}\sum_{u=1}^{T_i} r_{iu}.
$$

This recentering step is essential. If clipping or `tanh` is applied before the final mean subtraction, the response average is no longer guaranteed to stay unchanged.

The resulting token-level target is

$$
A'_{it} = A_i + \lambda_i \tilde r_{it},
$$

with the exact invariant

$$
\frac{1}{T_i}\sum_{t=1}^{T_i} A'_{it} = A_i.
$$

That invariant is one of the main advantages of the additive residual design.

## 8. PPO/GRPO Interpretation

Using $A'_{it}$ inside the clipped PPO objective yields

$$
L_i^{\mathrm{PG}}
=
\sum_t
\min\Big(
\rho_{it} A'_{it},
\mathrm{clip}(\rho_{it}) A'_{it}
\Big).
$$

This has a clean interpretation:

- response-level trust still comes from $A_i$,
- token influence only reallocates gradient mass across tokens,
- the average response-level signal is preserved exactly.

So the design does **not** attempt to replace GRPO with a new token-level RL target. It refines GRPO by injecting a mean-preserving compatible residual.

## 9. Recommended Practical Form

For each response $i$:

1. Compute valid-token influence scores $s_{it}$.
2. Standardize within the response:

   $$
   z_{it} = \frac{s_{it} - \mu_i}{\sigma_i + \varepsilon}.
   $$

3. Apply a robust transform:

   $$
   r_{it} = \tanh(z_{it} / \tau)
   $$

   or

   $$
   r_{it} = \mathrm{clip}(z_{it}, -c, c).
   $$

4. Recenter:

   $$
   \tilde r_{it} = r_{it} - \mathrm{mean}(r_i).
   $$

5. Form token advantages:

   $$
   A'_{it} = A_i + \lambda_i \tilde r_{it}.
   $$

Recommended safeguards:

- If fewer than 2 valid tokens exist, fall back to $A'_{it} = A_i$.
- If $\sigma_i$ is too small, fall back to $A'_{it} = A_i$.
- Use a confidence-scaled $\lambda_i$ when influence quality is uncertain.

## 10. Optional Confidence Scaling

The additive residual rule does not require the sign of $\mu_i$ to be correct, but it still benefits from a confidence term:

$$
A'_{it} = A_i + \lambda_0 \, c_i \, \tilde r_{it},
\qquad
0 \le c_i \le 1.
$$

Possible choices for $c_i$ include:

- a response-level reliability score from trace diagnostics,
- a prompt-level accepted-vs-rejected separation score,
- a small-step schedule that increases $\lambda_0$ only after influence statistics stabilize.

The theory above does not force a unique confidence definition. It only requires that $c_i$ be treated as a reliability multiplier, not as a replacement for the response-level advantage.

## 11. Main Takeaway

The best theoretical statement is:

1. The current token influence score is interpretable as a noisy estimator of a **compatible token-advantage direction**.
2. Response-wise centering and scaling remove unstable response-level nuisance terms.
3. The additive rule

   $$
   A'_{it} = A_i + \lambda_i \tilde r_{it}
   $$

   is the mean-preserving minimum-change refinement of the original GRPO advantage.
4. Therefore, if we trust response-level GRPO advantage at the coarse level and trust token influence only for **relative within-response ranking**, additive residual is the cleanest design.

## 12. Design Implication for Archer2.0

For Archer2.0, the most defensible next step is:

- keep the current influence scorer focused on prompt-local contrastive signal,
- stop requiring the raw response-mean influence sign to be correct,
- replace multiplicative token weights with a token-level additive residual applied to the response advantage.

This directly targets the part of the token credit problem that influence scores are most capable of estimating: **within-response redistribution**, not absolute response-level sign.
