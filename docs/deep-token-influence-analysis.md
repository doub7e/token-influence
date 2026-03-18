# Deep Token-Level Influence Analysis

**Model**: Qwen3-4B-Base, step 300→301, 192 responses (81 acc, 111 rej, 12 prompt groups)
**Configs**: lastmlp-f16 (last MLP, factor=16), all-layers-f32 (252 modules), lmhead-f128, lastmlp-f32

## Executive Summary

Token-level influence scores contain **genuine per-token information** (97-99% of variance is within-response residual, not response-mean), but this information is **not interpretable as per-token credit**. Key findings:

1. **Cross-config agreement is low** — different module choices produce very different token rankings
2. **Entropy is uncorrelated with influence** — model uncertainty ≠ influence magnitude
3. **Boxed/answer tokens have NEGATIVE influence** across all configs
4. **Late-position AUC is highest** (0.76), confirming cleaner signal near final answer
5. **Transition words (Therefore, So) have 2.4× the acc/rej gap** compared to other tokens
6. **The fraction-positive feature (>50% positive → accepted)** is as discriminative as the mean

---

## 1. Cross-Config Token Ranking Agreement

| Config Pair | Spearman ρ | Top-20% Jaccard |
|-------------|-----------|-----------------|
| lastmlp-f16 vs lastmlp-f32 | **0.839** | 0.536 |
| lastmlp-f16 vs lmhead-f128 | 0.384 | 0.239 |
| lastmlp-f16 vs all-layers-f32 | 0.201 | 0.178 |
| all-layers-f32 vs lmhead-f128 | 0.148 | 0.155 |

**Insight**: Same-module-different-projection configs (lastmlp-f16 vs f32) agree strongly (ρ=0.84), confirming the signal is from the module, not random projection noise. But **different modules disagree** (ρ=0.15-0.38). The all-layers config captures a fundamentally different signal from last-layer-only configs — this is the cross-position contamination effect documented in `per-token-gradient-semantics.md`.

**Cross-level agreement**: Response-level Pearson between lastmlp and all-layers is 0.619, but token-level Pearson is only 0.236. This confirms: **contaminated configs agree on WHICH responses are good, but disagree on WHICH tokens are responsible**.

## 2. Entropy-Influence Correlation

| Entropy Bucket | Gap (acc-rej) lastmlp-f16 | Gap all-layers |
|---------------|--------------------------|----------------|
| Very low (0-0.1) | +0.0022 | +0.136 |
| Low (0.1-0.5) | **+0.0046** | **+0.516** |
| Medium (0.5-1.0) | +0.0030 | +0.360 |
| High (1.0-2.0) | +0.0034 | +0.355 |
| Very high (2.0-5.0) | +0.0014 | +0.141 |

**Insight**: Low-entropy tokens (confident predictions) have the **second-best** discrimination. The best discrimination is at **low entropy (0.1-0.5)** — tokens where the model is "slightly uncertain." Very high entropy tokens have the weakest signal.

Per-response entropy-|influence| correlation is essentially zero (−0.03 for acc, −0.07 for rej). **Entropy and influence are orthogonal signals** — influence is not simply measuring prediction confidence.

## 3. Error Localization

| Config | Mean Drop Position | Median Drop Position | % in First 20% |
|--------|-------------------|---------------------|-----------------|
| lastmlp-f16 | 0.375 | 0.304 | 39.6% |
| all-layers-f32 | 0.269 | **0.081** | **64.0%** |
| lmhead-f128 | 0.412 | 0.348 | 27.0% |
| lastmlp-f32 | 0.401 | 0.314 | 36.9% |

**Insight**: All-layers config places the maximum influence drop in the **first 20% of the response** for 64% of rejected responses — this is the trajectory contamination: early tokens carry the signal of the entire (bad) future trajectory. The cleaner lastmlp configs spread the drop more uniformly, with lmhead-f128 being the most uniform (27% in first 20%).

**This confirms**: For genuine error localization, last-layer-only configs are more appropriate, even though their overall AUC is lower. The all-layers config gets better AUC by "cheating" — giving early tokens future information.

## 4. Answer Token (\\boxed) Influence

| Config | Boxed Token Gap | Near-Boxed Gap | Other Token Gap |
|--------|----------------|----------------|-----------------|
| lastmlp-f16 | **−0.003** | +0.004 | +0.003 |
| all-layers-f32 | +0.107 | +0.249 | +0.214 |
| lmhead-f128 | **−0.004** | +0.003 | +0.002 |
| lastmlp-f32 | **−0.005** | +0.003 | +0.001 |

**Major Insight**: Across all clean configs (lastmlp, lmhead), **boxed tokens have NEGATIVE acc/rej gap** — meaning boxed tokens in accepted responses have *lower* influence than in rejected responses! This is counterintuitive.

**Explanation**: The `\boxed` token itself is a formatting token that occurs in both correct and incorrect answers. Its gradient reflects the prediction difficulty of the answer digits that follow, not the correctness. In accepted responses, the answer is often a "predictable" continuation, yielding small gradients. In rejected responses, the wrong answer may be less predictable, yielding larger gradients.

The **near-boxed tokens** (±10 positions) DO have positive gap, suggesting the reasoning leading to and following the answer carries the real signal.

## 5. Distribution Shape

| Config | Kurtosis (acc) | Kurtosis (rej) | Overlap |
|--------|---------------|----------------|---------|
| lastmlp-f16 | +0.38 | +0.50 | 0.459 |
| all-layers-f32 | +3.91 | +3.03 | 0.435 |
| lmhead-f128 | +0.55 | +0.75 | 0.466 |
| lastmlp-f32 | +0.24 | +0.42 | 0.488 |

**Insight**: Clean configs (lastmlp, lmhead) have near-Gaussian distributions (kurtosis < 1). The all-layers config is **heavy-tailed** (kurtosis ≈ 3-4), meaning a few tokens get extreme scores. In all configs, **rejected distributions have slightly higher kurtosis** — rejected responses have more extreme-scored tokens.

Overlap is 43-49%: there is no clean separation at the token level. Even the best config has nearly half of rejected tokens scoring above the accepted median.

## 6. Response-Level vs Token-Level Decomposition

| Config | Response-Mean Var (%) | Residual Var (%) | Residual-Std AUC |
|--------|----------------------|-----------------|------------------|
| lastmlp-f16 | 3.3% | **97.4%** | 0.644 |
| all-layers-f32 | 4.6% | 96.1% | 0.638 |
| lmhead-f128 | 1.7% | **98.5%** | **0.703** |
| lastmlp-f32 | 1.5% | 98.8% | 0.596 |

**Key Insight**: 97-99% of token-level variance is **within-response residual**, not response-mean signal. The scores are NOT just "every token in accepted response gets +x, every token in rejected response gets −x."

Moreover, the **residual standard deviation itself is discriminative** (AUC 0.60-0.70) — accepted responses have higher within-response spread than rejected ones. This means accepted responses have more token-level variation in their influence scores, possibly because correct reasoning involves more diverse gradient patterns.

## 7. Influence Gradient (Trend Along Sequence)

| Config | Acc Slope | Rej Slope | Acc Late-Early Gap | Rej Late-Early Gap | Slope AUC |
|--------|-----------|-----------|-------------------|-------------------|-----------|
| lastmlp-f16 | −1e-6 | −1e-6 | −0.0008 | −0.0016 | 0.558 |
| all-layers-f32 | +8e-5 | −1e-4 | +0.085 | −0.056 | 0.571 |

**Insight**: For the clean config (lastmlp), there is **no systematic slope** — influence does not consistently increase or decrease along the sequence. For all-layers, accepted responses trend upward (late tokens more positive) while rejected trend downward — but this is the contamination effect again.

## 8. Position-Dependent AUC (Fine-Grained Deciles)

| Decile | lastmlp-f16 AUC | all-layers AUC |
|--------|----------------|----------------|
| 0-10% | 0.711 | 0.672 |
| 10-20% | 0.707 | 0.654 |
| 20-30% | 0.676 | 0.656 |
| 30-40% | 0.670 | 0.668 |
| 40-50% | 0.718 | 0.664 |
| 50-60% | 0.698 | 0.674 |
| 60-70% | 0.687 | 0.659 |
| 70-80% | 0.681 | 0.695 |
| 80-90% | **0.737** | 0.705 |
| **90-100%** | **0.762** | **0.764** |

**Key Insight**: For lastmlp-f16, AUC is U-shaped: high at start (0.71), dips in the middle (0.67), and **peaks at the end** (0.76). This makes sense:
- **Early tokens**: carry prompt-continuation signal (partially shared across responses)
- **Middle tokens**: intermediate reasoning, most noisy
- **Late tokens** (90-100%): near the final answer, cleanest per-token signal

Both configs converge to the same AUC (0.76) at the last decile — **at the very end, even contaminated configs become clean** because there are no future tokens to leak signal from.

## 9. Transition Token Analysis

| Category | Acc/Rej Gap | Other Token Gap | Ratio |
|----------|-----------|----------------|-------|
| **Transition words** (Therefore, So, ...) | **+0.0063** | +0.0026 | **2.4×** |
| Equals sign (=) | −0.0027 | +0.0026 | negative! |

**Insight**: Reasoning transition words have **2.4× stronger acc/rej discrimination** than average tokens. These tokens mark where the model commits to a reasoning step — in accepted responses they align with correct reasoning, in rejected they don't.

The equals sign (=) has **negative gap** — similar to boxed tokens, it's a structural token that doesn't carry credit. The actual discrimination happens at the tokens expressing the reasoning, not the structural/formatting ones.

## 10. Feature-Level Discrimination Power

| Feature | lastmlp AUC | all-layers AUC |
|---------|------------|----------------|
| resp_mean | **0.749** | **0.779** |
| frac_positive | 0.715 | 0.771 |
| resp_median | 0.714 | 0.774 |
| token_std | 0.604 | 0.636 |
| p90−p10 range | 0.602 | 0.617 |

**Insight**: The **fraction of positive tokens** (frac_positive) is almost as discriminative as the response mean (AUC 0.715 vs 0.749). This suggests a simple binary signal: accepted responses have **more tokens with positive influence than negative**, regardless of magnitude. For a token-weighting scheme, this validates the mask-based approach (set negative-influence tokens to 0).

## 11. Concentration (Gini)

| Config | Gini (acc) | Gini (rej) | Top-10% Share (acc) | Top-10% Share (rej) |
|--------|-----------|-----------|--------------------|--------------------|
| lastmlp-f16 | 0.426 | 0.433 | 26.7% | 27.3% |
| all-layers | 0.461 | 0.456 | 30.6% | 30.0% |

**Insight**: Influence is moderately concentrated (Gini ≈ 0.43, top 10% of tokens capture 27% of total |influence|). This is moderate — not dominated by a few tokens, but not uniform either.

Rejected responses have slightly higher Gini (more concentrated) — a few tokens carry disproportionate influence, possibly the error-inducing tokens.

---

## Actionable Conclusions for Token Credit Assignment

1. **Use fraction-positive as a simpler proxy**: Instead of continuous influence scores, the binary sign of per-token influence (positive vs negative) captures most of the useful signal. This aligns with the mask-based approach.

2. **Focus weighting on late-response tokens**: The last 10-20% of tokens have the cleanest signal (AUC 0.76). Consider applying token-level modifications only to late-position tokens, leaving early tokens at the response-level advantage.

3. **Weight transition tokens more heavily**: Reasoning transition words carry 2.4× more discrimination. These are natural candidates for per-token advantage modification.

4. **Don't trust structural token scores**: `\boxed`, `=`, formatting tokens have unreliable or inverted signals. Exclude them from token-level weighting.

5. **Use within-response spread as a secondary signal**: The std of token-level influence within a response is weakly discriminative (AUC 0.60-0.70). Responses with higher spread may benefit more from token-level weighting.

6. **The "clean" per-token signal is weak**: Even the best position (last decile, AUC 0.76) still has 46% overlap between acc and rej token distributions. Token-level credit is inherently noisy — any weighting scheme must be robust to this noise.
