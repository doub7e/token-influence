"""Influence-weighted token loss: modulate per-token PG loss using influence scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class InfluenceTokenWeightConfig:
    """Configuration for influence-based token loss weighting."""

    enabled: bool = False
    threshold: float = 1.0  # z-score threshold for masking (used by zero/flip/soft)
    mode: str = "zero"  # "zero" | "flip" | "soft" | "softmax" | "tanh" | "direct"
    normalize: str = "zscore"  # "zscore" or "none" (ignored by direct)
    apply_epochs: list[int] = field(default_factory=lambda: [1, 2])
    # softmax mode params
    softmax_temperature: float = 1.0
    weight_clamp_min: float = 0.1
    weight_clamp_max: float = 5.0
    # tanh mode params: w = 1 + alpha * tanh(z / tau), non-competitive
    tanh_alpha: float = 0.5  # modulation strength: w ∈ [1-alpha, 1+alpha] before clamp
    tanh_tau: float = 1.0  # temperature for tanh input scaling
    # direct mode params: m_t = 1 + λ * score_t, applied to advantage
    # Requires score_normalization="h_inv_norm" in tracer to remove gradient magnitude.
    direct_lambda: float = 0.3  # scaling factor for influence score
    direct_clamp_min: float = 0.0  # min weight (0 = no sign flip, <0 = allow reversal)
    direct_clamp_max: float = 2.0  # max weight
    adv_target: str = "advantage"  # "advantage" or "loss" — where to apply weights
    # ratio mode params: w_t = s_t / μ_i (proposition corollary, exact credit)
    # No λ needed, no sign orientation needed — ratio handles sign naturally.
    ratio_clamp_min: float = -1.0  # min weight after ratio (<0 allows sign flip)
    ratio_clamp_max: float = 3.0  # max weight after ratio
    ratio_snr_threshold: float = 0.3  # fallback to uniform when |μ_i/σ_i| < threshold
    # additive mode params: w_t = 1 + λ * z_t where z_t = (s_t - mean) / std
    # Redistributes response advantage budget by token influence. mean(w_t) = 1.
    additive_lambda: float = 0.5  # redistribution strength
    additive_clamp_min: float = -1.0  # min weight (< 0 allows sign flip)
    additive_clamp_max: float = 3.0  # max weight
    apply_to: str = "all"  # "all" | "positive" | "negative" — which responses to weight
    # return mode params: G_t = Σ_{k=t}^T γ^{k-t} s_k (suffix-sum return-to-go)
    # Treats influence scores as token-level rewards; computes RL-style return.
    # Applied as additive correction: A'_t = A_t + λ * G_t
    return_gamma: float = 1.0  # discount factor (1.0 = undiscounted, <1 = exponential decay)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InfluenceTokenWeightConfig":
        if not d or not d.get("enabled", False):
            return cls(enabled=False)
        return cls(
            enabled=True,
            threshold=float(d.get("threshold", 1.0)),
            mode=str(d.get("mode", "zero")),
            normalize=str(d.get("normalize", "zscore")),
            apply_epochs=list(d.get("apply_epochs", [1, 2])),
            softmax_temperature=float(d.get("softmax_temperature", 1.0)),
            weight_clamp_min=float(d.get("weight_clamp_min", 0.1)),
            weight_clamp_max=float(d.get("weight_clamp_max", 5.0)),
            tanh_alpha=float(d.get("tanh_alpha", 0.5)),
            tanh_tau=float(d.get("tanh_tau", 1.0)),
            direct_lambda=float(d.get("direct_lambda", 0.3)),
            direct_clamp_min=float(d.get("direct_clamp_min", 0.0)),
            direct_clamp_max=float(d.get("direct_clamp_max", 2.0)),
            adv_target=str(d.get("adv_target", "advantage")),
            ratio_clamp_min=float(d.get("ratio_clamp_min", -1.0)),
            ratio_clamp_max=float(d.get("ratio_clamp_max", 3.0)),
            ratio_snr_threshold=float(d.get("ratio_snr_threshold", 0.3)),
            additive_lambda=float(d.get("additive_lambda", 0.5)),
            additive_clamp_min=float(d.get("additive_clamp_min", -1.0)),
            additive_clamp_max=float(d.get("additive_clamp_max", 3.0)),
            apply_to=str(d.get("apply_to", "all")),
            return_gamma=float(d.get("return_gamma", 1.0)),
        )


def build_token_loss_weights(
    influence_cache: dict[int, np.ndarray],
    row_ids: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: InfluenceTokenWeightConfig,
    accepted: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build per-token loss weights from cached influence scores.

    Args:
        influence_cache: row_id -> influence array (np.float16, shape [response_len])
        row_ids: [batch_size] integer row IDs for each sample in micro-batch
        advantages: [batch_size, response_length] advantage values
        response_mask: [batch_size, response_length] binary mask
        config: weighting configuration
        accepted: [batch_size] bool tensor from influence_trace_accepted labels.
            If provided, used directly; otherwise falls back to advantage sign.

    Returns:
        (weights, stats, per_row_weights): weights is [batch_size, response_length] tensor;
            stats is a dict of extra metrics (e.g. sign_guard_filtered count);
            per_row_weights is a dict[int, np.ndarray] mapping row_id to float16
            weight arrays (for NPZ persistence / visualization).
    """
    bsz, seq_len = advantages.shape
    # For additive correction modes, weights are corrections (init 0); otherwise multiplicative (init 1)
    _init_val = 0.0 if config.mode in ("credit", "return") else 1.0
    weights = torch.full((bsz, seq_len), _init_val, device=advantages.device, dtype=advantages.dtype)
    stats: dict[str, float] = {}
    per_row_weights: dict[int, np.ndarray] = {}

    if not config.enabled:
        return weights, stats, per_row_weights

    if not influence_cache:
        return weights, stats, per_row_weights

    # Determine accepted/rejected per row
    with torch.no_grad():
        if accepted is not None:
            is_accepted = accepted.bool()
        else:
            # Fallback: infer from advantage sign
            adv_sum = (advantages * response_mask).sum(dim=-1)  # [bsz]
            is_accepted = adv_sum > 0  # [bsz]

    # Track which tokens actually received influence weights (for random shuffle)
    influenced = torch.zeros(bsz, seq_len, device=advantages.device, dtype=torch.bool)
    sign_guard_filtered = 0  # responses where sign(μ_i) != expected sign
    sign_guard_total = 0  # total responses that entered ratio mode with valid SNR
    cache_hits = 0
    cache_misses = 0
    skip_snr = 0  # responses caught by SNR filter

    for i in range(bsz):
        rid = int(row_ids[i].item())
        if rid not in influence_cache:
            cache_misses += 1
            continue  # no influence data for this row → keep weight=1.0
        cache_hits += 1

        infl = influence_cache[rid]  # np.float16 array
        infl_t = torch.from_numpy(infl.astype(np.float32)).to(advantages.device)

        # Truncate or pad to match seq_len
        infl_len = infl_t.shape[0]
        if infl_len > seq_len:
            infl_t = infl_t[:seq_len]
        elif infl_len < seq_len:
            pad = torch.full(
                (seq_len - infl_len,), float("nan"),
                device=advantages.device, dtype=torch.float32,
            )
            infl_t = torch.cat([infl_t, pad])

        # Only consider valid (non-NaN, masked) tokens
        non_nan = (~torch.isnan(infl_t)).sum().item()
        rmask_sum = response_mask[i].bool().sum().item()
        valid = (~torch.isnan(infl_t)) & response_mask[i].bool()
        n_valid = valid.sum()
        if n_valid < 1:
            stats.setdefault("_skip_all_nan", 0.0)
            stats["_skip_all_nan"] += 1.0
            stats[f"_dbg_rid_{i}"] = float(rid)
            stats[f"_dbg_infl_len_{i}"] = float(infl_len)
            stats[f"_dbg_seq_len_{i}"] = float(seq_len)
            stats[f"_dbg_non_nan_{i}"] = float(non_nan)
            stats[f"_dbg_rmask_{i}"] = float(rmask_sum)
            continue

        scores = infl_t.clone()

        if config.mode == "ratio":
            # Option (a): w_t = s_t / μ_i  (proposition corollary)
            # Requires sign(μ_i) consistent with accepted/rejected:
            #   Accepted (μ_i > 0): s_t > 0 → w > 0, s_t < 0 → w < 0
            #   Rejected (μ_i < 0): s_t < 0 → w > 0, s_t > 0 → w < 0
            # When sign(μ_i) is inconsistent, credit assignment inverts → fallback.
            valid_s = scores[valid]
            mu_i = valid_s.mean()
            sigma_i = valid_s.std()
            snr = mu_i.abs() / (sigma_i + 1e-8)
            if snr < config.ratio_snr_threshold:
                skip_snr += 1
                continue  # fallback to uniform (weights already 1.0)
            # Sign guard: μ_i sign must match expected direction
            sign_guard_total += 1
            expected_positive = is_accepted[i].item()
            if (expected_positive and mu_i < 0) or (not expected_positive and mu_i > 0):
                sign_guard_filtered += 1
                continue  # sign mismatch → fallback to uniform
            w = valid_s / mu_i
            w = w.clamp(config.ratio_clamp_min, config.ratio_clamp_max)
            # Re-normalize to mean=1 after clamping
            w_mean = w.mean()
            if w_mean.abs() > 1e-8:
                w = w / w_mean
            else:
                continue  # degenerate after clamp → uniform
            weights[i, valid] = w.to(weights.dtype)
            influenced[i] = valid
            continue

        if config.mode in ("mask", "mask_random", "mask_soft", "mask_threshold"):
            # Mask family: modulate advantage based on influence sign.
            # mask: hard 0/1 based on sign
            # mask_random: same fraction masked but randomly (ablation control)
            # mask_soft: wrong-direction tokens get weight=mask_soft_value instead of 0
            # mask_threshold: only mask when |score| > threshold (z-scored), else keep weight=1
            valid_s = scores[valid]

            if config.mode == "mask_threshold":
                # Z-score within response, then mask only confident wrong-direction tokens
                mu = valid_s.mean()
                sigma = valid_s.std()
                if sigma < 1e-8:
                    continue
                z = (valid_s - mu) / sigma
                if is_accepted[i]:
                    mask_bad = z < -config.threshold  # confidently bad tokens
                else:
                    mask_bad = z > config.threshold   # confidently good tokens in rejected
            else:
                if is_accepted[i]:
                    mask_bad = valid_s < 0
                else:
                    mask_bad = valid_s > 0

            n_mask = int(mask_bad.sum().item())
            if config.mode == "mask_random" and n_mask > 0:
                mask_bad = torch.zeros_like(valid_s, dtype=torch.bool)
                perm = torch.randperm(valid_s.numel(), device=valid_s.device)[:n_mask]
                mask_bad[perm] = True

            w = torch.ones_like(valid_s)
            if config.mode == "mask_soft":
                w[mask_bad] = config.weight_clamp_min  # soft value (e.g. 0.5)
            else:
                w[mask_bad] = 0.0
            weights[i, valid] = w.to(weights.dtype)
            influenced[i] = valid
            continue

        if config.mode == "credit":
            # Credit mode: A'_t = A_t + correction_t (additive to advantage)
            # Uses sign flip for rejected responses — same dense credit semantics
            # as multiplicative sign-flip, but additive form should be more stable
            # because correction is independent of A_t magnitude.
            valid_s = scores[valid]
            if not is_accepted[i]:
                valid_s = -valid_s  # sign flip: high-influence tokens in rejected → negative correction
            mu = valid_s.mean()
            sigma = valid_s.std()
            if sigma < 1e-8:
                continue  # all same score → keep correction=0
            z = (valid_s - mu) / sigma
            correction = config.additive_lambda * z
            correction = correction.clamp(config.additive_clamp_min, config.additive_clamp_max)
            weights[i, valid] = correction.to(weights.dtype)
            influenced[i] = valid
            continue

        if config.mode == "return":
            # Return-to-go mode: treat influence scores as token-level rewards
            # and compute RL-style return G_t = Σ_{k=t}^T γ^{k-t} s_k.
            # Sign flip for rejected responses (high influence in rejected = bad).
            # Applied as additive correction: A'_t = A_t + λ * G_t
            if config.apply_to == "positive" and not is_accepted[i]:
                continue
            if config.apply_to == "negative" and is_accepted[i]:
                continue
            valid_s = scores[valid]
            if not is_accepted[i]:
                valid_s = -valid_s  # sign flip for rejected
            # Compute suffix-sum return-to-go
            gamma = config.return_gamma
            T = valid_s.shape[0]
            G = torch.zeros_like(valid_s)
            G[T - 1] = valid_s[T - 1]
            for t in range(T - 2, -1, -1):
                G[t] = valid_s[t] + gamma * G[t + 1]
            # Scale and clamp
            correction = config.additive_lambda * G
            correction = correction.clamp(config.additive_clamp_min, config.additive_clamp_max)
            weights[i, valid] = correction.to(weights.dtype)
            influenced[i] = valid
            continue

        if config.mode == "additive":
            # Additive mode: w_t = 1 + λ × z_t, z_t = (s_t - mean) / std
            # Dense credit assignment: for rejected responses, flip score sign
            # so that "good tokens" (high influence, promoting accepted) get
            # w < 1 (less suppression) and "bad tokens" get w > 1 (more
            # suppression). Clamp range should be tight (e.g. [0, 2]) to
            # limit noise amplification when scores are unreliable.
            # Skip responses based on apply_to setting
            if config.apply_to == "positive" and not is_accepted[i]:
                continue  # keep w=1 for rejected responses
            if config.apply_to == "negative" and is_accepted[i]:
                continue  # keep w=1 for accepted responses
            valid_s = scores[valid]
            # Sign flip for rejected: "all" flips (dense credit), "noflip_neg"
            # keeps original score direction for rejected (saliency on neg side).
            if not is_accepted[i] and config.apply_to not in ("noflip_neg",):
                valid_s = -valid_s
            mu = valid_s.mean()
            sigma = valid_s.std()
            if sigma < 1e-8:
                continue  # all same score → keep uniform
            z = (valid_s - mu) / sigma
            w = 1.0 + config.additive_lambda * z
            w = w.clamp(config.additive_clamp_min, config.additive_clamp_max)
            weights[i, valid] = w.to(weights.dtype)
            influenced[i] = valid
            continue

        if config.mode in ("direct", "random", "direct_no_baseline", "direct_no_baseline_random"):
            # Direct mode: m_t = 1 + λ * score_t (with baseline)
            # direct_no_baseline: m_t = λ * score_t (no baseline, pure influence)
            #
            # For accepted responses: positive score → beneficial → m > 0
            # For rejected responses: flip sign so that bad tokens get positive m
            valid_s = scores[valid]
            if not is_accepted[i]:
                valid_s = -valid_s
            if config.mode in ("direct_no_baseline", "direct_no_baseline_random"):
                m = config.direct_lambda * valid_s
            else:
                m = 1.0 + config.direct_lambda * valid_s
            m = m.clamp(config.direct_clamp_min, config.direct_clamp_max)
            weights[i, valid] = m.to(weights.dtype)
            influenced[i] = valid
            continue

        # --- Legacy modes below (unchanged) ---
        # Z-score needs at least 2 tokens for std
        if n_valid < 2:
            continue

        # Normalize
        if config.normalize == "zscore":
            valid_scores = scores[valid]
            mu = valid_scores.mean()
            std = valid_scores.std()
            if std > 1e-8:
                scores = (scores - mu) / std
            else:
                continue  # all same score → no meaningful weighting

        if config.mode == "softmax":
            valid_z = scores[valid]
            if is_accepted[i]:
                logits = valid_z / config.softmax_temperature
            else:
                logits = -valid_z / config.softmax_temperature
            softmax_w = torch.softmax(logits, dim=0)
            softmax_w = softmax_w * n_valid.float()
            softmax_w = softmax_w.clamp(
                min=config.weight_clamp_min,
                max=config.weight_clamp_max,
            )
            softmax_w = softmax_w * (n_valid.float() / softmax_w.sum())
            weights[i, valid] = softmax_w.to(weights.dtype)
            influenced[i] = valid
        elif config.mode == "tanh":
            valid_z = scores[valid]
            if is_accepted[i]:
                z_eff = valid_z
            else:
                z_eff = -valid_z
            tanh_w = 1.0 + config.tanh_alpha * torch.tanh(z_eff / config.tanh_tau)
            tanh_w = tanh_w.clamp(min=config.weight_clamp_min, max=config.weight_clamp_max)
            weights[i, valid] = tanh_w.to(weights.dtype)
            influenced[i] = valid
        else:
            # Threshold-based modes: zero, flip, soft
            if is_accepted[i]:
                mask_tokens = valid & (scores < -config.threshold)
            else:
                mask_tokens = valid & (scores > config.threshold)

            if config.mode == "zero":
                weights[i, mask_tokens] = 0.0
            elif config.mode == "flip":
                weights[i, mask_tokens] = -1.0
            elif config.mode == "soft":
                if is_accepted[i]:
                    w = torch.sigmoid(scores + config.threshold)
                else:
                    w = torch.sigmoid(-scores + config.threshold)
                weights[i, valid] = w[valid]
            influenced[i] = valid

    # Random shuffle ablation: keep the same weight distribution per response,
    # but randomly permute the token-weight assignment ONLY among tokens that
    # actually received influence weights (not padding or missing-cache tokens).
    if config.mode in ("random", "direct_no_baseline_random"):
        for i in range(bsz):
            mask = influenced[i]
            n_inf = int(mask.sum().item())
            if n_inf > 1:
                inf_w = weights[i, mask]
                perm = torch.randperm(n_inf, device=inf_w.device)
                weights[i, mask] = inf_w[perm]

    # Build per-row weight arrays for NPZ persistence / visualization
    for i in range(bsz):
        rid = int(row_ids[i].item())
        row_w = weights[i].detach().cpu().to(torch.float32).numpy().astype(np.float16)
        per_row_weights[rid] = row_w

    # Populate stats
    stats["cache_hits"] = float(cache_hits)
    stats["cache_misses"] = float(cache_misses)
    stats["cache_total"] = float(cache_hits + cache_misses)
    skip_nan_count = stats.get("_skip_all_nan", 0.0)
    stats["skip_nan"] = skip_nan_count  # non-underscore key for logging
    if sign_guard_total > 0:
        stats["sign_guard_filtered"] = float(sign_guard_filtered)
        stats["sign_guard_total"] = float(sign_guard_total)
        stats["sign_guard_frac"] = float(sign_guard_filtered / sign_guard_total)

    # Debug print: show breakdown for first micro-batch per step
    import torch.distributed as dist
    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0 and cache_hits > 0:
        print(
            f"[token_weight_dbg] bsz={bsz} cache_hits={cache_hits} cache_misses={cache_misses} "
            f"skip_nan={int(skip_nan_count)} skip_snr={skip_snr} sg_total={sign_guard_total} sg_filtered={sign_guard_filtered} "
            f"cache_size={len(influence_cache)} snr_threshold={config.ratio_snr_threshold} mode={config.mode}"
        )
        # Print details for NaN-skipped responses (only if any)
        if skip_nan_count > 0:
            for k, v in stats.items():
                if k.startswith("_dbg_"):
                    print(f"  {k}={v}")

    return weights, stats, per_row_weights
