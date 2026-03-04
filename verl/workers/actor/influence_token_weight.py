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
    mode: str = "zero"  # "zero" | "flip" | "soft" | "softmax" | "tanh"
    normalize: str = "zscore"  # "zscore" or "none"
    apply_epochs: list[int] = field(default_factory=lambda: [1, 2])
    # softmax mode params
    softmax_temperature: float = 1.0
    weight_clamp_min: float = 0.1
    weight_clamp_max: float = 5.0
    # tanh mode params: w = 1 + alpha * tanh(z / tau), non-competitive
    tanh_alpha: float = 0.5  # modulation strength: w ∈ [1-alpha, 1+alpha] before clamp
    tanh_tau: float = 1.0  # temperature for tanh input scaling

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
        )


def build_token_loss_weights(
    influence_cache: dict[int, np.ndarray],
    row_ids: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: InfluenceTokenWeightConfig,
    accepted: torch.Tensor | None = None,
) -> torch.Tensor:
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
        weights: [batch_size, response_length] tensor, same device as advantages
    """
    bsz, seq_len = advantages.shape
    weights = torch.ones(bsz, seq_len, device=advantages.device, dtype=advantages.dtype)

    if not config.enabled or not influence_cache:
        return weights

    # Determine accepted/rejected per row
    with torch.no_grad():
        if accepted is not None:
            is_accepted = accepted.bool()
        else:
            # Fallback: infer from advantage sign
            adv_sum = (advantages * response_mask).sum(dim=-1)  # [bsz]
            is_accepted = adv_sum > 0  # [bsz]

    for i in range(bsz):
        rid = int(row_ids[i].item())
        if rid not in influence_cache:
            continue  # no influence data for this row → keep weight=1.0

        infl = influence_cache[rid]  # np.float16 array
        infl_t = torch.from_numpy(infl.astype(np.float32)).to(advantages.device)

        # Truncate or pad to match seq_len
        if infl_t.shape[0] > seq_len:
            infl_t = infl_t[:seq_len]
        elif infl_t.shape[0] < seq_len:
            pad = torch.full(
                (seq_len - infl_t.shape[0],), float("nan"),
                device=advantages.device, dtype=torch.float32,
            )
            infl_t = torch.cat([infl_t, pad])

        # Only consider valid (non-NaN, masked) tokens
        valid = (~torch.isnan(infl_t)) & response_mask[i].bool()
        n_valid = valid.sum()
        if n_valid < 2:
            continue

        scores = infl_t.clone()

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
            # Softmax mode: smooth weighting over all tokens in the response.
            # Accepted: softmax(z) → high-influence tokens get more weight
            # Rejected: softmax(-z) → low-influence tokens (truly bad) get more weight
            valid_z = scores[valid]
            if is_accepted[i]:
                logits = valid_z / config.softmax_temperature
            else:
                logits = -valid_z / config.softmax_temperature
            softmax_w = torch.softmax(logits, dim=0)
            # Scale so sum(weights) = n_valid → same effective lr as uniform
            softmax_w = softmax_w * n_valid.float()
            # Clamp to prevent extreme weights
            softmax_w = softmax_w.clamp(
                min=config.weight_clamp_min,
                max=config.weight_clamp_max,
            )
            # Re-normalize after clamping to preserve effective lr
            softmax_w = softmax_w * (n_valid.float() / softmax_w.sum())
            weights[i, valid] = softmax_w.to(weights.dtype)
        elif config.mode == "tanh":
            # Tanh mode: bounded, non-competitive per-token weights.
            # w_t = 1 + alpha * tanh(z_eff / tau)
            # Each token's weight depends only on its own z-score.
            # If all z ≈ 0 (noise), all weights ≈ 1.0 (no effect).
            valid_z = scores[valid]
            if is_accepted[i]:
                z_eff = valid_z  # high z → beneficial → boost
            else:
                z_eff = -valid_z  # low z → truly bad → boost
            tanh_w = 1.0 + config.tanh_alpha * torch.tanh(z_eff / config.tanh_tau)
            tanh_w = tanh_w.clamp(min=config.weight_clamp_min, max=config.weight_clamp_max)
            weights[i, valid] = tanh_w.to(weights.dtype)
        else:
            # Threshold-based modes: zero, flip, soft
            if is_accepted[i]:
                # Accepted: flag tokens with z < -threshold (harmful tokens)
                mask_tokens = valid & (scores < -config.threshold)
            else:
                # Rejected: flag tokens with z > +threshold (good despite rejection)
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

    return weights
