#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities to persist rollout token-level entropy traces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_numpy(value: Any) -> np.ndarray | None:
    """Best-effort conversion for metadata arrays."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    try:
        return np.asarray(value)
    except Exception:
        return None


def _to_unicode_array(value: Any) -> np.ndarray | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    try:
        return arr.astype(str)
    except Exception:
        return None


class RolloutEntropyTraceWriter:
    """Write one compressed entropy record per rollout step."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | Path,
        project_name: str,
        experiment_name: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.steps_dir = self.output_dir / "steps"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "summary.json"

        if not self.enabled:
            return

        self.steps_dir.mkdir(parents=True, exist_ok=True)
        if not self.summary_path.exists():
            summary = {
                "created_at": _now_iso(),
                "project_name": project_name,
                "experiment_name": experiment_name,
                "format_version": 1,
                "step_file_pattern": "steps/step_*.npz",
            }
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _append_manifest(self, record: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_step(
        self,
        *,
        step: int,
        batch,
        entropies: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        ent = entropies.detach().to(torch.float32).cpu()
        mask = response_mask.detach().to(torch.bool).cpu()

        responses = batch.batch.get("responses")
        if responses is None:
            max_response_len = ent.shape[-1]
            responses = batch.batch["input_ids"][:, -max_response_len:]
        responses = responses.detach().to(torch.int32).cpu()

        if ent.shape != mask.shape:
            raise ValueError(f"entropy/mask shape mismatch: {ent.shape} vs {mask.shape}")
        if ent.shape != responses.shape:
            raise ValueError(f"entropy/responses shape mismatch: {ent.shape} vs {responses.shape}")

        prompt_lengths = None
        if "attention_mask" in batch.batch:
            attn = batch.batch["attention_mask"].detach().to(torch.int32).cpu()
            prompt_lengths = attn.sum(-1) - mask.to(torch.int32).sum(-1)

        file_name = f"step_{int(step):06d}.npz"
        file_path = self.steps_dir / file_name

        arrays: dict[str, np.ndarray] = {
            "entropies": ent.numpy().astype(np.float16),
            "response_mask": mask.numpy().astype(np.bool_),
            "responses": responses.numpy(),
            "response_index": np.arange(ent.shape[0], dtype=np.int32),
        }

        if prompt_lengths is not None:
            arrays["prompt_length"] = prompt_lengths.numpy().astype(np.int32)

        uid = _to_unicode_array(batch.non_tensor_batch.get("uid"))
        if uid is not None and uid.shape[0] == ent.shape[0]:
            arrays["uid"] = uid

        sample_index = _to_numpy(batch.non_tensor_batch.get("index"))
        if sample_index is not None and sample_index.shape[0] == ent.shape[0]:
            arrays["sample_index"] = sample_index

        np.savez_compressed(file_path, **arrays)

        valid_ent = ent[mask]
        if valid_ent.numel() > 0:
            entropy_min = float(valid_ent.min().item())
            entropy_max = float(valid_ent.max().item())
            entropy_mean = float(valid_ent.mean().item())
            entropy_std = float(valid_ent.std(unbiased=False).item())
        else:
            entropy_min = entropy_max = entropy_mean = entropy_std = float("nan")

        record = {
            "step": int(step),
            "created_at": _now_iso(),
            "file": str(Path("steps") / file_name),
            "num_responses": int(ent.shape[0]),
            "response_len": int(ent.shape[1]),
            "num_valid_tokens": int(mask.sum().item()),
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
            "entropy_mean": entropy_mean,
            "entropy_std": entropy_std,
        }
        self._append_manifest(record)
