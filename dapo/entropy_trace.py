#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities to persist rollout token-level entropy traces."""

from __future__ import annotations

import json
import os
import uuid
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
        write_every: int = 1,
        update_summary_every: int = 1,
        atomic_write: bool = True,
        fsync: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.steps_dir = self.output_dir / "steps"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.latest_step_path = self.output_dir / "latest_step.txt"

        self.write_every = max(int(write_every), 1)
        self.update_summary_every = max(int(update_summary_every), 1)
        self.atomic_write = bool(atomic_write)
        self.fsync = bool(fsync)

        if not self.enabled:
            return

        self.steps_dir.mkdir(parents=True, exist_ok=True)
        created_at = _now_iso()
        summary: dict[str, Any] = {}
        if self.summary_path.exists():
            try:
                summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        summary.setdefault("created_at", created_at)
        summary["updated_at"] = created_at
        summary["project_name"] = project_name
        summary["experiment_name"] = experiment_name
        summary["format_version"] = 2
        summary["step_file_pattern"] = "steps/step_*.npz"
        summary.setdefault("latest_step", None)
        summary.setdefault("num_steps_written", 0)
        summary.setdefault("write_every", self.write_every)
        summary.setdefault("atomic_write", self.atomic_write)
        summary.setdefault("fsync", self.fsync)

        self._summary = summary
        self._write_summary()

    def _atomic_write_text(self, path: Path, content: str) -> None:
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        if self.fsync:
            try:
                with tmp_path.open("rb") as f:
                    os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
        if self.fsync:
            try:
                dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass

    def _write_summary(self) -> None:
        if not self.enabled:
            return
        self._atomic_write_text(
            self.summary_path,
            json.dumps(self._summary, indent=2, ensure_ascii=True) + "\n",
        )

    def _append_manifest(self, record: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
            if self.fsync:
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass

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
        if int(step) % self.write_every != 0:
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

        if self.atomic_write:
            tmp_path = self.steps_dir / f".{file_name}.{uuid.uuid4().hex}.tmp.npz"
            np.savez_compressed(tmp_path, **arrays)
            if self.fsync:
                try:
                    with tmp_path.open("rb") as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, file_path)
        else:
            np.savez_compressed(file_path, **arrays)
            if self.fsync:
                try:
                    with file_path.open("rb") as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass

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

        # Small, watch-friendly progress file for "how far did we get?" checks mid-run.
        self._atomic_write_text(self.latest_step_path, f"{int(step)}\n")

        if int(step) % self.update_summary_every == 0:
            self._summary["updated_at"] = _now_iso()
            self._summary["latest_step"] = int(step)
            self._summary["num_steps_written"] = int(self._summary.get("num_steps_written", 0)) + 1
            self._write_summary()
