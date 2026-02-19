#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities to persist rollout token influence traces."""

from __future__ import annotations

import hashlib
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


def _uid_to_hash64(uid: str) -> int:
    digest = hashlib.sha1(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _to_unicode_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.astype(str)


class RolloutInfluenceTraceWriter:
    """Write one compressed influence record per rollout step."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | Path,
        project_name: str,
        experiment_name: str,
        write_every: int = 1,
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
        self.atomic_write = bool(atomic_write)
        self.fsync = bool(fsync)
        if not self.enabled:
            return
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        created_at = _now_iso()
        summary: dict[str, Any] = {}
        if self.summary_path.exists():
            summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        summary.setdefault("created_at", created_at)
        summary["updated_at"] = created_at
        summary["project_name"] = project_name
        summary["experiment_name"] = experiment_name
        summary["format_version"] = 1
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
            with tmp_path.open("rb") as f:
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
        if self.fsync:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _write_summary(self) -> None:
        if not self.enabled:
            return
        self._atomic_write_text(self.summary_path, json.dumps(self._summary, indent=2, ensure_ascii=True) + "\n")

    def _append_manifest(self, record: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())

    def write_step(
        self,
        *,
        step: int,
        batch,
        entropies: torch.Tensor,
        response_mask: torch.Tensor,
        influence_rows: np.ndarray | None,
    ) -> None:
        if not self.enabled:
            return
        if int(step) % self.write_every != 0:
            return
        if influence_rows is None or len(influence_rows) == 0:
            return
        row_to_influence: dict[int, np.ndarray] = {}
        for rec in influence_rows.tolist():
            if not isinstance(rec, dict):
                continue
            row_id = int(rec["row_id"])
            inf = np.asarray(rec["influence"], dtype=np.float16)
            row_to_influence[row_id] = inf
        if not row_to_influence:
            return
        selected_rows = np.array(sorted(row_to_influence.keys()), dtype=np.int64)
        ent = entropies.detach().to(torch.float32).cpu().numpy()[selected_rows]
        mask = response_mask.detach().to(torch.bool).cpu().numpy()[selected_rows]
        responses = batch.batch["responses"].detach().to(torch.int32).cpu().numpy()[selected_rows]
        influence = np.full(ent.shape, np.nan, dtype=np.float16)
        for i, row_id in enumerate(selected_rows.tolist()):
            inf = row_to_influence[row_id]
            target_len = int(influence.shape[1])
            src_len = int(inf.shape[0])
            copy_len = min(src_len, target_len)
            if copy_len > 0:
                influence[i, :copy_len] = inf[:copy_len]
        uid = _to_unicode_array(batch.non_tensor_batch.get("uid"))
        uid_sel = uid[selected_rows] if uid is not None else None
        uid_hash = np.array([_uid_to_hash64(str(x)) for x in uid_sel], dtype=np.int64) if uid_sel is not None else None
        prompt_ids_obj = batch.non_tensor_batch.get("raw_prompt_ids")
        if prompt_ids_obj is None:
            prompt_ids = np.array(["[]"] * len(selected_rows), dtype=object)
        else:
            prompt_ids = np.asarray(prompt_ids_obj, dtype=object)[selected_rows]
        rewards = batch.batch["influence_trace_reward"].detach().to(torch.float32).cpu().numpy()[selected_rows]
        accepted = batch.batch["influence_trace_accepted"].detach().to(torch.bool).cpu().numpy()[selected_rows]
        group_ids = batch.batch["influence_trace_group_id"].detach().to(torch.int32).cpu().numpy()[selected_rows]
        file_name = f"step_{int(step):06d}.npz"
        file_path = self.steps_dir / file_name
        arrays: dict[str, np.ndarray] = {
            "entropies": ent.astype(np.float16),
            "influence": influence,
            "response_mask": mask.astype(np.bool_),
            "responses": responses.astype(np.int32),
            "selected_row_id": selected_rows.astype(np.int32),
            "reward": rewards.astype(np.float32),
            "accepted": accepted.astype(np.bool_),
            "group_id": group_ids.astype(np.int32),
            "prompt_ids": prompt_ids,
        }
        if uid_sel is not None:
            arrays["uid"] = uid_sel
        if uid_hash is not None:
            arrays["uid_hash"] = uid_hash
        if self.atomic_write:
            tmp_path = self.steps_dir / f".{file_name}.{uuid.uuid4().hex}.tmp.npz"
            np.savez_compressed(tmp_path, **arrays)
            if self.fsync:
                with tmp_path.open("rb") as f:
                    os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
        else:
            np.savez_compressed(file_path, **arrays)
            if self.fsync:
                with file_path.open("rb") as f:
                    os.fsync(f.fileno())
        valid_inf = influence[np.isfinite(influence) & mask]
        record = {
            "step": int(step),
            "created_at": _now_iso(),
            "file": str(Path("steps") / file_name),
            "num_responses": int(ent.shape[0]),
            "response_len": int(ent.shape[1]),
            "num_valid_tokens": int(mask.sum()),
            "num_valid_influence": int(valid_inf.size),
            "influence_sum": float(valid_inf.sum()) if valid_inf.size > 0 else float("nan"),
            "influence_mean": float(valid_inf.mean()) if valid_inf.size > 0 else float("nan"),
        }
        self._append_manifest(record)
        self._atomic_write_text(self.latest_step_path, f"{int(step)}\n")
        self._summary["updated_at"] = _now_iso()
        self._summary["latest_step"] = int(step)
        self._summary["num_steps_written"] = int(self._summary.get("num_steps_written", 0)) + 1
        self._write_summary()
