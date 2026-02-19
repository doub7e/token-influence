"""GPU-side token influence tracing utilities for PPO actor training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def _stable_hash32(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _make_rademacher_proj(
    in_dim: int,
    out_dim: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    signs = torch.randint(0, 2, (out_dim, in_dim), generator=g, dtype=torch.int8)
    proj = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    proj.div_(math.sqrt(float(out_dim)))
    return proj.to(device=device, dtype=dtype, non_blocking=True)


@dataclass
class InfluenceTraceConfig:
    enable: bool = False
    reg_lambda: float = -1.0
    module_name_filter: tuple[str, ...] = ("self_attn.o_proj", "mlp.down_proj")
    max_modules: int = 2
    max_proj_vector_sum: int = 64
    max_hessian_dim: int = 2500

    @staticmethod
    def from_meta(meta: dict[str, Any]) -> "InfluenceTraceConfig":
        raw = meta.get("influence_trace_cfg", {}) if meta is not None else {}
        if not isinstance(raw, dict):
            return InfluenceTraceConfig(enable=False)
        filters = raw.get("module_name_filter", ("self_attn.o_proj", "mlp.down_proj"))
        if isinstance(filters, str):
            filters = (filters,)
        return InfluenceTraceConfig(
            enable=bool(raw.get("enable", False)),
            reg_lambda=float(raw.get("reg_lambda", -1.0)),
            module_name_filter=tuple(filters),
            max_modules=max(int(raw.get("max_modules", 2)), 1),
            max_proj_vector_sum=max(int(raw.get("max_proj_vector_sum", 64)), 2),
            max_hessian_dim=max(int(raw.get("max_hessian_dim", 2500)), 1),
        )


class TokenInfluenceTracer:
    """Capture projected per-token gradients from real training backward pass."""

    def __init__(self, cfg: InfluenceTraceConfig):
        self.cfg = cfg
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.modules: dict[str, nn.Linear] = {}
        self.module_dims: dict[str, tuple[int, int]] = {}
        self.current_capture: dict[str, torch.Tensor] | None = None
        self.storage: dict[str, dict[str, list[torch.Tensor]]] = {}

    def _pick_projection_dims(self, in_features: int, out_features: int) -> tuple[int, int]:
        max_sum = self.cfg.max_proj_vector_sum
        ratio = float(in_features) / float(in_features + out_features)
        k_in = max(1, int(round(max_sum * ratio)))
        k_out = max(1, max_sum - k_in)
        k_in = min(k_in, in_features)
        k_out = min(k_out, out_features)
        while (k_in * k_out) > self.cfg.max_hessian_dim and (k_in + k_out) > 2:
            if k_in >= k_out and k_in > 1:
                k_in -= 1
            elif k_out > 1:
                k_out -= 1
            else:
                break
        if k_in + k_out > self.cfg.max_proj_vector_sum:
            raise ValueError("Projection vectors exceed configured max sum.")
        if (k_in * k_out) > self.cfg.max_hessian_dim:
            raise ValueError("Projected Hessian dimension exceeds configured max.")
        return k_in, k_out

    def _iter_target_modules(self, model: torch.nn.Module) -> list[tuple[str, nn.Linear]]:
        by_filter: dict[str, list[tuple[str, nn.Linear]]] = {f: [] for f in self.cfg.module_name_filter}
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            for filt in self.cfg.module_name_filter:
                if filt in name:
                    by_filter[filt].append((name, mod))
                    break
        picked: list[tuple[str, nn.Linear]] = []
        for filt in self.cfg.module_name_filter:
            mods = by_filter[filt]
            if not mods:
                continue
            picked.extend(mods[-self.cfg.max_modules :])
        unique: dict[str, nn.Linear] = {}
        for name, mod in picked:
            unique[name] = mod
        return list(unique.items())

    def register(self, model: torch.nn.Module) -> None:
        if self.handles:
            return
        target_modules = self._iter_target_modules(model)
        for name, mod in target_modules:
            k_in, k_out = self._pick_projection_dims(mod.in_features, mod.out_features)
            self.modules[name] = mod
            self.module_dims[name] = (k_in, k_out)
            p_in = _make_rademacher_proj(
                mod.in_features,
                k_in,
                _stable_hash32(name + ":pin"),
                device=mod.weight.device,
                dtype=torch.float32,
            )
            p_out = _make_rademacher_proj(
                mod.out_features,
                k_out,
                _stable_hash32(name + ":pout"),
                device=mod.weight.device,
                dtype=torch.float32,
            )
            mod.register_buffer("_inftrace_p_in", p_in, persistent=False)
            mod.register_buffer("_inftrace_p_out", p_out, persistent=False)
            self.handles.append(mod.register_forward_hook(self._make_forward_hook()))
            self.handles.append(mod.register_full_backward_hook(self._make_backward_hook(name)))
        self.storage = {
            name: {"u": [], "v": [], "row_id": [], "token_pos": [], "group_id": [], "accepted": []}
            for name in self.modules
        }

    def _make_forward_hook(self):
        def hook(module: nn.Module, inputs, output):
            if self.current_capture is None:
                return
            x = inputs[0]
            if x.dim() == 3:
                if x.shape[0] != 1:
                    raise ValueError("Influence trace currently expects rmpad input with batch axis = 1.")
                x_tokens = x.squeeze(0)
            elif x.dim() == 2:
                x_tokens = x
            else:
                raise ValueError(f"Unsupported linear input dim for influence trace: {x.dim()}")
            capture_idx = self.current_capture["capture_idx"]
            if capture_idx.numel() == 0:
                return
            x_sel = x_tokens.index_select(0, capture_idx).to(torch.float32)
            p_in = getattr(module, "_inftrace_p_in")
            v = x_sel @ p_in.t()
            module._inftrace_v = v
        return hook

    def _make_backward_hook(self, module_name: str):
        def hook(module: nn.Module, grad_input, grad_output):
            if self.current_capture is None:
                return
            if not hasattr(module, "_inftrace_v"):
                return
            dy = grad_output[0]
            if dy.dim() == 3:
                if dy.shape[0] != 1:
                    raise ValueError("Influence trace currently expects rmpad grad output with batch axis = 1.")
                dy_tokens = dy.squeeze(0)
            elif dy.dim() == 2:
                dy_tokens = dy
            else:
                raise ValueError(f"Unsupported linear grad dim for influence trace: {dy.dim()}")
            capture_idx = self.current_capture["capture_idx"]
            dy_sel = dy_tokens.index_select(0, capture_idx).to(torch.float32)
            p_out = getattr(module, "_inftrace_p_out")
            u = dy_sel @ p_out.t()
            v = module._inftrace_v
            bucket = self.storage[module_name]
            bucket["u"].append(u.detach())
            bucket["v"].append(v.detach())
            bucket["row_id"].append(self.current_capture["row_id"])
            bucket["token_pos"].append(self.current_capture["token_pos"])
            bucket["group_id"].append(self.current_capture["group_id"])
            bucket["accepted"].append(self.current_capture["accepted"])
            delattr(module, "_inftrace_v")
        return hook

    def clear_storage(self) -> None:
        for name in self.storage:
            for key in self.storage[name]:
                self.storage[name][key].clear()
        self.current_capture = None

    def begin_rmpad_capture(
        self,
        *,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        response_mask: torch.Tensor,
        selected_rows: torch.Tensor,
        row_ids: torch.Tensor,
        group_ids: torch.Tensor,
        accepted: torch.Tensor,
    ) -> bool:
        response_len = response_mask.shape[1]
        if response_len <= 0:
            self.current_capture = None
            return False
        if selected_rows.dtype != torch.bool:
            selected_rows = selected_rows.bool()
        token_pos = torch.arange(response_len, device=response_mask.device, dtype=torch.long)
        row_pos = torch.arange(batch_size, device=response_mask.device, dtype=torch.long)
        full_pos = (seqlen - response_len - 1) + token_pos[None, :]
        full_pos = full_pos.expand(batch_size, response_len)
        row_grid = row_pos[:, None].expand(batch_size, response_len)
        valid = response_mask.bool() & selected_rows[:, None]
        if not valid.any():
            self.current_capture = None
            return False
        row_valid = row_grid[valid]
        tok_valid = token_pos[None, :].expand(batch_size, response_len)[valid]
        flat_idx = row_valid * seqlen + full_pos[valid]
        lookup = torch.full((batch_size * seqlen,), -1, device=indices.device, dtype=torch.long)
        lookup[indices] = torch.arange(indices.numel(), device=indices.device, dtype=torch.long)
        capture_idx = lookup[flat_idx]
        keep = capture_idx >= 0
        if not keep.any():
            self.current_capture = None
            return False
        row_valid = row_valid[keep]
        self.current_capture = {
            "capture_idx": capture_idx[keep],
            "row_id": row_ids[row_valid].to(torch.int64),
            "token_pos": tok_valid[keep].to(torch.int64),
            "group_id": group_ids[row_valid].to(torch.int64),
            "accepted": accepted[row_valid].to(torch.bool),
            "response_len": torch.tensor(response_len, device=indices.device, dtype=torch.int64),
        }
        return True

    def end_microbatch(self) -> None:
        self.current_capture = None

    def _compute_kernel_solve(
        self,
        g_resp: torch.Tensor,
        g_tok: torch.Tensor,
        reg_lambda: float,
    ) -> torch.Tensor:
        dim = g_resp.shape[1]
        hessian = g_resp.T @ g_resp
        if reg_lambda > 0:
            reg = reg_lambda
        else:
            reg = float((hessian.trace() / max(dim, 1)).item()) * 0.1
        hessian = hessian + reg * torch.eye(dim, device=hessian.device, dtype=hessian.dtype)
        chol = torch.linalg.cholesky(hessian)
        solved = torch.cholesky_solve(g_tok.T, chol)
        return solved

    def _module_token_scores(self, module_name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        bucket = self.storage[module_name]
        if not bucket["u"]:
            return (
                torch.empty(0, device="cpu", dtype=torch.int64),
                torch.empty(0, device="cpu", dtype=torch.float32),
                0,
            )
        u = torch.cat(bucket["u"], dim=0).to(torch.float32)
        v = torch.cat(bucket["v"], dim=0).to(torch.float32)
        row_id = torch.cat(bucket["row_id"], dim=0)
        token_pos = torch.cat(bucket["token_pos"], dim=0)
        group_id = torch.cat(bucket["group_id"], dim=0)
        accepted = torch.cat(bucket["accepted"], dim=0)
        response_len = int(self.current_capture["response_len"].item()) if self.current_capture is not None else int(token_pos.max().item()) + 1
        stride = response_len + 1
        pair_ids = row_id * stride + token_pos
        unique_groups = torch.unique(group_id, sorted=True)
        pair_out: list[torch.Tensor] = []
        score_out: list[torch.Tensor] = []
        for gid in unique_groups.tolist():
            g_mask = group_id == gid
            if int(g_mask.sum().item()) == 0:
                continue
            row_g = row_id[g_mask]
            tok_g = token_pos[g_mask]
            u_g = u[g_mask]
            v_g = v[g_mask]
            acc_g = accepted[g_mask]
            g_tok = torch.bmm(u_g.unsqueeze(2), v_g.unsqueeze(1)).reshape(u_g.shape[0], -1)
            uniq_rows, inv = torch.unique(row_g, sorted=True, return_inverse=True)
            row_cnt = torch.bincount(inv, minlength=uniq_rows.numel()).to(torch.float32).unsqueeze(1)
            g_resp = torch.zeros((uniq_rows.numel(), g_tok.shape[1]), device=g_tok.device, dtype=g_tok.dtype)
            g_resp.index_add_(0, inv, g_tok)
            g_resp = g_resp / row_cnt
            row_acc = torch.zeros((uniq_rows.numel(),), device=g_tok.device, dtype=torch.bool)
            row_acc.index_put_((inv,), acc_g, accumulate=False)
            if bool(row_acc.all().item()) or bool((~row_acc).all().item()):
                continue
            solved = self._compute_kernel_solve(g_resp=g_resp, g_tok=g_tok, reg_lambda=self.cfg.reg_lambda)
            infl = -(g_resp @ solved)  # [n_resp, n_tokens]
            score = infl[row_acc].sum(dim=0) - infl[~row_acc].sum(dim=0)
            pair = row_g * stride + tok_g
            pair_out.append(pair)
            score_out.append(score)
        if not pair_out:
            return (
                torch.empty(0, device="cpu", dtype=torch.int64),
                torch.empty(0, device="cpu", dtype=torch.float32),
                response_len,
            )
        return torch.cat(pair_out, dim=0), torch.cat(score_out, dim=0), response_len

    def pop_token_influence_rows(self) -> list[dict[str, Any]]:
        if not self.storage:
            return []
        all_pairs: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        response_len = 0
        for module_name in self.storage:
            pairs, scores, cur_response_len = self._module_token_scores(module_name)
            if pairs.numel() == 0:
                continue
            response_len = max(response_len, cur_response_len)
            all_pairs.append(pairs)
            all_scores.append(scores)
        if not all_pairs:
            self.clear_storage()
            return []
        pair_cat = torch.cat(all_pairs, dim=0)
        score_cat = torch.cat(all_scores, dim=0)
        uniq_pair, inv = torch.unique(pair_cat, sorted=True, return_inverse=True)
        summed = torch.zeros((uniq_pair.shape[0],), device=score_cat.device, dtype=torch.float32)
        summed.index_add_(0, inv, score_cat)
        stride = response_len + 1
        row_id = torch.div(uniq_pair, stride, rounding_mode="floor")
        token_pos = torch.remainder(uniq_pair, stride)
        uniq_rows = torch.unique(row_id, sorted=True)
        row_records: list[dict[str, Any]] = []
        for rid in uniq_rows.tolist():
            rmask = row_id == rid
            pos = token_pos[rmask]
            val = summed[rmask]
            influence = torch.full((response_len,), float("nan"), device=val.device, dtype=torch.float32)
            influence[pos] = val
            row_records.append(
                {
                    "row_id": int(rid),
                    "influence": influence.detach().cpu().numpy().astype(np.float16),
                }
            )
        self.clear_storage()
        return row_records

    def estimate_hessian_memory_mb(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, (k_in, k_out) in self.module_dims.items():
            dim = int(k_in * k_out)
            bytes_peak = 2 * dim * dim * 4
            out[name] = float(bytes_peak) / (1024.0 * 1024.0)
        return out

