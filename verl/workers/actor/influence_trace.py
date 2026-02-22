"""GPU-side token influence tracing utilities for PPO actor training."""

from __future__ import annotations

import hashlib
import math
import time
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
    hessian_mode: str = "inverse"
    output_function: str = "training_loss"
    accepted_rejected_scope: str = "per_prompt"
    module_name_filter: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    max_modules: int = -1
    projection_dim_factor: int = 512
    max_proj_vector_sum: int = -1
    max_hessian_dim: int = 2500
    max_tokens_per_response: int = -1
    skip_optimizer_step: bool = False
    grad_offload_to_cpu: bool = False
    force_gpu_compute: bool = True
    profile_timing: bool = False
    exclude_self_response: bool = False
    contrastive_agg: str = "sum"       # "sum" or "mean"
    hessian_source: str = "response"   # "response" or "token"

    @staticmethod
    def from_meta(meta: dict[str, Any]) -> "InfluenceTraceConfig":
        raw = meta.get("influence_trace_cfg", {}) if meta is not None else {}
        if not isinstance(raw, dict):
            return InfluenceTraceConfig(enable=False)
        filters = raw.get(
            "module_name_filter",
            (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ),
        )
        if isinstance(filters, str):
            filters = (filters,)
        hessian_mode = str(raw.get("hessian_mode", "inverse")).strip().lower()
        if hessian_mode not in {"inverse", "identity"}:
            hessian_mode = "inverse"
        output_function = str(raw.get("output_function", "training_loss")).strip().lower()
        if output_function not in {"training_loss", "log_prob_advantage", "log_prob_reward"}:
            output_function = "training_loss"
        accepted_rejected_scope = str(raw.get("accepted_rejected_scope", "per_prompt")).strip().lower()
        if accepted_rejected_scope not in {"per_prompt", "all_selected"}:
            accepted_rejected_scope = "per_prompt"
        contrastive_agg = str(raw.get("contrastive_agg", "sum")).strip().lower()
        if contrastive_agg not in {"sum", "mean"}:
            contrastive_agg = "sum"
        hessian_source = str(raw.get("hessian_source", "response")).strip().lower()
        if hessian_source not in {"response", "token"}:
            hessian_source = "response"
        return InfluenceTraceConfig(
            enable=bool(raw.get("enable", False)),
            reg_lambda=float(raw.get("reg_lambda", -1.0)),
            hessian_mode=hessian_mode,
            output_function=output_function,
            accepted_rejected_scope=accepted_rejected_scope,
            module_name_filter=tuple(filters),
            max_modules=int(raw.get("max_modules", -1)),
            projection_dim_factor=max(int(raw.get("projection_dim_factor", 512)), 1),
            max_proj_vector_sum=int(raw.get("max_proj_vector_sum", -1)),
            max_hessian_dim=max(int(raw.get("max_hessian_dim", 2500)), 1),
            max_tokens_per_response=int(raw.get("max_tokens_per_response", -1)),
            skip_optimizer_step=bool(raw.get("skip_optimizer_step", False)),
            grad_offload_to_cpu=bool(raw.get("grad_offload_to_cpu", False)),
            force_gpu_compute=bool(raw.get("force_gpu_compute", True)),
            profile_timing=bool(raw.get("profile_timing", False)),
            exclude_self_response=bool(raw.get("exclude_self_response", False)),
            contrastive_agg=contrastive_agg,
            hessian_source=hessian_source,
        )


class TokenInfluenceTracer:
    """Capture projected per-token gradients from real training backward pass."""

    def __init__(self, cfg: InfluenceTraceConfig):
        self.cfg = cfg
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.modules: dict[str, nn.Linear] = {}
        self.module_dims: dict[str, tuple[int, int]] = {}
        self.module_infos: list[dict[str, Any]] = []
        self.current_capture: dict[str, torch.Tensor] | None = None
        self.storage: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._anchor_module_name: str | None = None
        self._anchor_activation: torch.Tensor | None = None
        self._store_enabled: bool = True
        self._compute_mode_logged = False
        self._debug: dict[str, int] = {}
        self._timing: dict[str, float] = {}
        self._reset_debug()
        self._reset_timing()

    def _reset_debug(self) -> None:
        self._debug = {
            "capture_begin_calls": 0,
            "capture_begin_nonempty": 0,
            "capture_selected_tokens": 0,
            "anchor_tensor_ready": 0,
            "forward_capture_calls": 0,
            "forward_set_v_calls": 0,
            "backward_hook_calls": 0,
            "output_grad_hook_calls": 0,
            "stored_chunks": 0,
            "groups_total": 0,
            "groups_skipped_all_same": 0,
        }

    def _reset_timing(self) -> None:
        self._timing = {
            "grad_staging_s": 0.0,
            "hessian_solve_s": 0.0,
            "token_scoring_s": 0.0,
            "score_aggregation_s": 0.0,
        }

    def _pick_projection_dims(self, in_features: int, out_features: int) -> tuple[int, int]:
        factor = int(self.cfg.projection_dim_factor)
        max_sum = int(self.cfg.max_proj_vector_sum)
        max_hessian_dim = int(self.cfg.max_hessian_dim)
        if factor > 1:
            k_in = max(1, int(in_features // factor))
            k_out = max(1, int(out_features // factor))
        else:
            if max_sum <= 1:
                raise ValueError("Either projection_dim_factor>1 or max_proj_vector_sum>1 is required.")
            ratio = float(in_features) / float(in_features + out_features)
            k_in = max(1, int(round(max_sum * ratio)))
            k_out = max(1, max_sum - k_in)
        k_in = min(k_in, in_features)
        k_out = min(k_out, out_features)
        while max_sum > 1 and (k_in + k_out) > max_sum and (k_in + k_out) > 2:
            if k_in >= k_out and k_in > 1:
                k_in -= 1
            elif k_out > 1:
                k_out -= 1
            else:
                break
        while max_hessian_dim > 0 and (k_in * k_out) > max_hessian_dim and (k_in + k_out) > 2:
            if k_in >= k_out and k_in > 1:
                k_in -= 1
            elif k_out > 1:
                k_out -= 1
            else:
                break
        if max_sum > 1 and (k_in + k_out) > max_sum:
            raise ValueError("Projection vectors exceed configured max sum.")
        if max_hessian_dim > 0 and (k_in * k_out) > max_hessian_dim:
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
            if self.cfg.max_modules <= 0:
                picked.extend(mods)
            else:
                picked.extend(mods[-self.cfg.max_modules :])
        unique: dict[str, nn.Linear] = {}
        for name, mod in picked:
            unique[name] = mod
        return list(unique.items())

    def register(self, model: torch.nn.Module) -> None:
        if self.handles:
            return
        self.module_infos = []
        target_modules = self._iter_target_modules(model)
        for name, mod in target_modules:
            k_in, k_out = self._pick_projection_dims(mod.in_features, mod.out_features)
            self.modules[name] = mod
            self.module_dims[name] = (k_in, k_out)
            dim = int(k_in * k_out)
            if self.cfg.hessian_mode == "identity":
                hessian_peak_mb = 0.0
            else:
                hessian_peak_mb = float((2 * dim * dim * 4) / (1024.0 * 1024.0))
            self.module_infos.append(
                {
                    "name": name,
                    "in_features": int(mod.in_features),
                    "out_features": int(mod.out_features),
                    "k_in": int(k_in),
                    "k_out": int(k_out),
                    "proj_dim": dim,
                    "hessian_peak_mb": hessian_peak_mb,
                }
            )
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
            self.handles.append(mod.register_forward_hook(self._make_forward_hook(name)))
            self.handles.append(mod.register_full_backward_hook(self._make_backward_hook(name)))
        self.storage = {
            name: {"u": [], "v": [], "row_id": [], "token_pos": [], "group_id": [], "accepted": []}
            for name in self.modules
        }
        if self.modules:
            self._anchor_module_name = next(iter(self.modules.keys()))

    def projection_report(self) -> list[dict[str, Any]]:
        return list(self.module_infos)

    def anchor_tensor(self) -> torch.Tensor | None:
        return self._anchor_activation

    def anchor_parameter(self) -> torch.nn.Parameter | None:
        if self._anchor_module_name is None:
            return None
        mod = self.modules.get(self._anchor_module_name)
        if mod is None:
            return None
        return mod.weight

    def _make_forward_hook(self, module_name: str):
        def hook(module: nn.Module, inputs, output):
            x = inputs[0]
            if x.dim() == 3:
                if x.shape[0] != 1:
                    raise ValueError("Influence trace currently expects rmpad input with batch axis = 1.")
                x_tokens = x.squeeze(0)
            elif x.dim() == 2:
                x_tokens = x
            else:
                raise ValueError(f"Unsupported linear input dim for influence trace: {x.dim()}")
            if (
                self._anchor_module_name is not None
                and module_name == self._anchor_module_name
                and self._anchor_activation is None
            ):
                if x_tokens.numel() > 0:
                    x0 = x_tokens.reshape(-1)[0]
                    if x0.requires_grad:
                        self._anchor_activation = x0
            if self.current_capture is None:
                return
            capture_idx = self.current_capture["capture_idx"]
            if capture_idx.numel() == 0:
                return
            self._debug["forward_capture_calls"] += 1
            x_sel = x_tokens.index_select(0, capture_idx).to(torch.float32)
            p_in = getattr(module, "_inftrace_p_in")
            v = x_sel @ p_in.t()
            setattr(module, "_inftrace_v", v)
            self._debug["forward_set_v_calls"] += 1
        return hook

    def _consume_output_grad(
        self,
        *,
        module_name: str,
        module: nn.Module,
        dy: torch.Tensor,
        v: torch.Tensor,
        capture: dict[str, torch.Tensor] | None,
    ) -> None:
        if capture is None or not self._store_enabled:
            return
        self._debug["output_grad_hook_calls"] += 1
        if dy.dim() == 3:
            if dy.shape[0] != 1:
                raise ValueError("Influence trace currently expects rmpad grad output with batch axis = 1.")
            dy_tokens = dy.squeeze(0)
        elif dy.dim() == 2:
            dy_tokens = dy
        else:
            raise ValueError(f"Unsupported linear grad dim for influence trace: {dy.dim()}")
        capture_idx = capture["capture_idx"]
        dy_sel = dy_tokens.index_select(0, capture_idx).to(torch.float32)
        p_out = getattr(module, "_inftrace_p_out")
        u = dy_sel @ p_out.t()
        if self.cfg.grad_offload_to_cpu:
            u = u.detach().cpu()
            v = v.detach().cpu()
            row_id = capture["row_id"].detach().cpu()
            token_pos = capture["token_pos"].detach().cpu()
            group_id = capture["group_id"].detach().cpu()
            accepted = capture["accepted"].detach().cpu()
        else:
            u = u.detach()
            v = v.detach()
            row_id = capture["row_id"]
            token_pos = capture["token_pos"]
            group_id = capture["group_id"]
            accepted = capture["accepted"]
        bucket = self.storage[module_name]
        bucket["u"].append(u)
        bucket["v"].append(v)
        bucket["row_id"].append(row_id)
        bucket["token_pos"].append(token_pos)
        bucket["group_id"].append(group_id)
        bucket["accepted"].append(accepted)
        self._debug["stored_chunks"] += 1

    def _make_backward_hook(self, module_name: str):
        def hook(module: nn.Module, grad_input, grad_output):
            self._debug["backward_hook_calls"] += 1
            if self.current_capture is None:
                if hasattr(module, "_inftrace_v"):
                    delattr(module, "_inftrace_v")
                return
            if not hasattr(module, "_inftrace_v"):
                return
            if not self._store_enabled:
                delattr(module, "_inftrace_v")
                return
            dy = grad_output[0]
            if dy is None:
                delattr(module, "_inftrace_v")
                return
            self._debug["output_grad_hook_calls"] += 1
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
            if self.cfg.grad_offload_to_cpu:
                u = u.detach().cpu()
                v = v.detach().cpu()
                row_id = self.current_capture["row_id"].detach().cpu()
                token_pos = self.current_capture["token_pos"].detach().cpu()
                group_id = self.current_capture["group_id"].detach().cpu()
                accepted = self.current_capture["accepted"].detach().cpu()
            else:
                u = u.detach()
                v = v.detach()
                row_id = self.current_capture["row_id"]
                token_pos = self.current_capture["token_pos"]
                group_id = self.current_capture["group_id"]
                accepted = self.current_capture["accepted"]
            bucket = self.storage[module_name]
            bucket["u"].append(u)
            bucket["v"].append(v)
            bucket["row_id"].append(row_id)
            bucket["token_pos"].append(token_pos)
            bucket["group_id"].append(group_id)
            bucket["accepted"].append(accepted)
            delattr(module, "_inftrace_v")
        return hook

    def clear_storage(self) -> None:
        for name in self.storage:
            for key in self.storage[name]:
                self.storage[name][key].clear()
        self.current_capture = None
        self._anchor_activation = None

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
        self._debug["capture_begin_calls"] += 1
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
        token_cap = int(self.cfg.max_tokens_per_response)
        if token_cap > 0 and valid.any():
            capped = torch.zeros_like(valid)
            selected_idx = torch.nonzero(selected_rows, as_tuple=False).flatten()
            for rid in selected_idx.tolist():
                tok_idx = torch.nonzero(valid[rid], as_tuple=False).flatten()
                if tok_idx.numel() == 0:
                    continue
                if tok_idx.numel() > token_cap:
                    pick = torch.linspace(
                        0,
                        tok_idx.numel() - 1,
                        steps=token_cap,
                        device=tok_idx.device,
                    ).round().to(torch.long)
                    tok_idx = tok_idx.index_select(0, pick)
                capped[rid, tok_idx] = True
            valid = capped
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
        self._anchor_activation = None
        self._store_enabled = True
        self._debug["capture_begin_nonempty"] += 1
        self._debug["capture_selected_tokens"] += int(keep.sum().item())
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
        self._debug["anchor_tensor_ready"] += int(self._anchor_activation is not None)
        self._anchor_activation = None
        self._store_enabled = True

    def suspend_storage(self) -> None:
        self._store_enabled = False

    def debug_stats(self, *, reset: bool = False) -> dict[str, int]:
        out = dict(self._debug)
        if reset:
            self._reset_debug()
        return out

    def pop_timing(self, *, reset: bool = True) -> dict[str, float]:
        out = dict(self._timing)
        if reset:
            self._reset_timing()
        return out

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

    def _is_rank0(self) -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _cat_and_stage(
        self,
        chunks: list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not chunks:
            out_dtype = dtype if dtype is not None else torch.float32
            return torch.empty(0, device=device, dtype=out_dtype)
        out = torch.cat(chunks, dim=0)
        want_dtype = dtype if dtype is not None else out.dtype
        if out.device.type == "cpu" and device.type == "cuda" and self.cfg.grad_offload_to_cpu and not out.is_pinned():
            out = out.pin_memory()
        if out.device != device or out.dtype != want_dtype:
            out = out.to(device=device, dtype=want_dtype, non_blocking=(out.device.type == "cpu" and device.type == "cuda"))
        return out

    def _module_compute_device(self, module_name: str) -> torch.device:
        mod = self.modules.get(module_name)
        if self.cfg.force_gpu_compute and torch.cuda.is_available():
            if mod is not None and mod.weight.device.type == "cuda":
                return mod.weight.device
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if mod is None:
            return torch.device("cpu")
        device = mod.weight.device
        if device.type == "cuda":
            return device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return device

    def _module_token_scores_identity(
        self,
        *,
        row_g: torch.Tensor,
        tok_g: torch.Tensor,
        u_g: torch.Tensor,
        v_g: torch.Tensor,
        acc_g: torch.Tensor,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uniq_rows, inv = torch.unique(row_g, sorted=True, return_inverse=True)
        row_acc = torch.zeros((uniq_rows.numel(),), device=u_g.device, dtype=torch.bool)
        row_acc.index_put_((inv,), acc_g, accumulate=False)
        if bool(row_acc.all().item()) or bool((~row_acc).all().item()):
            return (
                torch.empty(0, device=u_g.device, dtype=torch.int64),
                torch.empty(0, device=u_g.device, dtype=torch.float32),
            )

        if self.cfg.contrastive_agg == "mean":
            n_acc = row_acc.sum().float().clamp(min=1)
            n_rej = (~row_acc).sum().float().clamp(min=1)
            row_weight = torch.where(row_acc, 1.0 / n_acc, -1.0 / n_rej)
        else:
            row_weight = torch.where(row_acc, 1.0, -1.0)
        token_weight = row_weight.index_select(0, inv)
        weighted_u = u_g * token_weight.unsqueeze(-1)

        n_tokens = int(u_g.shape[0])
        if n_tokens == 0:
            return (
                torch.empty(0, device=u_g.device, dtype=torch.int64),
                torch.empty(0, device=u_g.device, dtype=torch.float32),
            )

        # For identity mode:
        # infl(t->r) = g_r^T g_t
        # g_r is response-gradient sum, and each token gradient is rank-1 vec(u v^T).
        # Let M = sum_i (w_i * u_i v_i^T), then score_t = u_t^T M v_t.
        # This is O(n_tokens * D) and avoids token-token O(n^2) work.
        m = weighted_u.transpose(0, 1) @ v_g  # [k_out, k_in]
        uv = u_g @ m  # [n_tokens, k_in]
        scores = torch.sum(uv * v_g, dim=-1)

        if self.cfg.exclude_self_response:
            for ridx in range(uniq_rows.numel()):
                rmask = inv == ridx
                u_r = u_g[rmask]
                v_r = v_g[rmask]
                m_r = u_r.transpose(0, 1) @ v_r  # [k_out, k_in]
                correction = torch.sum((u_g[rmask] @ m_r) * v_g[rmask], dim=-1)
                scores[rmask] -= row_weight[ridx] * correction

        pair = row_g * stride + tok_g
        return pair, scores

    def _module_token_scores(self, module_name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        bucket = self.storage[module_name]
        if not bucket["u"]:
            return (
                torch.empty(0, device="cpu", dtype=torch.int64),
                torch.empty(0, device="cpu", dtype=torch.float32),
                0,
            )
        do_timing = self.cfg.profile_timing and torch.cuda.is_available()

        def _sync() -> None:
            if do_timing:
                torch.cuda.synchronize()

        compute_device = self._module_compute_device(module_name)
        _sync()
        t_stage = time.perf_counter()
        u = self._cat_and_stage(bucket["u"], device=compute_device, dtype=torch.float32)
        v = self._cat_and_stage(bucket["v"], device=compute_device, dtype=torch.float32)
        row_id = self._cat_and_stage(bucket["row_id"], device=compute_device, dtype=torch.int64)
        token_pos = self._cat_and_stage(bucket["token_pos"], device=compute_device, dtype=torch.int64)
        group_id = self._cat_and_stage(bucket["group_id"], device=compute_device, dtype=torch.int64)
        accepted = self._cat_and_stage(bucket["accepted"], device=compute_device, dtype=torch.bool)
        _sync()
        self._timing["grad_staging_s"] += time.perf_counter() - t_stage

        if not self._compute_mode_logged and self._is_rank0():
            print(
                "[influence_trace] runtime compute="
                f"{compute_device}, grad_storage={'cpu' if self.cfg.grad_offload_to_cpu else 'gpu'}, "
                f"hessian_mode={self.cfg.hessian_mode}, force_gpu_compute={self.cfg.force_gpu_compute}"
            )
            self._compute_mode_logged = True
        response_len = int(self.current_capture["response_len"].item()) if self.current_capture is not None else int(token_pos.max().item()) + 1
        stride = response_len + 1
        use_all_selected = self.cfg.accepted_rejected_scope == "all_selected"
        unique_groups = [None] if use_all_selected else torch.unique(group_id, sorted=True).tolist()
        pair_out: list[torch.Tensor] = []
        score_out: list[torch.Tensor] = []
        for gid in unique_groups:
            if gid is None:
                g_mask = torch.ones_like(group_id, dtype=torch.bool)
            else:
                g_mask = group_id == gid
            if int(g_mask.sum().item()) == 0:
                continue
            self._debug["groups_total"] += 1
            row_g = row_id[g_mask]
            tok_g = token_pos[g_mask]
            u_g = u[g_mask]
            v_g = v[g_mask]
            acc_g = accepted[g_mask]
            if self.cfg.hessian_mode == "identity":
                _sync()
                t_tok = time.perf_counter()
                pair, score = self._module_token_scores_identity(
                    row_g=row_g,
                    tok_g=tok_g,
                    u_g=u_g,
                    v_g=v_g,
                    acc_g=acc_g,
                    stride=stride,
                )
                _sync()
                self._timing["token_scoring_s"] += time.perf_counter() - t_tok
                if pair.numel() == 0:
                    continue
            else:
                _sync()
                t_tok = time.perf_counter()
                g_tok = torch.bmm(u_g.unsqueeze(2), v_g.unsqueeze(1)).reshape(u_g.shape[0], -1)
                uniq_rows, inv = torch.unique(row_g, sorted=True, return_inverse=True)
                g_resp = torch.zeros((uniq_rows.numel(), g_tok.shape[1]), device=g_tok.device, dtype=g_tok.dtype)
                g_resp.index_add_(0, inv, g_tok)
                row_acc = torch.zeros((uniq_rows.numel(),), device=g_tok.device, dtype=torch.bool)
                row_acc.index_put_((inv,), acc_g, accumulate=False)
                if bool(row_acc.all().item()) or bool((~row_acc).all().item()):
                    self._debug["groups_skipped_all_same"] += 1
                    _sync()
                    self._timing["token_scoring_s"] += time.perf_counter() - t_tok
                    continue
                _sync()
                self._timing["token_scoring_s"] += time.perf_counter() - t_tok

                _sync()
                t_hess = time.perf_counter()
                hess_source = g_tok if self.cfg.hessian_source == "token" else g_resp
                solved = self._compute_kernel_solve(g_resp=hess_source, g_tok=g_tok, reg_lambda=self.cfg.reg_lambda)
                _sync()
                self._timing["hessian_solve_s"] += time.perf_counter() - t_hess

                _sync()
                t_tok2 = time.perf_counter()
                infl = g_resp @ solved  # [n_resp, n_tokens]
                if self.cfg.contrastive_agg == "mean":
                    score = infl[row_acc].mean(dim=0) - infl[~row_acc].mean(dim=0)
                else:
                    score = infl[row_acc].sum(dim=0) - infl[~row_acc].sum(dim=0)

                if self.cfg.exclude_self_response:
                    self_infl = infl[inv, torch.arange(inv.numel(), device=inv.device)]
                    if self.cfg.contrastive_agg == "mean":
                        n_acc = row_acc.sum().float()
                        n_rej = (~row_acc).sum().float()
                        sum_acc = infl[row_acc].sum(dim=0)
                        sum_rej = infl[~row_acc].sum(dim=0)
                        score = torch.where(
                            acc_g,
                            (sum_acc - self_infl) / (n_acc - 1).clamp(min=1) - sum_rej / n_rej.clamp(min=1),
                            sum_acc / n_acc.clamp(min=1) - (sum_rej - self_infl) / (n_rej - 1).clamp(min=1),
                        )
                    else:
                        sign_per_token = torch.where(acc_g, 1.0, -1.0)
                        score -= sign_per_token * self_infl

                pair = row_g * stride + tok_g
                _sync()
                self._timing["token_scoring_s"] += time.perf_counter() - t_tok2

            _sync()
            t_agg = time.perf_counter()
            pair_out.append(pair)
            score_out.append(score)
            _sync()
            self._timing["score_aggregation_s"] += time.perf_counter() - t_agg

        if not pair_out:
            return (
                torch.empty(0, device="cpu", dtype=torch.int64),
                torch.empty(0, device="cpu", dtype=torch.float32),
                response_len,
            )
        _sync()
        t_agg_final = time.perf_counter()
        result = torch.cat(pair_out, dim=0), torch.cat(score_out, dim=0), response_len
        _sync()
        self._timing["score_aggregation_s"] += time.perf_counter() - t_agg_final
        return result

    def pop_token_influence_rows(self) -> list[dict[str, Any]]:
        if not self.storage:
            return []
        score_map: torch.Tensor | None = None
        token_seen: torch.Tensor | None = None
        row_seen: torch.Tensor | None = None
        response_len = 0

        module_names = list(self.storage.keys())
        n_modules = len(module_names)
        t0 = torch.cuda.Event(enable_timing=True) if self.cfg.profile_timing and torch.cuda.is_available() else None
        t1 = torch.cuda.Event(enable_timing=True) if self.cfg.profile_timing and torch.cuda.is_available() else None
        for mod_idx, module_name in enumerate(module_names):
            if self.cfg.profile_timing and self._is_rank0() and (mod_idx == 0 or (mod_idx + 1) % 20 == 0 or (mod_idx + 1) == n_modules):
                if t0 is not None and t1 is not None:
                    t0.record()
            pairs, scores, cur_response_len = self._module_token_scores(module_name)
            if self.cfg.profile_timing and self._is_rank0() and (mod_idx == 0 or (mod_idx + 1) % 20 == 0 or (mod_idx + 1) == n_modules):
                if t0 is not None and t1 is not None:
                    t1.record()
                    torch.cuda.synchronize()
                    elapsed_ms = float(t0.elapsed_time(t1))
                    print(
                        f"[influence_trace][pop_rows] module={mod_idx + 1}/{n_modules} "
                        f"name={module_name} elapsed_ms={elapsed_ms:.2f}"
                    )
            if pairs.numel() == 0:
                continue

            local_response_len = int(cur_response_len)
            local_stride = local_response_len + 1
            row_id = torch.div(pairs, local_stride, rounding_mode="floor")
            token_pos = torch.remainder(pairs, local_stride)

            if score_map is None:
                response_len = local_response_len
                max_row = int(row_id.max().item()) if row_id.numel() > 0 else -1
                score_map = torch.zeros((max_row + 1, response_len), device=scores.device, dtype=torch.float32)
                token_seen = torch.zeros((max_row + 1, response_len), device=scores.device, dtype=torch.bool)
                row_seen = torch.zeros((max_row + 1,), device=scores.device, dtype=torch.bool)
            else:
                assert token_seen is not None
                assert row_seen is not None
                if local_response_len > response_len:
                    grow = local_response_len - response_len
                    score_map = torch.nn.functional.pad(score_map, (0, grow))
                    token_seen = torch.nn.functional.pad(token_seen, (0, grow))
                    response_len = local_response_len
                max_row = int(row_id.max().item()) if row_id.numel() > 0 else -1
                if max_row + 1 > score_map.shape[0]:
                    grow_rows = max_row + 1 - score_map.shape[0]
                    score_map = torch.cat(
                        [score_map, torch.zeros((grow_rows, response_len), device=score_map.device, dtype=score_map.dtype)], dim=0
                    )
                    token_seen = torch.cat(
                        [token_seen, torch.zeros((grow_rows, response_len), device=token_seen.device, dtype=token_seen.dtype)], dim=0
                    )
                    row_seen = torch.cat(
                        [row_seen, torch.zeros((grow_rows,), device=row_seen.device, dtype=row_seen.dtype)], dim=0
                    )

                if scores.device != score_map.device:
                    scores = scores.to(score_map.device, non_blocking=True)
                    row_id = row_id.to(score_map.device, non_blocking=True)
                    token_pos = token_pos.to(score_map.device, non_blocking=True)

            if response_len <= 0:
                continue
            valid = token_pos < response_len
            if not bool(valid.all().item()):
                row_id = row_id[valid]
                token_pos = token_pos[valid]
                scores = scores[valid]
                if row_id.numel() == 0:
                    continue

            assert score_map is not None
            assert token_seen is not None
            assert row_seen is not None
            flat = row_id * response_len + token_pos
            score_map.view(-1).index_add_(0, flat, scores)
            token_seen.view(-1)[flat] = True
            row_seen[row_id] = True

        if score_map is None or token_seen is None or row_seen is None:
            self.clear_storage()
            return []

        uniq_rows = torch.nonzero(row_seen, as_tuple=False).flatten()
        if uniq_rows.numel() == 0:
            self.clear_storage()
            return []
        row_records: list[dict[str, Any]] = []
        for rid in uniq_rows.tolist():
            influence = torch.full((response_len,), float("nan"), device=score_map.device, dtype=torch.float32)
            mask = token_seen[rid]
            influence[mask] = score_map[rid][mask]
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
        if self.cfg.hessian_mode == "identity":
            for name in self.module_dims:
                out[name] = 0.0
            return out
        for name, (k_in, k_out) in self.module_dims.items():
            dim = int(k_in * k_out)
            bytes_peak = 2 * dim * dim * 4
            out[name] = float(bytes_peak) / (1024.0 * 1024.0)
        return out
