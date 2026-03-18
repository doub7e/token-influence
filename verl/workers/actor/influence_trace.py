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
    self_influence_scale: float = 0.0  # when exclude_self_response=True, add back this fraction of self-influence (0=full exclude, 1=full include)
    contrastive_agg: str = "sum"       # "sum" or "mean"
    hessian_source: str = "response"   # "response" or "token"
    debug_hessian_similarity: bool = False  # log cross-prompt Hessian similarity
    score_normalization: str = "none"  # "none" | "h_inv_norm" — per-token normalization to remove gradient magnitude
    token_unit_norm: bool = False  # normalize per-token gradient vectors to unit norm before scoring (TrackStar-style)

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
        if output_function == "log_prob_reward":
            output_function = "log_prob"  # backward-compatible alias
        if output_function not in {"training_loss", "log_prob_advantage", "log_prob"}:
            output_function = "training_loss"
        accepted_rejected_scope = str(raw.get("accepted_rejected_scope", "per_prompt")).strip().lower()
        if accepted_rejected_scope not in {"per_prompt", "all_selected", "global_selected"}:
            accepted_rejected_scope = "per_prompt"
        contrastive_agg = str(raw.get("contrastive_agg", "sum")).strip().lower()
        if contrastive_agg not in {"sum", "mean", "advantage"}:
            contrastive_agg = "sum"
        hessian_source = str(raw.get("hessian_source", "response")).strip().lower()
        if hessian_source not in {"response", "token"}:
            hessian_source = "response"
        score_normalization = str(raw.get("score_normalization", "none")).strip().lower()
        if score_normalization not in {"none", "h_inv_norm"}:
            score_normalization = "none"
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
            max_hessian_dim=int(raw.get("max_hessian_dim", 2500)),
            max_tokens_per_response=int(raw.get("max_tokens_per_response", -1)),
            skip_optimizer_step=bool(raw.get("skip_optimizer_step", False)),
            grad_offload_to_cpu=bool(raw.get("grad_offload_to_cpu", False)),
            force_gpu_compute=bool(raw.get("force_gpu_compute", True)),
            profile_timing=bool(raw.get("profile_timing", False)),
            exclude_self_response=bool(raw.get("exclude_self_response", False)),
            self_influence_scale=float(raw.get("self_influence_scale", 0.0)),
            contrastive_agg=contrastive_agg,
            hessian_source=hessian_source,
            debug_hessian_similarity=bool(raw.get("debug_hessian_similarity", False)),
            score_normalization=score_normalization,
            token_unit_norm=bool(raw.get("token_unit_norm", False)),
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
        # Shared metadata storage: stored once per microbatch, referenced by all modules
        self._shared_meta: dict[str, list[torch.Tensor]] = {
            "row_id": [], "token_pos": [], "group_id": [], "accepted": [], "advantage": [],
        }
        self._shared_meta_count: int = 0  # number of shared metadata chunks
        self._anchor_module_name: str | None = None
        self._anchor_activation: torch.Tensor | None = None
        self._store_enabled: bool = True
        self._compute_mode_logged = False
        self._debug: dict[str, int] = {}
        self._timing: dict[str, float] = {}
        self._hessian_diag: dict[str, list[torch.Tensor]] = {}  # module -> list of H per group
        self._grad_dot_accum: torch.Tensor | None = None  # [max_row+1, max_row+1] accumulated dot product
        self._grad_hinv_dot_accum: torch.Tensor | None = None  # [max_row+1, max_row+1] H^{-1}-weighted dot
        self._grad_row_meta: dict[int, tuple[bool, int]] = {}  # row_id -> (accepted, group_id)
        self._debug_unproj_module: str | None = None  # one module for unprojected comparison
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
            name: {"u": [], "v": []}
            for name in self.modules
        }
        self._shared_meta = {
            "row_id": [], "token_pos": [], "group_id": [], "accepted": [], "advantage": [],
        }
        self._shared_meta_count = 0
        # Pick a middle-layer k_proj for unprojected vs projected comparison
        # (layer 0 may lack data due to anchor parameter interaction)
        if self.cfg.debug_hessian_similarity:
            k_proj_names = [n for n in self.modules if "k_proj" in n]
            if k_proj_names:
                pick = k_proj_names[len(k_proj_names) // 2]
                self._debug_unproj_module = pick
                self.storage[pick]["raw_x"] = []
                self.storage[pick]["raw_dy"] = []
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
            if self._debug_unproj_module == module_name:
                setattr(module, "_inftrace_x_raw", x_sel.detach())
            self._debug["forward_set_v_calls"] += 1
        return hook

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
            # Store u, v in bf16 (convert to fp32 only during scoring)
            bucket = self.storage[module_name]
            if self.cfg.grad_offload_to_cpu:
                bucket["u"].append(u.detach().to(torch.bfloat16).cpu())
                bucket["v"].append(v.detach().to(torch.bfloat16).cpu())
            else:
                bucket["u"].append(u.detach().to(torch.bfloat16))
                bucket["v"].append(v.detach().to(torch.bfloat16))
            self._debug["stored_chunks"] += 1
            # Store shared metadata only once per microbatch (first module stores it)
            if self._shared_meta_count == len(bucket["u"]) - 1:
                # This is the first module to store for this microbatch chunk
                _to_cpu = self.cfg.grad_offload_to_cpu
                self._shared_meta["row_id"].append(self.current_capture["row_id"].detach().cpu() if _to_cpu else self.current_capture["row_id"].detach())
                self._shared_meta["token_pos"].append(self.current_capture["token_pos"].detach().cpu() if _to_cpu else self.current_capture["token_pos"].detach())
                self._shared_meta["group_id"].append(self.current_capture["group_id"].detach().cpu() if _to_cpu else self.current_capture["group_id"].detach())
                self._shared_meta["accepted"].append(self.current_capture["accepted"].detach().cpu() if _to_cpu else self.current_capture["accepted"].detach())
                self._shared_meta["advantage"].append(self.current_capture["advantage"].detach().cpu() if _to_cpu else self.current_capture["advantage"].detach())
                self._shared_meta_count += 1
            if self._debug_unproj_module == module_name:
                x_raw = getattr(module, "_inftrace_x_raw", None)
                if x_raw is not None:
                    bucket.setdefault("raw_x", []).append(x_raw)
                    bucket.setdefault("raw_dy", []).append(dy_sel.detach())
                    delattr(module, "_inftrace_x_raw")
            delattr(module, "_inftrace_v")
        return hook

    def clear_storage(self) -> None:
        for name in self.storage:
            for key in self.storage[name]:
                self.storage[name][key].clear()
        for key in self._shared_meta:
            self._shared_meta[key].clear()
        self._shared_meta_count = 0
        self._staged_meta = None
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
        advantage: torch.Tensor | None = None,
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
            "advantage": advantage[row_valid].to(torch.float32) if advantage is not None else torch.zeros(row_valid.shape[0], device=indices.device, dtype=torch.float32),
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
        return_chol: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dim = g_resp.shape[1]
        hessian = g_resp.T @ g_resp
        if reg_lambda > 0:
            reg = reg_lambda
        else:
            reg = float((hessian.trace() / max(dim, 1)).item()) * 0.1
        hessian = hessian + reg * torch.eye(dim, device=hessian.device, dtype=hessian.dtype)
        chol = torch.linalg.cholesky(hessian)
        solved = torch.cholesky_solve(g_tok.T, chol)
        if return_chol:
            return solved, chol
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
        adv_g: torch.Tensor | None,
        stride: int,
        use_global: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uniq_rows, inv = torch.unique(row_g, sorted=True, return_inverse=True)

        n_tokens = int(u_g.shape[0])
        _empty = (
            torch.empty(0, device=u_g.device, dtype=torch.int64),
            torch.empty(0, device=u_g.device, dtype=torch.float32),
        )
        if n_tokens == 0:
            return _empty

        # TrackStar-style: normalize per-token projected gradients to unit norm
        if self.cfg.token_unit_norm:
            u_g = u_g / u_g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_g = v_g / v_g.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if self.cfg.contrastive_agg == "advantage":
            if adv_g is None:
                return _empty
            row_adv = torch.zeros((uniq_rows.numel(),), device=u_g.device, dtype=torch.float32)
            row_adv.index_put_((inv,), adv_g, accumulate=False)
            if row_adv.abs().max() < 1e-8:
                return _empty
            # Advantage-weighted: M_full = Σ_k A_k · u_k v_k^T
            token_adv = row_adv[inv]
            weighted_u = u_g * token_adv.unsqueeze(-1)
            m_full = weighted_u.transpose(0, 1) @ v_g  # [k_out, k_in]
            scores = torch.sum((u_g @ m_full) * v_g, dim=-1)  # [n_tokens]
            # Leave-one-out: subtract per-response self-contribution
            for ridx in range(uniq_rows.numel()):
                rmask = inv == ridx
                u_r, v_r = u_g[rmask], v_g[rmask]
                m_r = u_r.transpose(0, 1) @ v_r
                self_contrib = torch.sum((u_g[rmask] @ m_r) * v_g[rmask], dim=-1)
                scores[rmask] -= row_adv[ridx] * self_contrib
            pair = row_g * stride + tok_g
            return pair, scores

        # --- Binary contrastive modes (sum / mean) ---
        row_acc = torch.zeros((uniq_rows.numel(),), device=u_g.device, dtype=torch.bool)
        row_acc.index_put_((inv,), acc_g, accumulate=False)

        if use_global and self.cfg.contrastive_agg in ("sum", "mean"):
            # --- Cross-GPU contrastive M (global_selected, identity mode) ---
            # M = Σ u_i v_i^T for acc/rej groups. All-reduce M_acc, M_rej, counts.
            # Counts are per-response (not per-token) to match mean-mode semantics.
            k_out, k_in = u_g.shape[1], v_g.shape[1]
            _dev, _dt = u_g.device, u_g.dtype
            m_acc = (u_g[acc_g].T @ v_g[acc_g]) if acc_g.any() else torch.zeros(k_out, k_in, device=_dev, dtype=_dt)
            m_rej = (u_g[~acc_g].T @ v_g[~acc_g]) if (~acc_g).any() else torch.zeros(k_out, k_in, device=_dev, dtype=_dt)
            counts = torch.tensor([float(row_acc.sum()), float((~row_acc).sum())], device=_dev, dtype=_dt)
            buf = torch.cat([m_acc.reshape(-1), m_rej.reshape(-1), counts])
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            flat_size = k_out * k_in
            m_acc_g = buf[:flat_size].reshape(k_out, k_in)
            m_rej_g = buf[flat_size:2*flat_size].reshape(k_out, k_in)
            n_acc_g, n_rej_g = buf[2*flat_size].clamp(min=1), buf[2*flat_size+1].clamp(min=1)

            if n_acc_g < 1 or n_rej_g < 1:
                return _empty

            if self.cfg.contrastive_agg == "mean":
                m_global = m_acc_g / n_acc_g - m_rej_g / n_rej_g
            else:
                m_global = m_acc_g - m_rej_g
            scores = torch.sum((u_g @ m_global) * v_g, dim=-1)  # [n_tokens]

            if self.cfg.exclude_self_response:
                for ridx in range(uniq_rows.numel()):
                    rmask = inv == ridx
                    u_r, v_r = u_g[rmask], v_g[rmask]
                    m_r = u_r.T @ v_r  # [k_out, k_in]
                    correction = torch.sum((u_g[rmask] @ m_r) * v_g[rmask], dim=-1)
                    is_acc = bool(row_acc[ridx].item())
                    if self.cfg.contrastive_agg == "mean":
                        if is_acc:
                            w_self = 1.0 / n_acc_g
                            w_loo = 1.0 / (n_acc_g - 1).clamp(min=1)
                        else:
                            w_self = -1.0 / n_rej_g
                            w_loo = -1.0 / (n_rej_g - 1).clamp(min=1)
                        scores[rmask] -= w_self * correction
                        _scale = self.cfg.self_influence_scale
                        if _scale <= 0:
                            pass  # full exclude: already removed
                        elif _scale < 1:
                            scores[rmask] += _scale * w_self * correction
                        else:
                            scores[rmask] += w_self * correction
                    else:
                        sign = 1.0 if is_acc else -1.0
                        _scale = self.cfg.self_influence_scale
                        if _scale <= 0:
                            scores[rmask] -= sign * correction
                        elif _scale < 1:
                            scores[rmask] -= (1.0 - _scale) * sign * correction

            pair = row_g * stride + tok_g
            return pair, scores

        # --- Local contrastive (all_selected / per_prompt) ---
        if bool(row_acc.all().item()) or bool((~row_acc).all().item()):
            return _empty

        if self.cfg.contrastive_agg == "mean":
            n_acc = row_acc.sum().float().clamp(min=1)
            n_rej = (~row_acc).sum().float().clamp(min=1)
            row_weight = torch.where(row_acc, 1.0 / n_acc, -1.0 / n_rej)
        else:
            row_weight = torch.where(row_acc, 1.0, -1.0)
        token_weight = row_weight.index_select(0, inv)
        weighted_u = u_g * token_weight.unsqueeze(-1)

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
                # Add back scaled self-influence
                if self.cfg.self_influence_scale > 0:
                    scores[rmask] += self.cfg.self_influence_scale * row_weight[ridx] * correction

        pair = row_g * stride + tok_g
        return pair, scores

    def _stage_shared_meta(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Stage shared metadata once, cache for reuse across modules."""
        if not hasattr(self, "_staged_meta") or self._staged_meta is None:
            row_id = self._cat_and_stage(self._shared_meta["row_id"], device=device, dtype=torch.int64)
            token_pos = self._cat_and_stage(self._shared_meta["token_pos"], device=device, dtype=torch.int64)
            group_id = self._cat_and_stage(self._shared_meta["group_id"], device=device, dtype=torch.int64)
            accepted = self._cat_and_stage(self._shared_meta["accepted"], device=device, dtype=torch.bool)
            advantage = self._cat_and_stage(self._shared_meta["advantage"], device=device, dtype=torch.float32)
            self._staged_meta = (row_id, token_pos, group_id, accepted, advantage)
        return self._staged_meta

    def _module_token_scores(self, module_name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        bucket = self.storage[module_name]
        if not bucket["u"]:
            return (
                torch.empty(0, device="cpu", dtype=torch.int64),
                torch.empty(0, device="cpu", dtype=torch.float32),
                0,
            )

        compute_device = self._module_compute_device(module_name)
        t_stage = time.perf_counter()
        # Stage u, v from bf16 to fp32 on compute device
        u = self._cat_and_stage(bucket["u"], device=compute_device, dtype=torch.float32)
        v = self._cat_and_stage(bucket["v"], device=compute_device, dtype=torch.float32)
        # Use shared metadata (staged once, reused across all modules)
        row_id, token_pos, group_id, accepted, advantage = self._stage_shared_meta(compute_device)
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
        use_all_selected = self.cfg.accepted_rejected_scope in ("all_selected", "global_selected")
        use_global = (
            self.cfg.accepted_rejected_scope == "global_selected"
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        unique_groups = [None] if use_all_selected else torch.unique(group_id, sorted=True).tolist()
        pair_out: list[torch.Tensor] = []
        score_out: list[torch.Tensor] = []
        _diag_grads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        _diag_hess_accum: torch.Tensor | None = None  # [D, D] accumulated hessian (GPU, float64)
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
            adv_g = advantage[g_mask]
            if self.cfg.hessian_mode == "identity":
                t_tok = time.perf_counter()
                pair, score = self._module_token_scores_identity(
                    row_g=row_g,
                    tok_g=tok_g,
                    u_g=u_g,
                    v_g=v_g,
                    acc_g=acc_g,
                    adv_g=adv_g,
                    stride=stride,
                    use_global=use_global,
                )
                self._timing["token_scoring_s"] += time.perf_counter() - t_tok
                if pair.numel() == 0:
                    continue
            else:
                t_tok = time.perf_counter()
                uniq_rows, inv = torch.unique(row_g, sorted=True, return_inverse=True)
                D = u_g.shape[1] * v_g.shape[1]  # k_out * k_in

                # --- Memory-efficient chunked gradient reconstruction ---
                # Instead of materializing full [n_tokens, D] g_tok at once,
                # we process chunks to bound peak GPU memory.
                # g_tok = vec(u ⊗ v) has shape [n_tokens, D].
                # With 200k tokens and D=50k, that's 40 GiB — too much.
                # We chunk to keep each chunk under ~4 GiB.
                _MAX_CHUNK_BYTES = 4 * 1024**3  # 4 GiB
                _chunk_size = max(1, _MAX_CHUNK_BYTES // (D * 4))  # fp32
                n_tokens = u_g.shape[0]
                _dev, _dt = u_g.device, u_g.dtype

                # Step 1: Build g_resp and hessian via chunked accumulation.
                # g_resp[r] = Σ_{t in r} g_t, hessian = G^T G (or g_resp^T g_resp)
                g_resp = torch.zeros((uniq_rows.numel(), D), device=_dev, dtype=_dt)
                use_token_hessian = (self.cfg.hessian_source == "token")
                hessian = torch.zeros((D, D), device=_dev, dtype=_dt)

                for c_start in range(0, n_tokens, _chunk_size):
                    c_end = min(c_start + _chunk_size, n_tokens)
                    g_chunk = torch.bmm(
                        u_g[c_start:c_end].unsqueeze(2),
                        v_g[c_start:c_end].unsqueeze(1),
                    ).reshape(c_end - c_start, -1)
                    g_resp.index_add_(0, inv[c_start:c_end], g_chunk)
                    if use_token_hessian:
                        hessian.addmm_(g_chunk.T, g_chunk)
                    del g_chunk

                row_acc = torch.zeros((uniq_rows.numel(),), device=_dev, dtype=torch.bool)
                row_acc.index_put_((inv,), acc_g, accumulate=False)
                if self.cfg.contrastive_agg == "advantage":
                    row_adv = torch.zeros((uniq_rows.numel(),), device=_dev, dtype=torch.float32)
                    row_adv.index_put_((inv,), adv_g, accumulate=False)
                    if row_adv.abs().max() < 1e-8:
                        self._debug["groups_skipped_all_same"] += 1
                        self._timing["token_scoring_s"] += time.perf_counter() - t_tok
                        continue
                elif not use_global and (bool(row_acc.all().item()) or bool((~row_acc).all().item())):
                    self._debug["groups_skipped_all_same"] += 1
                    self._timing["token_scoring_s"] += time.perf_counter() - t_tok
                    continue
                self._timing["token_scoring_s"] += time.perf_counter() - t_tok

                # Step 2: Build and factorize hessian.
                t_hess = time.perf_counter()
                if not use_token_hessian:
                    hessian = g_resp.T @ g_resp
                reg = self.cfg.reg_lambda
                if reg <= 0:
                    reg = float((hessian.trace() / max(D, 1)).item()) * 0.1
                hessian.add_(torch.eye(D, device=_dev, dtype=_dt), alpha=reg)
                chol = torch.linalg.cholesky(hessian)
                del hessian

                # Step 3: Compute solved = H^{-1} g_tok in chunks and derive scores.
                # We never hold both g_tok and solved in full; instead we stream
                # chunks of g_tok through the pre-computed Cholesky factor.

                # First compute the contrastive direction from g_resp.
                if use_global and self.cfg.contrastive_agg in ("sum", "mean"):
                    _D = g_resp.shape[1]
                    sum_acc = g_resp[row_acc].sum(0) if row_acc.any() else torch.zeros(_D, device=_dev, dtype=_dt)
                    sum_rej = g_resp[~row_acc].sum(0) if (~row_acc).any() else torch.zeros(_D, device=_dev, dtype=_dt)
                    counts = torch.tensor([float(row_acc.sum()), float((~row_acc).sum())], device=_dev, dtype=_dt)
                    buf = torch.cat([sum_acc, sum_rej, counts])
                    torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
                    sum_acc_g, sum_rej_g = buf[:_D], buf[_D:2*_D]
                    n_acc_g, n_rej_g = buf[2*_D].clamp(min=1), buf[2*_D+1].clamp(min=1)
                    if n_acc_g < 1 or n_rej_g < 1:
                        self._debug["groups_skipped_all_same"] += 1
                        self._timing["hessian_solve_s"] += time.perf_counter() - t_hess
                        continue
                    if self.cfg.contrastive_agg == "mean":
                        d_contrastive = sum_acc_g / n_acc_g - sum_rej_g / n_rej_g
                    else:
                        d_contrastive = sum_acc_g - sum_rej_g
                    _is_global = True
                else:
                    d_contrastive = None
                    _is_global = False

                # Now stream g_tok chunks to compute per-token scores.
                score = torch.empty(n_tokens, device=_dev, dtype=_dt)

                for c_start in range(0, n_tokens, _chunk_size):
                    c_end = min(c_start + _chunk_size, n_tokens)
                    g_chunk = torch.bmm(
                        u_g[c_start:c_end].unsqueeze(2),
                        v_g[c_start:c_end].unsqueeze(1),
                    ).reshape(c_end - c_start, -1)
                    # solved_chunk = H^{-1} g_chunk^T → [D, chunk_size]
                    solved_chunk = torch.cholesky_solve(g_chunk.T, chol)

                    if self.cfg.token_unit_norm:
                        mahal_sq = (g_chunk * solved_chunk.T).sum(dim=-1)
                        mahal = mahal_sq.clamp(min=1e-16).sqrt()
                        solved_chunk = solved_chunk / mahal.unsqueeze(0)

                    if _is_global:
                        score[c_start:c_end] = d_contrastive @ solved_chunk
                        if self.cfg.exclude_self_response:
                            # self_infl for leave-one-out
                            local_infl_chunk = g_resp @ solved_chunk  # [n_resp, chunk]
                            self_infl = local_infl_chunk[inv[c_start:c_end], torch.arange(c_end - c_start, device=_dev)]
                            acc_chunk = acc_g[c_start:c_end]
                            if self.cfg.contrastive_agg == "mean":
                                sum_acc_score = sum_acc_g @ solved_chunk
                                sum_rej_score = sum_rej_g @ solved_chunk
                                score_exclude = torch.where(
                                    acc_chunk,
                                    (sum_acc_score - self_infl) / (n_acc_g - 1).clamp(min=1) - sum_rej_score / n_rej_g,
                                    sum_acc_score / n_acc_g - (sum_rej_score - self_infl) / (n_rej_g - 1).clamp(min=1),
                                )
                            else:
                                sign_per_token = torch.where(acc_chunk, 1.0, -1.0)
                                score_exclude = score[c_start:c_end] - sign_per_token * self_infl
                            _scale = self.cfg.self_influence_scale
                            if _scale <= 0:
                                score[c_start:c_end] = score_exclude
                            elif _scale < 1:
                                score[c_start:c_end] = (1.0 - _scale) * score_exclude + _scale * score[c_start:c_end]
                    else:
                        # Local contrastive scoring
                        infl_chunk = g_resp @ solved_chunk  # [n_resp, chunk]
                        inv_chunk = inv[c_start:c_end]
                        if self.cfg.contrastive_agg == "advantage":
                            full_score = (row_adv.unsqueeze(1) * infl_chunk).sum(dim=0)
                            self_contrib = row_adv[inv_chunk] * infl_chunk[inv_chunk, torch.arange(c_end - c_start, device=_dev)]
                            score[c_start:c_end] = full_score - self_contrib
                        elif self.cfg.contrastive_agg == "mean":
                            score[c_start:c_end] = infl_chunk[row_acc].mean(dim=0) - infl_chunk[~row_acc].mean(dim=0)
                        else:
                            score[c_start:c_end] = infl_chunk[row_acc].sum(dim=0) - infl_chunk[~row_acc].sum(dim=0)
                    del g_chunk, solved_chunk

                del chol
                self._timing["hessian_solve_s"] += time.perf_counter() - t_hess

                # --- Hessian similarity diagnostic ---
                if self.cfg.debug_hessian_similarity and self._is_rank0():
                    # Rebuild hessian for diagnostic (small cost)
                    hess_diag = g_resp.T @ g_resp
                    h_norm = hess_diag / hess_diag.norm().clamp(min=1e-12)
                    if module_name not in self._hessian_diag:
                        self._hessian_diag[module_name] = []
                    self._hessian_diag[module_name].append(h_norm.detach().cpu())
                    _diag_grads.append((
                        uniq_rows.detach().cpu(),
                        g_resp.detach().cpu(),
                        row_acc.detach().cpu(),
                    ))
                    _hs64 = g_resp.to(torch.float64)
                    _hh = _hs64.T @ _hs64
                    if _diag_hess_accum is None:
                        _diag_hess_accum = _hh
                    else:
                        _diag_hess_accum += _hh
                    del _hs64, _hh
                    _gid_int = int(gid) if gid is not None else -1
                    for k in range(uniq_rows.numel()):
                        rid = int(uniq_rows[k].item())
                        self._grad_row_meta[rid] = (bool(row_acc[k].item()), _gid_int)

                pair = row_g * stride + tok_g

            pair_out.append(pair)
            score_out.append(score)

        # Accumulate gradient dot products for this module across all groups
        if _diag_grads and self.cfg.debug_hessian_similarity and self._is_rank0():
            self._accumulate_grad_dot(_diag_grads, _diag_hess_accum)
            del _diag_hess_accum

        if not pair_out:
            return (
                torch.empty(0, device="cpu", dtype=torch.int64),
                torch.empty(0, device="cpu", dtype=torch.float32),
                response_len,
            )
        t_agg_final = time.perf_counter()
        result = torch.cat(pair_out, dim=0), torch.cat(score_out, dim=0), response_len
        self._timing["score_aggregation_s"] += time.perf_counter() - t_agg_final
        return result

    def pop_token_influence_rows(self) -> list[dict[str, Any]]:
        if not self.storage:
            return []
        # Reset staged metadata cache so it gets rebuilt for scoring
        self._staged_meta = None
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
        # --- Log diagnostics ---
        if self.cfg.debug_hessian_similarity and self._is_rank0():
            if self._hessian_diag:
                self._log_hessian_similarity()
                self._hessian_diag.clear()
            if self._grad_dot_accum is not None:
                self._log_gradient_similarity()
                self._grad_dot_accum = None
                self._grad_hinv_dot_accum = None
                self._grad_row_meta.clear()
            if self._debug_unproj_module is not None:
                self._log_unprojected_comparison()

        self.clear_storage()
        return row_records

    def _log_hessian_similarity(self) -> None:
        """Compute and log pairwise cosine similarity of Hessians across prompt groups."""
        # Pick a few representative modules to log
        module_types = {}  # type_suffix -> first module_name
        for name in self._hessian_diag:
            suffix = name.rsplit(".", 1)[-1] if "." in name else name
            if suffix not in module_types:
                module_types[suffix] = name

        print("[influence_trace][hessian_similarity] ===== Cross-Prompt Hessian Similarity =====")
        for suffix, name in sorted(module_types.items()):
            h_list = self._hessian_diag[name]
            n = len(h_list)
            if n < 2:
                continue
            # Stack and compute pairwise cosine similarity
            flat = torch.stack([h.flatten() for h in h_list])  # [n, D*D]
            # Cosine similarity matrix
            norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
            flat_normed = flat / norms
            sim = flat_normed @ flat_normed.T  # [n, n]
            # Extract upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            pairwise = sim[mask]
            # Also compute eigenvalue spectrum similarity: compare top-k eigenvalues
            eig_list = []
            for h in h_list:
                eigvals = torch.linalg.eigvalsh(h)
                eigvals = eigvals / eigvals.abs().max().clamp(min=1e-12)  # normalize
                eig_list.append(eigvals)
            eig_stack = torch.stack(eig_list)  # [n, D]
            eig_norms = eig_stack.norm(dim=1, keepdim=True).clamp(min=1e-12)
            eig_normed = eig_stack / eig_norms
            eig_sim = eig_normed @ eig_normed.T
            eig_pairwise = eig_sim[mask]

            D = h_list[0].shape[0]
            print(
                f"[influence_trace][hessian_similarity] module_type={suffix} "
                f"(example={name}) D={D} n_groups={n} | "
                f"H cosine: mean={pairwise.mean():.4f} std={pairwise.std():.4f} "
                f"min={pairwise.min():.4f} max={pairwise.max():.4f} | "
                f"eigval cosine: mean={eig_pairwise.mean():.4f} std={eig_pairwise.std():.4f} "
                f"min={eig_pairwise.min():.4f} max={eig_pairwise.max():.4f}"
            )

    def _accumulate_grad_dot(
        self,
        chunks: list[tuple],
        hess_accum: torch.Tensor | None = None,
    ) -> None:
        """Accumulate per-module response dot products into global matrices.

        For each module, concatenate g_resp across all groups and compute the
        full pairwise dot-product matrix, then scatter-add into the global
        ``_grad_dot_accum`` matrix indexed by global row IDs.  After all 196
        modules are processed, ``_grad_dot_accum[i, j]`` equals
        ``Σ_m g_i^m · g_j^m`` — the dot product of concatenated full-model
        response gradients.

        If ``hess_accum`` is provided (the accumulated ``Σ_t g_t g_t^T`` across
        all groups for this module), builds a global Hessian ``H = hess_accum + λI``,
        solves ``H^{-1} g_resp^T`` for ALL responses, and computes the full
        pairwise H^{-1}-weighted dot product matrix (including cross-prompt pairs).
        """
        all_rows = torch.cat([c[0] for c in chunks])  # global row IDs
        all_g = torch.cat([c[1] for c in chunks])      # [N, D] on CPU
        dot = all_g.to(torch.float64) @ all_g.to(torch.float64).T  # [N, N]

        max_row = int(all_rows.max().item())
        new_size = max_row + 1

        def _ensure_accum(accum: torch.Tensor | None, sz: int) -> torch.Tensor:
            if accum is None:
                return torch.zeros(sz, sz, dtype=torch.float64)
            if accum.shape[0] < sz:
                old = accum
                accum = torch.zeros(sz, sz, dtype=torch.float64)
                accum[: old.shape[0], : old.shape[1]] = old
            return accum

        self._grad_dot_accum = _ensure_accum(self._grad_dot_accum, new_size)

        ii = all_rows.unsqueeze(1).expand(-1, all_rows.numel()).reshape(-1).long()
        jj = all_rows.unsqueeze(0).expand(all_rows.numel(), -1).reshape(-1).long()
        self._grad_dot_accum.index_put_((ii, jj), dot.reshape(-1), accumulate=True)

        # Global H^{-1}-weighted dot products using accumulated Hessian
        if hess_accum is not None:
            D = hess_accum.shape[0]
            if self.cfg.reg_lambda > 0:
                reg = self.cfg.reg_lambda
            else:
                reg = float((hess_accum.trace() / max(D, 1)).item()) * 0.1
            H = hess_accum + reg * torch.eye(D, device=hess_accum.device, dtype=hess_accum.dtype)
            chol = torch.linalg.cholesky(H)
            all_g_dev = all_g.to(device=chol.device, dtype=chol.dtype)
            solved_resp = torch.cholesky_solve(all_g_dev.T, chol)  # [D, N]
            hinv_dot = (all_g_dev @ solved_resp).cpu().to(torch.float64)  # [N, N]
            del chol, all_g_dev, solved_resp, H

            self._grad_hinv_dot_accum = _ensure_accum(self._grad_hinv_dot_accum, new_size)
            self._grad_hinv_dot_accum.index_put_(
                (ii, jj), hinv_dot.reshape(-1), accumulate=True,
            )

    def _log_gradient_similarity(self) -> None:
        """Log accumulated response gradient dot-products across ALL modules.

        Reports both raw dot products ``Σ_m g_i^m · g_j^m`` and (if available)
        H^{-1}-weighted dot products ``Σ_m g_i^{m,T} H_m^{-1} g_j^m``.
        """
        if self._grad_dot_accum is None or not self._grad_row_meta:
            return

        row_ids = sorted(self._grad_row_meta.keys())
        n = len(row_ids)
        if n < 2:
            return

        acc_flags = torch.tensor(
            [self._grad_row_meta[r][0] for r in row_ids], dtype=torch.bool
        )
        group_ids = torch.tensor(
            [self._grad_row_meta[r][1] for r in row_ids], dtype=torch.long
        )
        idx = torch.tensor(row_ids, dtype=torch.long)

        # Report raw dot products
        self._log_dot_matrix(
            label="RAW_DOT",
            dot_full=self._grad_dot_accum,
            idx=idx, acc_flags=acc_flags, group_ids=group_ids, n=n,
            report_norms=True,
        )

        # Report H^{-1}-weighted dot products
        if self._grad_hinv_dot_accum is not None:
            self._log_dot_matrix(
                label="HINV_DOT",
                dot_full=self._grad_hinv_dot_accum,
                idx=idx, acc_flags=acc_flags, group_ids=group_ids, n=n,
                report_norms=True,
            )

    def _log_dot_matrix(
        self,
        label: str,
        dot_full: torch.Tensor,
        idx: torch.Tensor,
        acc_flags: torch.Tensor,
        group_ids: torch.Tensor,
        n: int,
        report_norms: bool = True,
    ) -> None:
        """Generic analysis of a pairwise dot-product matrix."""
        sub = dot_full[idx][:, idx]  # [n, n]

        def _stats(vals: list[float]) -> str:
            if not vals:
                return "n=0"
            t = torch.tensor(vals, dtype=torch.float64)
            return f"n={len(vals)} mean={t.mean():.6g} std={t.std():.6g} min={t.min():.6g} max={t.max():.6g}"

        print(
            f"[influence_trace][grad_dot] ===== {label} "
            f"(accumulated across all modules) n_resp={n} "
            f"(acc={acc_flags.sum().item()} rej={(~acc_flags).sum().item()}) ====="
        )

        if report_norms:
            diag = sub.diagonal()
            norms = diag.clamp(min=0).sqrt()
            norms_acc = [float(norms[i]) for i in range(n) if acc_flags[i]]
            norms_rej = [float(norms[i]) for i in range(n) if not acc_flags[i]]
            print(
                f"[influence_trace][grad_dot][{label}] self_dot  "
                f"acc: {_stats(norms_acc)} | rej: {_stats(norms_rej)}"
            )

        # --- Cosine similarity ---
        diag = sub.diagonal().clamp(min=0).sqrt()
        norm_outer = diag.unsqueeze(1) * diag.unsqueeze(0)
        cosine = sub / norm_outer.clamp(min=1e-30)

        # --- Per-prompt analysis ---
        unique_groups = group_ids.unique().tolist()
        pp_aa: list[float] = []
        pp_rr: list[float] = []
        pp_ar: list[float] = []
        pp_cos_aa: list[float] = []
        pp_cos_rr: list[float] = []
        pp_cos_ar: list[float] = []
        for g in unique_groups:
            if g == -1:
                continue
            gm = group_ids == g
            gi = torch.where(gm)[0]
            ga = acc_flags[gm]
            g_sub = sub[gi][:, gi]
            g_cos = cosine[gi][:, gi]
            a_local = torch.where(ga)[0]
            r_local = torch.where(~ga)[0]
            for i in range(a_local.numel()):
                for j in range(i + 1, a_local.numel()):
                    pp_aa.append(float(g_sub[a_local[i], a_local[j]]))
                    pp_cos_aa.append(float(g_cos[a_local[i], a_local[j]]))
            for i in range(r_local.numel()):
                for j in range(i + 1, r_local.numel()):
                    pp_rr.append(float(g_sub[r_local[i], r_local[j]]))
                    pp_cos_rr.append(float(g_cos[r_local[i], r_local[j]]))
            for i in a_local:
                for j in r_local:
                    pp_ar.append(float(g_sub[i, j]))
                    pp_cos_ar.append(float(g_cos[i, j]))

        print(
            f"[influence_trace][grad_dot][{label}] PER_PROMPT dot  "
            f"acc-acc: {_stats(pp_aa)} | "
            f"rej-rej: {_stats(pp_rr)} | "
            f"acc-rej: {_stats(pp_ar)}"
        )
        print(
            f"[influence_trace][grad_dot][{label}] PER_PROMPT cos  "
            f"acc-acc: {_stats(pp_cos_aa)} | "
            f"rej-rej: {_stats(pp_cos_rr)} | "
            f"acc-rej: {_stats(pp_cos_ar)}"
        )

        # --- All-prompt analysis ---
        a_idx = torch.where(acc_flags)[0]
        r_idx = torch.where(~acc_flags)[0]
        max_pairs = 5000

        def _collect_pairs(
            mat: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor,
            max_n: int, same: bool,
        ) -> list[float]:
            na, nb = idx_a.numel(), idx_b.numel()
            total = na * (na - 1) // 2 if same else na * nb
            if total == 0:
                return []
            vals: list[float] = []
            if total <= max_n:
                if same:
                    for i in range(na):
                        for j in range(i + 1, na):
                            vals.append(float(mat[idx_a[i], idx_a[j]]))
                else:
                    for i in range(na):
                        for j in range(nb):
                            vals.append(float(mat[idx_a[i], idx_b[j]]))
            else:
                for _ in range(max_n):
                    if same:
                        i, j = int(torch.randint(0, na, (1,))), int(torch.randint(0, na, (1,)))
                        while i == j:
                            j = int(torch.randint(0, na, (1,)))
                        vals.append(float(mat[idx_a[i], idx_a[j]]))
                    else:
                        i = int(torch.randint(0, na, (1,)))
                        j = int(torch.randint(0, nb, (1,)))
                        vals.append(float(mat[idx_a[i], idx_b[j]]))
            return vals

        all_aa = _collect_pairs(sub, a_idx, a_idx, max_pairs, same=True)
        all_rr = _collect_pairs(sub, r_idx, r_idx, max_pairs, same=True)
        all_ar = _collect_pairs(sub, a_idx, r_idx, max_pairs, same=False)
        all_cos_aa = _collect_pairs(cosine, a_idx, a_idx, max_pairs, same=True)
        all_cos_rr = _collect_pairs(cosine, r_idx, r_idx, max_pairs, same=True)
        all_cos_ar = _collect_pairs(cosine, a_idx, r_idx, max_pairs, same=False)

        print(
            f"[influence_trace][grad_dot][{label}] ALL_PROMPT dot  "
            f"acc-acc: {_stats(all_aa)} | "
            f"rej-rej: {_stats(all_rr)} | "
            f"acc-rej: {_stats(all_ar)}"
        )
        print(
            f"[influence_trace][grad_dot][{label}] ALL_PROMPT cos  "
            f"acc-acc: {_stats(all_cos_aa)} | "
            f"rej-rej: {_stats(all_cos_rr)} | "
            f"acc-rej: {_stats(all_cos_ar)}"
        )

    def _log_unprojected_comparison(self) -> None:
        """Compare projected vs unprojected response gradient dot-products.

        For ONE debug module (k_proj), computes the pairwise response dot
        product using both the projected (u, v) and raw (dy, x) representations.
        Projected:   dot_proj(r, r') = Σ_{t∈r,t'∈r'} (u_t·u_{t'}) * (v_t·v_{t'})
        Unprojected: dot_raw(r, r')  = Σ_{t∈r,t'∈r'} (dy_t·dy_{t'}) * (x_t·x_{t'})
        """
        name = self._debug_unproj_module
        if name is None or name not in self.storage:
            print(f"[influence_trace][unproj_cmp] SKIP: name={name} in_storage={name in self.storage if name else 'N/A'}")
            return
        bucket = self.storage[name]
        has_raw = "raw_x" in bucket
        n_u = len(bucket["u"]) if bucket["u"] else 0
        n_raw = len(bucket["raw_x"]) if has_raw else 0
        if not bucket["u"] or not has_raw or not bucket["raw_x"]:
            return

        compute_device = self._module_compute_device(name)
        u = self._cat_and_stage(bucket["u"], device=compute_device, dtype=torch.float32)
        v = self._cat_and_stage(bucket["v"], device=compute_device, dtype=torch.float32)
        raw_x = self._cat_and_stage(bucket["raw_x"], device=compute_device, dtype=torch.float32)
        raw_dy = self._cat_and_stage(bucket["raw_dy"], device=compute_device, dtype=torch.float32)
        row_id = self._cat_and_stage(bucket["row_id"], device=compute_device, dtype=torch.int64)
        group_id = self._cat_and_stage(bucket["group_id"], device=compute_device, dtype=torch.int64)
        accepted = self._cat_and_stage(bucket["accepted"], device=compute_device, dtype=torch.bool)

        n_tokens = u.shape[0]
        if n_tokens == 0:
            return

        mod = self.modules[name]
        k_in, k_out = self.module_dims[name]
        print(
            f"[influence_trace][unproj_cmp] ===== Unprojected vs Projected Comparison ====="
            f"\n[influence_trace][unproj_cmp] module={name} "
            f"in_features={mod.in_features} out_features={mod.out_features} "
            f"k_in={k_in} k_out={k_out} D_proj={k_in * k_out} n_tokens={n_tokens}"
        )

        # Process the first mixed group
        use_all_selected = self.cfg.accepted_rejected_scope in ("all_selected", "global_selected")
        unique_groups = [None] if use_all_selected else torch.unique(group_id, sorted=True).tolist()
        for gid in unique_groups:
            if gid is None:
                g_mask = torch.ones_like(group_id, dtype=torch.bool)
            else:
                g_mask = group_id == gid
            if int(g_mask.sum().item()) == 0:
                continue
            row_g = row_id[g_mask]
            acc_g = accepted[g_mask]
            u_g = u[g_mask]
            v_g = v[g_mask]
            x_g = raw_x[g_mask]
            dy_g = raw_dy[g_mask]

            uniq_rows, inv = torch.unique(row_g, sorted=True, return_inverse=True)
            row_acc = torch.zeros(uniq_rows.numel(), device=u_g.device, dtype=torch.bool)
            row_acc.index_put_((inv,), acc_g, accumulate=False)
            if bool(row_acc.all().item()) or bool((~row_acc).all().item()):
                continue  # skip all-same groups

            n_resp = uniq_rows.numel()
            proj_dot = torch.zeros(n_resp, n_resp, dtype=torch.float64, device=u_g.device)
            raw_dot = torch.zeros(n_resp, n_resp, dtype=torch.float64, device=u_g.device)

            for ri in range(n_resp):
                mi = inv == ri
                u_ri, v_ri = u_g[mi], v_g[mi]
                x_ri, dy_ri = x_g[mi], dy_g[mi]
                for rj in range(ri, n_resp):
                    mj = inv == rj
                    u_rj, v_rj = u_g[mj], v_g[mj]
                    x_rj, dy_rj = x_g[mj], dy_g[mj]
                    # Projected: Σ (u_t·u_{t'}) * (v_t·v_{t'})
                    Mu = u_ri.to(torch.float64) @ u_rj.to(torch.float64).T
                    Mv = v_ri.to(torch.float64) @ v_rj.to(torch.float64).T
                    pd = (Mu * Mv).sum()
                    proj_dot[ri, rj] = pd
                    proj_dot[rj, ri] = pd
                    # Unprojected: Σ (dy_t·dy_{t'}) * (x_t·x_{t'})
                    Mdy = dy_ri.to(torch.float64) @ dy_rj.to(torch.float64).T
                    Mx = x_ri.to(torch.float64) @ x_rj.to(torch.float64).T
                    rd = (Mdy * Mx).sum()
                    raw_dot[ri, rj] = rd
                    raw_dot[rj, ri] = rd

            # Extract statistics
            diag_proj = proj_dot.diagonal()
            diag_raw = raw_dot.diagonal()
            mask_ut = torch.triu(torch.ones(n_resp, n_resp, dtype=torch.bool, device=u_g.device), diagonal=1)

            def _pair_stats(mat: torch.Tensor, ra: torch.Tensor, label: str) -> None:
                aa_mask = mask_ut & ra.unsqueeze(1) & ra.unsqueeze(0)
                rr_mask = mask_ut & (~ra).unsqueeze(1) & (~ra).unsqueeze(0)
                ar_mask = mask_ut & (ra.unsqueeze(1) ^ ra.unsqueeze(0))
                for cat, cm in [("acc-acc", aa_mask), ("rej-rej", rr_mask), ("acc-rej", ar_mask)]:
                    vals = mat[cm]
                    if vals.numel() == 0:
                        print(f"[influence_trace][unproj_cmp] {label} {cat}: n=0")
                    else:
                        print(
                            f"[influence_trace][unproj_cmp] {label} {cat}: "
                            f"n={vals.numel()} mean={vals.mean():.6g} std={vals.std():.6g} "
                            f"min={vals.min():.6g} max={vals.max():.6g}"
                        )

            gid_str = f"group={gid}" if gid is not None else "all_selected"
            print(
                f"[influence_trace][unproj_cmp] {gid_str} n_resp={n_resp} "
                f"(acc={row_acc.sum().item()} rej={(~row_acc).sum().item()})"
            )
            print(
                f"[influence_trace][unproj_cmp] PROJ  self_dot: "
                f"mean={diag_proj.mean():.6g} std={diag_proj.std():.6g}"
            )
            print(
                f"[influence_trace][unproj_cmp] RAW   self_dot: "
                f"mean={diag_raw.mean():.6g} std={diag_raw.std():.6g}"
            )
            _pair_stats(proj_dot, row_acc, "PROJ ")
            _pair_stats(raw_dot, row_acc, "RAW  ")

            # Also compute cosine similarity for both
            proj_norms = diag_proj.clamp(min=0).sqrt()
            raw_norms = diag_raw.clamp(min=0).sqrt()
            proj_cos = proj_dot / (proj_norms.unsqueeze(1) * proj_norms.unsqueeze(0)).clamp(min=1e-30)
            raw_cos = raw_dot / (raw_norms.unsqueeze(1) * raw_norms.unsqueeze(0)).clamp(min=1e-30)
            _pair_stats(proj_cos, row_acc, "PROJ_COS ")
            _pair_stats(raw_cos, row_acc, "RAW_COS  ")

            # Only process first mixed group
            break

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
