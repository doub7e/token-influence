# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
import time
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_archer, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.influence_trace import InfluenceTraceConfig, TokenInfluenceTracer

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self._influence_cfg: InfluenceTraceConfig | None = None
        self._influence_tracer: TokenInfluenceTracer | None = None
        self._influence_rows: list[dict[str, Any]] = []

    def pop_influence_trace_rows(self) -> list[dict[str, Any]]:
        rows = self._influence_rows
        self._influence_rows = []
        return rows

    def _setup_influence_trace(self, meta_info: dict[str, Any]) -> bool:
        cfg = InfluenceTraceConfig.from_meta(meta_info)
        self._influence_cfg = cfg
        if not cfg.enable:
            self._influence_tracer = None
            return False
        if not self.use_remove_padding:
            raise ValueError("influence_trace requires actor_rollout_ref.model.use_remove_padding=True")
        if self.use_ulysses_sp:
            raise ValueError("influence_trace does not support ulysses sequence parallelism yet")
        if self._influence_tracer is None:
            self._influence_tracer = TokenInfluenceTracer(cfg)
            self._influence_tracer.register(self.actor_module)
            mem_map = self._influence_tracer.estimate_hessian_memory_mb()
            if torch.distributed.get_rank() == 0 and mem_map:
                report = self._influence_tracer.projection_report()
                print(
                    "[influence_trace] projection setup: "
                    f"hessian_mode={cfg.hessian_mode}, factor={cfg.projection_dim_factor}, max_modules={cfg.max_modules}, "
                    f"max_proj_vector_sum={cfg.max_proj_vector_sum}, max_hessian_dim={cfg.max_hessian_dim}, "
                    f"max_tokens_per_response={cfg.max_tokens_per_response}, "
                    f"skip_optimizer_step={cfg.skip_optimizer_step}, "
                    f"grad_offload_to_cpu={cfg.grad_offload_to_cpu}, output_function={cfg.output_function}, "
                    f"accepted_rejected_scope={cfg.accepted_rejected_scope}"
                )
                for row in report:
                    print(
                        "[influence_trace] module="
                        f"{row['name']}, in={row['in_features']}, out={row['out_features']}, "
                        f"k_in={row['k_in']}, k_out={row['k_out']}, D={row['proj_dim']}, "
                        f"hessian_peak_mb={row['hessian_peak_mb']:.2f}"
                    )
                total_mb = float(sum(mem_map.values()))
                print(f"[influence_trace] Hessian peak estimate total (2xD^2 fp32): {total_mb:.2f}MB")
        else:
            self._influence_tracer.cfg = cfg
        self._influence_tracer.clear_storage()
        return True

    def _compute_log_prob_advantage_objective(
        self,
        *,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
    ) -> torch.Tensor:
        obj = agg_loss(loss_mat=log_prob * advantages, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        if self.config.use_dynamic_bsz:
            obj = obj * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
        else:
            obj = obj / self.gradient_accumulation
        return obj

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        influence_payload: dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                if influence_payload is not None:
                    tracer = self._influence_tracer
                    if tracer is None:
                        raise ValueError("influence_payload is provided but influence tracer is not initialized.")
                    tracer.begin_rmpad_capture(
                        indices=indices,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        response_mask=influence_payload["response_mask"],
                        selected_rows=influence_payload["selected_rows"],
                        row_ids=influence_payload["row_ids"],
                        group_ids=influence_payload["group_ids"],
                        accepted=influence_payload["accepted"],
                    )

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        trace_profile = bool(self._influence_cfg is not None and self._influence_cfg.profile_timing)
        is_rank0 = (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if trace_profile and is_rank0:
            print("[influence_trace][optimizer] clip_start")

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        if trace_profile and is_rank0:
            print("[influence_trace][optimizer] clip_done")

        if self._influence_cfg is not None and self._influence_cfg.skip_optimizer_step:
            if trace_profile and is_rank0:
                print("[influence_trace][optimizer] step_skipped")
                print("[influence_trace][optimizer] step_done")
            return grad_norm

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        if trace_profile and is_rank0:
            print("[influence_trace][optimizer] step_done")
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        self._influence_rows = []
        influence_enabled = self._setup_influence_trace(data.meta_info)

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        influence_keys = [
            "influence_trace_selected",
            "influence_trace_group_id",
            "influence_trace_row_id",
            "influence_trace_accepted",
            "influence_trace_reward",
        ]
        if influence_enabled:
            for key in influence_keys:
                if key in data.batch.keys():
                    select_keys.append(key)
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        profile_influence_timing = (
            influence_enabled
            and self._influence_cfg is not None
            and bool(getattr(self._influence_cfg, "profile_timing", False))
        )
        influence_timing = {
            "logprob_backward": 0.0,
            "loss_backward": 0.0,
            "pop_rows": 0.0,
            "forward_1": 0.0,
            "forward_2": 0.0,
        }
        saw_influence_payload = False

        def _sync_for_timing():
            if profile_influence_timing and torch.cuda.is_available():
                torch.cuda.synchronize()

        def _rank0() -> bool:
            return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = True
                    # if entropy_coeff != 0:
                    #     calculate_entropy = True
                    influence_payload = None
                    if influence_enabled and epoch == 0 and "influence_trace_selected" in data:
                        selected_rows = data["influence_trace_selected"].bool()
                        if bool(selected_rows.any().item()):
                            saw_influence_payload = True
                        influence_payload = {
                            "selected_rows": selected_rows,
                            "response_mask": response_mask.bool(),
                            "row_ids": data["influence_trace_row_id"].to(torch.long),
                            "group_ids": data["influence_trace_group_id"].to(torch.long),
                            "accepted": data["influence_trace_accepted"].bool(),
                        }
                    influence_cfg = self._influence_cfg
                    capture_with_training_backward = (
                        influence_payload is not None and influence_cfg is not None and influence_cfg.output_function == "training_loss"
                    )
                    capture_with_logprob_backward = (
                        influence_payload is not None and influence_cfg is not None and influence_cfg.output_function == "log_prob_advantage"
                    )
                    capture_with_influence_backward = capture_with_training_backward or capture_with_logprob_backward
                    if profile_influence_timing and _rank0():
                        sel_rows = int(influence_payload["selected_rows"].sum().item()) if influence_payload is not None else 0
                        print(
                            f"[influence_trace][micro] idx={micro_idx} "
                            f"capture_train={int(capture_with_training_backward)} "
                            f"capture_logprob={int(capture_with_logprob_backward)} "
                            f"selected_rows={sel_rows}"
                        )
                    _sync_for_timing()
                    t_forward_1 = time.perf_counter()
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        influence_payload=influence_payload if capture_with_influence_backward else None,
                    )
                    _sync_for_timing()
                    influence_timing["forward_1"] += time.perf_counter() - t_forward_1
                    if capture_with_logprob_backward:
                        tracer = self._influence_tracer
                        if tracer is None:
                            raise ValueError("influence_payload is provided but influence tracer is not initialized.")
                        influence_obj = self._compute_log_prob_advantage_objective(
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        anchor_param = tracer.anchor_parameter()
                        if anchor_param is not None:
                            _sync_for_timing()
                            t_logprob_bw = time.perf_counter()
                            torch.autograd.backward(
                                influence_obj,
                                inputs=[anchor_param],
                                retain_graph=False,
                            )
                            if anchor_param.grad is not None:
                                anchor_param.grad = None
                            _sync_for_timing()
                            influence_timing["logprob_backward"] += time.perf_counter() - t_logprob_bw
                        tracer.end_microbatch()
                        # Build a fresh graph for the real training backward.
                        _sync_for_timing()
                        t_forward_2 = time.perf_counter()
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            influence_payload=None,
                        )
                        _sync_for_timing()
                        influence_timing["forward_2"] += time.perf_counter() - t_forward_2

                    # high entropy token mask
                    with torch.no_grad():
                        token_entropy_quantile = self.config.get("token_entropy_quantile", 0.8)
                        masked_entropy = torch.where(response_mask.bool(), entropy.detach(), torch.nan)  # (bsz, response_length)
                        q80 = torch.nanquantile(masked_entropy, q=token_entropy_quantile, dim=-1, keepdim=True)  # (bsz, 1)
                        high_entropy_mask = (masked_entropy <= q80) & response_mask # only low entropy token is True
                        low_entropy_mask = (masked_entropy > q80) & response_mask #  only high entropy token is True

                    if self.config.get("use_archer_policy_loss", False):
                        pg_loss, pg_clipfrac_upper, pg_clipfrac_lower, negative_pg_clipfrac_dual, positive_pg_clipfrac_dual = compute_policy_loss_archer(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            high_entropy_mask=high_entropy_mask,
                            negative_low_entropy_clip_ratio_low=0.2,
                            negative_high_entropy_clip_ratio_low=0.4,
                            positive_low_entropy_clip_ratio_high=0.2,
                            positive_high_entropy_clip_ratio_high=0.4,
                            negative_clip_ratio_c=3.0,
                            positive_clip_ratio_c=3.0,
                            use_dynamic_clip=self.config.get("use_dynamic_clip", False),
                        )
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            use_token_entropy_separate=self.config.get("use_token_entropy_separate", False),
                            high_entropy_mask=high_entropy_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            low_entropy_clip_ratio_low=self.config.get("low_entropy_clip_ratio_low", clip_ratio_low),
                            low_entropy_clip_ratio_high=self.config.get("low_entropy_clip_ratio_high", clip_ratio_low),
                            high_entropy_clip_ratio_low=self.config.get("high_entropy_clip_ratio_low", clip_ratio_high),
                            high_entropy_clip_ratio_high=self.config.get("high_entropy_clip_ratio_high", clip_ratio_high),
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    use_token_entropy_separate = self.config.get("use_token_entropy_separate", False)
                    if self.config.get("use_kl_loss", False):
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        if use_token_entropy_separate:
                            low_entropy_kl_loss = agg_loss(loss_mat=kld, loss_mask=high_entropy_mask, loss_agg_mode=loss_agg_mode)
                            high_entropy_kl_loss = agg_loss(loss_mat=kld, loss_mask=low_entropy_mask, loss_agg_mode=loss_agg_mode)
                            kl_loss = low_entropy_kl_loss + high_entropy_kl_loss
                            policy_loss = policy_loss + low_entropy_kl_loss * self.config.kl_loss_coef + high_entropy_kl_loss * self.config.high_entropy_kl_loss_scale_coef
                        else:
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        if use_token_entropy_separate:
                            metrics["actor/low_entropy_kl_loss"] = low_entropy_kl_loss.detach().item()
                            metrics["actor/high_entropy_kl_loss"] = high_entropy_kl_loss.detach().item()
                            metrics["actor/high_entropy_kl_loss_scale_coef"] = self.config.high_entropy_kl_loss_scale_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    _sync_for_timing()
                    t_loss_bw = time.perf_counter()
                    loss.backward()
                    _sync_for_timing()
                    influence_timing["loss_backward"] += time.perf_counter() - t_loss_bw
                    if capture_with_training_backward:
                        tracer = self._influence_tracer
                        if tracer is not None:
                            tracer.end_microbatch()

                    if self.config.get("use_archer_policy_loss", False):
                        data = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac_upper": pg_clipfrac_upper.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            "actor/negative_pg_clipfrac_dual": negative_pg_clipfrac_dual.detach().item(),
                            "actor/positive_pg_clipfrac_dual": positive_pg_clipfrac_dual.detach().item(),
                        }
                    else:
                        data = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }

                    action_reward = verl_F.masked_sum(log_prob, response_mask) / responses.numel()
                    data["actor/action_reward"] = action_reward.detach().item()

                    append_to_dict(metrics, data)

                if profile_influence_timing and _rank0():
                    print("[influence_trace][phase] before_optimizer_step")
                grad_norm = self._optimizer_step()
                if profile_influence_timing and _rank0():
                    print("[influence_trace][phase] after_optimizer_step")
                if self._influence_cfg is not None and self._influence_cfg.skip_optimizer_step:
                    grad_norm_value = float("nan")
                elif torch.is_tensor(grad_norm):
                    grad_norm_value = float(grad_norm.detach().cpu().item())
                else:
                    grad_norm_value = float(grad_norm)
                data = {"actor/grad_norm": grad_norm_value}
                append_to_dict(metrics, data)
        if influence_enabled and self._influence_tracer is not None:
            if profile_influence_timing and _rank0():
                print("[influence_trace][phase] before_pop_rows")
            _sync_for_timing()
            t_pop_rows = time.perf_counter()
            self._influence_rows = self._influence_tracer.pop_token_influence_rows()
            _sync_for_timing()
            influence_timing["pop_rows"] += time.perf_counter() - t_pop_rows
            if profile_influence_timing and _rank0():
                print(f"[influence_trace][phase] after_pop_rows rows={len(self._influence_rows)}")
            rows_emitted = float(len(self._influence_rows))
            append_to_dict(metrics, {"influence_trace/rows_emitted": rows_emitted})
            dbg = self._influence_tracer.debug_stats(reset=True)
            append_to_dict(
                metrics,
                {
                    "influence_trace/debug_capture_begin_calls": float(dbg.get("capture_begin_calls", 0)),
                    "influence_trace/debug_capture_begin_nonempty": float(dbg.get("capture_begin_nonempty", 0)),
                    "influence_trace/debug_capture_selected_tokens": float(dbg.get("capture_selected_tokens", 0)),
                    "influence_trace/debug_anchor_tensor_ready": float(dbg.get("anchor_tensor_ready", 0)),
                    "influence_trace/debug_forward_capture_calls": float(dbg.get("forward_capture_calls", 0)),
                    "influence_trace/debug_forward_set_v_calls": float(dbg.get("forward_set_v_calls", 0)),
                    "influence_trace/debug_backward_hook_calls": float(dbg.get("backward_hook_calls", 0)),
                    "influence_trace/debug_output_grad_hook_calls": float(dbg.get("output_grad_hook_calls", 0)),
                    "influence_trace/debug_stored_chunks": float(dbg.get("stored_chunks", 0)),
                    "influence_trace/debug_groups_total": float(dbg.get("groups_total", 0)),
                    "influence_trace/debug_groups_skipped_all_same": float(dbg.get("groups_skipped_all_same", 0)),
                },
            )
            if saw_influence_payload and rows_emitted == 0.0 and torch.distributed.get_rank() == 0:
                print(
                    "[influence_trace][warn] selected_rows were present but no influence rows were emitted. "
                    f"debug={dbg}"
                )
        if profile_influence_timing:
            append_to_dict(
                metrics,
                {
                    "timing_s/influence_forward_1": influence_timing["forward_1"],
                    "timing_s/influence_forward_2": influence_timing["forward_2"],
                    "timing_s/influence_logprob_backward": influence_timing["logprob_backward"],
                    "timing_s/influence_loss_backward": influence_timing["loss_backward"],
                    "timing_s/influence_pop_rows": influence_timing["pop_rows"],
                },
            )
        self.actor_optimizer.zero_grad()
        return metrics
