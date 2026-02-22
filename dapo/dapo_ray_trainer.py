#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:17:50
@Author  :   wangjiakang
@File    :   dapo_ray_trainer.py
'''


import hashlib
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import math
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask

from .entropy_trace import RolloutEntropyTraceWriter
from .influence_trace import RolloutInfluenceTraceWriter


def _uid_hash64(uid: str) -> int:
    digest = hashlib.sha1(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _prepare_influence_trace_batch(
    batch: DataProto,
    *,
    world_size: int,
    max_prompts_per_step: int,
) -> tuple[int, int, dict[str, int]]:
    if "token_level_rewards" not in batch.batch:
        raise ValueError("token_level_rewards is required for influence trace.")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("uid is required for influence trace.")
    seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().to(torch.float32).cpu()
    accepted = torch.zeros_like(seq_rewards, dtype=torch.bool)
    uids = np.asarray(batch.non_tensor_batch["uid"]).astype(str)
    uid_to_indices: dict[str, np.ndarray] = {}
    for uid in np.unique(uids):
        uid_to_indices[uid] = np.where(uids == uid)[0]
        uid_rewards = seq_rewards[uid_to_indices[uid]]
        accepted[uid_to_indices[uid]] = uid_rewards == uid_rewards.max()

    batch_size = len(batch)
    if batch_size % world_size != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by world size {world_size}.")
    chunk_size = batch_size // world_size

    selected_uid: list[str] = []
    uid_constant = 0
    uid_mixed = 0
    uid_cross_rank = 0
    prompt_cap = int(max_prompts_per_step)
    if prompt_cap <= 0:
        prompt_cap = -1  # no cap: keep all eligible prompts

    for uid, idx in uid_to_indices.items():
        if idx.size == 0:
            continue
        uid_rewards = seq_rewards[idx]
        if bool((uid_rewards.max() == uid_rewards.min()).item()):
            uid_constant += 1
            continue
        local_acc = accepted[idx]
        if (not bool(local_acc.any().item())) or (not bool((~local_acc).any().item())):
            continue
        uid_mixed += 1
        rank_lo = int(idx.min() // chunk_size)
        rank_hi = int(idx.max() // chunk_size)
        if rank_lo != rank_hi:
            uid_cross_rank += 1
            continue
        selected_uid.append(uid)
        if prompt_cap > 0 and len(selected_uid) >= prompt_cap:
            break

    selected_mask = np.zeros((batch_size,), dtype=np.bool_)
    group_id = np.full((batch_size,), -1, dtype=np.int32)
    uid_hash = np.array([_uid_hash64(str(x)) for x in uids], dtype=np.int64)
    for gid, uid in enumerate(selected_uid):
        idx = uid_to_indices[uid]
        selected_mask[idx] = True
        group_id[idx] = gid

    batch.batch["influence_trace_selected"] = torch.from_numpy(selected_mask)
    batch.batch["influence_trace_group_id"] = torch.from_numpy(group_id)
    batch.batch["influence_trace_row_id"] = torch.arange(batch_size, dtype=torch.int32)
    batch.batch["influence_trace_reward"] = seq_rewards.to(torch.float32)
    batch.batch["influence_trace_accepted"] = accepted.to(torch.bool)
    batch.batch["influence_trace_uid_hash"] = torch.from_numpy(uid_hash)
    debug_counts = {
        "uid_total": int(len(uid_to_indices)),
        "uid_constant": int(uid_constant),
        "uid_mixed": int(uid_mixed),
        "uid_cross_rank": int(uid_cross_rank),
        "uid_selected": int(len(selected_uid)),
    }
    return int(selected_mask.sum()), len(selected_uid), debug_counts


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        entropy_trace_cfg = self.config.trainer.get("entropy_trace", {})
        entropy_trace_output_dir = entropy_trace_cfg.get("output_dir", f"{self.config.trainer.default_local_dir}/entropy_trace")
        entropy_trace_writer = RolloutEntropyTraceWriter(
            enabled=bool(entropy_trace_cfg.get("enable", False)),
            output_dir=entropy_trace_output_dir,
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            write_every=int(entropy_trace_cfg.get("write_every", 1)),
            update_summary_every=int(entropy_trace_cfg.get("update_summary_every", 1)),
            atomic_write=bool(entropy_trace_cfg.get("atomic_write", True)),
            fsync=bool(entropy_trace_cfg.get("fsync", False)),
        )
        influence_trace_cfg = self.config.trainer.get("influence_trace", {})
        influence_trace_output_dir = influence_trace_cfg.get("output_dir", f"{self.config.trainer.default_local_dir}/influence_trace")
        influence_trace_writer = RolloutInfluenceTraceWriter(
            enabled=bool(influence_trace_cfg.get("enable", False)),
            output_dir=influence_trace_output_dir,
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            write_every=int(influence_trace_cfg.get("write_every", 1)),
            atomic_write=bool(influence_trace_cfg.get("atomic_write", True)),
            fsync=bool(influence_trace_cfg.get("fsync", False)),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)


                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        sample_scores = reward_tensor.sum(-1).cpu().tolist()

                        if "thinking_tokens_info" in reward_result:
                            thinking_tokens_infos_dict = reward_result["thinking_tokens_info"]
                            for key_info in list(thinking_tokens_infos_dict.keys()):
                                lst = thinking_tokens_infos_dict[key_info]
                                assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                                for value, score in zip(lst, sample_scores):
                                    if score > 0:
                                        thinking_tokens_infos_dict['pos_'+key_info].append(value)
                                    else:
                                        thinking_tokens_infos_dict['neg_'+key_info].append(value)

                            for key_info, lst in thinking_tokens_infos_dict.items():
                                metrics[key_info] = sum(lst) / len(lst)

                        if "repetition_info" in reward_result:
                            repetition_infos_dict = reward_result["repetition_info"]
                            for key_info in list(repetition_infos_dict.keys()):
                                lst = repetition_infos_dict[key_info]
                                assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                                for value, score in zip(lst, sample_scores):
                                    if score > 0:
                                        repetition_infos_dict['pos_'+key_info].append(value)
                                    else:
                                        repetition_infos_dict['neg_'+key_info].append(value)

                            for key_info, lst in repetition_infos_dict.items():
                                metrics[key_info] = sum(lst) / len(lst)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]


                    # Rejection sampling based on rewards
                    # Group rewards by uid
                    uids = new_batch.non_tensor_batch['uid']
                    unique_uids = np.unique(uids)
                    valid_mask = torch.ones(len(uids), dtype=torch.bool)
                    solve_none = 0
                    solve_all = 0
                    for uid in unique_uids:
                        uid_mask = uids == uid
                        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                        # Check if all rewards are 0 or all are 1 for this uid
                        if (uid_rewards == 0).all():
                            valid_mask[uid_mask] = False
                            solve_none += 1
                        elif (uid_rewards == 1).all():
                            valid_mask[uid_mask] = False
                            solve_all += 1

                    # Log to metrics
                    metrics['batch/solve_none'] = solve_none
                    metrics['batch/solve_all'] = solve_all
                    metrics['batch/valid'] = len(unique_uids) - solve_all - solve_none
                    print("valid prompt:", len(unique_uids) - solve_all - solve_none, solve_none, solve_all)

                    if self.config.trainer.rejection_sample:
                        # If no valid samples remain, skip this batch and get a new one
                        if not valid_mask.any():
                            continue
                        # Filter batch to keep only valid samples
                        batch = new_batch[valid_mask]

                    max_response_length = batch.batch['responses'].shape[-1]
                    response_mask = batch.batch['attention_mask'][:, -max_response_length:]
                    response_length = response_mask.sum(-1).float()
                    response_clip_mask = ~torch.ge(response_length, max_response_length)
                    metrics['batch/clip_overlong'] = len(batch) - response_clip_mask.sum()
                    if self.config.trainer.enable_overlong_filter:
                        batch = batch[response_clip_mask]

                    def get_sorted_indices(lst):
                        return [index for index, _ in sorted(enumerate(lst), key=lambda x: x[1])]
                    sorted_indices = torch.tensor(get_sorted_indices(batch.non_tensor_batch['index']))
                    batch.reorder(sorted_indices)

                    # Round down to the nearest multiple of world size
                    num_trainer_replicas = self.actor_rollout_wg.world_size
                    if batch.batch['input_ids'].shape[0] < num_trainer_replicas and num_trainer_replicas/batch.batch['input_ids'].shape[0] <= 2:
                        batch = batch.repeat(repeat_times=math.ceil(num_trainer_replicas/batch.batch['input_ids'].shape[0]), interleave=False)

                    max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                    if not max_batch_size:
                        continue
                    batch = batch[:max_batch_size]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        try:
                            entropy_trace_writer.write_step(
                                step=self.global_steps,
                                batch=batch,
                                entropies=entropys,
                                response_mask=response_masks,
                            )
                        except Exception as e:
                            print(f"[WARN] failed to write entropy trace at step {self.global_steps}: {e}")
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm
                        )
                        if "inverse_pair" in batch.meta_info:
                            metrics["batch/inverse_pair"] = batch.meta_info["inverse_pair"] / metrics["batch/valid"]

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        if bool(influence_trace_cfg.get("enable", False)):
                            selected_num, selected_prompts, selected_debug = _prepare_influence_trace_batch(
                                batch,
                                world_size=self.actor_rollout_wg.world_size,
                                max_prompts_per_step=int(influence_trace_cfg.get("max_prompts_per_step", 2)),
                            )
                            metrics["influence_trace/selected_responses"] = selected_num
                            metrics["influence_trace/selected_prompts"] = selected_prompts
                            metrics["influence_trace/uid_total"] = float(selected_debug["uid_total"])
                            metrics["influence_trace/uid_constant"] = float(selected_debug["uid_constant"])
                            metrics["influence_trace/uid_mixed"] = float(selected_debug["uid_mixed"])
                            metrics["influence_trace/uid_cross_rank"] = float(selected_debug["uid_cross_rank"])
                            batch.meta_info["influence_trace_cfg"] = {
                                "enable": True,
                                "reg_lambda": float(influence_trace_cfg.get("reg_lambda", -1.0)),
                                "hessian_mode": str(influence_trace_cfg.get("hessian_mode", "inverse")),
                                "output_function": str(influence_trace_cfg.get("output_function", "training_loss")),
                                "accepted_rejected_scope": str(influence_trace_cfg.get("accepted_rejected_scope", "per_prompt")),
                                "module_name_filter": list(
                                    influence_trace_cfg.get(
                                        "module_name_filter",
                                        [
                                            "self_attn.q_proj",
                                            "self_attn.k_proj",
                                            "self_attn.v_proj",
                                            "self_attn.o_proj",
                                            "mlp.gate_proj",
                                            "mlp.up_proj",
                                            "mlp.down_proj",
                                        ],
                                    )
                                ),
                                "max_modules": int(influence_trace_cfg.get("max_modules", -1)),
                                "projection_dim_factor": int(influence_trace_cfg.get("projection_dim_factor", 512)),
                                "max_proj_vector_sum": int(influence_trace_cfg.get("max_proj_vector_sum", -1)),
                                "max_hessian_dim": int(influence_trace_cfg.get("max_hessian_dim", 2500)),
                                "max_tokens_per_response": int(influence_trace_cfg.get("max_tokens_per_response", -1)),
                                "skip_optimizer_step": bool(influence_trace_cfg.get("skip_optimizer_step", False)),
                                "grad_offload_to_cpu": bool(influence_trace_cfg.get("grad_offload_to_cpu", False)),
                                "force_gpu_compute": bool(influence_trace_cfg.get("force_gpu_compute", True)),
                                "profile_timing": bool(influence_trace_cfg.get("profile_timing", False)),
                                "exclude_self_response": bool(influence_trace_cfg.get("exclude_self_response", False)),
                                "contrastive_agg": str(influence_trace_cfg.get("contrastive_agg", "sum")),
                                "hessian_source": str(influence_trace_cfg.get("hessian_source", "response")),
                            }
                        else:
                            batch.meta_info["influence_trace_cfg"] = {"enable": False}
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if bool(influence_trace_cfg.get("enable", False)):
                            influence_rows = actor_output.non_tensor_batch.get("influence_trace_rows")
                            influence_trace_writer.write_step(
                                step=self.global_steps,
                                batch=batch,
                                entropies=entropys,
                                response_mask=response_masks,
                                influence_rows=influence_rows,
                            )

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
