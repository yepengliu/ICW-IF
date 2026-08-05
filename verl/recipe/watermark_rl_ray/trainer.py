"""WatermarkRLTrainer — Stage 2 on-policy RL with per-sample sequence reward.

Design decisions (per 2026-04-20 mentor meeting):
  - No ref model, no KD losses (use_reference_policy=False, no KL-in-reward)
  - GRPO advantage (group by prompt uid, normalize by group std)
  - Per-sample task/seed/fraction piped through to reward fn via non_tensor_batch
  - Native verl ActorRolloutRef (async) worker; native update_actor (PPO loss)

fit() loop per step:
  1. Prompt batch → _get_gen_batch (pops input_ids + non_tensor fields)
  2. Restore {task, wm_seed, wm_fraction} in batch so they survive rollout union
  3. Rollout via vLLM agent loop (n rollouts per prompt)
  4. batch_rep.union(gen_out) → full sequences
  5. compute_reward (our PerSampleWatermarkZScoreRewardFn)
  6. compute_response_mask → compute old_log_probs → compute_advantage(grpo)
  7. actor_rollout_wg.update_actor(batch_rep)

Inherits WatermarkKDRayTrainer for:
  - Worker infrastructure (resource pool + worker groups)
  - _validate() using WatermarkZScoreRewardFn
  - Checkpoint save/load
  - _create_dataloader() (accepts our prompt dataloader via kd_train_dataloader arg)
"""

import uuid
from collections import defaultdict
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import Role
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from recipe.watermark_kd_ray.trainer import WatermarkKDRayTrainer


# Non-tensor batch keys that must survive rollout (they're popped by _get_gen_batch
# into gen_batch but not restored by gen_batch.union with rollout output).
# acrostic_target added 2026-05-02 — required by per-sample acrostic detector
# in PerSampleWatermarkZScoreRewardFn (no fallback to a fixed default anymore).
# raw_prompt_ref_ids added 2026-05-25 — clean-prompt tokens used to build ref
# input for KD-style KL (ref sees clean prompt + actor's rollout response).
_PASSTHROUGH_KEYS = (
    "wm_seed", "wm_fraction", "task", "acrostic_target", "raw_prompt_ref_ids",
)


class WatermarkRLTrainer(WatermarkKDRayTrainer):
    """On-policy RL trainer with per-sample sequence reward (GRPO)."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        prompt_train_dataloader: Optional[StatefulDataLoader] = None,
        device_name: Optional[str] = None,
    ):
        # Pass our prompt loader as kd_train_dataloader so parent stores it
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            kd_train_dataloader=prompt_train_dataloader,
            device_name=device_name,
        )
        self.reward_fn = reward_fn
        self.use_critic = False
        # Enable ref policy when KL loss is requested. When True, fit() will
        # build a KD-style ref input (clean prompt + rollout response) and
        # call compute_ref_log_prob; dp_actor consumes ref_log_prob to add
        # use_kl_loss term to the policy loss.
        self.use_reference_policy = bool(
            config.actor_rollout_ref.actor.get("use_kl_loss", False)
        )

    # ------------------------------------------------------------------ #
    #  Worker init — use native async actor/rollout worker                #
    # ------------------------------------------------------------------ #

    def init_workers(self):
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # When use_reference_policy=True, co-locate ref model in the same worker
        # by passing role="actor_rollout_ref" (which flips _is_ref=True inside
        # AsyncActorRolloutRefWorker so ref_module_fsdp gets built and
        # compute_ref_log_prob asserts pass). Otherwise stay as "actor_rollout"
        # to avoid loading a ref model needlessly.
        role_str = (
            str(Role.ActorRolloutRef) if self.use_reference_policy else str(Role.ActorRollout)
        )
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role=role_str,
        )
        self.resource_pool_to_cls[resource_pool][role_str] = actor_rollout_cls

        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )

        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[role_str]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = self.config.actor_rollout_ref.rollout.mode == "async"
        if self.async_rollout_mode:
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=None,
            )

    # ------------------------------------------------------------------ #
    #  Fit loop                                                           #
    # ------------------------------------------------------------------ #

    def fit(self):
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        test_freq = self.config.trainer.get("test_freq", -1)
        save_freq = self.config.trainer.get("save_freq", -1)
        val_before_train = self.config.trainer.get("val_before_train", False)

        steps_per_epoch = len(self.train_dataloader)
        if save_freq == "after_each_epoch":
            save_freq = steps_per_epoch
        if test_freq == "after_each_epoch":
            test_freq = steps_per_epoch

        if val_before_train and self.val_dataloader is not None:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        n = self.config.actor_rollout_ref.rollout.n
        size_divisor = (
            self.config.actor_rollout_ref.rollout.agent.num_workers
            if self.async_rollout_mode
            else self.actor_rollout_wg.world_size
        )

        adv_estimator = self.config.algorithm.adv_estimator
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        for epoch in range(self.config.trainer.total_epochs):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )
            for prompt_batch_data in pbar:
                self.global_steps += 1
                metrics: dict = {}

                # ---- Build initial batch ----
                batch = DataProto.from_single_dict(prompt_batch_data)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # ---- Get gen batch (pops input_ids + non-tensor fields) ----
                gen_batch = self._get_gen_batch(batch)
                # Restore wm passthrough fields so they survive union with gen_out
                for key in _PASSTHROUGH_KEYS:
                    if key in gen_batch.non_tensor_batch and key not in batch.non_tensor_batch:
                        batch.non_tensor_batch[key] = gen_batch.non_tensor_batch[key].copy()

                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": False,
                    "global_steps": self.global_steps,
                }

                # GRPO: n rollouts per prompt
                gen_batch_rep = gen_batch.repeat(n, interleave=True)
                gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch_rep, size_divisor)

                rollout_start = perf_counter()
                if self.async_rollout_mode:
                    gen_out_padded = self.async_rollout_manager.generate_sequences(gen_padded)
                else:
                    gen_out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                rollout_s = perf_counter() - rollout_start

                gen_out = unpad_dataproto(gen_out_padded, pad_size=pad_size)

                # Repeat prompt-side batch to align with n-per-prompt rollouts
                batch_rep = batch.repeat(n, interleave=True)
                full_batch = batch_rep.union(gen_out)

                # Sanity on first step
                if self.global_steps == 1:
                    for key in _PASSTHROUGH_KEYS:
                        assert key in full_batch.non_tensor_batch, (
                            f"{key} missing from full_batch.non_tensor_batch"
                        )
                    assert "responses" in full_batch.batch.keys(), "responses missing"

                # ---- Reward ----
                reward_start = perf_counter()
                reward_tensor, reward_extra = compute_reward(full_batch, self.reward_fn)
                full_batch.batch["token_level_scores"] = reward_tensor
                full_batch.batch["token_level_rewards"] = reward_tensor
                if reward_extra:
                    full_batch.non_tensor_batch.update(
                        {k: np.array(v) for k, v in reward_extra.items()}
                    )
                reward_s = perf_counter() - reward_start

                # ---- Response mask + old_log_probs (needed for PPO ratio) ----
                full_batch.batch["response_mask"] = compute_response_mask(full_batch)

                # Recompute old_log_probs through the actor (needed for PPO clip ratio).
                logp_start = perf_counter()
                old_logp = self.actor_rollout_wg.compute_log_prob(full_batch)
                if "entropys" in old_logp.batch.keys():
                    old_logp.batch.pop("entropys")
                # Diagnose + sanitize forward log_probs. dp_actor's agg_loss path
                # under loss_agg_mode=seq-mean-token-mean does `loss * mask` then
                # sum, and `NaN * 0 = NaN` in torch — even one bad value poisons
                # the entire batch loss scalar. We also surface the count so we
                # can correlate forward-NaN with `skip_step` reasons in wandb.
                _olp = old_logp.batch["old_log_probs"]
                _olp_nan = torch.isnan(_olp).sum().item()
                _olp_inf = torch.isinf(_olp).sum().item()
                if _olp_nan + _olp_inf > 0:
                    print(
                        f"[DIAG] step {self.global_steps}: old_log_probs has "
                        f"nan={_olp_nan} inf={_olp_inf} (sanitizing to 0)",
                        flush=True,
                    )
                metrics["sanity/old_logp_nan"] = int(_olp_nan)
                metrics["sanity/old_logp_inf"] = int(_olp_inf)
                old_logp.batch["old_log_probs"] = torch.nan_to_num(
                    _olp, nan=0.0, posinf=0.0, neginf=0.0
                )
                full_batch = full_batch.union(old_logp)
                logp_s = perf_counter() - logp_start

                # ---- Ref log probs on clean-prompt + rollout response (KD-style) ----
                ref_logp_s = 0.0
                _ref_nan = 0
                _ref_inf = 0
                if self.use_reference_policy:
                    ref_start = perf_counter()
                    ref_input = self._build_ref_input(full_batch)
                    ref_out = self.actor_rollout_wg.compute_ref_log_prob(ref_input)
                    # compute_ref_log_prob returns tensor over its own input
                    # layout; because we built ref_input with the same response
                    # placement (left-padded prompt of identical max_prompt_length
                    # + same response tokens), the per-position log_probs align
                    # 1:1 with actor's response tokens. Just union the field.
                    _rlp = ref_out.batch["ref_log_prob"]
                    _ref_nan = torch.isnan(_rlp).sum().item()
                    _ref_inf = torch.isinf(_rlp).sum().item()
                    if _ref_nan + _ref_inf > 0:
                        print(
                            f"[DIAG] step {self.global_steps}: ref_log_prob has "
                            f"nan={_ref_nan} inf={_ref_inf} (sanitizing to 0)",
                            flush=True,
                        )
                    full_batch.batch["ref_log_prob"] = torch.nan_to_num(
                        _rlp, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    ref_logp_s = perf_counter() - ref_start
                metrics["sanity/ref_logp_nan"] = int(_ref_nan)
                metrics["sanity/ref_logp_inf"] = int(_ref_inf)

                # ---- Advantages: valid-subset GRPO + invalid loss mask ----
                # Only samples with z_score_valid=True participate in the group
                # mean/std. Invalid samples (response too short) get advantage=0
                # AND response_mask=0 so they contribute nothing to PG / entropy
                # / KL loss aggregation. Reward value for invalid samples is
                # therefore inert (the MIN_LEN sentinel never enters the loss).
                full_batch.meta_info["global_token_num"] = torch.sum(
                    full_batch.batch["attention_mask"], dim=-1
                ).tolist()

                valid_np = full_batch.non_tensor_batch["z_score_valid"].astype(bool)
                uids_np  = full_batch.non_tensor_batch["uid"]
                scores   = full_batch.batch["token_level_rewards"].sum(dim=-1)
                device   = scores.device

                advantages_scalar = torch.zeros_like(scores)
                valid_t = torch.from_numpy(valid_np).to(device)

                group_all_invalid = 0
                group_partial_invalid = 0
                group_singleton_valid = 0
                unique_uids = np.unique(uids_np)
                for uid in unique_uids:
                    g_mask_np  = (uids_np == uid)
                    g_valid_np = g_mask_np & valid_np
                    n_valid = int(g_valid_np.sum())
                    n_total = int(g_mask_np.sum())

                    if n_valid == 0:
                        group_all_invalid += 1
                        continue
                    if n_valid < n_total:
                        group_partial_invalid += 1
                    if n_valid < 2:
                        group_singleton_valid += 1
                        continue  # no within-group signal; advantage stays 0

                    g_valid_t = torch.from_numpy(g_valid_np).to(device)
                    s = scores[g_valid_t]
                    mean_g = s.mean()
                    if norm_adv_by_std_in_grpo:
                        std_g = s.std(unbiased=False) + 1e-6
                        advantages_scalar[g_valid_t] = (s - mean_g) / std_g
                    else:
                        advantages_scalar[g_valid_t] = s - mean_g

                adv_tok = advantages_scalar.unsqueeze(-1) * full_batch.batch["response_mask"]
                full_batch.batch["advantages"] = adv_tok
                full_batch.batch["returns"]    = adv_tok

                invalid_t = ~valid_t
                full_batch.batch["response_mask"][invalid_t] = 0

                metrics["reward/n_valid"] = int(valid_np.sum())
                metrics["reward/n_total"] = int(valid_np.size)
                metrics["reward/group_count"] = int(len(unique_uids))
                metrics["reward/group_all_invalid"] = group_all_invalid
                metrics["reward/group_partial_invalid"] = group_partial_invalid
                metrics["reward/group_singleton_valid"] = group_singleton_valid

                # ---- Pre-update NaN guard: skip update_actor if forward had NaN ----
                # If either old_logp (actor forward in compute_log_prob) or
                # ref_log_prob (ref forward) had any NaN/Inf, skip update_actor
                # entirely rather than letting NaN propagate through pg/kl
                # backward, where it may sneak past FSDP's grad-finite guard
                # (e.g. when total_norm aggregation drops sub-grad NaN).
                forward_nan = (_olp_nan + _olp_inf + _ref_nan + _ref_inf) > 0
                update_s = 0.0
                actor_metrics: dict = {}
                if forward_nan:
                    print(
                        f"[WARN] step {self.global_steps}: skip update_actor "
                        f"(forward NaN/Inf: old_logp nan={_olp_nan} inf={_olp_inf}, "
                        f"ref_logp nan={_ref_nan} inf={_ref_inf})",
                        flush=True,
                    )
                else:
                    update_start = perf_counter()
                    actor_out = self.actor_rollout_wg.update_actor(full_batch)
                    update_s = perf_counter() - update_start
                    actor_metrics = reduce_metrics(actor_out.meta_info.get("metrics", {}))
                    metrics.update(actor_metrics)

                # ---- NaN soft-skip: log, mark as bad, continue ----
                # 2026-05-25: step-24 deterministic NaN observed for the same
                # prompt batch across v1 (silent-froze) and v2 (raise). Root
                # cause appears to be actor FSDP forward producing NaN log_prob
                # on a specific prompt-mix in step-24's batch (flash_attention_2
                # + fused triton kernels at 60k context). dp_actor's grad-finite
                # guard (line 328 of dp_actor.py) already zeroes grad and skips
                # optimizer.step() when grad_norm is non-finite, so the safest
                # response is to log + continue rather than abort.
                _bad_metrics = {
                    k: v for k, v in actor_metrics.items()
                    if isinstance(v, (int, float, np.floating, np.integer))
                    and not np.isfinite(v)
                }
                bad_step = bool(_bad_metrics) or forward_nan
                metrics["sanity/bad_step"] = int(bad_step)
                metrics["sanity/forward_nan_step"] = int(forward_nan)
                if _bad_metrics:
                    print(
                        f"[WARN] step {self.global_steps}: non-finite actor "
                        f"metrics {_bad_metrics}. Optimizer skipped this batch "
                        f"by dp_actor.grad-finite guard. Continuing.",
                        flush=True,
                    )

                # ---- Log ----
                metrics.update(self._compute_reward_metrics(reward_extra, full_batch))
                metrics["timing/rollout_s"] = rollout_s
                metrics["timing/reward_s"] = reward_s
                metrics["timing/log_prob_s"] = logp_s
                metrics["timing/ref_log_prob_s"] = ref_logp_s
                metrics["timing/update_s"] = update_s

                log_metrics = {
                    (k if (k.startswith("train/") or k.startswith("timing/") or k.startswith("val/") or k.startswith("actor/")) else f"train/{k}"): v
                    for k, v in metrics.items()
                }
                tracking.log(data=log_metrics, step=self.global_steps)
                pbar.set_postfix(
                    rwd_mean=f"{metrics.get('reward/z_mean', 0.0):.2f}",
                    g_z=f"{metrics.get('reward/z_green_mean', 0.0):.2f}",
                    i_z=f"{metrics.get('reward/z_initials_mean', 0.0):.2f}",
                )

                if test_freq > 0 and self.global_steps % test_freq == 0:
                    val_metrics = self._validate()
                    tracking.log(data=val_metrics, step=self.global_steps)

                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                if self.global_steps >= self.total_training_steps:
                    break

            if self.global_steps >= self.total_training_steps:
                break

        # Final validation only if test_freq > 0 was requested
        if self.val_dataloader is not None and test_freq > 0:
            val_metrics = self._validate()
            tracking.log(data=val_metrics, step=self.global_steps)

        self._save_checkpoint()
        print(f"Training complete. Total steps: {self.global_steps}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_ref_input(self, full_batch: DataProto) -> DataProto:
        """Construct ref forward input: clean prompt + actor's rollout response.

        KD-style ref: instead of feeding the same ICW prompt that actor sees,
        ref reads the clean (no-watermark) prompt followed by the same response
        tokens actor generated. Per-position log_probs over the response are
        then 1:1 aligned with actor's response positions (both end with the
        same R response tokens), so ref_log_prob can be unioned without re-
        indexing.

        Inputs (from full_batch):
          - batch["input_ids"]      (B, L)  L = P + R
          - batch["attention_mask"] (B, L)
          - batch["responses"]      (B, R)
          - non_tensor_batch["raw_prompt_ref_ids"]  array of list[int]

        Output DataProto fields (only what compute_log_prob reads):
          - input_ids, attention_mask, position_ids, responses
        """
        actor_input_ids = full_batch.batch["input_ids"]
        actor_attn = full_batch.batch["attention_mask"]
        actor_responses = full_batch.batch["responses"]
        device = actor_input_ids.device

        B, L = actor_input_ids.shape
        R = actor_responses.shape[1]
        P = L - R  # max_prompt_length

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        clean_ids_list = full_batch.non_tensor_batch["raw_prompt_ref_ids"]

        new_prompt_ids = torch.full((B, P), pad_id, dtype=actor_input_ids.dtype, device=device)
        new_prompt_attn = torch.zeros((B, P), dtype=actor_attn.dtype, device=device)
        n_clean_per_sample = torch.zeros((B,), dtype=torch.long, device=device)

        for i in range(B):
            clean = clean_ids_list[i]
            if not isinstance(clean, list):
                clean = list(clean)
            n = len(clean)
            if n == 0:
                # Should not happen if parquet has prompt_ref. Fall back to
                # actor's own prompt slice — KL term becomes trivially small.
                new_prompt_ids[i] = actor_input_ids[i, :P]
                new_prompt_attn[i] = actor_attn[i, :P]
                n_clean_per_sample[i] = int(actor_attn[i, :P].sum().item())
                continue
            if n > P:
                clean = clean[-P:]
                n = P
            pad_len = P - n
            new_prompt_ids[i, pad_len:] = torch.tensor(clean, dtype=actor_input_ids.dtype, device=device)
            new_prompt_attn[i, pad_len:] = 1
            n_clean_per_sample[i] = n

        # Response part: identical token ids; attention validity follows actor's
        # (response_mask is derived from attention_mask[:, -R:]).
        actor_resp_attn = actor_attn[:, P:]

        new_input_ids = torch.cat([new_prompt_ids, actor_responses], dim=1)
        new_attn = torch.cat([new_prompt_attn, actor_resp_attn], dim=1)

        # position_ids: left-padded prompt has 0s for pad slots and arange(n)
        # for real tokens; response continues from n_clean for each sample.
        new_pos = torch.zeros((B, L), dtype=torch.long, device=device)
        for i in range(B):
            n = int(n_clean_per_sample[i].item())
            if n > 0:
                pad_len = P - n
                new_pos[i, pad_len:P] = torch.arange(n, dtype=torch.long, device=device)
                new_pos[i, P:] = torch.arange(n, n + R, dtype=torch.long, device=device)
            else:
                # degenerate fallback path (shouldn't trigger with prompt_ref)
                new_pos[i] = full_batch.batch["position_ids"][i]

        ref_input = DataProto.from_dict(
            tensors={
                "input_ids": new_input_ids,
                "attention_mask": new_attn,
                "position_ids": new_pos,
                "responses": actor_responses,
            }
        )
        return ref_input

    @staticmethod
    def _compute_reward_metrics(reward_extra: dict, batch: DataProto) -> dict:
        """Aggregate per-sample reward metrics into scalars."""
        m: dict = {}
        if not reward_extra:
            return m

        z = np.array(reward_extra.get("z_score", []), dtype=np.float32)
        valid = np.array(reward_extra.get("z_score_valid", []), dtype=bool)
        if valid.sum() > 0:
            m["reward/z_mean"] = float(z[valid].mean())
            m["reward/z_std"] = float(z[valid].std())
        m["reward/valid_ratio"] = float(valid.mean()) if valid.size else 0.0

        # Per-task aggregates (use nanmean to ignore non-matching samples)
        tasks = batch.non_tensor_batch.get("task")
        if tasks is not None:
            for t in ("green", "initials", "acrostics"):
                key = f"z_score_{t}"
                if key in reward_extra:
                    arr = np.array(reward_extra[key], dtype=np.float32)
                    # count = samples actually from this task with valid response
                    task_mask = np.array([str(x) == t for x in tasks])
                    good = task_mask & valid
                    if good.sum() > 0:
                        m[f"reward/z_{t}_mean"] = float(arr[good].mean())
                        m[f"reward/z_{t}_std"]  = float(arr[good].std())
                        m[f"reward/z_{t}_count"] = int(good.sum())

        # Response length stats
        rl = reward_extra.get("response_len", [])
        if rl:
            rl_np = np.array(rl, dtype=np.float32)
            m["response/len_mean"] = float(rl_np.mean())
            m["response/len_min"]  = float(rl_np.min())
            m["response/len_max"]  = float(rl_np.max())

        return m
