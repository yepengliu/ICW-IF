"""
Entry point for watermark KD Ray trainer.

Usage:
    python -m recipe.watermark_kd_ray.main \
        actor_rollout_ref.model.path=<model_path> \
        ref_model.path=<ref_path> \
        data.train_files=<train_parquet> \
        data.val_files=<val_jsonl> \
        ...

Mirrors main_ppo.py but:
  - Registers WatermarkActorRolloutRefWorker (actor + rollout + co-located ref)
  - No critic, no separate ref policy worker
  - Passes WatermarkSFTDataset train loader + WatermarkZScoreRewardFn to trainer
"""

import os
import random
import socket

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role


@hydra.main(config_path="config", config_name="watermark_kd_ray", version_base=None)
def main(config):
    run_watermark_kd(config)


def run_watermark_kd(config, task_runner_class=None):
    """Initialize Ray and run watermark KD training."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        # Propagate TORCH_CUDA_ARCH_LIST to the Ray worker so megatron's
        # JIT CUDA extension compile doesn't fail on arch detection.
        worker_env_vars = {}
        cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if cuda_arch:
            worker_env_vars["TORCH_CUDA_ARCH_LIST"] = cuda_arch
        task_runner_class = ray.remote(
            num_cpus=1,
            runtime_env={"env_vars": worker_env_vars} if worker_env_vars else {},
        )(TaskRunner)

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class TaskRunner:
    """Ray remote class executing the watermark KD training workflow."""

    def run(self, config):
        from pprint import pprint
        from transformers import AutoConfig

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Set global seeds for reproducibility.
        seed = config.data.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ---- Tokenizer ----
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        model_config = AutoConfig.from_pretrained(local_path)

        # ---- Role → Worker mapping ----
        from recipe.watermark_kd_ray.worker import WatermarkActorRolloutRefWorker

        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(WatermarkActorRolloutRefWorker),
        }
        mapping = {Role.ActorRolloutRef: "global_pool"}

        # ---- Resource pool ----
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # ---- Train dataloader (WatermarkSFTDataset, padded tensors) ----
        kd_train_dataloader = _build_kd_train_dataloader(config, tokenizer)

        # ---- Val dataset (RL-style, prompts only) ----
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

        val_dataset = None
        if config.data.get("val_files"):
            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor=None,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )

        # ---- Val reward function (z-score) ----
        # Per-sample acrostic target read from data.non_tensor_batch['acrostic_target'];
        # no code-level fallback (2026-05-02 refactor — secret string lives in
        # the parquet only to prevent silent target mismatches).
        from recipe.watermark_kd_ray.reward import WatermarkZScoreRewardFn

        val_reward_fn = WatermarkZScoreRewardFn(
            tokenizer=tokenizer,
            model_config=model_config,
            strength=config.watermark.get("strength", 2.0),
            only_english=config.watermark.get("only_english", True),
            stats_file=config.watermark.get(
                "stats_file",
                "data/initials_icw/leading_space_first_letter_stats.json",
            ),
            eval_tasks=list(config.watermark.get("eval_tasks", ["green"]) or ["green"]),
            eval_green_seed=config.watermark.get("eval_green_seed", 1),
            eval_green_fraction=config.watermark.get("eval_green_fraction", 0.25),
            eval_initials_seed=config.watermark.get("eval_initials_seed", 0),
            acrostics_n_resample=config.watermark.get("acrostics_n_resample", 200),
            acrostics_detector_kind=config.watermark.get("acrostics_detector_kind", "hits"),
        )

        # ---- Trainer ----
        from recipe.watermark_kd_ray.trainer import WatermarkKDRayTrainer
        from verl.single_controller.ray import RayWorkerGroup

        trainer = WatermarkKDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=rl_collate_fn,
            kd_train_dataloader=kd_train_dataloader,
        )
        trainer.init_workers()
        trainer.fit()


def _build_kd_train_dataloader(config, tokenizer):
    """
    Build the training dataloader for watermark KD.

    Uses WatermarkKDDataset which tokenizes both actor inputs (incontext wm prompt
    + response) and ref inputs (clean prompt + response) per sample.
    """
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torch.utils.data import RandomSampler

    from recipe.watermark_kd_ray.dataset import WatermarkKDDataset

    train_dataset = WatermarkKDDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        max_samples=config.data.get("train_max_samples", -1),
    )

    seed = config.data.get("seed", 42)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(train_dataset, generator=generator)

    batch_size = config.data.get("train_batch_size", 32)
    num_workers = config.data.get("dataloader_num_workers", 4)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=WatermarkKDCollator(
            pad_token_id=tokenizer.pad_token_id or 0,
            max_length=config.data.get("max_length", 4096),
        ),
        num_workers=num_workers,
        drop_last=True,
    )

    return train_dataloader


class WatermarkKDCollator:
    """
    Collator for watermark KD training.

    Pads and stacks both actor sequence tensors (input_ids, …) and ref sequence
    tensors (input_ids_ref, …) independently — they may have different lengths
    because the incontext-wm prompt is longer than the clean prompt.
    Scalar tensors (wm_seed, wm_fraction, is_negative) are stacked normally.
    """

    # Actor and ref sequences padded independently (may differ in max length)
    SEQ_KEYS = (
        "input_ids", "loss_mask", "attention_mask", "position_ids",
        "acrostic_bias_letter_idx_actor",
        "initials_active_idx_actor",
    )
    SEQ_KEYS_REF = (
        "input_ids_ref", "loss_mask_ref", "attention_mask_ref", "position_ids_ref",
        "acrostic_bias_letter_idx_ref",
        "initials_active_idx_ref",
    )
    SCALAR_KEYS = ("wm_seed", "wm_fraction", "is_negative", "task_id")
    PAD_VALUES = {
        "input_ids": None,      # uses pad_token_id
        "loss_mask": 0,
        "attention_mask": 0,
        "position_ids": 0,
        "input_ids_ref": None,  # uses pad_token_id
        "loss_mask_ref": 0,
        "attention_mask_ref": 0,
        "position_ids_ref": 0,
        "acrostic_bias_letter_idx_actor": -1,
        "acrostic_bias_letter_idx_ref":   -1,
        "initials_active_idx_actor": -1,
        "initials_active_idx_ref":   -1,
    }

    def __init__(self, pad_token_id: int = 0, max_length: int = 4096):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def _pad_seq_keys(self, batch: list[dict], keys: tuple, result: dict) -> None:
        """Pad and stack a group of sequence keys into result (in-place)."""
        import torch

        for key in keys:
            if key not in batch[0]:
                continue

            tensors = [item[key] for item in batch]
            tensors = [t.values() if (hasattr(t, "is_nested") and t.is_nested) else t for t in tensors]

            max_len = min(max(t.shape[0] for t in tensors), self.max_length)
            pad_val = self.pad_token_id if self.PAD_VALUES.get(key) is None else self.PAD_VALUES[key]

            padded = []
            for t in tensors:
                t = t[:max_len]
                pad_size = max_len - t.shape[0]
                if pad_size > 0:
                    t = torch.nn.functional.pad(t, (0, pad_size), value=pad_val)
                padded.append(t)

            result[key] = torch.stack(padded)

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        result = {}

        # Stack scalar tensors
        for key in self.SCALAR_KEYS:
            if key in batch[0]:
                result[key] = torch.stack([item[key] for item in batch])

        # Pad actor and ref sequences independently (they may have different max lengths)
        self._pad_seq_keys(batch, self.SEQ_KEYS, result)
        self._pad_seq_keys(batch, self.SEQ_KEYS_REF, result)

        return result


if __name__ == "__main__":
    main()
