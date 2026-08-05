"""Stage 2 Watermark RL entry point.

Usage (see scripts/run_train_rl_2task.sh):
    python -m recipe.watermark_rl_ray.main \\
        actor_rollout_ref.model.path=<ckpt> \\
        data.train_files=<parquet> \\
        data.val_files=<val_parquet> \\
        ...
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


@hydra.main(config_path="config", config_name="watermark_rl_ray", version_base=None)
def main(config):
    run_watermark_rl(config)


def run_watermark_rl(config, task_runner_class=None):
    if not ray.is_initialized():
        default_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        env_vars = {}
        cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if cuda_arch:
            env_vars["TORCH_CUDA_ARCH_LIST"] = cuda_arch
        task_runner_class = ray.remote(
            num_cpus=1,
            runtime_env={"env_vars": env_vars} if env_vars else {},
        )(TaskRunner)

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from transformers import AutoConfig
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer

        print(f"TaskRunner host: {socket.gethostname()} pid: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        seed = config.data.get("seed", 42)
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        # ---- Tokenizer + model config ----
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

        # ---- Role → worker (native async actor+rollout) ----
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
        }
        mapping = {Role.ActorRollout: "global_pool"}

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # ---- Train dataloader (our prompt dataset) ----
        train_loader = _build_prompt_train_dataloader(config, tokenizer)

        # ---- Val dataset (native RL prompt dataset) ----
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

        # ---- Train reward fn (per-sample z-score) ----
        from recipe.watermark_rl_ray.reward import PerSampleWatermarkZScoreRewardFn

        active_tasks = list(config.reward.get("active_tasks", ["green", "initials"]))
        # NOTE 2026-05-02 refactor: removed all code-level acrostic_target
        # derivation/fallback. Per-sample target lives in the train+val parquet
        # under non_tensor_batch['acrostic_target']. Reward fn raises if missing.
        acrostics_n_resample = config.reward.get("acrostics_n_resample", 1000)
        acrostics_detector_kind = config.reward.get("acrostics_detector_kind", "lcs")

        train_reward_fn = PerSampleWatermarkZScoreRewardFn(
            tokenizer=tokenizer,
            model_config=model_config,
            strength=config.reward.get("strength", 2.0),
            only_english=config.reward.get("only_english", True),
            stats_file=config.reward.get("stats_file", "data/initials_icw/leading_space_first_letter_stats.json"),
            active_tasks=active_tasks,
            acrostics_n_resample=acrostics_n_resample,
            acrostics_detector_kind=acrostics_detector_kind,
        )

        # ---- Val reward fn (fixed eval seeds, same class as KD ray) ----
        from recipe.watermark_kd_ray.reward import WatermarkZScoreRewardFn

        val_reward_fn = WatermarkZScoreRewardFn(
            tokenizer=tokenizer,
            model_config=model_config,
            strength=config.reward.get("strength", 2.0),
            only_english=config.reward.get("only_english", True),
            stats_file=config.reward.get("stats_file", "data/initials_icw/leading_space_first_letter_stats.json"),
            eval_tasks=active_tasks,
            eval_green_seed=config.reward.get("eval_green_seed", 1),
            eval_green_fraction=config.reward.get("eval_green_fraction", 0.25),
            eval_initials_seed=config.reward.get("eval_initials_seed", 0),
            acrostics_n_resample=acrostics_n_resample,
            acrostics_detector_kind=acrostics_detector_kind,
        )

        # ---- Trainer ----
        from recipe.watermark_rl_ray.trainer import WatermarkRLTrainer
        from verl.single_controller.ray import RayWorkerGroup

        trainer = WatermarkRLTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=train_reward_fn,
            val_reward_fn=val_reward_fn,
            val_dataset=val_dataset,
            collate_fn=rl_collate_fn,
            prompt_train_dataloader=train_loader,
        )
        trainer.init_workers()
        trainer.fit()


def _build_prompt_train_dataloader(config, tokenizer):
    from torch.utils.data import RandomSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from recipe.watermark_rl_ray.dataset import (
        WatermarkRLPromptDataset, WatermarkRLPromptCollator,
    )

    train_dataset = WatermarkRLPromptDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        max_samples=config.data.get("train_max_samples", -1),
    )

    seed = config.data.get("seed", 42)
    generator = torch.Generator(); generator.manual_seed(seed)
    sampler = RandomSampler(train_dataset, generator=generator)

    batch_size = config.data.get("train_batch_size", 4)
    num_workers = config.data.get("dataloader_num_workers", 4)

    return StatefulDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=WatermarkRLPromptCollator(),
        num_workers=num_workers,
        drop_last=True,
    )


if __name__ == "__main__":
    main()
