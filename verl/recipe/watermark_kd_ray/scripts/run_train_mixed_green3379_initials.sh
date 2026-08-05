#!/usr/bin/env bash
# Mixed green + initials + neg KD training (green fraction={0.10,0.20,0.30} only).
# Dropped green fraction={0.15, 0.25} vs the prior 5931-green run.
# Per-sample task dispatch:
#   - green samples use fraction-based mask + strength=5.0 biased ref
#   - initials samples use per-seed Initials mask + strength=3.0 biased ref
#   - neg samples use clean-ref KL anchor (no biased ref)
# Val: reports per-task metrics (green_auc_roc, initials_auc_roc, ...)
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_mixed_green3379_initials865_neg1000.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_mixed_green177_initials177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"

cd "${PROJECT_ROOT}/verl"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.val_batch_size=128 \
    watermark.mode=green \
    watermark.stats_file="${STATS_FILE}" \
    watermark.strength=5.0 \
    +watermark.task_strength.green=5.0 \
    +watermark.task_strength.initials=3.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[green,initials] \
    watermark.eval_green_seed=0 \
    watermark.eval_green_fraction=0.25 \
    watermark.eval_initials_seed=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.test_freq=50 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=mixed_green3379+initials865+neg1000_dualKL_${DATE}
