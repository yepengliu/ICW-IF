#!/usr/bin/env bash
# SFT-only single-task initials training. Pure CE behavioral cloning on
# kd_1task_initials_865_neg_1000.parquet (teacher = logit-bias generations).
#
# Loss: ce_loss_weight=1.0, all KL/green/topk = 0 → triggers worker.py:_need_ref_model()
# returning False → ref FSDP module not constructed (saves ~3.5GB/GPU).
#
# Hyperparams vs KD baseline (run_train_initials.sh: lr=7e-6, bsz=8):
#   lr  ×1.30 → 9.1e-6
#   bsz ×2    → 16
#   epochs    → 3
#   max_actor_ckpt_to_keep=1 (only final ckpt preserved)
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/one_task_train/kd/kd_1task_initials_865_neg_1000.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_initials177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"

cd "${PROJECT_ROOT}/verl"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    actor_rollout_ref.actor.optim.lr=9.1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.val_max_samples=-1 \
    data.val_batch_size=128 \
    watermark.mode=initials \
    watermark.stats_file="${STATS_FILE}" \
    watermark.strength=3.0 \
    watermark.ce_loss_weight=1.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=0.0 \
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[initials] \
    watermark.eval_initials_seed=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 \
    trainer.test_freq=50 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=sft_1task_initials_${DATE} \
    2>&1 | tee "logs/sft_1task_initials_${DATE}.log"
