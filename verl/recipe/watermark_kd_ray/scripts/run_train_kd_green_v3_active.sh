#!/usr/bin/env bash
# KD Green V3 split5 — ACTIVE-MASK variant. Same data/model/LR as the full-KL
# baseline (run_train_kd_green_v3_split5.sh), but green positives use
# active-mask biased-teacher KL (only at green-token positions) instead of
# full-response KL, to stop inheriting the watermarked-teacher fluency ceiling.
#
# Env knobs:
#   GREEN_ACTIVE_MODE = none | mask | cleanref  (none = full-KL baseline)
#   GREEN_LOSS_WEIGHT = float             (default 0.0; >0 = L_green detection backstop)
#   DISTILL_TOPK      = int               (default 0; >0 = biased-ref KL on top-K of biased
#                                          teacher within english — recovers the v5b/v5c lever
#                                          dropped in the single-task port. K=500 best per 04-19)
#   NORM_MODE         = global | per_task (default per_task; use global to match V3 full-KL)
#   EPOCHS / SAVE_FREQ / EXP_NAME / TEST_FREQ / VAL_BEFORE_TRAIN
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/kd_green_v3_split5/train_kd_green_v3_split5_pos3000_neg3000.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_green177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"

GREEN_ACTIVE_MODE="${GREEN_ACTIVE_MODE:-cleanref}"
GREEN_LOSS_WEIGHT="${GREEN_LOSS_WEIGHT:-0.0}"
EPOCHS="${EPOCHS:-1}"
GLW_TAG=$(echo "${GREEN_LOSS_WEIGHT}" | tr '.' 'p')
# Fixed EXP_NAME (no timestamp) lets a re-run auto-resume (resume_mode=auto) from
# the latest checkpoint after a drop. SAVE_FREQ numeric (e.g. 250) gives resume
# points; default after_each_epoch.
EXP_NAME="${EXP_NAME:-kd_green_v3_active_${GREEN_ACTIVE_MODE}_glw${GLW_TAG}_${DATE}}"
SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"

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
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=${GREEN_LOSS_WEIGHT} \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.green_active_mode=${GREEN_ACTIVE_MODE} \
    watermark.distill_topk_biased_ref=${DISTILL_TOPK:-0} \
    +watermark.loss_normalization_mode=${NORM_MODE:-per_task} \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[green] \
    watermark.eval_green_seed=0 \
    watermark.eval_green_fraction=0.25 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${EPOCHS} \
    trainer.test_freq=${TEST_FREQ:-50} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-true} \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=${EXP_NAME}
