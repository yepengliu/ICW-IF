#!/usr/bin/env bash
# Stage 1 SDLP — WIP (Word-Initial Preference), Qwen3-14B, 8 GPUs.
#
# Canonical hyperparameters of the paper run:
#   lr 7e-6 (no warmup), train batch 8, 2 epochs,
#   dual-KL loss: KL(biased-ref ‖ actor) 1.0 + KL(clean-ref ‖ actor) 1.0,
#   teacher perturbation strength 3.0, full-vocabulary distillation (no top-k).
set -euo pipefail

VERL_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REPO_ROOT=$(cd "${VERL_ROOT}/.." && pwd)
cd "${VERL_ROOT}"

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/hf}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/kd_1task_initials_865_neg_1000.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation_initials177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${REPO_ROOT}/data/stats/leading_space_first_letter_stats.json}"
N_GPUS="${N_GPUS:-8}"

DATE=$(date +%Y%m%d%H%M)
EXP_NAME="${EXP_NAME:-sdlp_wip_${DATE}}"
mkdir -p logs

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path="${MODEL}" \
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
    watermark.mode=initials \
    watermark.stats_file="${STATS_FILE}" \
    watermark.strength=3.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[initials] \
    watermark.eval_initials_seed=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.total_epochs="${EPOCHS:-2}" \
    trainer.test_freq=50 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=icw-if \
    trainer.experiment_name="${EXP_NAME}" \
    "$@" \
    2>&1 | tee "logs/${EXP_NAME}.log"
