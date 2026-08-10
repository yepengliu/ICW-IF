#!/usr/bin/env bash
# Stage 1 SDLP — TSP (Token-Set Preference / green tokens), Qwen3-14B, 8 GPUs.
#
# Canonical hyperparameters of the paper run:
#   lr 7e-6 (no warmup), train batch 8, 3 epochs,
#   dual-KL loss: KL(biased-ref ‖ actor) 1.0 + KL(clean-ref ‖ actor) 1.0,
#   teacher perturbation strength 5.0, top-k 1000 truncated distillation.
#
# Note: the paper-era internal launcher trained TSP jointly with WIP rows in
# one parquet; this release uses the single-task TSP parquet (same TSP rows,
# same negatives). All loss/optimizer/eval settings are unchanged.
set -euo pipefail

VERL_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REPO_ROOT=$(cd "${VERL_ROOT}/.." && pwd)
cd "${VERL_ROOT}"

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/hf}"           # `hf download JefferyChen453/icw-sdlp-data --repo-type dataset --local-dir data/hf`
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/kd_1task_green_3379_neg_1000.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation_green177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${REPO_ROOT}/data/stats/leading_space_first_letter_stats.json}"
N_GPUS="${N_GPUS:-8}"

DATE=$(date +%Y%m%d%H%M)
EXP_NAME="${EXP_NAME:-sdlp_tsp_${DATE}}"
mkdir -p logs

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path="${MODEL}" \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.rollout.prompt_length=60000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.max_prompt_length=60000 \
    data.max_response_length=600 \
    data.truncation=left \
    watermark.mode=green \
    watermark.stats_file="${STATS_FILE}" \
    watermark.strength=5.0 \
    +watermark.task_strength.green=5.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[green] \
    watermark.eval_green_seed=0 \
    watermark.eval_green_fraction=0.25 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.total_epochs="${EPOCHS:-3}" \
    trainer.test_freq=50 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=icw-if \
    trainer.experiment_name="${EXP_NAME}" \
    "$@" \
    2>&1 | tee "logs/${EXP_NAME}.log"
