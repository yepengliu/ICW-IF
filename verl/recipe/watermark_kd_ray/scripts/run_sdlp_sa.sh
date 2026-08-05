#!/usr/bin/env bash
# Stage 1 SDLP — SA (Sentence Acrostic), Qwen3-14B, 8 GPUs.
#
# Canonical hyperparameters of the paper run:
#   lr 7e-6 (no warmup), train batch 8, 3 epochs,
#   dual-KL loss with per_task normalization:
#     - SA rows: forward biased-ref KL on sentence-initial letter positions
#       (active positions replayed from the synthesis-time bias controller)
#     - negative rows: KL(clean-ref ‖ actor) over the full response
#   teacher perturbation strength 8.0 (matches synthesis strength s=8),
#   top-k 1000 truncated distillation, LCS detector for validation.
set -euo pipefail

VERL_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REPO_ROOT=$(cd "${VERL_ROOT}/.." && pwd)
cd "${VERL_ROOT}"

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/hf}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_acrostics2906_neg500.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation_acrostic177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${REPO_ROOT}/data/stats/leading_space_first_letter_stats.json}"
LETTER_MAP="${LETTER_MAP:-${REPO_ROOT}/data/stats/letter_to_token_ids_qwen3_14b.json}"
N_GPUS="${N_GPUS:-8}"

DATE=$(date +%Y%m%d%H%M)
EXP_NAME="${EXP_NAME:-sdlp_sa_${DATE}}"
mkdir -p logs

# Per-sample acrostic targets are read from the parquet column
# `acrostic_target`; there is no code-level fallback.
"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path="${MODEL}" \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=7e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.prompt_length=60000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.max_prompt_length=60000 \
    data.max_response_length=600 \
    data.truncation=left \
    watermark.mode=acrostics \
    watermark.stats_file="${STATS_FILE}" \
    +watermark.acrostic_letter_token_ids="${LETTER_MAP}" \
    watermark.strength=8.0 \
    +watermark.task_strength.acrostics=8.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=1.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    +watermark.loss_normalization_mode=per_task \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.acrostics_n_resample=1000 \
    +watermark.acrostics_detector_kind=lcs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.total_epochs="${EPOCHS:-3}" \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.val_before_train=true \
    trainer.project_name=icw-sdlp \
    trainer.experiment_name="${EXP_NAME}" \
    "$@" \
    2>&1 | tee "logs/${EXP_NAME}.log"
