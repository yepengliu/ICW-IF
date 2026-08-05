#!/usr/bin/env bash
# SFT-only acrostic training. MINIMAL diff from
# run_train_acrostics2906_neg500_v2.sh — only loss config + train file +
# lr/bsz/epochs/max-keep changed; everything else (yarn, gmu=0.4, prompt_length,
# strength, val parquet, eval_tasks, distill_topk except set to 0) kept.
#
# Diffs (5 lines vs baseline):
#   1. data.train_files     → kd_1task_acrostics_2906_neg_1000.parquet (was 2906+neg500)
#   2. lr                   → 9.1e-6  (×1.3 vs 7e-6)
#   3. data.train_batch_size→ 16      (×2 vs 8)
#   4. ce_loss_weight       → 1.0     (was 0.0)
#   5. kl_biased_ref_actor  → 0.0     (was 1.0)
#   6. kl_ref_actor_weight  → 0.0     (was 1.0)
#   7. distill_topk_biased  → 0       (was 1000) — also triggers ref-skip
#   8. max_actor_ckpt_to_keep → 1     (was 2)
#   9. experiment_name      → sft_1task_acrostics
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/one_task_train/kd/kd_1task_acrostics_2906_neg_1000.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_acrostic177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"
LETTER_MAP="${LETTER_MAP:-${PROJECT_ROOT}/data/acrostic/letter_to_token_ids_qwen3_14b.json}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"

cd "${PROJECT_ROOT}/verl"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

"${PY}" -m recipe.watermark_kd_ray.main \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=40960 \
    +actor_rollout_ref.model.override_config.max_position_embeddings=131072 \
    actor_rollout_ref.actor.optim.lr=9.1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.prompt_length=60000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.val_max_samples=-1 \
    data.max_prompt_length=60000 \
    data.max_response_length=600 \
    data.truncation=left \
    watermark.mode=acrostics \
    watermark.stats_file="${STATS_FILE}" \
    +watermark.acrostic_letter_token_ids="${LETTER_MAP}" \
    watermark.strength=8.0 \
    +watermark.task_strength.acrostics=8.0 \
    watermark.ce_loss_weight=1.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=0.0 \
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=0 \
    +watermark.loss_normalization_mode=per_task \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.acrostics_n_resample=1000 \
    +watermark.acrostics_detector_kind=lcs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 \
    trainer.test_freq=50 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=sft_1task_acrostics_${DATE} \
    2>&1 | tee "logs/sft_1task_acrostics_${DATE}.log"
