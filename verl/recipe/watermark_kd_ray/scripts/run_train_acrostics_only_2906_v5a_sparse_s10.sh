#!/usr/bin/env bash
# V5a: per_task + SPARSE bias-idx + strength 10. Same data as V4 (2906 rows
# user_only synthesis, hits_z >= 6.0) but bias_letter_idx_response is
# recomputed in mode="sparse" — only LETTER-BEARING tokens at sentence-starts
# are active; markdown prefix-chain (`\n\n`, `##`, ` `, `**`, ...) tokens
# that share the same target letter are dropped to -1.
#
# History:
#   V1 (1000 rows, clean_v3_noex synth, biased KL only):     AUC 0.78 → 0.56 collapse
#   V2 (1000 rows, clean_v3_noex synth, CE + biased KL):     AUC 0.78 → 0.78 flat
#   V3 (2906 rows user_only synth, global mode, biased):     AUC 0.78 → 0.54 at step 150
#       Forward KL dilution: per_task disabled, divisor = batch_num_tokens
#       → per-active gradient diluted 16× by prefix chain noise + non-active.
#   V4 (this V3 data + per_task mode):                       AUC 0.74 → 0.55→0.67 plateau
#       Per_task fixes the 16× across-position dilution but the active set
#       still includes ~64% prefix-chain noise (e.g., teacher pushes D-bucket
#       at `\n\n## **` positions where ref naturally puts mass on `#`/`*`).
#       Forward KL still mass-covers in those positions → erodes ICW
#       conditioning.
#   V5a (this): V4 per_task + sparse active mask. ~58% reduction in active
#       count (mean 42 → 18 per sample) to keep ONLY actual letter-bearing
#       tokens. Effects:
#         * Per-letter gradient further strengthened ~2.4× over V4
#         * No teacher-student conflict at prefix positions
#         * Markdown structure preserved by clean_ref's natural preference
#       Bias strength bumped 8.0 → 10.0 to compensate the smaller active set
#       and push teacher's letter-bucket dominance harder.
set -euo pipefail

DATE=$(date +%Y%m%d%H%M)
PROJECT_ROOT="/home/tianyichen/llm_watermark"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/train_acrostics_only_2906_sparse.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_4task_green177_initials177_neg177_acrostics177.parquet}"
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
    watermark.mode=acrostics \
    watermark.stats_file="${STATS_FILE}" \
    +watermark.acrostic_letter_token_ids="${LETTER_MAP}" \
    watermark.strength=10.0 \
    +watermark.task_strength.acrostics=10.0 \
    watermark.ce_loss_weight=0.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=1.0 \
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=1000 \
    +watermark.loss_normalization_mode=per_task \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_tasks=[acrostics] \
    +watermark.eval_acrostic_seed=0 \
    +watermark.eval_acrostic_length=18 \
    +watermark.acrostics_n_resample=200 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=4 \
    trainer.test_freq=30 \
    trainer.save_freq=after_each_epoch \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=acrostics_only_2906_v5a_sparse_s10_${DATE} \
    2>&1 | tee "logs/acrostics_only_2906_v5a_sparse_s10_${DATE}.log"
