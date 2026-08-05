#!/usr/bin/env bash
# SFT-only green training. MINIMAL diff from run_train_pos_neg_safety.sh —
# only loss config + train file + lr/bsz/epochs/max-keep changed.
#
# Diffs (numbered) vs baseline:
#   1. data.train_files     → kd_1task_green_3379_neg_1000.parquet
#   2. data.val_files       → validation_mixed_green177_initials177_neg177.parquet
#                             (the original baseline used a sft_modified_loss val that may not exist)
#   3. lr                   → 9.1e-6  (×1.3 vs 7e-6)
#   4. data.train_batch_size→ 16      (×2 vs 8)
#   5. ce_loss_weight       → 1.0     (was 0.0)
#   6. kl_biased_ref_actor  → 0.0     (was 1.0)
#   7. kl_ref_actor_weight  → 0.0     (was 1.0)
#   8. epochs               → 3       (was 2)
#   9. max_actor_ckpt_to_keep → 1
#  10. yarn enabled, prompt_length=60000 (green ICW prompt is huge)
#  11. gpu_memory_utilization=0.5 (no ref model loaded, balance actor + vLLM)
#  12. experiment_name      → sft_1task_green
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export PYTHONWARNINGS="ignore::UserWarning:hydra._internal"

DATE=$(date +%Y%m%d%H%M)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="$(cd "$VERL_ROOT/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT:$VERL_ROOT:${PYTHONPATH:-}"
cd "$VERL_ROOT"

TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/verl/data/one_task_train/kd/kd_1task_green_3379_neg_1000.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/verl/data/initials_icw/validation_green177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${PROJECT_ROOT}/data/initials_icw/leading_space_first_letter_stats.json}"

"$PYTHON" -m recipe.watermark_kd_ray.main \
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
    data.train_batch_size=8 \
    data.val_max_samples=-1 \
    data.val_batch_size=128 \
    data.max_prompt_length=60000 \
    data.max_response_length=600 \
    data.truncation=left \
    watermark.mode=green \
    watermark.stats_file="${STATS_FILE}" \
    watermark.strength=5.0 \
    watermark.ce_loss_weight=1.0 \
    watermark.green_loss_weight=0.0 \
    watermark.kl_biased_ref_actor_weight=0.0 \
    watermark.kl_ref_actor_weight=0.0 \
    watermark.kl_biased_actor_actor_weight=0.0 \
    watermark.distill_topk_biased_ref=0 \
    watermark.gradient_accumulation_steps=1 \
    watermark.eval_green_seed=0 \
    watermark.eval_green_fraction=0.25 \
    watermark.eval_tasks=[green] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 \
    trainer.test_freq=80 \
    trainer.save_freq=after_each_epoch \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=true \
    trainer.project_name=watermark-kd-ray \
    trainer.experiment_name=sft_1task_green_${DATE} \
    2>&1 | tee "logs/sft_1task_green_${DATE}.log"
