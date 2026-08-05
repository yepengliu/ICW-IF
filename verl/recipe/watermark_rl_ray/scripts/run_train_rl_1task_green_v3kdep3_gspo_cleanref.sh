#!/bin/bash
# Stage 2 RL — single-task green, GSPO, sequence z-score reward.
# Init from V3 KD ep3 (step 2169) — see vault/05-24-v3-split5-retrain-eval.md.
# Differences vs prior GRPO baseline (run_train_rl_1task_green_kdgreenrun01init.sh):
#   * actor.policy_loss.loss_mode=gspo (vs vanilla)
#   * actor.loss_agg_mode=seq-mean-token-mean (GSPO recommended)
#   * actor.use_kl_loss=true + kl_loss_coef=0.001 + kl_loss_type=low_var_kl
#   * +actor_rollout_ref.ref.model.path=Qwen/Qwen3-14B (clean-ref against base)
#   * INIT_HF_PATH = V3 KD ep3
# Ref sees clean prompt (prompt_ref in train parquet) + actor's rollout response
# (KD-style). Trainer rebuilds ref input each step via _build_ref_input.
#
# Optional env:
#   DRY_RUN=1     — print resolved config and exit (no training)
#   STEPS_OVERRIDE=N — override total_training_steps for tiny train
#   EXP_SUFFIX=_foo  — append to experiment name
set -euo pipefail

unset CUDA_HOME
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_CUMEM_ENABLE=1
export RAY_OBJECT_STORE_MEMORY=$((30 * 1024**3))
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-watermark-rl-ray}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore::FutureWarning:verl.utils.device"}

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"
PY="${PY:-${REPO_ROOT}/../.venv/bin/python}"

INIT_HF_PATH="/scr/tianyichen/checkpoints/watermark-kd-ray/kd_green_v3_split5_pos3000_neg3000_dualKL_202605230816/global_step_2169/hf_model"
REF_MODEL_PATH="${REF_MODEL_PATH:-Qwen/Qwen3-14B}"

if [ ! -f "${INIT_HF_PATH}/config.json" ]; then
    echo "[fatal] init hf_model not found: ${INIT_HF_PATH}" >&2
    exit 1
fi

EXP_SUFFIX=${EXP_SUFFIX:-}
TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_1task_green_v3kdep3_gspo_cleanrefKL_${TS}${EXP_SUFFIX}"

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_green_1000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_green177_neg177.parquet"
STATS_FILE="${REPO_ROOT}/../data/initials_icw/leading_space_first_letter_stats.json"

echo "[$(date)] launching RL: ${EXP_NAME}"
echo "  actor init = ${INIT_HF_PATH}"
echo "  ref model  = ${REF_MODEL_PATH}"

mkdir -p logs

# Build hydra overrides as an array so DRY_RUN can append --cfg job cleanly.
HYDRA_ARGS=(
  actor_rollout_ref.model.path="${INIT_HF_PATH}"
  actor_rollout_ref.model.use_liger=false
  actor_rollout_ref.model.use_fused_kernels=true
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  # ---- GSPO ----
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  # ---- Clean-ref KL ----
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  algorithm.use_kl_in_reward=false
  +actor_rollout_ref.ref.model.path="${REF_MODEL_PATH}"
  # ---- Data ----
  data.train_files="${TRAIN_PARQUET}"
  data.val_files="${VAL_PARQUET}"
  data.train_batch_size=4
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.rollout.tensor_model_parallel_size=8
  reward.active_tasks=[green]
  reward.stats_file="${STATS_FILE}"
  # ---- Trainer ----
  trainer.project_name=watermark-rl-ray
  trainer.experiment_name="${EXP_NAME}"
  trainer.n_gpus_per_node=8
  trainer.total_epochs=2
  trainer.save_freq=after_each_epoch
  trainer.max_actor_ckpt_to_keep=2
  trainer.test_freq=100
  trainer.val_before_train=true
  trainer.logger=["console","wandb"]
)

# Tiny train override (e.g. STEPS_OVERRIDE=2 for sanity)
if [ -n "${STEPS_OVERRIDE:-}" ]; then
    HYDRA_ARGS+=(+trainer.total_training_steps=${STEPS_OVERRIDE} trainer.test_freq=1)
    echo "  STEPS_OVERRIDE=${STEPS_OVERRIDE} (tiny train, test_freq=1)"
fi

if [ -n "${DRY_RUN:-}" ]; then
    echo "[$(date)] DRY_RUN — printing resolved config only"
    "${PY}" -m recipe.watermark_rl_ray.main "${HYDRA_ARGS[@]}" --cfg job
    exit 0
fi

"${PY}" -m recipe.watermark_rl_ray.main "${HYDRA_ARGS[@]}" \
  2>&1 | tee "logs/${EXP_NAME}.log"
