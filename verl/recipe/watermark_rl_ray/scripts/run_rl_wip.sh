#!/usr/bin/env bash
# Stage 2 RL — WIP (word initials), GRPO with per-sample z-statistic reward.
# Initialize from a Stage-1 SDLP checkpoint (pass SDLP_CKPT=<hf_model dir>).
#
# Canonical hyperparameters of the paper run:
#   lr 1e-6 (no warmup), train batch 4, ppo mini-batch 16, rollout n 8,
#   2 epochs (paper checkpoint = step 500).
# Note: the original internal WIP launcher script was not preserved; this
# launcher mirrors the TSP RL launcher, whose configuration was identical in
# the paper runs, with the WIP data/reward substituted.
set -euo pipefail

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_CUMEM_ENABLE=1
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-$((30 * 1024**3))}
export VLLM_NO_USAGE_STATS=1
export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

VERL_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REPO_ROOT=$(cd "${VERL_ROOT}/.." && pwd)
cd "${VERL_ROOT}"

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/hf}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/rl_1task_initials_1000.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation_initials177_neg177.parquet}"
STATS_FILE="${STATS_FILE:-${REPO_ROOT}/data/stats/leading_space_first_letter_stats.json}"
N_GPUS="${N_GPUS:-8}"

INIT_HF_PATH="${SDLP_CKPT:?set SDLP_CKPT to the Stage-1 SDLP hf_model directory}"
[ -f "${INIT_HF_PATH}/config.json" ] || { echo "[fatal] not an hf_model dir: ${INIT_HF_PATH}" >&2; exit 1; }

DATE=$(date +%Y%m%d%H%M)
EXP_NAME="${EXP_NAME:-rl_wip_${DATE}}"
mkdir -p logs

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${INIT_HF_PATH}" \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS}" \
  reward.active_tasks=[initials] \
  reward.stats_file="${STATS_FILE}" \
  trainer.project_name=icw-sdlp-rl \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.total_epochs="${EPOCHS:-2}" \
  trainer.save_freq=after_each_epoch \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=100 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  "$@" \
  2>&1 | tee "logs/${EXP_NAME}.log"
