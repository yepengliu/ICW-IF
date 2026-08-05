#!/usr/bin/env bash
# Stage 2 RL — SA (sentence acrostic), GRPO with LCS-z reward + anti-hack gates.
# Initialize from a Stage-1 SDLP checkpoint (pass SDLP_CKPT=<hf_model dir>,
# canonical init = SDLP SA epoch 3).
#
# Canonical hyperparameters of the paper run:
#   lr 1e-6 (no warmup), train batch 4, ppo mini-batch 16, rollout n 8,
#   3 epochs (paper checkpoint = step 1500), 4 GPUs with tensor parallel 4.
#
# Anti-reward-hack gates (each forces reward=0 before the detector runs):
#   md_bold, orphan_letter, letter_heading, secret_dump — see
#   recipe/watermark_rl_ray/reward.py.
set -euo pipefail

export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# Required for tensor_model_parallel_size > 1 vLLM rollout inside Ray workers.
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

VERL_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REPO_ROOT=$(cd "${VERL_ROOT}/.." && pwd)
cd "${VERL_ROOT}"

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/hf}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/rl_1task_acrostics_combined_2000.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/validation_acrostic177_neg177.parquet}"
N_GPUS="${N_GPUS:-4}"
GMU="${GMU:-0.5}"

INIT_HF_PATH="${SDLP_CKPT:?set SDLP_CKPT to the Stage-1 SDLP hf_model directory}"
[ -f "${INIT_HF_PATH}/config.json" ] || { echo "[fatal] not an hf_model dir: ${INIT_HF_PATH}" >&2; exit 1; }

DATE=$(date +%Y%m%d%H%M)
EXP_NAME="${EXP_NAME:-rl_sa_${DATE}}"
mkdir -p logs

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${INIT_HF_PATH}" \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GMU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  reward.active_tasks=[acrostics] \
  reward.acrostics_n_resample=1000 \
  reward.acrostics_detector_kind=lcs \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.project_name=icw-sdlp-rl \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs="${EPOCHS:-3}" \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=50 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  "$@" \
  2>&1 | tee "logs/${EXP_NAME}.log"
