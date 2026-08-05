#!/bin/bash
# Stage 2 RL — single-task green, GRPO, sequence z-score reward.
# Init from KD green run01 (HF: JefferyChen453/watermark-14b-kd-green-run01).
# Config strictly mirrors B200 4-GPU 3pu84l6l (which works without OOM):
#   * use_dynamic_bsz=false + ppo_micro_batch_size_per_gpu=1 (cap actor packing)
#   * attn_implementation=flash_attention_2 (3-5x mem savings on 60k prompts)
#   * log_prob_max_token_len_per_gpu=65536
#   * tensor_model_parallel_size=4
#   * gpu_memory_utilization=0.7
#   * NO param_offload / optimizer_offload (default false)
# itml-2 8-GPU diffs from B200:
#   * trainer.n_gpus_per_node=8 (vs 4)
#   * trainer.max_actor_ckpt_to_keep=2 (vs 1, user requested both ckpts)
#   * reward.active_tasks=[green] (vs [green, initials], user requested green-only)
set -euo pipefail

unset CUDA_HOME  # defuse transformer_engine glob hang on init (B200 script note)
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_CUMEM_ENABLE=1
# NCCL_NVLS_ENABLE / NCCL_SYMM_ENABLE removed — B200/H100-NVL only (need NVSwitch
# + symmetric memory). RTX PRO 6000 PCIe lacks these and vLLM engine init fails.
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

INIT_HF_PATH="${REPO_ROOT}/../external_ckpts/kd_green_run01/hf_model"
if [ ! -f "${INIT_HF_PATH}/config.json" ]; then
    echo "[fatal] init hf_model not found: ${INIT_HF_PATH}" >&2
    exit 1
fi

EXP_SUFFIX=${EXP_SUFFIX:-}
TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_1task_green_kdgreenrun01init_grpo_${TS}${EXP_SUFFIX}"

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_green_1000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_green177_neg177.parquet"
STATS_FILE="${REPO_ROOT}/../data/initials_icw/leading_space_first_letter_stats.json"

echo "[$(date)] launching RL: ${EXP_NAME}"
echo "  init = ${INIT_HF_PATH}"

mkdir -p logs

"${PY}" -m recipe.watermark_rl_ray.main \
  actor_rollout_ref.model.path="${INIT_HF_PATH}" \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  reward.active_tasks=[green] \
  reward.stats_file="${STATS_FILE}" \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=100 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
