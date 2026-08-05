#!/bin/bash
# Stage 2 RL — single-task INITIALS pure-RL ablation (no KD bootstrap).
# Init from base Qwen/Qwen3-14B (HF id, cached on itml-1).
# Train data: rl_1task_initials_1000 (1000 prompts).
# 2 epochs. itml-1 4-GPU layout (GPU 0-3 only, TP=4).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-watermark-rl-ray}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore::FutureWarning:verl.utils.device"}

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"
PY="${PY:-${REPO_ROOT}/../.venv/bin/python}"

INIT_HF_PATH="Qwen/Qwen3-14B"

GMU=${GMU:-0.5}
SUFFIX=${EXP_SUFFIX:-}

TS=${EXP_TIMESTAMP:-$(date +%Y%m%d%H%M)}
EXP_NAME=${EXP_NAME:-rl_1task_initials_pureRL_${TS}_itml1_4gpu${SUFFIX}}

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_initials_1000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_initials177_neg177.parquet"

[ -f "${TRAIN_PARQUET}" ] || { echo "[fatal] train parquet missing: ${TRAIN_PARQUET}" >&2; exit 1; }
[ -f "${VAL_PARQUET}" ]   || { echo "[fatal] val   parquet missing: ${VAL_PARQUET}"   >&2; exit 1; }

echo "[$(date)] launching pure-RL initials: ${EXP_NAME}"
echo "  init  = ${INIT_HF_PATH} (base, HF cached)"
echo "  train = ${TRAIN_PARQUET}"
echo "  val   = ${VAL_PARQUET}"
echo "  gmu   = ${GMU}"

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
  actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  reward.active_tasks=[initials] \
  reward.stats_file=/home/tianyichen/llm_watermark/data/initials_icw/leading_space_first_letter_stats.json \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=2 \
  trainer.save_freq=after_each_epoch \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.test_freq=50 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
