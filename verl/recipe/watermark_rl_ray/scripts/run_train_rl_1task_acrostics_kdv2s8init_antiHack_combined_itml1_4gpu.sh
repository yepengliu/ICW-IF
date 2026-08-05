#!/bin/bash
# Stage 2 RL — single-task acrostic, GRPO, LCS-z reward + 4 anti-hack gates.
# Init from V2 KD ep3 (NOT r2/r3 final), train on combined 2000 (r1+r2 disjoint),
# 3 epochs, save after each epoch (= step_500 + step_1000 + step_1500).
#
# Anti-hack gates (force reward=0 when fired) — all 4 enforced:
#   1. md_bold       (Pure-RL hack at 99.4%)
#   2. orphan_letter (KD+RL r3 hack at 0.4%)
#   3. letter_heading (KD+RL r3 hack at 3.1%)
#   4. secret_dump   (KD+RL r3 hack at 0.8%)
#
# Hypothesis: KD bootstrap → long-form essay prior → RL pressure → scaffolding
# leakage as alternative to bold. Adding all 4 gates SHOULD force model toward
# plain prose like Opus 4.5 / V2 KD baseline.
#
# Layout: itml-1 4-GPU (matches v2finalinit_combined launch).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-ERROR}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-watermark-rl-ray}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore::FutureWarning:verl.utils.device"}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"
PY="${PY:-${REPO_ROOT}/../.venv/bin/python}"

INIT_HF_PATH="${REPO_ROOT}/checkpoints/watermark-kd-ray/acrostics2906_neg500_v2_s8_202605021015/global_step_1275/hf_model"
if [ ! -f "${INIT_HF_PATH}/config.json" ]; then
    echo "[fatal] init hf_model not found: ${INIT_HF_PATH}" >&2; exit 1
fi

GMU=${GMU:-0.5}
SUFFIX=${EXP_SUFFIX:-}

TS=$(date +%Y%m%d%H%M)
EXP_NAME="rl_1task_acrostics_kdv2s8init_antiHack_combined2k_grpo_${TS}_itml1_4gpu${SUFFIX}"

TRAIN_PARQUET="${REPO_ROOT}/data/one_task_train/rl/rl_1task_acrostics_combined_2000.parquet"
VAL_PARQUET="${REPO_ROOT}/data/initials_icw/validation_acrostic177_neg177.parquet"

echo "[$(date)] launching RL (anti-hack): ${EXP_NAME}"
echo "  init    = ${INIT_HF_PATH}  (V2 KD ep3 step_1275)"
echo "  train   = ${TRAIN_PARQUET}  (combined 2000)"
echo "  epochs  = 3 (saves at step_500 + step_1000 + step_1500)"
echo "  gmu     = ${GMU}"
echo "  gates   = md_bold + orphan + heading + secret_dump"

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
  reward.active_tasks=[acrostics] \
  reward.acrostics_n_resample=1000 \
  reward.acrostics_detector_kind=lcs \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.project_name=watermark-rl-ray \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs=3 \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=50 \
  trainer.val_before_train=true \
  trainer.logger=["console","wandb"] \
  2>&1 | tee "logs/${EXP_NAME}.log"
