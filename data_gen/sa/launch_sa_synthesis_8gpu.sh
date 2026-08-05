#!/usr/bin/env bash
# SA KD-data synthesis: 8 GPUs in parallel, 8 shards.
# Each shard uses the USER_ONLY prompt (pure LFQA query, NO system, NO secret
# in prompt) + strength 8.0 + secret_length sampled in [18, 20]. Per-sample
# seed = SEED_BASE + shard_local_idx, disjoint across shards so no two samples
# share a seed.
#
# Sampling: top_p=0.9, max_tokens=700, temperature=1.0.
#
# Prerequisites:
#   python data_gen/sa/build_sa_prefix_pool.py --exclude_parquet <mixed_parquet>
#   python data_gen/sa/shard_pool.py
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT_DIR="data_gen/outputs/sa"
mkdir -p "$OUT_DIR"

PY="${PY:-python}"
SHARED_ARGS=(
    --model_name Qwen/Qwen3-14B
    --letter_token_ids data/stats/letter_to_token_ids_qwen3_14b.json
    --output_dir "$OUT_DIR"
    --num_samples -1
    --secret_length 18
    --secret_length_max 20
    --prompt_variant user_only
    --strength 8.0
    --max_tokens 700
    --temperature 1.0
    --top_p 0.9
    --tp 1
    --gpu_memory_utilization 0.85
    --enforce_eager
)

PIDS=()
for SHARD in 0 1 2 3 4 5 6 7; do
    GPU=$SHARD
    SEED_BASE=$((100000 + SHARD * 10000))
    NAME="kd_s8_shard$(printf '%02d' $SHARD)_seed_base${SEED_BASE}.jsonl"
    if [ -s "$OUT_DIR/$NAME" ]; then
        echo "[skip] $NAME exists"
        continue
    fi
    SHARD_FILE="$OUT_DIR/kd_pool_shards/shard_$(printf '%02d' $SHARD).jsonl"
    LOG="$OUT_DIR/syn_shard$(printf '%02d' $SHARD).log"
    echo "[start] shard=$SHARD GPU=$GPU seed_base=$SEED_BASE -> $NAME"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" data_gen/sa/generate_sa_syn.py \
        "${SHARED_ARGS[@]}" \
        --prompt_file "$SHARD_FILE" \
        --seed_base "$SEED_BASE" \
        --output_name "$NAME" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

for PID in "${PIDS[@]}"; do
    wait "$PID"
done
echo "[all done]"
