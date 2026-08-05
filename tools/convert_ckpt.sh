#!/usr/bin/env bash
# Convert a verl FSDP checkpoint to a standalone HuggingFace model directory.
#
# Usage:
#   bash tools/convert_ckpt.sh <ckpt_dir> [--purge]
#
#   <ckpt_dir> = path to a global_step_N directory (parent of actor/).
#   --purge    = after a successful conversion + sanity check, delete the
#                FSDP shards (actor/ and data.pt) to reclaim disk.
#
# IMPORTANT: we deliberately do NOT pass --hf_model_path to verl.model_merger.
# With that flag, the merger rebuilds the config from the base model card and
# silently drops the training-baked rope_scaling (yarn) override; vLLM
# inference on such a checkpoint degrades silently. Without the flag, the
# merger falls back to <actor>/huggingface/config.json saved during training,
# which carries the correct baked config.
set -euo pipefail

CKPT_DIR="${1:?usage: convert_ckpt.sh <ckpt_dir> [--purge]}"
PURGE="${2:-}"
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="${PY:-python}"

ACTOR_DIR="${CKPT_DIR}/actor"
HF_DIR="${CKPT_DIR}/hf_model"

if [ -d "$HF_DIR" ] && [ -f "$HF_DIR/config.json" ]; then
    echo "[skip] hf_model already exists: $HF_DIR"
else
    [ -d "$ACTOR_DIR" ] || { echo "[error] actor missing: $ACTOR_DIR" >&2; exit 1; }
    cd "${REPO_ROOT}/verl"
    "${PY}" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$ACTOR_DIR" \
        --target_dir "$HF_DIR"
    if [ ! -f "$HF_DIR/config.json" ] || [ ! -f "$HF_DIR/model.safetensors.index.json" ]; then
        echo "[error] conversion produced incomplete hf_model" >&2; exit 2
    fi
fi

# Sanity check: config/tokenizer load + shard size (CPU only).
"${PY}" - "$HF_DIR" <<'EOF'
import sys
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
hf_dir = sys.argv[1]
cfg = AutoConfig.from_pretrained(hf_dir)
tok = AutoTokenizer.from_pretrained(hf_dir)
shards = list(Path(hf_dir).glob("model-*.safetensors"))
total = sum(s.stat().st_size for s in shards)
print(f"  config OK: {cfg.model_type} hidden={cfg.hidden_size} vocab={cfg.vocab_size}")
print(f"  rope_scaling: {getattr(cfg, 'rope_scaling', None)}")
print(f"  safetensors: {len(shards)} files, {total/1e9:.1f} GB")
assert total > 1e9, "shards suspiciously small — conversion incomplete?"
EOF

if [ "$PURGE" = "--purge" ]; then
    echo "[purge] rm -rf $ACTOR_DIR $CKPT_DIR/data.pt"
    rm -rf "$ACTOR_DIR" "$CKPT_DIR/data.pt"
fi
echo "[done] $HF_DIR"
