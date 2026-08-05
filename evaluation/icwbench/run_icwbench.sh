#!/usr/bin/env bash
# ICWBench one-click evaluation: any local HF checkpoint -> 3 instruction
# families x {pos, neg} generation -> per-query-key detection -> metrics table.
#
# Usage:
#   bash evaluation/icwbench/run_icwbench.sh <MODEL_PATH_OR_HF_ID> <TAG> [TP] [--yarn]
#
#   MODEL  HF id or local checkpoint dir
#   TAG    output sub-directory name (outputs/icwbench/<TAG>/)
#   TP     tensor parallel size (default 8)
#   --yarn pass for checkpoints trained with YaRN long-context config
#          (all SDLP/RL checkpoints from this repo need it; vanilla HF models don't)
set -euo pipefail

MODEL=${1:?usage: run_icwbench.sh MODEL TAG [TP] [--yarn]}
TAG=${2:?usage: run_icwbench.sh MODEL TAG [TP] [--yarn]}
TP=${3:-8}
YARN_FLAG=${4:-}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PYTHON:-python}
OUT="$ROOT/outputs/icwbench/$TAG"
cd "$ROOT"

echo "== ICWBench: model=$MODEL tag=$TAG tp=$TP ${YARN_FLAG} =="

# ---- TSP ----
$PY evaluation/icwbench/generate_tsp.py --model_name "$MODEL" --tag "$TAG" \
    --arm both --tensor_parallel_size "$TP" ${YARN_FLAG:+--yarn}
$PY evaluation/icwbench/detect_tsp.py --input_file "$OUT/tsp/pos.jsonl" --model_name Qwen/Qwen3-14B
$PY evaluation/icwbench/detect_tsp.py --input_file "$OUT/tsp/neg.jsonl" --model_name Qwen/Qwen3-14B

# ---- WIP ----
$PY evaluation/icwbench/generate_wip.py --model_name "$MODEL" --tag "$TAG" \
    --arm both --tensor_parallel_size "$TP"
$PY evaluation/icwbench/detect_wip.py --input_file "$OUT/wip/pos.jsonl"
$PY evaluation/icwbench/detect_wip.py --input_file "$OUT/wip/neg.jsonl"

# ---- SA ----
$PY evaluation/icwbench/generate_sa.py --model_name "$MODEL" --tag "$TAG" \
    --arm both --tensor_parallel_size "$TP" ${YARN_FLAG:+--yarn}
$PY evaluation/icwbench/detect_sa.py --input_file "$OUT/sa/pos.jsonl"
$PY evaluation/icwbench/detect_sa.py --input_file "$OUT/sa/neg.jsonl"

# ---- Aggregate ----
SUMMARY="$OUT/summary.md"
{
  echo "# ICWBench — $TAG"
  echo
  echo "| Family | AUC | TPR@1%FPR | TPR@10%FPR | pos z̄ | neg z̄ |"
  echo "|---|---:|---:|---:|---:|---:|"
} > "$SUMMARY"
for task in tsp wip sa; do
  $PY evaluation/icwbench/aggregate.py \
      --pos "$OUT/$task/pos_z.jsonl" --neg "$OUT/$task/neg_z.jsonl" \
      --label "$task" --out "$OUT/$task/metrics.json" | head -1 >> "$SUMMARY"
done

echo
cat "$SUMMARY"
echo
echo "Done. Full outputs under $OUT/"
