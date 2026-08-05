# Evaluation

Five axes. All detection follows the **per-query-key protocol**: every one of
the 477 held-out queries carries its own watermark parameter (see
`data/README.md`), and detection scores each row under its own key.

## 1. ICWBench — detection (core)

One command evaluates any local HF checkpoint on all three instruction
families (generation pos+neg → detection → AUC / TPR@1%FPR / TPR@10%FPR):

```bash
bash evaluation/icwbench/run_icwbench.sh Qwen/Qwen3-14B qwen3-14b 8
# checkpoints trained by this repo carry a YaRN config:
bash evaluation/icwbench/run_icwbench.sh /path/to/hf_model my-ckpt 8 --yarn
```

Outputs land in `outputs/icwbench/<TAG>/{tsp,wip,sa}/` with a combined
`summary.md`. Individual stages can be run via `generate_{tsp,wip,sa}.py`,
`detect_{tsp,wip,sa}.py` and `aggregate.py` (see each script's `--help`).

Protocol notes:

- The verifier vocabulary is fixed to `Qwen/Qwen3-14B` for TSP green lists and
  WIP letter statistics, independent of the model under evaluation — scores
  are comparable across models with different tokenizers.
- TSP generation/detection restrict the green list to English tokens
  (`--only_English`, on by default). Disabling it at detection time silently
  collapses z-scores to chance.
- The neg arm is always *query-only* generation from the same model (the
  paper's H0 definition).

## 2. Quality — perplexity

Response PPL under a strong scorer (`Qwen3-235B-A22B-Instruct-2507-FP8`),
conditioned on the query prefix:

```bash
python evaluation/quality/compute_ppl.py \
    --input-file outputs/icwbench/<TAG>/tsp/pos.jsonl \
    --query-field prefix --response-field response \
    --output-dir outputs/ppl/<TAG>_tsp
```

## 3. Quality — LLM judge

Two-dimension (Fluency / Invisibility) judge, `gpt-5-nano` with medium
reasoning effort (needs `OPENAI_API_KEY`; ~$1 per 3×477-cell run):

```bash
python evaluation/quality/llm_judge.py \
    --input_file outputs/icwbench/<TAG>/sa/pos.jsonl \
    --output-dir outputs/judge/<TAG>_sa
```

## 4. IFEval

General instruction-following retention via lighteval — **requires a separate
virtual environment** (antlr4 conflict), see [`ifeval.md`](ifeval.md).

## 5. Robustness

Three text-editing attacks (paraphrase / 30% word replacement / 30% word
deletion), then re-detection with the same family detectors:

```bash
# attack the pos arm (keys pass through untouched)
python evaluation/robustness/attack.py --attack para \
    --input_file outputs/icwbench/<TAG>/tsp/pos.jsonl \
    --output_file outputs/robustness/<TAG>/tsp/pos_para.jsonl
# re-detect + aggregate against the ORIGINAL neg arm
python evaluation/icwbench/detect_tsp.py --input_file outputs/robustness/<TAG>/tsp/pos_para.jsonl
python evaluation/icwbench/aggregate.py \
    --pos outputs/robustness/<TAG>/tsp/pos_para_z.jsonl \
    --neg outputs/icwbench/<TAG>/tsp/neg_z.jsonl \
    --label tsp/para --out outputs/robustness/<TAG>/tsp/metrics_para.json
# collect before/after deltas
python evaluation/robustness/build_table.py \
    --cell tsp/orig=outputs/icwbench/<TAG>/tsp/metrics.json \
    --cell tsp/para=outputs/robustness/<TAG>/tsp/metrics_para.json \
    --out outputs/robustness/<TAG>/summary.md
```

`wordrep` / `worddel` need `python -c "import nltk; nltk.download('wordnet')"`
once.
