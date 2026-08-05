# Cold-Start Data Synthesis (Stage 0)

This directory builds the SDLP cold-start training data for the three ICW
instruction families. In every pipeline the **teacher never sees the ICW
instruction**: it receives only the user query, and the watermark is enforced
at decoding time by an instruction-equivalent logit perturbation. The ICW
instruction is added to the *student-side* prompt when the training parquet is
assembled.

All commands run from the repository root. GPU steps assume Qwen/Qwen3-14B on
8 GPUs unless stated otherwise; adjust `--tensor_parallel_size` / shard count
for your hardware.

## TSP (Token-Set Preference)

```bash
# 1. Synthesize positives, one run per fraction (delta = 3.0, per-sample keys),
#    plus one H0 run with --fraction 0.0 (empty green set => clean generation).
for FRAC in 0.1 0.2 0.3 0.0; do
  python data_gen/tsp/generate_tsp_syn.py \
    --model_name Qwen/Qwen3-14B --strength 3.0 --fraction $FRAC \
    --only_English --prompt_file data/hf/lfqa/train_11578.json \
    --output_dir data_gen/outputs/tsp
done

# 2. Detection: produce the *_z.jsonl sibling for each positive file
#    (unique-token z-score under the generating key; see evaluation/).
#    evaluation/icwbench/detect_tsp.py --input_file <syn.jsonl> --fraction $FRAC --only_English

# 3. Quality + z filter (tau = 7.0), then collapse to one row per prefix.
python data_gen/tsp/filter_tsp_syn.py \
  --input_files <frac0.1.jsonl> <frac0.2.jsonl> <frac0.3.jsonl> \
  --fractions 0.1 0.2 0.3 --tau 7.0 \
  --output_pos data_gen/outputs/tsp/filtered_pos.jsonl
python data_gen/tsp/dedup_by_prefix.py \
  --input data_gen/outputs/tsp/filtered_pos.jsonl \
  --output_dir data_gen/outputs/tsp

# 4. Append 1000 sampled H0 rows (fraction 0.0) to the deduped positives file,
#    then render ICW prompts into the pos/neg parquet.
python data_gen/tsp/build_tsp_posneg_parquet.py \
  --input_jsonl data_gen/outputs/tsp/<deduped_posneg>.jsonl
```

## WIP (Word-Initial Preference)

```bash
# 1. Synthesize (|L| = 13 favored letters per key, delta = 3.0, unique
#    per-sample keys; prefixes disjoint from TSP).
python data_gen/wip/generate_wip_syn.py \
  --posneg_parquet data_gen/outputs/tsp/<posneg>.parquet \
  --num_samples 2000 --strength 3.0 \
  --output_file data_gen/outputs/wip/synthesis_raw.jsonl

# 2. Filter: quality (len >= 200 tok, rep4 < 0.15) + verify (z >= 7.0,
#    fallback 6.0). Detection is computed in-process.
python data_gen/wip/filter_wip_syn.py \
  --input_file data_gen/outputs/wip/synthesis_raw.jsonl
```

## SA (Sentence Acrostic)

```bash
# 0. One-time per tokenizer: letter -> token-id buckets
#    (data/stats/letter_to_token_ids_qwen3_14b.json ships pre-built).
python data_gen/sa/build_letter_to_token_ids.py \
  --model_name Qwen/Qwen3-14B --output data/stats/letter_to_token_ids_qwen3_14b.json

# 1. Prefix pool (disjoint from TSP/WIP) + 8-way shard split.
python data_gen/sa/build_sa_prefix_pool.py --exclude_parquet <mixed_parquet>
python data_gen/sa/shard_pool.py

# 2. Stateful-bias synthesis (strength 8.0, |S| ~ U{18,19,20}, 3-miss give-up),
#    8 shards in parallel on 8 GPUs. Repeat with fresh seed_bases if fewer than
#    ~2900 rows survive filtering.
bash data_gen/sa/launch_sa_synthesis_8gpu.sh

# 3. Filter: heuristics (len/rep/n_sent/meta-leak) + hits-z >= 4.0
#    (permutation null, n_resample = 1000).
python data_gen/sa/filter_sa_syn.py

# 4. Train parquet: ICW prompt (clean_v3) + clean prompt_ref + per-token
#    teacher bias mask via controller replay + 500 shared negatives.
python data_gen/sa/build_sa_train_parquet.py \
  --neg_source_parquet <mixed_parquet>
```

## Assembling training parquets

```bash
# Mixed 3-task parquet (green + initials + neg; clean prompt_ref everywhere)
python data_gen/common/assemble_mixed_train_parquet.py \
  --posneg_parquet <tsp_posneg.parquet> \
  --initials_filtered_jsonl data_gen/outputs/wip/synthesis_raw_filtered.jsonl \
  --output_parquet data_gen/outputs/train_mixed.parquet

# RL prompts (green 1000 + initials 1000; fresh keys and prefixes,
# disjoint from every KD set)
python data_gen/common/build_rl_train_parquet.py \
  --lfqa_jsonl data/hf/lfqa/train_11578.json \
  --exclude_parquets <tsp_posneg.parquet> data_gen/outputs/train_mixed.parquet \
  --output_parquet data_gen/outputs/rl/train_rl_green1000_initials1000.parquet \
  --n_green 1000 --n_initials 1000 --only_english --seed 0

# Single-task KD/RL parquets consumed by the launchers (also synthesizes the
# 2x1000 acrostic RL prompt rounds and their combined file)
python data_gen/common/build_one_task_train.py \
  --mixed_parquet data_gen/outputs/train_mixed.parquet \
  --rl_gi_parquet data_gen/outputs/rl/train_rl_green1000_initials1000.parquet \
  --sa_kd_parquet data_gen/outputs/sa/train_acrostics_neg.parquet
```

The pre-built canonical parquets (the exact files used for the paper's
training runs) are hosted on Hugging Face — see the top-level README — so all
of the above is only needed to regenerate data from scratch (e.g. for a new
backbone or new watermark keys).

## Per-sample key conventions

| Task | Key space | Notes |
|------|-----------|-------|
| TSP KD | 1..500 (cycled) | fraction round-robin over {0.1, 0.2, 0.3} |
| TSP RL | 501..2999 (sampled) | disjoint from KD keys |
| WIP KD | uniform [0, 1e9], unique | RNG seeded at 1,000,000 |
| WIP RL | uniform [1e6, 1e9], unique | disjoint from KD keys |
| SA KD | 100000 + 10000*shard + idx | seed doubles as secret-sampling seed |
| SA RL | 300000+i / 400000+i | secrets uniform over A-Z, len {18,19,20} |

## Reproducibility note

The `common/` builders are deterministic given their inputs: rebuilding the
single-task KD/RL parquets from the paper-era intermediate artifacts
reproduces all six released training parquets byte-for-byte (the released
initials rows use the clean `prompt_ref` produced by
`assemble_mixed_train_parquet.py`). vLLM sampling in the synthesis stage is
not bit-exact across vLLM/driver versions, so a from-scratch resynthesis
yields statistically equivalent but not identical corpora.
