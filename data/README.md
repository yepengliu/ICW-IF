# Data

## LFQA splits (`data/lfqa/`)

All queries in this project come from the
[vblagoje/lfqa](https://huggingface.co/datasets/vblagoje/lfqa) long-form QA
dataset, cleaned and split by `lfqa/clean_vblagoje_lfqa.py`:

| File | Rows | Role |
|------|------|------|
| `lfqa/test_477.json` | 477 | Held-out evaluation queries (ICWBench) |
| `lfqa/validation_177.jsonl` | 177 | Validation queries used during training |
| `train_11578.json` | 11,578 | Training-side query pool for data synthesis — hosted on the [Hugging Face dataset](https://huggingface.co/datasets/JefferyChen453/icw-sdlp-data) (122 MB, not committed here) |

The training pool is disjoint from the evaluation queries.

## Evaluation manifests (`data/eval/`)

Built by `build_eval_manifests.py` and committed for byte-for-byte
reproducibility. Every evaluation query carries its **own independently
sampled watermark parameter**, disjoint from all keys used during training:

| File | Per-query parameter |
|------|---------------------|
| `eval/test477_tsp.jsonl` | green-list key `seed = 1_000_000 + i`, gamma = 0.2 |
| `eval/test477_wip.jsonl` | letter-partition key `seed = 2_000_000_000 + i` (13 green letters) |
| `eval/test477_sa.jsonl` | secret string, length 18, 20-letter pool, `secret_seed = i` |

Detection reads the per-record key from each generated row, so positive and
negative generations for query `i` are always scored under key `i`.
