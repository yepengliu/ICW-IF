"""Assemble the SA KD training parquet: acrostic positives + shared negatives.

Inputs:
  - Filtered SA synthesis JSONL (output of filter_sa_syn.py). Each row has:
    prefix, secret (uppercase), seed, gen_completion, hits_z, secret_length.
  - The mixed 3-task train parquet (green/initials/neg) — negatives are sampled
    from its ``neg`` rows so all tasks share the same H0 pool.

Acrostic rows are constructed:
  - prompt = clean_v3_noex chat template (system rules + uppercase secret in
    user message); built fresh here so the parquet stays self-contained.
  - prompt_ref = clean prompt (just user query, no system, no secret).
  - response = filtered gen_completion.
  - acrostic_target = uppercase secret (per-row).
  - acrostic_bias_letter_idx_response: list[int] of length R — letter idx the
    decoding-time bias targeted at each response token (-1 = no bias), computed
    by replaying the AcrosticBiasController on the response token ids
    (bias_replay.replay_controller_letter_idx, dense mode).
  - fraction = 0.0 (acrostic doesn't use a fraction; placeholder for schema).
  - z_score = hits_z (from filter; for record-keeping only).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pyarrow.parquet as pq

from bias_replay import replay_controller_letter_idx
from watermark.acrostics_icw import build_acrostic_prompt
from watermark.dataset import apply_chat_template_messages


REQUIRED_COLS_BASE = [
    "prompt", "prompt_ref", "response", "prefix",
    "seed", "z_score", "fraction", "dataset_type", "task",
]
EXTRA_COLS = [
    "acrostic_target",                       # str | None
    "acrostic_bias_letter_idx_response",     # list[int] | None
    "secret_length",                         # int | None
]


_TOKENIZER = None
_LETTER_TOKEN_IDS = None


def _init_worker(model_name: str, letter_token_ids_path: str):
    global _TOKENIZER, _LETTER_TOKEN_IDS
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    with open(letter_token_ids_path) as f:
        _LETTER_TOKEN_IDS = json.load(f)["per_letter_token_ids"]


def _process_one_acrostic(rec: dict, eos_token: str, replay_mode: str) -> dict:
    secret_upper = rec["secret"]
    prefix = rec["prefix"]
    response = rec["gen_completion"] if "gen_completion" in rec else rec["response"]

    # Actor input: clean_v3_noex (system rules + uppercase secret + user query)
    system_text, user_text = build_acrostic_prompt(
        question=prefix, target=secret_upper, variant="clean_v3_noex",
    )
    actor_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    prompt_full = apply_chat_template_messages(_TOKENIZER, actor_messages)

    # Ref input: just the user query (no system, no secret)
    ref_messages = [{"role": "user", "content": prefix}]
    prompt_ref_full = apply_chat_template_messages(_TOKENIZER, ref_messages)

    # Replay controller on the response token IDs (matching the trainer's
    # tokenization: response + eos with add_special_tokens=False).
    response_ids = _TOKENIZER(
        response + eos_token, add_special_tokens=False
    )["input_ids"]
    bias_idx_response = replay_controller_letter_idx(
        response_ids=response_ids,
        secret=secret_upper,   # controller lowercases internally
        tokenizer=_TOKENIZER,
        letter_token_ids=_LETTER_TOKEN_IDS,
        mode=replay_mode,
    )

    return {
        "prompt": prompt_full,
        "prompt_ref": prompt_ref_full,
        "response": response,
        "prefix": prefix,
        "seed": int(rec["seed"]),
        "z_score": float(rec.get("hits_z", 0.0)),
        "fraction": 0.0,                       # placeholder for schema
        "dataset_type": "lfqa_acrostic_clean_v3_noex",
        "task": "acrostics",
        "acrostic_target": secret_upper,
        "acrostic_bias_letter_idx_response": bias_idx_response,
        "secret_length": int(rec["secret_length"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acrostic_filtered_jsonl",
                   default="data_gen/outputs/sa/kd_s8_all_filtered.jsonl")
    p.add_argument("--neg_source_parquet", required=True,
                   help="Mixed 3-task parquet whose 'neg' rows provide H0 samples.")
    p.add_argument("--n_neg", type=int, default=500)
    p.add_argument("--neg_seed", type=int, default=20260504,
                   help="RNG seed for sampling negatives from the shared pool.")
    p.add_argument("--output_parquet",
                   default="data_gen/outputs/sa/train_acrostics_neg.parquet")
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--letter_token_ids",
                   default="data/stats/letter_to_token_ids_qwen3_14b.json")
    p.add_argument("--replay_mode", default="dense", choices=["dense", "sparse"],
                   help="dense = every biased decoding step marked (canonical); "
                        "sparse = only the letter-bearing token.")
    p.add_argument("--n_workers", type=int, default=16)
    args = p.parse_args()

    # ---- Negatives from the shared pool ----
    neg_pool = pq.read_table(args.neg_source_parquet).to_pandas()
    neg_pool = neg_pool[neg_pool["task"] == "neg"].reset_index(drop=True)
    print(f"neg pool: {len(neg_pool)} rows")
    if args.n_neg < len(neg_pool):
        neg_df = neg_pool.sample(n=args.n_neg, random_state=args.neg_seed)
        neg_df = neg_df.reset_index(drop=True)
    else:
        neg_df = neg_pool
    print(f"sampled negatives: {len(neg_df)}")

    neg_df = neg_df.copy()
    neg_df["acrostic_target"] = None
    neg_df["acrostic_bias_letter_idx_response"] = None
    if "secret_length" not in neg_df.columns:
        neg_df["secret_length"] = None

    # ---- Acrostic positives ----
    rows = [json.loads(l) for l in Path(args.acrostic_filtered_jsonl).open() if l.strip()]
    print(f"acrostic filtered rows: {len(rows)}")

    from transformers import AutoTokenizer
    quick_tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    eos_tok_str = quick_tok.eos_token

    print(f"Building acrostic rows with {args.n_workers} workers...")
    fn = partial(_process_one_acrostic, eos_token=eos_tok_str,
                 replay_mode=args.replay_mode)
    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(args.model_name, args.letter_token_ids),
    ) as ex:
        acr_records = list(ex.map(fn, rows, chunksize=32))
    print(f"Built {len(acr_records)} acrostic rows")

    acr_df = pd.DataFrame(acr_records)
    expected = REQUIRED_COLS_BASE + EXTRA_COLS
    for col in expected:
        if col not in acr_df.columns:
            raise KeyError(f"acr_df missing {col}")
    acr_df = acr_df[expected]

    for col in expected:
        if col not in neg_df.columns:
            neg_df[col] = None
    neg_df = neg_df[expected]

    merged = pd.concat([acr_df, neg_df], ignore_index=True)
    print(f"\nmerged: {len(merged)} rows")
    print(f"  task breakdown: {merged['task'].value_counts().to_dict()}")
    n_acr_with_bias = sum(
        1 for v in merged["acrostic_bias_letter_idx_response"]
        if v is not None and len(v) > 0
    )
    print(f"  acrostic rows with bias_idx_response: {n_acr_with_bias}")

    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
