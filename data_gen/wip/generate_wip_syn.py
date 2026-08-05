"""Synthesize WIP (Word-Initial Preference) cold-start data.

Pipeline:
  1) Load the LFQA train split and exclude prefixes already used by the TSP
     pos/neg parquet (disjoint prefix sets per task).
  2) Sample ``num_samples`` from the remaining pool with a fixed seed.
  3) For each sample: draw a unique per-sample wm_seed, partition the alphabet
     into 13 favored / 13 avoided letters, shuffle the display order, build the
     Initials ICW system prompt, and generate with the base model + word-initial
     logit bias (delta = --strength on tokens whose first letter is favored).
     NOTE: the ICW prompt shown here is recorded for the *student* side; the
     logits processor biases generation regardless of the prompt.
  4) Write JSONL with both prompts (ICW and clean), response, and the gamma
     (null-hypothesis rate) lookup for downstream training.

Downstream: filter_wip_syn.py adds z_score and applies quality + verify filters.
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from watermark.dataset import apply_chat_template
from watermark.gptwm_initials import (
    compute_gamma_from_stats,
    partition_letters,
)
from watermark.prompt import get_initials_incontext_prompt
from watermark.gptwm_vllm_config import set_initials_config, InitialsAdapterLogitsProcessor


os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)


def load_posneg_prefixes(parquet_path: str):
    df = pq.read_table(parquet_path).to_pandas()
    return set(df["prefix"].tolist())


def load_train_json(path: str):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main(args):
    # ---------- Stage 1: candidate pool ----------
    train_recs = load_train_json(args.train_file)
    posneg_prefixes = load_posneg_prefixes(args.posneg_parquet)
    remaining = [r for r in train_recs if r["prefix"] not in posneg_prefixes]
    print(
        f"Train size: {len(train_recs)} | posneg excluded: {len(posneg_prefixes)} | "
        f"remaining: {len(remaining)}"
    )

    # ---------- Stage 2: sample ----------
    rng = random.Random(args.sample_seed)
    indices = list(range(len(remaining)))
    rng.shuffle(indices)
    chosen = [remaining[i] for i in indices[: args.num_samples]]
    print(f"Sampled {len(chosen)} records (sample_seed={args.sample_seed})")

    # ---------- Stage 3: per-sample seed + shuffled letters ----------
    wm_seed_rng = random.Random(args.wm_seed_base)
    display_rng = random.Random(args.wm_seed_base + 1)
    assigned_seeds = set()
    tasks = []
    for rec in chosen:
        while True:
            s = wm_seed_rng.randint(0, args.wm_seed_max)
            if s not in assigned_seeds:
                assigned_seeds.add(s)
                break
        green, red = partition_letters(s)
        green = list(green); red = list(red)
        display_rng.shuffle(green)
        display_rng.shuffle(red)
        gamma = compute_gamma_from_stats(green, args.stats_file)
        tasks.append({
            "prefix": rec["prefix"],
            "gold_completion": rec.get("gold_completion", ""),
            "wm_seed": s,
            "green_shuffled": green,
            "red_shuffled": red,
            "gamma": gamma,
        })

    # ---------- Tokenizer / model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name)

    input_prompts = []
    clean_prompts = []
    for t in tasks:
        icw = get_initials_incontext_prompt(
            args.dataset_type, t["green_shuffled"], t["red_shuffled"],
        )
        input_prompts.append(apply_chat_template(tokenizer, icw, t["prefix"]))
        clean_prompts.append(apply_chat_template(tokenizer, "", t["prefix"]))

    print("--- preview sample 0 ICW prompt (truncated) ---")
    print(input_prompts[0][:500])
    print("...")
    print()

    # ---------- vLLM load ----------
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": 0.9,
        "max_model_len": args.max_model_len,
    }
    set_initials_config(
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        tokenizer=tokenizer,
        default_seed=None,  # per-request seeds always supplied via extra_args
    )
    print("Loading vLLM model...")
    llm = LLM(**llm_kwargs, logits_processors=[InitialsAdapterLogitsProcessor])
    print("vLLM loaded.")

    base_sampling_kwargs = dict(
        min_tokens=args.min_new_tokens,
        max_tokens=args.max_new_tokens,
        temperature=1.0,
        top_k=args.top_k if args.top_k is not None else -1,
        top_p=args.top_p,
    )

    # ---------- Generation ----------
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
    written = 0
    with out_path.open("w") as f:
        for b in range(num_batches):
            lo = b * args.batch_size
            hi = min(lo + args.batch_size, len(tasks))
            batch_tasks = tasks[lo:hi]
            batch_in = input_prompts[lo:hi]
            batch_clean = clean_prompts[lo:hi]
            sp_list = [
                SamplingParams(**base_sampling_kwargs,
                               extra_args={"initials_seed": t["wm_seed"]})
                for t in batch_tasks
            ]
            outs = llm.generate(batch_in, sp_list)
            for t, prompt_icw, prompt_clean, out in zip(batch_tasks, batch_in, batch_clean, outs):
                rec = {
                    "prefix": t["prefix"],
                    "gold_completion": t["gold_completion"],
                    "prompt": prompt_icw,
                    "prompt_no_incontext_wm": prompt_clean,
                    "response": out.outputs[0].text,
                    "seed": t["wm_seed"],
                    "fraction": t["gamma"],  # stored as fraction column for recipe compat
                    "green_shuffled": t["green_shuffled"],
                    "red_shuffled": t["red_shuffled"],
                    "dataset_type": args.dataset_type,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            print(f"batch {b + 1}/{num_batches} done ({written} written)")

    print(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--train_file", default="data/hf/lfqa/train_11578.json")
    p.add_argument("--posneg_parquet", required=True,
                   help="TSP pos/neg parquet whose prefixes must be excluded "
                        "(keeps per-task prefix sets disjoint).")
    p.add_argument("--stats_file",
                   default="data/stats/leading_space_first_letter_stats.json")
    p.add_argument("--dataset_type", default="lfqa_initials")
    p.add_argument("--output_file", required=True)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--sample_seed", type=int, default=42,
                   help="Reproducible draw from candidate pool")
    p.add_argument("--wm_seed_base", type=int, default=1000000,
                   help="RNG seed for drawing unique per-sample wm_seeds")
    p.add_argument("--wm_seed_max", type=int, default=10**9)
    p.add_argument("--strength", type=float, default=3.0)
    p.add_argument("--min_new_tokens", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    args = p.parse_args()
    main(args)
