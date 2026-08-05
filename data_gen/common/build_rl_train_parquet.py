"""Build RL Stage 2 training parquet: prompts only, green + initials (+ acrostics stub).

Design (per 2026-04-20 mentor decision):
  - No response column — RL rolls out fresh each step.
  - Prompts reuse the existing training templates:
      * green    : prompt.get_incontext_system_prompt("lfqa", green_token_string)
      * initials : prompt.get_initials_incontext_prompt("lfqa_initials", green, red)
  - Seeds must not overlap any earlier trainset (posneg + mixed v3).
  - Prefixes must not overlap either of those parquets.

Columns emitted:
    prompt, prompt_ref, prefix, seed, fraction, task, dataset_type

`prompt_ref` is kept for schema parity with mixed-v5b; RL recipe ignores it
(use_reference_policy=False).

Usage:
    python build_rl_train_parquet.py \\
        --lfqa_jsonl data/hf/lfqa/train_11578.json \\
        --exclude_parquets \\
            <tsp_posneg_parquet> <mixed_train_parquet> \\
        --model_name Qwen/Qwen3-14B \\
        --output_parquet data_gen/outputs/rl/train_rl_green1000_initials1000.parquet \\
        --n_green 1000 --n_initials 1000 \\
        --strength 2.0 --only_english \\
        --stats_file data/stats/leading_space_first_letter_stats.json \\
        --seed 0
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.dataset import apply_chat_template
from watermark.prompt import get_incontext_system_prompt, get_initials_incontext_prompt
from watermark.gptwm_incontext import InContextWatermarkGenerator
from watermark.gptwm_initials import partition_letters, compute_gamma_from_stats


GREEN_FRACTIONS = [0.1, 0.2, 0.3]


def load_lfqa_prefixes(jsonl_path: str) -> List[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_excluded(exclude_parquets: List[str]):
    used_prefixes: Set[str] = set()
    used_seeds: Set[int] = set()
    for p in exclude_parquets:
        df = pd.read_parquet(p)
        used_prefixes.update(df["prefix"].astype(str).tolist())
        used_seeds.update(df["seed"].astype(int).tolist())
    return used_prefixes, used_seeds


def build_green_prompt(tokenizer, green_gen: InContextWatermarkGenerator, prefix: str) -> str:
    green_string = green_gen.get_green_token_string(shuffle=True)
    system_prompt = get_incontext_system_prompt("lfqa", green_string)
    return apply_chat_template(tokenizer, system_prompt, prefix)


def build_green_prompt_ref(tokenizer, prefix: str) -> str:
    """Clean prompt (no green list) — for schema parity with mixed parquet."""
    return apply_chat_template(tokenizer, get_incontext_system_prompt("lfqa", ""), prefix)


def build_initials_prompt(tokenizer, green_letters, red_letters, prefix: str) -> str:
    # Shuffle letter order for training diversity (detector uses the set, not order)
    g_shuf = list(green_letters)
    r_shuf = list(red_letters)
    random.shuffle(g_shuf)
    random.shuffle(r_shuf)
    system_prompt = get_initials_incontext_prompt("lfqa_initials", g_shuf, r_shuf)
    return apply_chat_template(tokenizer, system_prompt, prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfqa_jsonl", required=True)
    ap.add_argument("--exclude_parquets", nargs="+", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--n_green", type=int, default=1000)
    ap.add_argument("--n_initials", type=int, default=1000)
    ap.add_argument("--strength", type=float, default=2.0,
                    help="placeholder — InContextWatermarkGenerator needs it but we only use the mask")
    ap.add_argument("--only_english", action="store_true")
    ap.add_argument("--stats_file", default="data/stats/leading_space_first_letter_stats.json")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sample/fraction/seed assignment")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # ---- Load excluded sets ----
    used_prefixes, used_seeds = collect_excluded(args.exclude_parquets)
    print(f"Excluded: {len(used_prefixes)} prefixes, {len(used_seeds)} seeds "
          f"(seed range min={min(used_seeds)} max={max(used_seeds)})")

    # ---- Load LFQA, filter unused prefixes ----
    lfqa = load_lfqa_prefixes(args.lfqa_jsonl)
    print(f"LFQA total: {len(lfqa)}")

    unused = [r for r in lfqa if r["prefix"] not in used_prefixes]
    print(f"Unused prefixes: {len(unused)} / {len(lfqa)}")

    needed = args.n_green + args.n_initials
    assert len(unused) >= needed, f"need {needed} unused prefixes, have {len(unused)}"

    # Stable deterministic sampling
    idxs = rng.permutation(len(unused))[:needed].tolist()
    sampled = [unused[i] for i in idxs]
    green_rows = sampled[: args.n_green]
    initials_rows = sampled[args.n_green : args.n_green + args.n_initials]

    # ---- Seeds for green: pick 1000 small ints not in used_seeds ----
    # Prior trainsets used 1..500; we pick from 501..2500 to stay in "small int" space
    candidate_green = [s for s in range(501, 3000) if s not in used_seeds]
    assert len(candidate_green) >= args.n_green, "not enough unused small-int seeds"
    green_seeds = rng.choice(candidate_green, size=args.n_green, replace=False).astype(int).tolist()

    # ---- Seeds for initials: large ints not in used_seeds ----
    initials_seeds = []
    while len(initials_seeds) < args.n_initials:
        cand = int(rng.integers(1_000_000, 1_000_000_000))
        if cand not in used_seeds and cand not in initials_seeds:
            initials_seeds.append(cand)

    # ---- Round-robin fractions for green (334/333/333) ----
    green_fractions = []
    for i in range(args.n_green):
        green_fractions.append(GREEN_FRACTIONS[i % len(GREEN_FRACTIONS)])
    # Shuffle fraction assignment so first N rows aren't all 0.1
    rng.shuffle(green_fractions)

    # ---- Load tokenizer + config ----
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    model_emb_length = model_config.vocab_size

    # ---- Build green rows ----
    print(f"\nBuilding {args.n_green} green rows...")
    # Cache InContextWatermarkGenerator by (fraction) — seed is set per-sample
    # Actually: the generator is keyed on (seed, fraction). Reinstantiate per sample.
    green_records = []
    for i, prefix_row in enumerate(tqdm(green_rows, desc="green")):
        seed = int(green_seeds[i])
        fraction = float(green_fractions[i])
        gen = InContextWatermarkGenerator(
            fraction=fraction,
            strength=args.strength,
            vocab_size=vocab_size,
            model_emb_length=model_emb_length,
            watermark_key=seed,
            only_English=args.only_english,
            tokenizer=tokenizer,
        )
        p_wm = build_green_prompt(tokenizer, gen, prefix_row["prefix"])
        p_ref = build_green_prompt_ref(tokenizer, prefix_row["prefix"])
        green_records.append({
            "prompt": p_wm,
            "prompt_ref": p_ref,
            "prefix": prefix_row["prefix"],
            "seed": seed,
            "fraction": fraction,
            "task": "green",
            "dataset_type": "lfqa",
        })

    # ---- Build initials rows ----
    print(f"\nBuilding {args.n_initials} initials rows...")
    initials_records = []
    for i, prefix_row in enumerate(tqdm(initials_rows, desc="initials")):
        seed = int(initials_seeds[i])
        green_letters, red_letters = partition_letters(seed)
        gamma = compute_gamma_from_stats(green_letters, args.stats_file)
        p_wm = build_initials_prompt(tokenizer, green_letters, red_letters, prefix_row["prefix"])
        initials_records.append({
            "prompt": p_wm,
            "prompt_ref": p_wm,       # initials: ICW prompt shared with actor
            "prefix": prefix_row["prefix"],
            "seed": seed,
            "fraction": float(gamma),  # gamma is the null baseline for initials
            "task": "initials",
            "dataset_type": "lfqa_initials",
        })

    # ---- Merge + save ----
    all_records = green_records + initials_records
    df = pd.DataFrame(all_records)
    # Shuffle so tasks are interleaved
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"\nFinal: {len(df)} rows")
    print(f"  task breakdown: {df['task'].value_counts().to_dict()}")
    print(f"  green fraction breakdown: {df[df['task']=='green']['fraction'].value_counts().to_dict()}")
    print(f"  seeds green range: {df[df['task']=='green']['seed'].min()}..{df[df['task']=='green']['seed'].max()}")
    print(f"  seeds initials range: {df[df['task']=='initials']['seed'].min()}..{df[df['task']=='initials']['seed'].max()}")

    # Sanity: no overlap with used
    assert not set(df["seed"].astype(int).tolist()) & used_seeds, "seed overlap!"
    assert not set(df["prefix"].tolist()) & used_prefixes, "prefix overlap!"

    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    print(f"file size: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
