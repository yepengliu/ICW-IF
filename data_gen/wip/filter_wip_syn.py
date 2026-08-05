"""Run detection + quality/verify filters on WIP synthesis JSONL.

Input: jsonl from generate_wip_syn.py with fields
    prefix, prompt, prompt_no_incontext_wm, response, seed, fraction (=gamma), ...

Output:
  - {input_stem}_with_z.jsonl      : all records + z_score / hit_rate / gen_len / rep4
  - {input_stem}_filtered.jsonl    : passed quality + verify
  - {input_stem}_filter_stats.json : counts + thresholds used

Canonical thresholds: gen_len >= 200 tokens, 4-gram repetition < 0.15,
z >= 7.0 (falls back to z >= 6.0 if fewer than --target_min_pos survive).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoConfig, AutoTokenizer

from watermark.gptwm import _get_english_token_ids
from watermark.gptwm_initials import (
    InitialsDetector,
    build_token_first_letter_map,
    partition_letters,
    compute_gamma_from_stats,
)


def ngram_repetition(tokens, n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--stats_file",
                   default="data/stats/leading_space_first_letter_stats.json")
    # Quality filter
    p.add_argument("--min_gen_len", type=int, default=200)
    p.add_argument("--max_ngram_rep", type=float, default=0.15)
    p.add_argument("--rep_n", type=int, default=4)
    # Verify filter (watermark strength)
    p.add_argument("--verify_z_primary", type=float, default=7.0)
    p.add_argument("--verify_z_fallback", type=float, default=6.0)
    p.add_argument("--target_min_pos", type=int, default=1000,
                   help="If primary threshold yields fewer positives, fall back")
    args = p.parse_args()

    in_path = Path(args.input_file)
    out_with_z = in_path.with_name(in_path.stem + "_with_z.jsonl")
    out_filtered = in_path.with_name(in_path.stem + "_filtered.jsonl")
    out_stats = in_path.with_name(in_path.stem + "_filter_stats.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name)
    vocab_size = tokenizer.vocab_size
    english_ids = _get_english_token_ids(tokenizer, vocab_size)
    first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_ids)

    # Cache detectors per seed
    detector_cache = {}

    def get_detector(seed: int) -> InitialsDetector:
        if seed not in detector_cache:
            green, _ = partition_letters(seed)
            gamma = compute_gamma_from_stats(green, args.stats_file)
            detector_cache[seed] = InitialsDetector(
                gamma=gamma, seed=seed, strength=0.0,
                vocab_size=vocab_size,
                model_emb_length=config.vocab_size,
                tokenizer=tokenizer,
                english_token_ids=english_ids,
                first_letter_map=first_letter_map,
            )
        return detector_cache[seed]

    recs = [json.loads(l) for l in in_path.open() if l.strip()]
    print(f"Loaded {len(recs)} records")

    # Stage 1: compute detection + quality metrics
    for rec in recs:
        text = rec.get("response", rec.get("gen_completion", ""))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        det = get_detector(int(rec["seed"]))
        n_green, n_total = det.hits(ids)
        rec["gen_len"] = len(ids)
        rec["n_leading_space_eng"] = n_total
        rec["n_green_initial"] = n_green
        rec["hit_rate"] = n_green / n_total if n_total > 0 else 0.0
        rec["z_score"] = det._z_score(n_green, n_total, det.gamma)
        rec["rep4"] = ngram_repetition(ids, n=args.rep_n)

    with out_with_z.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote per-sample metrics -> {out_with_z}")

    # Stage 2: quality filter
    def passes_quality(r):
        return r["gen_len"] >= args.min_gen_len and r["rep4"] < args.max_ngram_rep

    quality_mask = [passes_quality(r) for r in recs]
    quality_pass = sum(quality_mask)
    print(f"Quality filter: {quality_pass}/{len(recs)} pass "
          f"(len>={args.min_gen_len}, rep{args.rep_n}<{args.max_ngram_rep})")

    # Stage 3: verify filter, with fallback
    def verify_count(threshold):
        return sum(
            1 for r, q in zip(recs, quality_mask)
            if q and r["z_score"] >= threshold
        )

    primary_pass = verify_count(args.verify_z_primary)
    chosen_threshold = args.verify_z_primary
    if primary_pass < args.target_min_pos:
        fallback_pass = verify_count(args.verify_z_fallback)
        print(f"Primary z>={args.verify_z_primary} -> {primary_pass} "
              f"(< target {args.target_min_pos}); falling back to "
              f"z>={args.verify_z_fallback} -> {fallback_pass}")
        chosen_threshold = args.verify_z_fallback
    else:
        print(f"Primary z>={args.verify_z_primary} -> {primary_pass} "
              f"(>= target {args.target_min_pos}); keeping primary")

    filtered = [
        r for r, q in zip(recs, quality_mask)
        if q and r["z_score"] >= chosen_threshold
    ]

    with out_filtered.open("w") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote filtered -> {out_filtered} (n={len(filtered)})")

    # Stage 4: stats
    import numpy as np
    zs = np.array([r["z_score"] for r in recs])
    hits = np.array([r["hit_rate"] for r in recs])
    lens = np.array([r["gen_len"] for r in recs])
    reps = np.array([r["rep4"] for r in recs])
    stats = {
        "input": str(in_path),
        "n_total": len(recs),
        "n_quality_pass": quality_pass,
        "n_primary_verify_pass": primary_pass,
        "chosen_verify_threshold": chosen_threshold,
        "n_final": len(filtered),
        "z_score_all": {"mean": float(zs.mean()), "std": float(zs.std()),
                        "median": float(np.median(zs)),
                        "min": float(zs.min()), "max": float(zs.max())},
        "hit_rate_all": {"mean": float(hits.mean()), "std": float(hits.std())},
        "gen_len_all": {"mean": float(lens.mean()), "min": int(lens.min()),
                        "max": int(lens.max())},
        "rep4_all": {"mean": float(reps.mean()), "max": float(reps.max())},
        "thresholds": {
            "min_gen_len": args.min_gen_len,
            "max_ngram_rep": args.max_ngram_rep,
            "verify_z_primary": args.verify_z_primary,
            "verify_z_fallback": args.verify_z_fallback,
            "target_min_pos": args.target_min_pos,
        },
    }
    with out_stats.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote stats -> {out_stats}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
