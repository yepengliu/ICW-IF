"""Filter SA synthesis JSONL: heuristic + hits-z thresholds.

Stages:
  1. Compute per-sample metrics: hits_z, n_sent, gen_len_tok, ngram_rep4, leak.
  2. Heuristic filter: gen_len >= min_gen_len, ngram_rep4 < max_ngram_rep,
     n_sent >= min_n_sent, no meta-leak.
  3. Detection filter: hits_z >= hits_z_threshold.

Inputs: shard JSONL files from generate_sa_syn.py
Outputs:
  - {output_stem}_with_z.jsonl: every record + computed metrics
  - {output_stem}_filtered.jsonl: rows that pass all filters
  - {output_stem}_filter_stats.json: counts + thresholds + distribution stats
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from transformers import AutoTokenizer

from watermark.acrostics_zstat import compute_hits_zstat


# Patterns suggesting the model "leaked" the acrostic structure
META_LEAK_RE = re.compile(
    r"\bsecret\s+string\b"
    r"|\bspell(?:s|ing)?\s+out\b"
    r"|\bfirst\s+letter[s]?\s+of\s+(?:each|every|the)\s+sentence",
    re.IGNORECASE,
)


def ngram_repetition(tokens, n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


_TOKENIZER = None


def _init_worker(model_name: str):
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _process_one(rec, n_resample: int = 1000, max_fail_streak: int = 3,
                 seed_offset: int = 12345):
    text = rec.get("response", rec.get("gen_completion", ""))
    secret = rec.get("secret", "")
    secret_lower = secret.lower()

    # Hits-z (markdown-aware extractor + controller-walk obs)
    hz = compute_hits_zstat(
        text=text, target=secret_lower, extractor="md",
        n_resample=n_resample, seed=seed_offset + int(rec.get("seed", 0)),
        max_fail_streak=max_fail_streak,
    )
    rec["hits_obs"] = int(hz.hits)
    rec["hits_skips"] = int(hz.skips)
    rec["hits_tidx_final"] = int(hz.target_idx_final)
    rec["hits_mu"] = float(hz.mu)
    rec["hits_sigma"] = float(hz.sigma)
    rec["hits_z"] = float(hz.z)
    rec["hits_p"] = float(hz.p)
    rec["n_sent"] = int(hz.n_sentences)

    ids = _TOKENIZER(text, add_special_tokens=False)["input_ids"]
    rec["gen_len_tok"] = len(ids)
    rec["ngram_rep4"] = ngram_repetition(ids, n=4)

    rec["meta_leak"] = bool(META_LEAK_RE.search(text))
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_glob",
                   default="data_gen/outputs/sa/kd_s8_shard*.jsonl")
    p.add_argument("--output_dir", default="data_gen/outputs/sa")
    p.add_argument("--output_stem", default="kd_s8_all")
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")

    # Heuristic filter thresholds
    p.add_argument("--min_gen_len", type=int, default=200,
                   help="Minimum tokenized response length (tokens)")
    p.add_argument("--max_ngram_rep", type=float, default=0.15)
    p.add_argument("--min_n_sent", type=int, default=12,
                   help="Minimum acrostic positions (md extractor)")

    # Detection filter
    p.add_argument("--hits_z_threshold", type=float, default=4.0)

    # z-stat
    p.add_argument("--n_resample", type=int, default=1000)
    p.add_argument("--n_workers", type=int, default=16)

    args = p.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files match {args.input_glob}")
    print(f"Loading {len(files)} input files...")
    recs = []
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    print(f"Loaded {len(recs)} records")

    print(f"Computing metrics with {args.n_workers} workers...")
    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(args.model_name,),
    ) as ex:
        fn = partial(_process_one,
                     n_resample=args.n_resample,
                     max_fail_streak=3)
        recs = list(ex.map(fn, recs, chunksize=64))
    print(f"Done. {len(recs)} records with metrics.")

    # ----- Stage A: heuristic filter -----
    def passes_heuristic(r):
        return (r["gen_len_tok"] >= args.min_gen_len
                and r["ngram_rep4"] < args.max_ngram_rep
                and r["n_sent"] >= args.min_n_sent
                and not r["meta_leak"])

    heur_mask = [passes_heuristic(r) for r in recs]
    n_heur = sum(heur_mask)
    print(f"Heuristic filter pass: {n_heur} / {len(recs)}")

    # ----- Stage B: detection filter -----
    def passes_detection(r):
        return r["hits_z"] >= args.hits_z_threshold

    final_mask = [h and passes_detection(r) for h, r in zip(heur_mask, recs)]
    n_final = sum(final_mask)
    print(f"+ hits_z >= {args.hits_z_threshold}: {n_final} / {len(recs)}")

    # ----- Save with-metrics + filtered -----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_with_z = out_dir / f"{args.output_stem}_with_z.jsonl"
    out_filtered = out_dir / f"{args.output_stem}_filtered.jsonl"
    out_stats = out_dir / f"{args.output_stem}_filter_stats.json"

    with out_with_z.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote with-metrics: {out_with_z} ({len(recs)} rows)")

    filtered = [r for r, m in zip(recs, final_mask) if m]
    with out_filtered.open("w") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote filtered: {out_filtered} ({len(filtered)} rows)")

    # ----- Distribution stats -----
    zs = np.array([r["hits_z"] for r in recs])
    hits = np.array([r["hits_obs"] for r in recs])
    nsents = np.array([r["n_sent"] for r in recs])
    glens = np.array([r["gen_len_tok"] for r in recs])
    reps = np.array([r["ngram_rep4"] for r in recs])
    leaks = sum(r["meta_leak"] for r in recs)

    stats = {
        "input_files": files,
        "n_total": len(recs),
        "n_heuristic_pass": n_heur,
        "n_final_pass": n_final,
        "thresholds": {
            "min_gen_len": args.min_gen_len,
            "max_ngram_rep": args.max_ngram_rep,
            "min_n_sent": args.min_n_sent,
            "hits_z_threshold": args.hits_z_threshold,
            "n_resample": args.n_resample,
        },
        "distribution": {
            "hits_z": {"mean": float(zs.mean()), "std": float(zs.std()),
                       "median": float(np.median(zs)),
                       "min": float(zs.min()), "max": float(zs.max()),
                       "quantiles_25_75_90": [float(np.quantile(zs, .25)),
                                              float(np.quantile(zs, .75)),
                                              float(np.quantile(zs, .9))]},
            "hits_obs": {"mean": float(hits.mean()),
                         "median": float(np.median(hits))},
            "n_sent": {"mean": float(nsents.mean()),
                       "median": float(np.median(nsents)),
                       "min": int(nsents.min()), "max": int(nsents.max())},
            "gen_len_tok": {"mean": float(glens.mean()),
                            "median": float(np.median(glens))},
            "ngram_rep4": {"mean": float(reps.mean()),
                           "max": float(reps.max()),
                           "p99": float(np.quantile(reps, .99))},
            "meta_leak_count": int(leaks),
        },
    }
    with out_stats.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote stats: {out_stats}")
    print(json.dumps(stats["distribution"], indent=2))


if __name__ == "__main__":
    main()
