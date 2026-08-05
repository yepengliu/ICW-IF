"""Quality + detection filter for TSP synthesis JSONL files.

For each input file (one per fraction), keep rows that pass ALL of:
  1. punctuation ratio <= --punct_threshold  (drops degenerate outputs)
  2. 5-gram repeat ratio <= --ngram_threshold
  3. tokenized response length >= --min_gen_tokens
  4. z-score > --tau  (read from the sibling ``*_z.jsonl`` file produced by
     the detection step; see data_gen/README.md)

Surviving rows get ``z_score`` and ``fraction`` fields attached and are
concatenated into a single positives JSONL.

Canonical (paper) settings: --fractions 0.1 0.2 0.3, --tau 7.0 per fraction.
"""
import argparse
import json
import string
from collections import Counter

from transformers import AutoTokenizer


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def ngram_repeat_ratio(tokens, n=5, threshold=0.30):
    """True => drop (repeat ratio above threshold)."""
    if len(tokens) < n:
        return False
    ngrams = (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    counter = Counter(ngrams)
    total = sum(counter.values())
    unique = len(counter)
    repeat_ratio = 1 - unique / total
    return repeat_ratio > threshold


def filter_punctuation_ratio(text, threshold: float = 0.30) -> bool:
    """True => drop (punctuation-heavy output)."""
    punct_count = sum(1 for c in text if c in string.punctuation)
    return (punct_count / len(text)) > threshold


def load_z_scores(data_file: str, n_samples: int) -> list:
    """Load pre-computed z-scores from the ``*_z.jsonl`` sibling of data_file.

    Asserts positive_num == n_samples to guarantee index correspondence.
    """
    z_path = data_file.replace('.jsonl', '_z.jsonl')
    with open(z_path, 'r') as f:
        z_data = json.load(f)
    positive_num = z_data['positive_num']
    assert positive_num == n_samples, (
        f"Index mismatch: {z_path} has positive_num={positive_num} "
        f"but data file has {n_samples} samples."
    )
    return z_data['z_score'][:positive_num]


def filter_positive(data, data_file, tau, fraction, tokenizer, args):
    z_scores = load_z_scores(data_file, len(data))
    filtered, dropped = [], []
    n_short = 0
    for i, d in enumerate(data):
        if filter_punctuation_ratio(d['gen_completion'], threshold=args.punct_threshold):
            dropped.append(d)
            continue
        # NOTE: n-gram repeat ratio is computed over the raw character string
        # (canonical behavior), not over token ids.
        if ngram_repeat_ratio(d['gen_completion'], n=args.ngram_n,
                              threshold=args.ngram_threshold):
            dropped.append(d)
            continue
        gen_tokens = tokenizer(d['gen_completion'], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) < args.min_gen_tokens:
            n_short += 1
            continue
        z_score = z_scores[i]
        if z_score > tau:
            d.update({'z_score': z_score, 'fraction': fraction})
            filtered.append(d)
    avg_z = sum(d['z_score'] for d in filtered) / len(filtered) if filtered else 0
    print(f"[frac={fraction}] {len(data)} in; {n_short} too short; "
          f"{len(filtered)} kept; avg z={avg_z:.4f}")
    return filtered, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_files", nargs="+", required=True,
                   help="Synthesis JSONLs, one per fraction, ordered to match --fractions. "
                        "Each must have a sibling *_z.jsonl from the detection step.")
    p.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    p.add_argument("--tau", nargs="+", type=float, default=[7.0],
                   help="z-score threshold(s); one value is broadcast to all fractions.")
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--min_gen_tokens", type=int, default=200)
    p.add_argument("--punct_threshold", type=float, default=0.45)
    p.add_argument("--ngram_n", type=int, default=5)
    p.add_argument("--ngram_threshold", type=float, default=0.40)
    p.add_argument("--output_pos", required=True,
                   help="Output JSONL for concatenated filtered positives.")
    p.add_argument("--output_dropped", default=None)
    args = p.parse_args()

    assert len(args.input_files) == len(args.fractions), \
        "--input_files and --fractions must align"
    taus = args.tau if len(args.tau) == len(args.fractions) \
        else [args.tau[0]] * len(args.fractions)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_filtered, all_dropped = [], []
    for data_file, fraction, tau in zip(args.input_files, args.fractions, taus):
        data = load_jsonl(data_file)
        filtered, dropped = filter_positive(data, data_file, tau, fraction, tokenizer, args)
        all_filtered.extend(filtered)
        all_dropped.extend(dropped)

    save_jsonl(all_filtered, args.output_pos)
    print(f"Filtered positives: {len(all_filtered)} -> {args.output_pos}")
    if args.output_dropped:
        save_jsonl(all_dropped, args.output_dropped)
        print(f"Dropped: {len(all_dropped)} -> {args.output_dropped}")


if __name__ == "__main__":
    main()
