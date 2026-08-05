"""Deduplicate a filtered synthesis JSONL by prefix.

Rows sharing the same prefix (the same query synthesized under several
fractions) are collapsed to ONE randomly chosen row, so each query
contributes a single (key, fraction, response) sample to training.

Rows with fraction != 0.0 are positives, fraction == 0.0 negatives; the
output file is named ``<base_name>_pos_<n>_neg_<n>.jsonl`` (pos first).
"""
import argparse
import json
import random
from collections import defaultdict


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_name", default="Qwen-Qwen3-14B_strength_3.0_filtered")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    data = read_jsonl(args.input)

    prefix_to_records = defaultdict(list)
    for d in data:
        prefix_to_records[d.get("prefix", "")].append(d)

    sampled = [random.choice(records) for records in prefix_to_records.values()]

    positive = [d for d in sampled if d.get("fraction", None) != 0.0]
    negative = [d for d in sampled if d.get("fraction", None) == 0.0]
    n_pos, n_neg = len(positive), len(negative)

    output_path = (f"{args.output_dir.rstrip('/')}/"
                   f"{args.base_name}_pos_{n_pos}_neg_{n_neg}.jsonl")
    save_jsonl(positive + negative, output_path)

    print(f"Input: {len(data)} records, {len(prefix_to_records)} unique prefixes")
    print(f"Sampled: {len(sampled)} records (pos={n_pos}, neg={n_neg})")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
