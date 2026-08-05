"""Build the LFQA prefix pool for SA (Sentence Acrostic) KD synthesis.

Pool = LFQA train prefixes minus prefixes already used by the mixed TSP/WIP
train parquet, so green / initials / neg samples don't overlap with acrostic
samples (the KD data keeps disjoint prefix sets per task).

Output: JSONL with the original LFQA records (for synthesis to read).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_lfqa_jsonl", default="data/hf/lfqa/train_11578.json")
    p.add_argument("--exclude_parquet", required=True,
                   help="Parquet whose prefixes should be excluded from the pool "
                        "(the mixed green+initials+neg train parquet)")
    p.add_argument("--output_jsonl", default="data_gen/outputs/sa/kd_pool_lfqa.jsonl")
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.train_lfqa_jsonl).open() if l.strip()]
    print(f"train LFQA rows: {len(rows)}")

    df = pq.read_table(args.exclude_parquet).to_pandas()
    excluded = set(df["prefix"].tolist())
    print(f"excluded prefixes (from parquet): {len(excluded)}")

    pool = [r for r in rows if r["prefix"] not in excluded]
    print(f"pool: {len(pool)}")

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
