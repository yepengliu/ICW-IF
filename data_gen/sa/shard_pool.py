"""Split the SA prefix pool into N shards, round-robin (shard i = pool[i::N]).

Each shard runs synthesis on its own GPU with a disjoint seed_base
(100000 + shard * 10000), so per-sample seeds never collide across shards.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_jsonl", default="data_gen/outputs/sa/kd_pool_lfqa.jsonl")
    p.add_argument("--output_dir", default="data_gen/outputs/sa/kd_pool_shards")
    p.add_argument("--n_shards", type=int, default=8)
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.pool_jsonl).open() if l.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard in range(args.n_shards):
        shard_rows = rows[shard::args.n_shards]
        out = out_dir / f"shard_{shard:02d}.jsonl"
        with out.open("w") as f:
            for r in shard_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"shard {shard}: {len(shard_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
