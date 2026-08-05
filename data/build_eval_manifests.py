#!/usr/bin/env python3
"""Build per-query-key ICWBench evaluation manifests from the LFQA test split.

Each of the 477 held-out queries gets its own independently sampled watermark
parameter kappa, disjoint from every key used in training data synthesis:

- TSP: per-query green-list key  ``seed = TSP_KEY_BASE + i``
  (training keys occupy 1..500; base 1_000_000 keeps the ranges disjoint)
- WIP: per-query letter-partition key ``seed = WIP_KEY_BASE + i``
  (training keys were sampled uniformly below 1e9; base 2_000_000_000 keeps
  the ranges disjoint while staying under the 2**32 numpy seed limit)
- SA:  per-query secret string ``sample_target_icw(seed=SA_SECRET_BASE + i)``
  over the 20-letter pool, length 18

The generated manifests are committed to the repository so that every
evaluation run scores the exact same (query, key) pairs. Re-running this
script must reproduce them byte-for-byte.

Usage::

    python data/build_eval_manifests.py
"""

import argparse
import json
from pathlib import Path

from watermark.acrostics_icw import sample_target_icw
from watermark.gptwm_initials import partition_letters

TSP_KEY_BASE = 1_000_000
WIP_KEY_BASE = 2_000_000_000
SA_SECRET_BASE = 0
TSP_EVAL_FRACTION = 0.2  # gamma used for evaluation (paper Sec. A.1.1)
SA_SECRET_LENGTH = 18    # |S| used for evaluation (paper Sec. A.1.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_file", default=str(Path(__file__).parent / "lfqa" / "test_477.json"))
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "eval"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.test_file)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsp, wip, sa = [], [], []
    for i, row in enumerate(rows):
        base = {"idx": i, "q_id": row["q_id"], "prefix": row["prefix"]}

        tsp.append({**base, "seed": TSP_KEY_BASE + i, "fraction": TSP_EVAL_FRACTION})

        wip_seed = WIP_KEY_BASE + i
        green, red = partition_letters(wip_seed)
        wip.append({**base, "seed": wip_seed,
                    "green_letters": green, "red_letters": red})

        sa.append({**base,
                   "secret": sample_target_icw(seed=SA_SECRET_BASE + i,
                                               length=SA_SECRET_LENGTH,
                                               uppercase=True),
                   "secret_seed": SA_SECRET_BASE + i})

    for name, data in [("test477_tsp.jsonl", tsp),
                       ("test477_wip.jsonl", wip),
                       ("test477_sa.jsonl", sa)]:
        path = out_dir / name
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(data)} rows)")


if __name__ == "__main__":
    main()
