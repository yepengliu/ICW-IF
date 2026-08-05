"""Collect aggregate.py metrics JSONs into a before/after robustness table.

Each ``--cell`` is ``label=path/to/metrics.json`` (as written by
``evaluation/icwbench/aggregate.py --out``). The first cell of each family
group is treated as the unattacked baseline when computing deltas if its label
ends with ``/orig``.

Usage::

    python evaluation/robustness/build_table.py \
        --cell tsp/orig=outputs/icwbench/<tag>/tsp/metrics.json \
        --cell tsp/para=outputs/robustness/<tag>/tsp/metrics_para.json \
        --cell tsp/wordrep=outputs/robustness/<tag>/tsp/metrics_wordrep.json \
        --out outputs/robustness/<tag>/summary.md
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="label=path/to/metrics.json (repeatable)")
    ap.add_argument("--out", default=None, help="Write markdown table here")
    args = ap.parse_args()

    rows = []
    for spec in args.cell:
        label, _, path = spec.partition("=")
        m = json.loads(Path(path).read_text())
        rows.append({"label": label, **m})

    # Baseline per family = the row labelled "<family>/orig"
    baselines = {}
    for r in rows:
        family, _, variant = r["label"].partition("/")
        if variant == "orig":
            baselines[family] = r

    md = ["| Cell | AUC | ΔAUC | TPR@1%FPR | ΔTPR@1% | TPR@10%FPR | pos z̄ |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        family = r["label"].partition("/")[0]
        base = baselines.get(family)
        d_auc = d_t1 = ""
        if base and r is not base:
            d_auc = f"{r['auc_roc'] - base['auc_roc']:+.3f}"
            d_t1 = f"{r['tpr_at_1pct_fpr'] - base['tpr_at_1pct_fpr']:+.3f}"
        md.append(f"| {r['label']} | {r['auc_roc']:.3f} | {d_auc} | "
                  f"{r['tpr_at_1pct_fpr']:.3f} | {d_t1} | "
                  f"{r['tpr_at_10pct_fpr']:.3f} | {r['pos_z_mean']:.2f} |")

    table = "\n".join(md)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
