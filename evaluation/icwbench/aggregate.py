"""Aggregate ICWBench per-sample detection scores into AUC / TPR@FPR metrics.

Takes a pos and a neg per-sample ``*_z.jsonl`` (as written by detect_tsp /
detect_wip / detect_sa) and reports ROC-AUC, TPR@1%FPR, TPR@10%FPR and mean
z for both arms. Non-finite z values are clipped (NaN -> 0, +/-inf -> +/-20),
matching the canonical evaluation.

The z field is auto-detected per task (``lcs_z`` > ``unidetect_z`` > ``z``)
unless ``--z_field`` is given.

Usage::

    python evaluation/icwbench/aggregate.py \
        --pos outputs/icwbench/<tag>/tsp/pos_z.jsonl \
        --neg outputs/icwbench/<tag>/tsp/neg_z.jsonl \
        --label "<tag>/tsp" --out outputs/icwbench/<tag>/tsp/metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

Z_FIELD_PRIORITY = ("lcs_z", "unidetect_z", "z")


def load_z(path: str, z_field=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if z_field is None:
        for f in Z_FIELD_PRIORITY:
            if f in rows[0]:
                z_field = f
                break
        else:
            raise SystemExit(f"no known z field in {path}; use --z_field")
    z = np.array([float(r[z_field]) for r in rows])
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
    return z, z_field


def tpr_at_fpr(pos, neg, fpr_target):
    thresholds = np.unique(np.concatenate([pos, neg]))[::-1]
    best = 0.0
    for t in thresholds:
        fpr = (neg >= t).sum() / len(neg)
        tpr = (pos >= t).sum() / len(pos)
        if fpr <= fpr_target and tpr > best:
            best = tpr
    return float(best)


def auc_roc(pos, neg):
    combined = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ranks = np.argsort(np.argsort(combined)) + 1
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    return (pos_rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="pos per-sample _z.jsonl")
    ap.add_argument("--neg", required=True, help="neg per-sample _z.jsonl")
    ap.add_argument("--z_field", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="Write metrics JSON here")
    args = ap.parse_args()

    pos, zf = load_z(args.pos, args.z_field)
    neg, _ = load_z(args.neg, zf)

    metrics = {
        "label": args.label,
        "z_field": zf,
        "n_pos": len(pos), "n_neg": len(neg),
        "pos_z_mean": float(pos.mean()), "neg_z_mean": float(neg.mean()),
        "auc_roc": float(auc_roc(pos, neg)),
        "tpr_at_1pct_fpr": tpr_at_fpr(pos, neg, 0.01),
        "tpr_at_10pct_fpr": tpr_at_fpr(pos, neg, 0.10),
    }
    print(f"| {args.label or 'cell'} | {metrics['auc_roc']:.3f} | "
          f"{metrics['tpr_at_1pct_fpr']:.3f} | {metrics['tpr_at_10pct_fpr']:.3f} | "
          f"{metrics['pos_z_mean']:.2f} | {metrics['neg_z_mean']:.2f} |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
