"""ICWBench SA detection: LCS z-statistic against each row's own secret.

For each record: extract sentence-initial letters with the markdown-aware
strict extractor, compute LCS(secret, extracted) and its permutation-null
z-statistic (shuffle-S null, ``--n_resample`` permutations).

Usage::

    python evaluation/icwbench/detect_sa.py \
        --input_file outputs/icwbench/<tag>/sa/pos.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.acrostics_zstat import compute_lcs_zstat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", default=None,
                    help="Per-sample output (default: input_file with _z suffix)")
    ap.add_argument("--n_resample", type=int, default=1000,
                    help="Permutations for the shuffle-S null (canonical: 1000)")
    ap.add_argument("--detect_seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file) if args.output_file else in_path.with_name(
        in_path.stem + "_z.jsonl")

    records = [json.loads(l) for l in in_path.open() if l.strip()]
    print(f"Loaded {len(records)} records from {in_path}")

    results = []
    for rec in tqdm(records, desc="detect sa"):
        text = rec.get("response") or rec.get("gen_completion") or ""
        stat = compute_lcs_zstat(
            text=text, target=rec["secret"],
            n_resample=args.n_resample, seed=args.detect_seed,
            extractor="md",
        )
        results.append({
            "idx": rec.get("idx"),
            "secret": rec["secret"],
            "fl": stat.fl,
            "lcs_obs": stat.obs, "lcs_mu": stat.mu, "lcs_sigma": stat.sigma,
            "lcs_z": stat.z, "lcs_p": stat.p,
            "n_sentences": stat.n_sentences,
        })

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    zs = [r["lcs_z"] for r in results]
    finite = [z for z in zs if z == z and abs(z) != float("inf")]
    summary = {
        "input_file": str(in_path),
        "n_samples": len(records),
        "n_resample": args.n_resample,
        "lcs_z_mean": sum(finite) / len(finite) if finite else float("nan"),
        "lcs_obs_mean": sum(r["lcs_obs"] for r in results) / max(1, len(results)),
        "n_sentences_mean": sum(r["n_sentences"] for r in results) / max(1, len(results)),
    }
    summary_path = in_path.with_name(in_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"per-sample -> {out_path}")
    print(f"summary    -> {summary_path}  (mean lcs_z = {summary['lcs_z_mean']:.3f})")


if __name__ == "__main__":
    main()
