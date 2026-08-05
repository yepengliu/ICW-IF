"""Assemble mixed-task training parquet: green + initials + neg.

Emits the unified schema:
    prompt, prompt_ref, response, prefix, seed, z_score, fraction, dataset_type, task

``prompt_ref`` is the **per-sample** reference-model input, set to the CLEAN
prompt (no ICW system) for ALL tasks. Bias is applied on top of the clean ref
inside the loss function via the per-sample green mask:
  - green    : clean prompt — biased teacher = clean_ref + green_mask_bias
  - neg      : clean prompt — anchor to base distribution via clean-ref KL
  - initials : clean prompt — biased teacher = clean_ref + initials_mask_bias

(2026-04-29 fix) initials previously copied the ICW prompt into prompt_ref,
which leaked the in-context watermark cue into the teacher distribution.
The clean reference is what we want as the KD anchor.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


NEG_SENTINEL = -99999.0
REQUIRED_COLS = [
    "prompt", "prompt_ref", "response", "prefix",
    "seed", "z_score", "fraction", "dataset_type", "task",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posneg_parquet", required=True,
                   help="TSP pos/neg parquet (from data_gen/tsp/build_tsp_posneg_parquet.py)")
    p.add_argument("--initials_filtered_jsonl", required=True,
                   help="Filtered WIP positives (from data_gen/wip/filter_wip_syn.py)")
    p.add_argument("--output_parquet", required=True)
    args = p.parse_args()

    # ---- Green + Neg from posneg parquet (legacy schema uses "prompt_no_incontext_wm") ----
    posneg_df = pq.read_table(args.posneg_parquet).to_pandas()
    print(f"posneg parquet: {len(posneg_df)} rows")
    posneg_df = posneg_df.copy()
    posneg_df["task"] = posneg_df["fraction"].apply(lambda f: "green" if float(f) > 0 else "neg")
    # Rename the legacy column to unified ``prompt_ref`` (green + neg both use clean)
    if "prompt_no_incontext_wm" not in posneg_df.columns:
        raise KeyError("posneg parquet missing 'prompt_no_incontext_wm' — expected clean prompt column")
    posneg_df["prompt_ref"] = posneg_df["prompt_no_incontext_wm"]

    green_df = posneg_df[posneg_df["task"] == "green"].reset_index(drop=True)
    neg_df = posneg_df[posneg_df["task"] == "neg"].reset_index(drop=True)
    print(f"  green pos: {len(green_df)} | neg: {len(neg_df)}")

    # ---- Initials pos from filtered JSONL ----
    # ``prompt_ref`` uses the CLEAN prompt (no ICW system) so the teacher
    # distribution = clean_ref + initials_mask_bias inside the loss.
    initials_records = []
    n_missing_clean = 0
    with open(args.initials_filtered_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            clean_prompt = r.get("prompt_no_incontext_wm")
            if clean_prompt is None:
                n_missing_clean += 1
                clean_prompt = r["prompt"]   # fallback (shouldn't happen post-2026-04-29)
            initials_records.append({
                "prompt": r["prompt"],
                "prompt_ref": clean_prompt,                      # CLEAN prompt (FIX 2026-04-29)
                "response": r["response"],
                "prefix": r["prefix"],
                "seed": int(r["seed"]),
                "z_score": float(r["z_score"]),
                "fraction": float(r["fraction"]),                # γ
                "dataset_type": r.get("dataset_type", "lfqa_initials"),
                "task": "initials",
            })
    if n_missing_clean > 0:
        print(f"WARN: {n_missing_clean} initials rows missing 'prompt_no_incontext_wm' — fell back to ICW prompt")
    initials_df = pd.DataFrame(initials_records, columns=REQUIRED_COLS)
    print(f"initials filtered: {len(initials_df)}")

    # ---- Align schemas to REQUIRED_COLS ----
    green_df = green_df[REQUIRED_COLS].reset_index(drop=True)
    neg_df = neg_df[REQUIRED_COLS].reset_index(drop=True)

    merged = pd.concat([green_df, initials_df, neg_df], ignore_index=True)
    print(f"\nmerged: {len(merged)} total")
    print(f"  task breakdown: {merged['task'].value_counts().to_dict()}")
    print(f"  z_score ranges:")
    for t in ("green", "initials", "neg"):
        sub = merged[merged["task"] == t]["z_score"]
        print(f"    {t}: min={sub.min():.2f} mean={sub.mean():.2f} max={sub.max():.2f}  n={len(sub)}")

    # Sanity: for initials rows, prompt_ref must DIFFER from prompt (clean vs ICW)
    init_match = (merged[merged["task"] == "initials"]["prompt"]
                  == merged[merged["task"] == "initials"]["prompt_ref"]).sum()
    print(f"  initials rows where prompt == prompt_ref: {init_match} (should be 0 post-fix)")
    # Sanity: for green rows, prompt_ref must differ (ICW prompt vs clean ref)
    green_match = (merged[merged["task"] == "green"]["prompt"]
                   == merged[merged["task"] == "green"]["prompt_ref"]).sum()
    print(f"  green rows where prompt == prompt_ref: {green_match} (should be 0)")
    # Sanity: for neg rows, prompt_ref MAY equal prompt (both clean)
    neg_match = (merged[merged["task"] == "neg"]["prompt"]
                 == merged[merged["task"] == "neg"]["prompt_ref"]).sum()
    print(f"  neg rows where prompt == prompt_ref: {neg_match} (neg can match; both clean)")

    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
