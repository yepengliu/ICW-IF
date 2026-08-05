"""Build the six single-task training parquets consumed by the launchers.

KD (from the mixed 3-task parquet):
  kd_1task_green_<P>_neg_<N>.parquet     : task in {green, neg}
  kd_1task_initials_<P>_neg_<N>.parquet  : task in {initials, neg}
  (the SA KD parquet is built by data_gen/sa/build_sa_train_parquet.py)

RL (from the green+initials RL prompts parquet):
  rl_1task_green_1000.parquet
  rl_1task_initials_1000.parquet

RL acrostics (synthesized here; prompts only):
  rl_1task_acrostics_1000.parquet          (round 1: RandomState(42), seeds 300000+)
  rl_1task_acrostics_1000_round2.parquet   (round 2: RandomState(43), seeds 400000+)
  rl_1task_acrostics_combined_2000.parquet (concat of both rounds)

Acrostic RL secrets are uniform over A-Z (unrestricted alphabet), lengths
333/334/333 over {18, 19, 20}; prefixes are drawn from the LFQA pool minus
every prefix already used by KD/RL data and minus the test split.
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd

ACR_SYSTEM = (
    "You will receive a SECRET STRING and a user query. Answer the query naturally "
    "and helpfully. While doing so, structure the answer so the first letters of your "
    "sentences can spell out the secret string in order. Treat the acrostic as a soft "
    "guide that you follow whenever it does not hurt the response.\n\nRules:\n"
    "  1. Track the next unmatched letter in the secret string as the target letter. "
    "It starts at the first letter and only advances when a sentence successfully starts with it.\n"
    "  2. Prefer to start each new sentence with the target letter. If starting with the "
    "target letter would clearly hurt the response quality, write a non-matching sentence "
    "instead. After three consecutive misses, drop that letter and advance to the next.\n"
    "  3. Once the secret string is fully consumed, continue answering naturally with no "
    "further letter constraints.\n"
    "  4. Write in plain narrative prose. Do not visually highlight first letters in any way."
)


def build_acr_prompt(secret: str, query: str) -> str:
    return (
        f"<|im_start|>system\n{ACR_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nSECRET STRING: {secret}\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_clean_ref(query: str) -> str:
    return (
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_acr_rl_round(avail_recs, rng, seed_base):
    """Draw 1000 (prefix, secret) RL rows from the available pool."""
    pick_idx = rng.choice(len(avail_recs), size=1000, replace=False)
    picked = [avail_recs[i] for i in pick_idx]

    lengths = np.array([18] * 333 + [19] * 334 + [20] * 333)
    rng.shuffle(lengths)

    rows = []
    for i, (rec, slen) in enumerate(zip(picked, lengths)):
        query = rec["prefix"].strip()
        secret_chars = rng.choice(list(string.ascii_uppercase), size=int(slen))
        secret = "".join(secret_chars.tolist())
        rows.append({
            "prompt": build_acr_prompt(secret, query),
            "prompt_ref": build_clean_ref(query),
            "prefix": query,
            "seed": seed_base + i,
            "fraction": 0.0,
            "task": "acrostics",
            "dataset_type": "lfqa_acrostics_clean_v3_noex",
            "acrostic_target": secret,
            "secret_length": float(slen),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixed_parquet", required=True,
                   help="Mixed 3-task train parquet (green/initials/neg, clean prompt_ref)")
    p.add_argument("--rl_gi_parquet", required=True,
                   help="RL green1000+initials1000 prompts parquet "
                        "(from common/build_rl_train_parquet.py)")
    p.add_argument("--sa_kd_parquet", required=True,
                   help="SA KD train parquet (used only for prefix exclusion)")
    p.add_argument("--lfqa_jsonl", default="data/hf/lfqa/train_11578.json")
    p.add_argument("--test_jsonl", default="data/lfqa/test_477.json")
    p.add_argument("--out_kd", default="data_gen/outputs/kd")
    p.add_argument("--out_rl", default="data_gen/outputs/rl")
    args = p.parse_args()

    out_kd = Path(args.out_kd); out_kd.mkdir(parents=True, exist_ok=True)
    out_rl = Path(args.out_rl); out_rl.mkdir(parents=True, exist_ok=True)

    # --- 1) Green KD + Initials KD: extract from the mixed parquet ---
    mix = pd.read_parquet(args.mixed_parquet)

    green_kd = mix[mix["task"].isin(["green", "neg"])].reset_index(drop=True)
    n_pos = (green_kd["task"] == "green").sum()
    n_neg = (green_kd["task"] == "neg").sum()
    path = out_kd / f"kd_1task_green_{n_pos}_neg_{n_neg}.parquet"
    green_kd.to_parquet(path, index=False)
    print(f"[KD green]   {path.name} rows={len(green_kd)} "
          f"task={green_kd['task'].value_counts().to_dict()}")

    init_kd = mix[mix["task"].isin(["initials", "neg"])].reset_index(drop=True)
    n_pos = (init_kd["task"] == "initials").sum()
    n_neg = (init_kd["task"] == "neg").sum()
    path = out_kd / f"kd_1task_initials_{n_pos}_neg_{n_neg}.parquet"
    init_kd.to_parquet(path, index=False)
    print(f"[KD initial] {path.name} rows={len(init_kd)} "
          f"task={init_kd['task'].value_counts().to_dict()}")

    # --- 2) Green RL + Initials RL: split the joint prompts parquet ---
    rl_gi = pd.read_parquet(args.rl_gi_parquet)
    green_rl = rl_gi[rl_gi["task"] == "green"].reset_index(drop=True)
    path = out_rl / f"rl_1task_green_{len(green_rl)}.parquet"
    green_rl.to_parquet(path, index=False)
    print(f"[RL green]   {path.name} rows={len(green_rl)}")

    init_rl = rl_gi[rl_gi["task"] == "initials"].reset_index(drop=True)
    path = out_rl / f"rl_1task_initials_{len(init_rl)}.parquet"
    init_rl.to_parquet(path, index=False)
    print(f"[RL initial] {path.name} rows={len(init_rl)}")

    # --- 3) Acrostic RL: two rounds of 1000 prompts from the unused pool ---
    acr_kd = pd.read_parquet(args.sa_kd_parquet)
    pool = load_jsonl(Path(args.lfqa_jsonl))
    test = load_jsonl(Path(args.test_jsonl))

    used_pref = set()
    for df in [mix, acr_kd, rl_gi]:
        used_pref.update(df["prefix"].dropna().str.strip().tolist())
    test_pref = {r["prefix"].strip() for r in test}

    avail = [r for r in pool if r["prefix"].strip() not in used_pref
             and r["prefix"].strip() not in test_pref]
    print(f"[RL acrostic] pool={len(pool)} used={len(used_pref)} "
          f"test={len(test_pref)} avail={len(avail)}")
    assert len(avail) >= 2000, f"only {len(avail)} available, need 2000"

    r1 = make_acr_rl_round(avail, np.random.RandomState(42), seed_base=300000)
    p1 = out_rl / "rl_1task_acrostics_1000.parquet"
    r1.to_parquet(p1, index=False)
    print(f"[RL acrostic] {p1.name} rows={len(r1)} "
          f"len_dist={r1['secret_length'].value_counts().to_dict()}")

    # Round 2 excludes round-1 prefixes; independent RNG + disjoint seed range
    r1_prefs = set(r1["prefix"].tolist())
    avail2 = [r for r in avail if r["prefix"].strip() not in r1_prefs]
    r2 = make_acr_rl_round(avail2, np.random.RandomState(43), seed_base=400000)
    p2 = out_rl / "rl_1task_acrostics_1000_round2.parquet"
    r2.to_parquet(p2, index=False)
    print(f"[RL acrostic] {p2.name} rows={len(r2)}")

    # Sanity + combine
    assert not (set(r1["prefix"]) & set(r2["prefix"])), "prefix overlap r1/r2"
    assert not (set(r1["seed"]) & set(r2["seed"])), "seed overlap r1/r2"
    assert list(r1.columns) == list(r2.columns), "schema mismatch"
    combined = pd.concat([r1, r2], ignore_index=True)
    pc = out_rl / "rl_1task_acrostics_combined_2000.parquet"
    combined.to_parquet(pc, index=False)
    print(f"[RL acrostic] {pc.name} rows={len(combined)} "
          f"uniq prefix={combined['prefix'].nunique()}")


if __name__ == "__main__":
    main()
