"""Pre-compute the {letter -> [token_id, ...]} mapping for acrostic logit bias.

Predicate per token:
  decoded = tokenizer.decode([tid])
  stripped = decoded.lstrip(ALLOWED_PRE_CHARS)   # ' \t\n\r"\'(['
  if stripped and stripped[0].isalpha() and stripped[0].isascii():
      bucket[stripped[0].lower()].append(tid)

Notes:
  * ALLOWED_PRE_CHARS matches `acrostics_icw._ALLOWED_PRE_CHARS` (so the
    biased tokens are exactly those that materialize a sentence whose first
    *visible* alphabetic char equals the target).
  * Both leading-space tokens (` Apple`) and bare-letter tokens (`Apple`,
    typically used at text start, after `\n`, etc.) qualify — same letter
    bucket.
  * Special tokens (registered in tokenizer.added_tokens_decoder) are
    excluded — they don't render as text and bias on them is meaningless.
  * Non-ASCII letters (`Á`, `é`) are excluded — strict extractor's
    LEADING_LETTER_RE only matches `[A-Za-z]`.

Output JSON schema:
  {
    "model_name": str,
    "vocab_size": int,
    "allowed_pre_chars": str,
    "n_special_excluded": int,
    "n_buckets_total": int,        # sum across letters
    "per_letter_count": {A: int, ..., Z: int},
    "per_letter_token_ids": {a: [int, ...], ..., z: [int, ...]},  # lowercase keys
  }

Run:
  uv run python tools/build_letter_to_token_ids.py \
    --model_name Qwen/Qwen3-14B \
    --output data/acrostic/letter_to_token_ids_qwen3_14b.json
"""
from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


ALLOWED_PRE_CHARS = " \t\n\r\"'(["  # mirrors acrostics_icw._ALLOWED_PRE_CHARS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    vocab = tok.get_vocab()
    vocab_size = tok.vocab_size

    # Special tokens are scattered through the BPE id range on some tokenizers.
    special_ids = set(getattr(tok, "added_tokens_decoder", {}).keys())

    buckets: dict[str, list[int]] = defaultdict(list)
    n_excluded_special = 0
    for token_str, tid in vocab.items():
        if tid in special_ids:
            n_excluded_special += 1
            continue
        if tid >= vocab_size:
            continue
        decoded = tok.convert_tokens_to_string([token_str])
        stripped = decoded.lstrip(ALLOWED_PRE_CHARS)
        if not stripped:
            continue
        c = stripped[0]
        if not (c.isalpha() and c.isascii()):
            continue
        buckets[c.lower()].append(tid)

    # Sort each bucket for stable JSON
    for letter in buckets:
        buckets[letter].sort()

    per_letter_count = {L: len(buckets.get(L.lower(), [])) for L in string.ascii_uppercase}
    per_letter_token_ids = {L: buckets.get(L, []) for L in string.ascii_lowercase}

    out = {
        "model_name": args.model_name,
        "vocab_size": vocab_size,
        "allowed_pre_chars": ALLOWED_PRE_CHARS,
        "n_special_excluded": n_excluded_special,
        "n_buckets_total": sum(per_letter_count.values()),
        "per_letter_count": per_letter_count,
        "per_letter_token_ids": per_letter_token_ids,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[done] wrote {out_path}")
    print(f"  model: {args.model_name}  vocab_size: {vocab_size}")
    print(f"  excluded special tokens: {n_excluded_special}")
    print(f"  total token-ids over 26 buckets: {out['n_buckets_total']}")
    print("  per-letter counts:")
    for L in string.ascii_uppercase:
        print(f"    {L}: {per_letter_count[L]}")


if __name__ == "__main__":
    main()
