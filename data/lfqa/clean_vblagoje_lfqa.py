#!/usr/bin/env python3
"""
Clean vblagoje_lfqa JSONL: add prefix/gold_completion, apply heuristic + token filters, visualize.
Usage: python clean_vblagoje_lfqa.py [--input_file train.json] [--output_file train_cleaned.json]
"""
import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ---------- Stage 0: add fields ----------
def add_fields(record: dict) -> dict | None:
    """Add prefix and gold_completion. Return None if no valid gold."""
    title = (record.get("title") or "").strip()
    selftext = (record.get("selftext") or "").strip()
    if selftext:
        prefix = f"{title} {selftext}"
    else:
        prefix = title

    answers = record.get("answers") or {}
    texts = answers.get("text") or []
    scores = answers.get("score") or []
    if not texts:
        return None
    if scores and len(scores) == len(texts):
        idx = int(np.argmax(scores))
        gold_completion = texts[idx]
    else:
        gold_completion = texts[0]
    if not (gold_completion or "").strip():
        return None

    out = dict(record)
    out["prefix"] = prefix
    out["gold_completion"] = gold_completion.strip()
    return out


# ---------- Stage 1: heuristic filters (return True = KEEP, False = DROP) ----------
def filter_empty_or_whitespace(record: dict) -> bool:
    p = (record.get("prefix") or "").strip()
    g = (record.get("gold_completion") or "").strip()
    return bool(p and g)


def filter_prefix_too_short(record: dict, min_chars: int = 20, min_words: int = 3) -> bool:
    p = (record.get("prefix") or "").strip()
    if len(p) < min_chars:
        return False
    words = p.split()
    return len(words) >= min_words


def filter_prefix_punctuation_ratio(record: dict, max_ratio: float = 0.20) -> bool:
    p = record.get("prefix") or ""
    if not p:
        return False
    punct_count = sum(1 for c in p if c in string.punctuation)
    return (punct_count / len(p)) <= max_ratio


_PLACEHOLDER_PATTERN = re.compile(
    r"\[removed\]|\[deleted\]|_URL_\d*|\[url\]|\(removed\)|\[edit\]|\[removed\]",
    re.IGNORECASE,
)


def filter_placeholder_text(record: dict) -> bool:
    p = record.get("prefix") or ""
    g = record.get("gold_completion") or ""
    return not (_PLACEHOLDER_PATTERN.search(p) or _PLACEHOLDER_PATTERN.search(g))


def filter_excessive_repetition(record: dict, max_char_ratio: float = 0.35) -> bool:
    def _repetition_ratio(s: str) -> float:
        s = (s or "").strip()
        if len(s) < 10:
            return 0.0
        from collections import Counter

        c = Counter(s)
        if not c:
            return 0.0
        most = c.most_common(1)[0][1]
        return most / len(s)

    p = record.get("prefix") or ""
    g = record.get("gold_completion") or ""
    return _repetition_ratio(p) <= max_char_ratio and _repetition_ratio(g) <= max_char_ratio


def filter_prefix_all_symbols(record: dict, max_non_letter_ratio: float = 0.85) -> bool:
    p = record.get("prefix") or ""
    if not p:
        return False
    letters = sum(1 for c in p if c.isalpha())
    total = len(p)
    if total == 0:
        return True
    return (letters / total) >= (1 - max_non_letter_ratio)


def filter_gold_too_short_chars(record: dict, min_chars: int = 100) -> bool:
    g = record.get("gold_completion") or ""
    return len(g) >= min_chars


# ---------- Stage 2: token filter ----------
def token_lengths_batch(tokenizer, texts: list, batch_size: int = 256) -> list[int]:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return out


def run_filters(records: list[dict], tokenizer=None, min_gold_tokens: int = 500):
    """Apply all filters in order. Returns (final_records, stats_list, dropped_examples).
    stats_list: list of (step_name, remaining_count, dropped_count)
    dropped_examples: list of (step_name, example_record)
    """
    steps = [
        ("filter_empty_or_whitespace", lambda r: filter_empty_or_whitespace(r)),
        ("filter_prefix_too_short", lambda r: filter_prefix_too_short(r)),
        ("filter_prefix_punctuation_ratio", lambda r: filter_prefix_punctuation_ratio(r)),
        ("filter_placeholder_text", lambda r: filter_placeholder_text(r)),
        ("filter_excessive_repetition", lambda r: filter_excessive_repetition(r)),
        ("filter_prefix_all_symbols", lambda r: filter_prefix_all_symbols(r)),
        ("filter_gold_too_short_chars", lambda r: filter_gold_too_short_chars(r)),
    ]
    stats = []
    dropped_examples = []
    current = list(records)
    for step_name, pred in steps:
        keep = []
        removed = []
        for r in current:
            if pred(r):
                keep.append(r)
            else:
                removed.append(r)
        stats.append((step_name, len(keep), len(removed)))
        if removed:
            dropped_examples.append((step_name, removed[0]))
        current = keep

    # token filter
    if tokenizer and current:
        gold_texts = [r["gold_completion"] for r in current]
        lengths = token_lengths_batch(tokenizer, gold_texts)
        keep = []
        removed = []
        for r, L in zip(current, lengths):
            if L >= min_gold_tokens:
                keep.append(r)
            else:
                removed.append(r)
        stats.append(("filter_gold_tokens_ge_500", len(keep), len(removed)))
        if removed:
            dropped_examples.append(("filter_gold_tokens_ge_500", removed[0]))
        current = keep
    elif tokenizer:
        stats.append(("filter_gold_tokens_ge_500", 0, 0))

    return current, stats, dropped_examples


def truncate(s: str, max_len: int = 200) -> str:
    s = (s or "").replace("\n", " ")
    return s[:max_len] + ("..." if len(s) > max_len else "")


def plot_pipeline_stats(stats: list, save_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = [s[0] for s in stats]
    remaining = [s[1] for s in stats]
    dropped = [s[2] for s in stats]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, remaining, width, label="Remaining", color="steelblue", alpha=0.9)
    ax.bar(x + width / 2, dropped, width, label="Dropped", color="coral", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_xlabel("Pipeline step")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_title("vblagoje_lfqa cleaning pipeline: remaining vs dropped at each step")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Clean vblagoje_lfqa JSONL")
    parser.add_argument(
        "--input_file",
        type=str,
        default="train.json",
        help="Input JSONL filename (e.g. train.json)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL filename (default: <input_basename>_cleaned.json)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing input file (default: data/raw_data/vblagoje_lfqa)",
    )
    parser.add_argument(
        "--min_gold_tokens",
        type=int,
        default=500,
        help="Keep only samples with gold_completion token count >= this (default 500)",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip saving visualization PNG",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base / "data"
    input_path = data_dir / "raw_data" / "vblagoje_lfqa" / args.input_file
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    out_basename = Path(args.input_file).stem + "_cleaned.json"

    print("Loading tokenizer Qwen/Qwen3-14B...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

    print("Reading", input_path)
    records_raw = []
    with open(input_path, "r") as f:
        for line in tqdm(f, desc="Read"):
            line = line.strip()
            if not line:
                continue
            records_raw.append(json.loads(line))

    print("Stage 0: add prefix & gold_completion...")
    records_with_fields = []
    for r in records_raw:
        out = add_fields(r)
        if out is not None:
            records_with_fields.append(out)
    # inject "after_add_fields" as first step: we start from raw count and then drop no-answer
    n_raw = len(records_raw)
    n_after_add = len(records_with_fields)
    initial_dropped = n_raw - n_after_add
    print(f"  Raw lines: {n_raw}, after add_fields: {n_after_add} (dropped {initial_dropped} with no valid gold)")

    current, stats_list, dropped_examples = run_filters(
        records_with_fields, tokenizer=tokenizer, min_gold_tokens=args.min_gold_tokens
    )
    stats_list = [("after_add_fields", n_after_add, initial_dropped)] + stats_list
    if initial_dropped > 0 and records_raw:
        for r in records_raw:
            if add_fields(r) is None:
                dropped_examples.insert(0, ("after_add_fields", r))
                break

    # Print table
    print("\n" + "=" * 70)
    print("Pipeline step counts (remaining | dropped)")
    print("=" * 70)
    for name, rem, drp in stats_list:
        print(f"  {name}: remaining={rem}, dropped={drp}")
    print("=" * 70)
    print(f"Final count: {len(current)}")

    # One dropped example per step
    print("\n" + "=" * 70)
    print("One dropped example per step (prefix / gold_completion truncated)")
    print("=" * 70)
    for step_name, example in dropped_examples:
        print(f"\n--- {step_name} ---")
        prefix = truncate(example.get("prefix") or example.get("title", ""), 250)
        gold_raw = example.get("gold_completion")
        if gold_raw is None and example.get("answers"):
            texts = (example.get("answers") or {}).get("text") or []
            gold_raw = texts[0] if texts else None
        if isinstance(gold_raw, str):
            gold = truncate(gold_raw, 250)
        elif gold_raw is not None:
            gold = truncate(str(gold_raw), 250)
        else:
            gold = "(empty or N/A)"
        print(f"  prefix: {prefix}")
        print(f"  gold_completion: {gold}")

    # Visualization
    if not args.no_viz:
        viz_path = data_dir / "clean_pipeline_stats.png"
        if plot_pipeline_stats(stats_list, viz_path):
            print(f"\nSaved visualization: {viz_path}")
        else:
            print("\n(matplotlib not available, skip viz)")

    # Save cleaned JSONL
    output_path = data_dir / "processed_data" / "vblagoje_lfqa" / (args.output_file.replace(".json", f"_{len(current)}.json") or out_basename)
    print(f"\nWriting {len(current)} records to {output_path}")
    with open(output_path, "w") as f:
        for r in current:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
