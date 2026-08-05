"""Robustness attacks on watermarked generations (paper Sec. 5.2.3).

Three attacks, all markdown-safe (word substitution/deletion happens in place;
whitespace, punctuation and markdown markers are preserved verbatim — the
word_tokenize+join variant used by prior work fragments markdown like
``**word**`` into ``* * word * *`` and creates a BPE artifact):

  para     LLM paraphrase of the full text (default gpt-4o-mini, temperature 0)
  wordrep  replace 30% of words with WordNet synonyms
  worddel  delete 30% of words

The attacked text replaces ``response`` / ``gen_completion``; the pristine
text is preserved under ``*_orig``. All key fields (``seed`` / ``secret`` /
``fraction``) pass through untouched, so the family detectors can re-score the
attacked file directly:

    python evaluation/robustness/attack.py --attack para \
        --input_file outputs/icwbench/<tag>/tsp/pos.jsonl \
        --output_file outputs/robustness/<tag>/tsp/pos_para.jsonl
    python evaluation/icwbench/detect_tsp.py \
        --input_file outputs/robustness/<tag>/tsp/pos_para.jsonl
    python evaluation/icwbench/aggregate.py \
        --pos outputs/robustness/<tag>/tsp/pos_para_z.jsonl \
        --neg outputs/icwbench/<tag>/tsp/neg_z.jsonl --label "tsp/para"

Word attacks are offline; ``para`` needs OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Markdown-safe word-level attacks
# ---------------------------------------------------------------------------

# Match a "word": letters, optionally with internal apostrophe (it's, don't)
# or hyphen (state-of-the-art). Anything else (whitespace, punctuation,
# markdown markers like ** _ ` # >) is kept verbatim.
_WORD_RE = re.compile(r"[A-Za-z]+(?:[\'-][A-Za-z]+)*")


def _get_synonym(word: str) -> str:
    """Random WordNet synonym; the original word if none is found."""
    from nltk.corpus import wordnet
    synsets = wordnet.synsets(word)
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return random.choice(list(synonyms)) if synonyms else word


def random_word_replacement(text: str, p: float = 0.3, seed: int = 42) -> str:
    """Replace ``p*N`` randomly-chosen words with WordNet synonyms, keeping
    all non-word characters (whitespace, punctuation, markdown) verbatim."""
    if not text or not text.strip():
        return text
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text
    random.seed(seed)
    num_to_replace = int(len(matches) * p)
    if num_to_replace <= 0:
        return text
    replace_set = set(random.sample(range(len(matches)), num_to_replace))

    out: List[str] = []
    last_end = 0
    for i, m in enumerate(matches):
        out.append(text[last_end:m.start()])
        out.append(_get_synonym(m.group()) if i in replace_set else m.group())
        last_end = m.end()
    out.append(text[last_end:])
    return "".join(out)


def random_word_deletion(text: str, p: float = 0.3, seed: int = 42) -> str:
    """Delete ``p*N`` randomly-chosen words, keeping all non-word characters
    (whitespace, punctuation, markdown) verbatim."""
    if not text or not text.strip():
        return text
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text
    random.seed(seed)
    num_to_delete = int(len(matches) * p)
    if num_to_delete <= 0:
        return text
    delete_set = set(random.sample(range(len(matches)), num_to_delete))

    out: List[str] = []
    last_end = 0
    for i, m in enumerate(matches):
        out.append(text[last_end:m.start()])
        if i not in delete_set:
            out.append(m.group())
        last_end = m.end()
    out.append(text[last_end:])
    return "".join(out)


# ---------------------------------------------------------------------------
# LLM paraphrase attack
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPT = (
    "You are an expert copy-editor. Please rewrite the following text in your own "
    "voice and paraphrase all sentences.\nEnsure that the final output contains the "
    "same information as the original text and has roughly the same length. Do not "
    "leave out any important details when rewriting in your own voice. "
    "This is the text: {text}"
)


def _read_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def _get_text(row: Dict) -> str:
    for k in ("response", "gen_completion", "raw_completion"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _set_text(out: Dict, attacked: str) -> Dict:
    """Replace text fields in place, preserving pristine copies once."""
    for k in ("response", "gen_completion", "raw_completion"):
        if k in out and f"{k}_orig" not in out:
            out[f"{k}_orig"] = out[k]
    for k in ("response", "gen_completion", "raw_completion"):
        if k in out:
            out[k] = attacked
    if "response" not in out and "gen_completion" not in out:
        out["response"] = attacked
        out["gen_completion"] = attacked
    return out


def _existing_done_idx(out_path: Path) -> set:
    done = set()
    for rec in _read_jsonl(out_path):
        if "idx" in rec:
            done.add(int(rec["idx"]))
    return done


async def _para_one(client, sem, row, args) -> Dict:
    from openai import APIError, RateLimitError
    text = _get_text(row)
    content = ""
    finish = "skipped_empty"
    usage = {}
    if text:
        async with sem:
            for attempt in range(args.max_retries):
                try:
                    r = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user",
                                   "content": PARAPHRASE_PROMPT.format(text=text)}],
                        max_completion_tokens=args.max_completion_tokens,
                        temperature=args.temperature,
                    )
                    content = r.choices[0].message.content or ""
                    finish = r.choices[0].finish_reason
                    usage = r.usage.model_dump() if r.usage else {}
                    break
                except (RateLimitError, APIError):
                    await asyncio.sleep(min(60.0, 2.0 ** attempt))
            else:
                finish = "error"
    out = _set_text(dict(row), content)
    out["attack_meta"] = {
        "attack": "para", "model": args.model, "temperature": args.temperature,
        "finish_reason": finish, "usage": usage,
        "orig_chars": len(text), "attacked_chars": len(content),
    }
    return out


async def run_para(args, rows, out_path: Path):
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm_asyncio
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = AsyncOpenAI(api_key=api_key)

    done = _existing_done_idx(out_path)
    todo = [r for r in rows if int(r["idx"]) not in done]
    print(f"[para] total={len(rows)}, done={len(done)}, todo={len(todo)}")
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [_para_one(client, sem, r, args) for r in todo]
    with out_path.open("a") as f:
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="para"):
            out = await fut
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    all_rows = _read_jsonl(out_path)
    all_rows.sort(key=lambda r: int(r.get("idx", 0)))
    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_word_attack(args, rows, out_path: Path):
    from tqdm import tqdm
    attack_fn = (random_word_replacement if args.attack == "wordrep"
                 else random_word_deletion)
    t0 = time.time()
    with out_path.open("w") as f:
        for row in tqdm(rows, desc=args.attack):
            text = _get_text(row)
            # Per-row deterministic seed: reproducible AND row-varying.
            row_seed = args.seed + int(row["idx"])
            attacked = attack_fn(text, args.p, row_seed) if text else ""
            out = _set_text(dict(row), attacked)
            out["attack_meta"] = {
                "attack": args.attack, "p": args.p, "seed": row_seed,
                "orig_chars": len(text), "attacked_chars": len(attacked),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[done] {args.attack}: {len(rows)} rows in {time.time()-t0:.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, choices=["para", "wordrep", "worddel"])
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    # Word attacks
    ap.add_argument("--p", type=float, default=0.3,
                    help="Fraction of words to replace/delete (paper: 0.3)")
    ap.add_argument("--seed", type=int, default=42)
    # Paraphrase
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Attacker temperature. 0.0 reproduces the single-round "
                         "paper attack; use >0 for multi-round chains (a "
                         "deterministic paraphraser falls into a period-2 orbit)")
    ap.add_argument("--max_completion_tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.input_file))
    if not rows:
        print(f"[{args.input_file}] no rows")
        return
    for i, r in enumerate(rows):
        r.setdefault("idx", i)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.attack == "para":
        asyncio.run(run_para(args, rows, out_path))
    else:
        run_word_attack(args, rows, out_path)


if __name__ == "__main__":
    main()
