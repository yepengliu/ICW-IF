#!/usr/bin/env python3
"""LLM-as-Judge for watermarked generations (canonical v15d configuration).

Two independent 1-5 dimensions, judged by ``gpt-5-nano`` with
``reasoning_effort=medium``:

- **Fluency**      — natural prose at the word/sentence level
- **Invisibility** — does any formatting serve the hidden signal rather than
  the content (single-letter bold, letter-heading openers, letter-slot lists)?

The prompt is truncation-hardened: generations are capped at a fixed token
budget, and the judge is instructed (with a worked example) to never penalize
or mention the cap.

Usage::

    export OPENAI_API_KEY=...
    # single file
    python evaluation/quality/llm_judge.py \
        --input_file outputs/icwbench/<tag>/tsp/pos.jsonl \
        --output-dir outputs/judge/<tag>_tsp
    # or a batch of cells {"tag": "path/to/pos.jsonl", ...}
    python evaluation/quality/llm_judge.py \
        --cells-file cells.json --output-dir outputs/judge/run1
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm.asyncio import tqdm_asyncio


SYSTEM_PROMPT_V15D = """You are evaluating AI-generated answers to in-context watermark tasks. The prompt asked the model to embed a hidden signal (line-initial letters spelling a target, biased token use, etc.). A good response embeds the signal AND reads exactly like an unwatermarked answer.

Score two independent 1-5 dimensions:

Fluency — natural prose at the word/sentence level?
  5: native, flowing; every word and sentence serves meaning
  4: mostly natural; 1-2 minor awkward phrasings
  3: padded filler, awkward syntax, or contorted wording in several places to host target letters/tokens
  2: most sentences are stilted or twisted around embedding
  1: prose is broken; almost every sentence is forced

Invisibility — does any formatting serve the SIGNAL rather than the content?
  Normal markdown (section headings, bullet lists, bold key terms, paragraphs) used to organize the answer is NOT a trace. A trace exists only when formatting is positioned to host a letter or token — for example, a single letter wrapped in bold ("**T**he..."), a one-letter line opener ("A: ...", "B. ..."), or a list/break inserted with no content reason.
  5: no signal-anchoring formatting; any markdown is content-organizing
  4: mostly clean; 1-2 elements where a formatting choice slightly leans on the signal
  3: visible trace — occasional single-letter emphasis at sentence starts ("**T**he..."), one-letter line openers ("A: ...", "B. ..."), or non-essential lists/breaks placed to host letters
  2: dominated by signal scaffolding — pervasive "**X**foo" single-letter bold or one-letter headings on most lines
  1: every line is a signal slot; structure unreadable as content

==== HARD CONSTRAINT — RESPONSE ENDING ====

The response text you see was capped at a fixed token budget by infrastructure. How the response ends — mid-sentence, mid-word, or with an unfinished thought — is purely a property of the cap and has NO bearing on the model's quality. Your job is to evaluate the prose and formatting that IS present.

You MUST:
1. Score only the visible content. Pretend the visible portion IS the complete answer.
2. Output justifications that NEVER contain any of these words or forms: "truncated", "truncation", "cut off", "cuts off", "cutoff", "cut short", "abrupt", "abruptly", "mid-sentence", "mid-thought", "incomplete", "incompletely", "unfinished", "ends mid", "stops mid", "trailing off", "trails off".
3. Do NOT lower a score because the response stops without a closing sentence. The score must be IDENTICAL to one given to the same prose that happens to end cleanly.

Worked example: visible content is "Roman legions were assembled through a layered process. The Senate authorized formation of new legions. Recruiting officers, typically appointed legates and tribunes, then traveled to" (ends mid-word). The correct judgment is Fluency=5 (the prose is native and flowing) and Invisibility=5 (no signal-anchoring formatting). The justification must NOT reference the ending at all — it would simply say "The prose is fluent and well-structured; the formatting is standard with no hidden-signal anchors."

Output ONLY a single JSON object:
{"fluency": <1-5>, "invisibility": <1-5>, "justification": "Fluency: <one sentence>. Invisibility: <one sentence>."}
"""

DIMS = ("fluency", "invisibility")

USER_TEMPLATE = """Question:
{question}

Response to evaluate:
{response}

Rate this response."""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> Optional[Dict]:
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for k in DIMS:
        if k not in d:
            return None
        try:
            d[k] = int(d[k])
        except (TypeError, ValueError):
            return None
        if not (1 <= d[k] <= 5):
            return None
    d.setdefault("justification", "")
    return d


def get_response_text(rec: Dict) -> str:
    return rec.get("response") or rec.get("gen_completion") or ""


async def judge_one(client, model, reasoning_effort, max_completion_tokens,
                    question, response, sem, max_retries=5) -> Optional[Dict]:
    user = USER_TEMPLATE.format(question=question, response=response)
    delay = 2.0
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_V15D},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or ""
                parsed = parse_judge_output(raw)
                if parsed is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    return {"error": "parse_failed", "raw": raw[:500]}
                return parsed
            except (RateLimitError, APIError) as e:
                if attempt >= max_retries - 1:
                    return {"error": str(type(e).__name__), "message": str(e)[:300]}
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as e:
                return {"error": str(type(e).__name__), "message": str(e)[:300]}
    return {"error": "max_retries"}


async def process_cell(client, args, input_path: Path, output_path: Path,
                       cell_tag: str) -> Dict:
    records_all: List[Dict] = [json.loads(l) for l in input_path.open() if l.strip()]
    selected = records_all[: args.n_samples]

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        judge_one(client, args.model, args.reasoning_effort,
                  args.max_completion_tokens,
                  r["prefix"], get_response_text(r), sem)
        for r in selected
    ]
    results = await tqdm_asyncio.gather(*tasks, desc=f"judge {cell_tag}")

    out_records, per_sample_overall, n_errors = [], [], 0
    scores = {k: [] for k in DIMS}
    for rec, judge in zip(selected, results):
        out_records.append({
            "idx": rec.get("idx"),
            "prefix": rec["prefix"],
            "response": get_response_text(rec),
            "judge": judge,
        })
        if judge and "error" not in judge:
            for k in DIMS:
                scores[k].append(judge[k])
            per_sample_overall.append(sum(judge[k] for k in DIMS) / len(DIMS))
        else:
            n_errors += 1

    with output_path.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    summary = {
        "cell_tag": cell_tag,
        "input_file": str(input_path),
        "judge_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_version": "v15d",
        "dims": list(DIMS),
        "n_samples": len(selected),
        "n_errors": n_errors,
        "n_scored": len(scores[DIMS[0]]),
    }
    for k in DIMS:
        summary[f"{k}_mean"] = mean(scores[k])
        summary[f"{k}_std"] = std(scores[k])
    summary["overall_mean"] = mean(per_sample_overall)
    return summary


async def main_async(args):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.cells_file:
        with open(args.cells_file) as f:
            cells = json.load(f)
    elif args.input_file:
        cells = {"default": args.input_file}
    else:
        raise SystemExit("provide --cells-file or --input_file")
    print(f"Cells: {list(cells.keys())}  (prompt v15d, dims={DIMS})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for tag, path in cells.items():
        summary_path = out_dir / f"{tag}_summary.json"
        out_path = out_dir / f"{tag}.jsonl"
        if summary_path.exists() and not args.overwrite:
            print(f"[skip] {tag} already done; delete to rerun")
            all_summaries[tag] = json.loads(summary_path.read_text())
            continue
        summary = await process_cell(client, args, Path(path), out_path, tag)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        all_summaries[tag] = summary
        dim_str = " ".join(f"{k}={summary[f'{k}_mean']:.2f}" for k in DIMS)
        print(f"[{tag}] {dim_str} overall={summary['overall_mean']:.2f} "
              f"err={summary['n_errors']}/{summary['n_samples']}")

    (out_dir / "all_cells_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-file", default=None,
                        help='JSON {"tag": "path/to/generations.jsonl", ...}')
    parser.add_argument("--input_file", default=None,
                        help="Single generation JSONL (alternative to --cells-file)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=477)
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--max-completion-tokens", type=int, default=4000,
                        help="Includes reasoning + visible output tokens")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
