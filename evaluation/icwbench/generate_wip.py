"""ICWBench WIP (Word-Initial Preference) generation with per-query letter keys.

Reads the evaluation manifest (one row per query, each with its own
letter-partition key) and generates:

  pos arm — ICW instruction (<green>/<red> letter lists for that query's key) + query
  neg arm — query only (null hypothesis H0)

The per-query key requires a per-sample system prompt (the letter partition
differs per row); the ``seed`` field is recorded on every output row for both
arms so detection scores each row under its own key.

Usage::

    python evaluation/icwbench/generate_wip.py \
        --model_name Qwen/Qwen3-14B --tag qwen3-14b --arm both
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.dataset import apply_chat_template  # noqa: E402
from watermark.gptwm_initials import partition_letters  # noqa: E402
from watermark.prompt import get_initials_incontext_prompt  # noqa: E402

os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: str, num_test=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:num_test] if num_test else rows


def main(args):
    from vllm import LLM, SamplingParams

    rows = load_manifest(args.manifest, args.num_test)
    out_dir = Path(args.output_dir) / args.tag / "wip"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        min_tokens=args.min_new_tokens,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k is not None else -1,
        top_p=args.top_p,
    )

    arms = ["pos", "neg"] if args.arm == "both" else [args.arm]
    for arm in arms:
        out_path = out_dir / f"{arm}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite)")
            continue

        input_prompts = []
        for row in rows:
            if arm == "pos":
                # Per-query letter partition; prompt lists letters alphabetically
                # (canonical inference-time convention).
                green, red = partition_letters(seed=row["seed"])
                system_prompt = get_initials_incontext_prompt(
                    args.dataset_type, sorted(green), sorted(red))
            else:
                system_prompt = ""
            input_prompts.append(
                apply_chat_template(tokenizer, system_prompt, row["prefix"]))

        records = []
        for start in tqdm(range(0, len(rows), args.batch_size),
                          desc=f"wip/{arm}"):
            batch_rows = rows[start:start + args.batch_size]
            batch_prompts = input_prompts[start:start + args.batch_size]
            outs = llm.generate(batch_prompts, sp)
            for row, prompt, out in zip(batch_rows, batch_prompts, outs):
                records.append(json.dumps({
                    "idx": row["idx"],
                    "prefix": row["prefix"],
                    "input_prompt": prompt,
                    "response": out.outputs[0].text,
                    "seed": row["seed"],   # per-query partition key, both arms
                    "arm": arm,
                    "model_name": args.model_name,
                }, ensure_ascii=False))
        out_path.write_text("\n".join(records) + "\n")
        print(f"[gen] wrote {out_path} ({len(records)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--manifest", type=str,
                        default=str(REPO_ROOT / "data/eval/test477_wip.jsonl"))
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "outputs/icwbench"))
    parser.add_argument("--num_test", type=int, default=None)
    parser.add_argument("--arm", choices=["pos", "neg", "both"], default="both")
    # Sampling (canonical evaluation settings)
    parser.add_argument("--min_new_tokens", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Infra
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dataset_type", type=str, default="lfqa_initials")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
