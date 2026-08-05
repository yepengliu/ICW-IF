"""ICWBench SA (Sentence Acrostic) generation with per-query secret strings.

Reads the evaluation manifest (one row per query, each with its own length-18
secret over the 20-letter pool) and generates:

  pos arm — clean_v3_noex acrostic instruction (system rules + "SECRET STRING:
            <S>" in the user turn) + query
  neg arm — query only (null hypothesis H0)

The ``secret`` is recorded on every output row for both arms so detection
scores each row against its own secret.

Usage::

    python evaluation/icwbench/generate_sa.py \
        --model_name Qwen/Qwen3-14B --tag qwen3-14b --arm both
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.acrostics_icw import build_acrostic_prompt  # noqa: E402
from watermark.dataset import apply_chat_template_messages  # noqa: E402

os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: str, num_test=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:num_test] if num_test else rows


def main(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_manifest(args.manifest, args.num_test)
    out_dir = Path(args.output_dir) / args.tag / "sa"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    llm_kwargs = dict(
        model=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )
    if args.yarn:
        llm_kwargs["max_model_len"] = args.max_model_len
        llm_kwargs.setdefault("hf_overrides", {})
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k is not None else -1,
        max_tokens=args.max_tokens,
        seed=args.gen_seed,
    )

    arms = ["pos", "neg"] if args.arm == "both" else [args.arm]
    for arm in arms:
        out_path = out_dir / f"{arm}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite)")
            continue

        prompts = []
        for row in rows:
            if arm == "pos":
                sys_text, user_text = build_acrostic_prompt(
                    question=row["prefix"], target=row["secret"],
                    variant=args.prompt_variant,
                )
                messages = [{"role": "system", "content": sys_text},
                            {"role": "user", "content": user_text}]
            else:
                messages = [{"role": "user", "content": row["prefix"]}]
            prompts.append(apply_chat_template_messages(tokenizer, messages))

        print(f"[gen] generating {len(prompts)} {arm.upper()}")
        outs = llm.generate(prompts, sp)
        with out_path.open("w") as f:
            for row, prompt, o in zip(rows, prompts, tqdm(outs, desc=f"sa/{arm}")):
                f.write(json.dumps({
                    "idx": row["idx"],
                    "prefix": row["prefix"],
                    "secret": row["secret"],           # per-query secret, both arms
                    "secret_length": len(row["secret"]),
                    "input_prompt": prompt,
                    "prompt_variant": args.prompt_variant if arm == "pos" else "user_only_neg",
                    "model_name": args.model_name,
                    "arm": arm,
                    "n_output_tokens": len(o.outputs[0].token_ids),
                    "finish_reason": o.outputs[0].finish_reason,
                    "response": o.outputs[0].text,
                }, ensure_ascii=False) + "\n")
        print(f"[gen] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--manifest", type=str,
                    default=str(REPO_ROOT / "data/eval/test477_sa.jsonl"))
    ap.add_argument("--output_dir", type=str,
                    default=str(REPO_ROOT / "outputs/icwbench"))
    ap.add_argument("--num_test", type=int, default=None)
    ap.add_argument("--arm", choices=["pos", "neg", "both"], default="both")
    ap.add_argument("--prompt_variant", type=str, default="clean_v3_noex")
    # Sampling (canonical evaluation settings; SA uses a 700-token cap)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--gen_seed", type=int, default=42)
    # Infra
    ap.add_argument("--tensor_parallel_size", "--tp", dest="tensor_parallel_size",
                    type=int, default=8)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--yarn", action="store_true",
                    help="Apply YaRN rope scaling (needed for ckpts trained with it)")
    ap.add_argument("--max_model_len", type=int, default=131072)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)
