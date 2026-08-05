"""ICWBench TSP (Token-Set Preference) generation with per-query watermark keys.

Reads the evaluation manifest (one row per query with its own green-list key)
and generates, for any local HF checkpoint:

  pos arm — ICW instruction (<green> token list for that query's key) + query
  neg arm — query only (null hypothesis H0)

Every output row records the query's key (``seed``) and ``fraction`` so that
detection scores each row under its own key, for both arms.

Usage::

    python evaluation/icwbench/generate_tsp.py \
        --model_name Qwen/Qwen3-14B --tag qwen3-14b \
        --arm both --yarn --max_model_len 131072 --tensor_parallel_size 8
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
from watermark.gptwm_incontext import InContextWatermarkGenerator  # noqa: E402
from watermark.prompt import get_incontext_system_prompt  # noqa: E402

os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: str, num_test=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:num_test] if num_test else rows


def build_llm(args):
    from vllm import LLM
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "hf_overrides": {},
        "max_model_len": args.max_model_len,
    }
    if args.yarn:
        # Long-context YaRN (required for checkpoints trained with baked YaRN config)
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = args.max_model_len
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768,
        }
    return LLM(**llm_kwargs)


def main(args):
    from vllm import SamplingParams

    rows = load_manifest(args.manifest, args.num_test)
    out_dir = Path(args.output_dir) / args.tag / "tsp"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The benchmark's verifier vocabulary is fixed (default Qwen/Qwen3-14B):
    # the green list embedded in the prompt and the detector operate in this
    # vocab regardless of the model under evaluation, so scores are comparable
    # across models with different tokenizers.
    from transformers import AutoConfig
    if args.wm_tokenizer == args.model_name:
        wm_tokenizer = tokenizer
    else:
        wm_tokenizer = AutoTokenizer.from_pretrained(args.wm_tokenizer, trust_remote_code=True)
        if wm_tokenizer.pad_token is None:
            wm_tokenizer.pad_token = wm_tokenizer.eos_token
    wm_config = AutoConfig.from_pretrained(args.wm_tokenizer, trust_remote_code=True)
    wm_max_id = max(wm_tokenizer.get_vocab().values())
    vocab_size = wm_tokenizer.vocab_size
    emb_length = max(wm_config.vocab_size, wm_max_id + 1, vocab_size + 1)

    llm = build_llm(args)
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
                wm_gen = InContextWatermarkGenerator(
                    fraction=row.get("fraction", args.fraction),
                    vocab_size=vocab_size,
                    model_emb_length=emb_length,
                    watermark_key=row["seed"],
                    only_English=args.only_English,
                    tokenizer=wm_tokenizer,
                )
                green_token_string = wm_gen.get_green_token_string(
                    shuffle=args.shuffle_green_tokens)
                system_prompt = get_incontext_system_prompt(
                    args.dataset_type, green_token_string)
            else:
                # H0: query only, no ICW instruction
                system_prompt = ""
            input_prompts.append(
                apply_chat_template(tokenizer, system_prompt, row["prefix"]))

        records = []
        for start in tqdm(range(0, len(rows), args.batch_size),
                          desc=f"tsp/{arm}"):
            batch_rows = rows[start:start + args.batch_size]
            batch_prompts = input_prompts[start:start + args.batch_size]
            outs = llm.generate(batch_prompts, sp)
            for row, prompt, out in zip(batch_rows, batch_prompts, outs):
                records.append(json.dumps({
                    "idx": row["idx"],
                    "prefix": row["prefix"],
                    "input_prompt": prompt,
                    "response": out.outputs[0].text,
                    "seed": row["seed"],                       # per-query key, both arms
                    "fraction": row.get("fraction", args.fraction),
                    "arm": arm,
                    "model_name": args.model_name,
                }, ensure_ascii=False))
        out_path.write_text("\n".join(records) + "\n")
        print(f"[gen] wrote {out_path} ({len(records)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True,
                        help="Output sub-directory name under --output_dir")
    parser.add_argument("--manifest", type=str,
                        default=str(REPO_ROOT / "data/eval/test477_tsp.jsonl"))
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
    parser.add_argument("--yarn", action="store_true",
                        help="Enable YaRN rope scaling (long ICW prompts)")
    parser.add_argument("--yarn_factor", type=float, default=4.0)
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    # Watermark
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="Fallback gamma when the manifest row has none")
    parser.add_argument("--only_English", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Restrict the green list to English tokens "
                             "(canonical; must match detection)")
    parser.add_argument("--shuffle_green_tokens", action="store_true",
                        help="Shuffle green token order per sample (training-style); "
                             "omit for evaluation to enable vLLM prefix caching")
    parser.add_argument("--dataset_type", type=str, default="lfqa")
    parser.add_argument("--wm_tokenizer", type=str, default="Qwen/Qwen3-14B",
                        help="Verifier vocabulary: tokenizer used to build the "
                             "green list (and later, detection). Fixed per "
                             "benchmark; independent of --model_name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
