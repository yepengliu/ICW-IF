"""Synthesize TSP (Token-Set Preference) cold-start data with decoding-time
logit perturbation.

The teacher sees ONLY the user query (no ICW instruction). A per-sample
watermark key selects the green token set T; the logits processor adds
delta * 1[v in T] at every decoding step (delta = --strength).

Canonical (paper) settings:
    --strength 3.0 --only_English --min_new_tokens 500 --max_new_tokens 600
    one run per fraction in {0.1, 0.2, 0.3} (H1) plus one run with
    --fraction 0.0 (H0 negatives: empty green set => unbiased generation).

Per-sample watermark key: seed_list[idx % seed_num] + seed_offset, with
seed_list = [1..seed_num].
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from watermark.dataset import load_generation_dataset, load_jsonl, make_multitask_prompt_mapper
from watermark.gptwm_vllm_config import set_watermark_config, GPTWatermarkAdapterLogitsProcessor

os.environ["VLLM_LOG_LEVEL"] = "ERROR"
logging.getLogger("vllm").setLevel(logging.ERROR)


def main(args):
    output_file = (
        f"{args.output_dir}/"
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test if args.num_test else len(load_jsonl(args.prompt_file))}_vllm.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace('.jsonl', '_only_English.jsonl')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model_name)
    set_watermark_config(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )

    seed_list = list(range(1, args.seed_num + 1))

    # Load dataset: apply chat template per example using dataset_type.
    # For "lfqa" the base system prompt is empty => pure user-query prompt.
    ds = load_generation_dataset(args.prompt_file, args.num_test)
    ds = ds.map(
        make_multitask_prompt_mapper(tokenizer),
        batched=False,
        with_indices=True,
    )

    print("Loading vLLM model...")
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "hf_overrides": {},
    }

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    if args.yarn:
        print("Using YaRN for long context")
        target_max_position_embeddings = args.max_model_len or 262144
        print(f"Overriding max_position_embeddings to {target_max_position_embeddings}")
        llm_kwargs["max_model_len"] = target_max_position_embeddings
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = target_max_position_embeddings
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768,
        }

    llm = LLM(**llm_kwargs, logits_processors=[GPTWatermarkAdapterLogitsProcessor])
    print("vLLM model loaded successfully")
    print("=" * 100)

    base_sampling_kwargs = dict(
        min_tokens=args.min_new_tokens,
        max_tokens=args.max_new_tokens,
        temperature=1.0,
        top_k=args.top_k if args.top_k is not None else -1,
        top_p=args.top_p,
    )

    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
        input_prompt = batch["input_prompt"]
        indices = batch["idx"]
        batch_seeds = [seed_list[idx % len(seed_list)] + args.seed_offset for idx in indices]

        sampling_params_list = [
            SamplingParams(
                **base_sampling_kwargs,
                extra_args={"watermark_key": seed},
            )
            for seed in batch_seeds
        ]

        outputs_vllm = llm.generate(input_prompt, sampling_params_list)

        outputs = []
        for i, out in enumerate(outputs_vllm):
            outputs.append(json.dumps({
                "prefix": batch["prefix"][i],
                "input_prompt": input_prompt[i],
                "gold_completion": batch["gold_completion"][i],
                "gen_completion": out.outputs[0].text,
                "seed": batch_seeds[i],
            }, ensure_ascii=False))

        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")

    print("Results saved to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument_group("Generation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--min_new_tokens", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--yarn", action="store_true", help="Enable YaRN for long context")
    parser.add_argument("--yarn_factor", type=float, default=4.0, help="YaRN scaling factor")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model length for vLLM")

    parser.add_argument_group("Watermark")
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="Green-set fraction gamma. 0.0 => H0 negatives (no bias).")
    parser.add_argument("--strength", type=float, default=3.0,
                        help="Logit bias delta added to green tokens.")
    parser.add_argument("--seed_num", type=int, default=500,
                        help="Per-sample watermark keys cycle over [1..seed_num].")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Add this offset to every per-sample watermark key. Use with "
                             "non-overlapping segments across multiple cells.")
    parser.add_argument("--only_English", action="store_true")

    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str,
                        default="data/hf/lfqa/train_11578.json")
    parser.add_argument("--output_dir", type=str, default="data_gen/outputs/tsp")
    parser.add_argument("--num_test", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
