"""Convert a deduplicated TSP pos/neg JSONL into the training parquet.

Per row:
  * prompt                 : ICW instruction (green-token list rendered from the
                             row's (seed, fraction)) + user query, chat-formatted.
                             For negatives (fraction 0.0) there is no green list,
                             so the prompt is the bare user query.
  * prompt_no_incontext_wm : clean prompt (user query only) — becomes
                             ``prompt_ref`` downstream.
  * response, prefix, seed, z_score, fraction, dataset_type
"""
import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from watermark.gptwm_incontext import InContextWatermarkGenerator
from watermark.prompt import get_incontext_system_prompt


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a JSONL file with synthetic data into a Parquet file with "
            'columns: "prompt", "prompt_no_incontext_wm", "response", "prefix", '
            '"seed", "z_score", "fraction", "dataset_type".'
        )
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", default=None,
                        help="Defaults to input path with .parquet extension.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--dataset_type", default="lfqa")
    return parser


def main():
    args = build_arg_parser().parse_args()

    input_path = args.input_jsonl
    if args.output_parquet is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".parquet"
    else:
        output_path = args.output_parquet

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)

    @lru_cache(maxsize=None)
    def get_generator(seed: int, fraction: float):
        generator = InContextWatermarkGenerator(
            fraction=fraction,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model_config.vocab_size,
            watermark_key=seed,
            only_English=True,
            tokenizer=tokenizer,
        )
        return generator

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    for line in tqdm(lines, total=len(lines)):
        obj = json.loads(line)

        prefix = obj.get("prefix")
        response = obj.get("gen_completion")
        seed = obj.get("seed")
        fraction = obj.get("fraction")
        z_score = obj.get("z_score")

        if not prefix or not response or seed is None:
            continue

        if fraction is None or fraction == 0.0:
            fraction = 0.0
            green_token_string = ""
        else:
            green_token_string = get_generator(
                int(seed), float(fraction)).get_green_token_string()

        # green_token_string == "" falls back to the base (empty) system prompt,
        # so negatives get the bare user-query prompt.
        system_prompt = get_incontext_system_prompt(args.dataset_type, green_token_string)
        if system_prompt:
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prefix},
            ]
        else:
            conversation = [{"role": "user", "content": prefix}]
        input_prompt = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        no_watermark_input_prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prefix}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        rows.append({
            "prompt": input_prompt,
            "prompt_no_incontext_wm": no_watermark_input_prompt,
            "response": response,
            "prefix": prefix,
            "seed": seed,
            "z_score": z_score,
            "fraction": fraction,
            "dataset_type": args.dataset_type,
        })

    print(f"Loaded {len(rows)} samples from {input_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Saved {output_path} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
