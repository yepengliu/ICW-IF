"""Synthesize SA (Sentence Acrostic) cold-start data using closed-loop
stateful logit bias.

Canonical setup:
  * Input: LFQA prefix as a pure user message; NO system prompt; secret NOT in
    the prompt (the teacher is unaware of the acrostic task) —
    --prompt_variant user_only.
  * Per-sample secret: |S| sampled uniformly in [18, 20] from the ICW 20-letter
    pool (letters with word-initial frequency >= 1.5%).
  * Generation: vLLM + AcrosticsBiasAdapterLogitsProcessor (strength 8.0,
    max_fail_streak=3) — a per-request state machine that biases the
    sentence-start token toward the current target letter and advances /
    drops targets exactly like the ICW instruction describes.
  * Output: JSONL with (prefix, secret, gen_completion, ...).

Run (single shard):
  python data_gen/sa/generate_sa_syn.py \
    --prompt_file data_gen/outputs/sa/kd_pool_shards/shard_00.jsonl \
    --seed_base 100000 --output_name kd_s8_shard00.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from watermark.acrostics_icw import (
    sample_target_icw, ICW_LETTER_POOL, build_acrostic_prompt,
)
from watermark.dataset import apply_chat_template_messages
from watermark.gptwm_vllm_config import (
    set_acrostics_bias_config,
    AcrosticsBiasAdapterLogitsProcessor,
)


os.environ.setdefault("VLLM_LOG_LEVEL", "INFO")


def load_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: str):
    """Some 'json' files are actually jsonl; handle both."""
    p = Path(path)
    text = p.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return load_jsonl(path)


def main(args):
    rng = random.Random(args.seed_base)

    rows = load_json(args.prompt_file)
    print(f"[load] {len(rows)} prefixes from {args.prompt_file}")
    if 0 < args.num_samples < len(rows):
        rng.shuffle(rows)
        rows = rows[: args.num_samples]
    print(f"[sample] using {len(rows)} prefixes")

    # ---- Tokenizer / config ----
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    vocab_size = tok.vocab_size
    model_emb_length = cfg.vocab_size

    # ---- Configure adapter ----
    set_acrostics_bias_config(
        strength=args.strength,
        vocab_size=vocab_size,
        model_emb_length=model_emb_length,
        tokenizer=tok,
        letter_token_ids_path=args.letter_token_ids,
        max_fail_streak=args.max_fail_streak,
    )

    # ---- vLLM ----
    print("[vllm] loading model")
    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed_base,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        logits_processors=[AcrosticsBiasAdapterLogitsProcessor],
    )

    # ---- Build per-sample (prompt, secret, sampling_params) ----
    targets = []
    prompts_text = []
    prompts_clean_text = []  # clean (no system, no secret) reference prompts
    sampling_params = []
    secret_len_lo, secret_len_hi = args.secret_length, args.secret_length
    if args.secret_length_max is not None:
        secret_len_hi = int(args.secret_length_max)
        assert secret_len_hi >= args.secret_length, "secret_length_max must be >= secret_length"
    for i, row in enumerate(rows):
        prefix = row.get("prefix") or row.get("input_prompt") or row["question"]
        seed_i = args.seed_base + i
        # Per-sample secret length sampled uniformly in [lo, hi] using seed for repro
        len_rng = random.Random(seed_i)
        sec_len = len_rng.randint(secret_len_lo, secret_len_hi)
        secret_upper = sample_target_icw(seed=seed_i,
                                         length=sec_len,
                                         pool=ICW_LETTER_POOL,
                                         uppercase=True)

        if args.prompt_variant == "user_only":
            # Canonical: pure user message, no system, secret NOT in prompt
            messages = [{"role": "user", "content": prefix}]
        elif args.prompt_variant in ("clean_v3_noex", "clean_v3_1ex"):
            system_text, user_text = build_acrostic_prompt(
                question=prefix, target=secret_upper, variant=args.prompt_variant,
            )
            messages = []
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})
        else:
            raise ValueError(
                f"unknown --prompt_variant: {args.prompt_variant!r}; "
                f"allowed: user_only / clean_v3_noex / clean_v3_1ex"
            )
        prompt_text = apply_chat_template_messages(tok, messages)
        prompt_clean_text = apply_chat_template_messages(
            tok, [{"role": "user", "content": prefix}]
        )

        sp = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=seed_i,
            extra_args={"acrostic_secret": secret_upper.lower()},
        )
        prompts_text.append(prompt_text)
        prompts_clean_text.append(prompt_clean_text)
        targets.append((prefix, secret_upper, sec_len, seed_i))
        sampling_params.append(sp)

    # ---- Generate ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    print(f"[gen] {len(prompts_text)} prompts -> {out_path}")

    outputs = llm.generate(prompts_text, sampling_params)

    # ---- Write ----
    n_written = 0
    with open(out_path, "w") as f:
        for i, out in enumerate(outputs):
            prefix, secret_upper, sec_len, seed_i = targets[i]
            gen = out.outputs[0].text
            row = {
                "idx": i,
                "prefix": prefix,
                "secret": secret_upper,
                "secret_length": sec_len,
                "seed": seed_i,
                "gen_completion": gen,
                "n_output_tokens": len(out.outputs[0].token_ids),
                "finish_reason": out.outputs[0].finish_reason,
                "input_prompt": prompts_text[i],
                "input_prompt_clean": prompts_clean_text[i],
                "prompt_variant": args.prompt_variant,
                "strength": args.strength,
                "max_fail_streak": args.max_fail_streak,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"[done] wrote {n_written} rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--prompt_file",
                   default="data_gen/outputs/sa/kd_pool_lfqa.jsonl")
    p.add_argument("--letter_token_ids",
                   default="data/stats/letter_to_token_ids_qwen3_14b.json")
    p.add_argument("--output_dir", default="data_gen/outputs/sa")
    p.add_argument("--output_name", default="syn.jsonl")
    p.add_argument("--num_samples", type=int, default=-1,
                   help="-1 means use ALL prefixes")
    p.add_argument("--secret_length", type=int, default=18,
                   help="If --secret_length_max not set, use this fixed length. "
                        "Otherwise this is the lower bound (inclusive).")
    p.add_argument("--secret_length_max", type=int, default=20,
                   help="If set, sample secret length uniformly in "
                        "[secret_length, secret_length_max] per sample (seeded).")
    p.add_argument("--prompt_variant", default="user_only",
                   choices=["user_only", "clean_v3_noex", "clean_v3_1ex"],
                   help="user_only=pure user msg (canonical: teacher unaware); "
                        "clean_v3_noex=system rules + uppercase secret in user msg.")
    p.add_argument("--strength", type=float, default=8.0,
                   help="Logit bias on the current target letter's token bucket.")
    p.add_argument("--max_fail_streak", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=700)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed_base", type=int, default=100000)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--enforce_eager", action="store_true",
                   help="Skip CUDA graph capture (faster startup, slower per-step).")
    args = p.parse_args()
    main(args)
