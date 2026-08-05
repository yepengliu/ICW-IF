#!/usr/bin/env python3
"""
Response-PPL via vLLM.

Input: a JSONL file where each line has at least ``query`` and ``response``.
Output: per-sample PPL (length-normalized) and a single corpus-level number
        that is comparable across responses of different lengths.

Scoring:
  - prefix  = chat template applied to the query, with add_generation_prompt=True
              (the model "sees" exactly what it would at generation time)
  - response is tokenized separately (add_special_tokens=False) and concatenated
  - per-sample PPL = exp(sum(-logp_t) / N_response_tokens)
        -> already per-token, so it doesn't grow with length
  - corpus PPL    = mean over samples of per-sample PPL
        -> every sample contributes equally regardless of response length,
           so longer outputs cannot dominate the average

Output files (in --output-dir):
  per_sample.jsonl     -- {query, response, ppl, n_response_tokens} per line
  summary.json         -- avg_ppl_sample_mean, plus token-weighted + median for ref
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompts(records, tokenizer, query_field, response_field, max_length):
    token_ids_list, comp_starts, comp_lens, response_ids_list = [], [], [], []
    for rec in records:
        query = rec[query_field]
        response = rec[response_field]
        prefix_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
        resp_ids = tokenizer.encode(response, add_special_tokens=False)
        full = prefix_ids + resp_ids
        if len(full) > max_length:
            full = full[:max_length]
            cids_eff = max(0, max_length - len(prefix_ids))
        else:
            cids_eff = len(resp_ids)
        token_ids_list.append(full)
        comp_starts.append(len(prefix_ids))
        comp_lens.append(cids_eff)
        response_ids_list.append(resp_ids)
    return token_ids_list, comp_starts, comp_lens, response_ids_list


def _logprob_value(lp_obj):
    if lp_obj is None:
        return None
    return lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True,
                        help="JSONL with query+response per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="0 = use all rows")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="hard cap on prefix+response token count")
    parser.add_argument("--model-name", default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="must divide num_attention_heads (Qwen3-235B-A22B has 64 heads -> 1/2/4/8)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--swap-space", type=float, default=4.0)
    args = parser.parse_args()

    records: List[Dict] = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if args.n_samples > 0:
        records = records[: args.n_samples]
    print(f"Loaded {len(records)} records from {args.input_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model_name} via vLLM (TP={args.tensor_parallel_size}, "
          f"max_model_len={args.max_model_len}) ...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        swap_space=args.swap_space,
    )

    token_ids_list, comp_starts, comp_lens, response_ids_list = build_prompts(
        records, tokenizer, args.query_field, args.response_field, args.max_length
    )

    prompts = [{"prompt_token_ids": ids} for ids in token_ids_list]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(records)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample.jsonl"
    summary_path = out_dir / "summary.json"

    per_sample_ppls: List[float] = []
    total_nll = 0.0
    total_tokens = 0
    out_records = []

    for rec, out, ids, cstart, clen, resp_ids in tqdm(
        list(zip(records, outputs, token_ids_list, comp_starts, comp_lens, response_ids_list)),
        desc="scoring",
    ):
        if clen == 0 or cstart == 0:
            ppl = float("inf")
            n_tok = 0
        else:
            plp = out.prompt_logprobs
            nll = 0.0
            n_tok = 0
            for pos in range(cstart, cstart + clen):
                if pos >= len(plp) or plp[pos] is None:
                    continue
                tok_id = ids[pos]
                lp = _logprob_value(plp[pos].get(tok_id))
                if lp is None:
                    continue
                nll += -lp
                n_tok += 1
            if n_tok == 0:
                ppl = float("inf")
            else:
                ppl = math.exp(nll / n_tok)
                total_nll += nll
                total_tokens += n_tok

        per_sample_ppls.append(ppl)
        out_records.append({
            args.query_field: rec[args.query_field],
            args.response_field: rec[args.response_field],
            "ppl": ppl,
            "n_response_tokens": n_tok,
        })

    with per_sample_path.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    finite = [p for p in per_sample_ppls if math.isfinite(p)]
    avg_ppl_sample_mean = sum(finite) / len(finite) if finite else float("inf")
    avg_ppl_token_weighted = (
        math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    )
    summary = {
        "input_file": args.input_file,
        "model_name": args.model_name,
        "n_samples": len(out_records),
        "n_finite": len(finite),
        "avg_ppl": avg_ppl_sample_mean,
        "avg_ppl_token_weighted": avg_ppl_token_weighted,
        "ppl_median": sorted(finite)[len(finite) // 2] if finite else float("inf"),
        "ppl_min": min(finite) if finite else float("inf"),
        "ppl_max": max(finite) if finite else float("inf"),
        "total_response_tokens": total_tokens,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\navg_ppl (sample-mean, length-comparable) = {avg_ppl_sample_mean:.4f}")
    print(f"avg_ppl (token-weighted, ref)            = {avg_ppl_token_weighted:.4f}")
    print(f"n_samples = {len(out_records)}, total_response_tokens = {total_tokens}")
    print(f"per-sample -> {per_sample_path}")
    print(f"summary    -> {summary_path}")


if __name__ == "__main__":
    main()
