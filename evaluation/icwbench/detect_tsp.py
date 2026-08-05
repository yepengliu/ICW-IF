"""ICWBench TSP detection: unique-token z-score under each row's own key.

Reads a generation JSONL (pos or neg arm), scores every row with the
green-list detector keyed by the row's ``seed`` and ``fraction`` fields, and
writes per-sample scores to ``<input>_z.jsonl`` plus a small summary JSON.

Usage::

    python evaluation/icwbench/detect_tsp.py \
        --input_file outputs/icwbench/<tag>/tsp/pos.jsonl \
        --model_name Qwen/Qwen3-14B --workers 8
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.gptwm import GPTWatermarkDetector  # noqa: E402

# Globals for worker processes (set by pool initializer)
_worker_tokenizer = None
_worker_emb_length = None


def _safe_emb_length(tokenizer, config) -> int:
    """Mask length covering every possible token id across tokenizer families."""
    max_id = max(tokenizer.get_vocab().values())
    cfg_size = getattr(config, "vocab_size", 0) or 0
    return max(cfg_size, max_id + 1, tokenizer.vocab_size + 1)


def _init_worker(model_name: str):
    global _worker_tokenizer, _worker_emb_length
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _worker_tokenizer.pad_token is None:
        _worker_tokenizer.pad_token = _worker_tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _worker_emb_length = _safe_emb_length(_worker_tokenizer, config)


def _get_text(rec: dict) -> str:
    return rec.get("response") or rec.get("gen_completion") or ""


def _score_chunk(chunk_and_args):
    chunk, a = chunk_and_args
    tokenizer, emb_length = _worker_tokenizer, _worker_emb_length
    detector_cache = {}

    def get_detector(seed: int, fraction: float) -> GPTWatermarkDetector:
        key = (seed, fraction)
        if key not in detector_cache:
            detector_cache[key] = GPTWatermarkDetector(
                fraction=fraction,
                strength=a["strength"],
                vocab_size=tokenizer.vocab_size,
                model_emb_length=emb_length,
                watermark_key=seed,
                only_English=a["only_English"],
                tokenizer=tokenizer,
            )
        return detector_cache[key]

    out = []
    for rec in chunk:
        ids = tokenizer(_get_text(rec), add_special_tokens=False)["input_ids"]
        if len(ids) < a["test_min_tokens"]:
            continue
        seed = int(rec.get("seed", a["wm_key"] or 0))
        fraction = float(rec.get("fraction", a["fraction"]))
        z = get_detector(seed, fraction).unidetect(ids)
        out.append({"idx": rec.get("idx"), "seed": seed, "fraction": fraction,
                    "n_tokens": len(ids), "z": z})
    return out


def main(args):
    records = [json.loads(l) for l in open(args.input_file) if l.strip()]
    a = {
        "strength": args.strength,
        "fraction": args.fraction,
        "wm_key": args.wm_key,
        "test_min_tokens": args.test_min_tokens,
        "only_English": args.only_English,
    }

    workers = max(1, args.workers)
    chunk_size = max(1, (len(records) + workers - 1) // workers)
    chunks = [(records[i:i + chunk_size], a) for i in range(0, len(records), chunk_size)]
    if workers == 1:
        _init_worker(args.model_name)
        results = [r for c in tqdm(chunks, desc="detect tsp") for r in _score_chunk(c)]
    else:
        with Pool(workers, initializer=_init_worker, initargs=(args.model_name,)) as pool:
            results = [r for c in tqdm(pool.imap(_score_chunk, chunks),
                                       total=len(chunks), desc="detect tsp")
                       for r in c]

    in_path = Path(args.input_file)
    out_path = Path(args.output_file) if args.output_file else in_path.with_name(
        in_path.stem + "_z.jsonl")
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    zs = [r["z"] for r in results]
    summary = {
        "input_file": str(in_path),
        "n_scored": len(zs),
        "n_skipped_short": len(records) - len(zs),
        "z_mean": sum(zs) / len(zs) if zs else float("nan"),
        "only_English": args.only_English,
        "strength": args.strength,
    }
    summary_path = in_path.with_name(in_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"per-sample -> {out_path}")
    print(f"summary    -> {summary_path}  (mean z = {summary['z_mean']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B",
                        help="Tokenizer used for detection (must match generation)")
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="Fallback gamma when a row has no fraction field")
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--wm_key", type=int, default=None,
                        help="Fallback key when a row has no seed field")
    parser.add_argument("--test_min_tokens", type=int, default=1)
    parser.add_argument("--only_English", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Restrict detection to English tokens (canonical; "
                             "disabling silently collapses z to chance)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
