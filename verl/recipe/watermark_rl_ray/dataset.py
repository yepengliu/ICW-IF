"""WatermarkRLPromptDataset — prompt-only dataset for Stage 2 RL training.

Parquet schema (3-task; see build_train_parquet_3task.py):
    prompt, prompt_ref, prefix, seed, fraction, task, dataset_type,
    acrostic_target  (str | None — only set when task=='acrostics')

Each __getitem__ returns a dict:
    input_ids        (max_prompt_length,) long  — left-padded actor prompt ids
    attention_mask   (max_prompt_length,) long
    position_ids     (max_prompt_length,) long
    raw_prompt_ids   list[int]                  — actor prompt ids for vLLM agent loop
    raw_prompt       str                        — for logging
    wm_seed          int                        — carries to reward fn
    wm_fraction      float                      — carries to reward fn
    task             str                        — {"green","initials","acrostics"}
    acrostic_target  str                        — per-sample target ("" for non-acrostics)

Collator packs torch.Tensor keys into DataProto.batch and str/list/scalar keys
into DataProto.non_tensor_batch (object numpy arrays) via DataProto.from_single_dict.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WatermarkRLPromptDataset(Dataset):
    TENSOR_KEYS = ("input_ids", "attention_mask", "position_ids")
    # Everything else is non-tensor (str / list / scalar).
    # raw_prompt_ref_ids carries clean-prompt token ids used by ref model in
    # KL loss (ref sees clean prompt + actor's rollout response, KD-style).
    NON_TENSOR_KEYS = (
        "raw_prompt_ids", "raw_prompt", "wm_seed", "wm_fraction", "task",
        "acrostic_target", "raw_prompt_ref_ids",
    )

    def __init__(self, parquet_files, tokenizer, config, max_samples: int = -1):
        import pandas as pd
        from verl.utils.fs import copy_to_local

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.max_prompt_length = int(config.get("max_prompt_length", 65536))
        prompt_column = config.get("prompt_column", "prompt")
        ref_prompt_column = config.get("ref_prompt_column", "prompt_ref")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        dfs = []
        for pf in parquet_files:
            local = copy_to_local(pf, verbose=True)
            dfs.append(pd.read_parquet(local))
        df = pd.concat(dfs, ignore_index=True)

        total = len(df)
        if max_samples > 0 and max_samples < total:
            rng = np.random.default_rng(42)
            idx = rng.choice(total, size=max_samples, replace=False)
            df = df.iloc[idx.tolist()]
            print(f"WatermarkRLPromptDataset: selected {max_samples} / {total} samples")
        print(f"WatermarkRLPromptDataset: {len(df)} rows loaded, "
              f"task breakdown: {df['task'].value_counts().to_dict()}")

        self.prompts     = df[prompt_column].tolist()
        self.wm_seeds    = df["seed"].tolist()
        self.wm_fracs    = df["fraction"].tolist()
        self.tasks       = df["task"].tolist()
        # acrostic_target column may not exist for legacy 2-task parquets
        if "acrostic_target" in df.columns:
            self.acrostic_targets = df["acrostic_target"].fillna("").astype(str).tolist()
        else:
            self.acrostic_targets = [""] * len(df)
        # Optional clean-prompt column for KL-against-clean-ref. Missing → empty
        # list per sample (trainer can detect and skip ref forward).
        if ref_prompt_column in df.columns:
            self.prompts_ref = df[ref_prompt_column].tolist()
        else:
            self.prompts_ref = [None] * len(df)
            print(f"WatermarkRLPromptDataset: WARNING ref_prompt_column "
                  f"{ref_prompt_column!r} missing — raw_prompt_ref_ids will be empty")

    def __len__(self):
        return len(self.prompts)

    def _tokenize(self, s: str) -> torch.Tensor:
        ids = self.tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.shape[0] > self.max_prompt_length:
            ids = ids[-self.max_prompt_length:]
        return ids

    def __getitem__(self, idx):
        prompt_str = self.prompts[idx]
        raw_ids = self._tokenize(prompt_str)
        n = raw_ids.shape[0]
        pad_len = self.max_prompt_length - n

        if pad_len > 0:
            pad = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([pad, raw_ids])
        else:
            input_ids = raw_ids

        attn = torch.zeros(self.max_prompt_length, dtype=torch.long)
        attn[pad_len:] = 1
        pos = torch.zeros(self.max_prompt_length, dtype=torch.long)
        pos[pad_len:] = torch.arange(n, dtype=torch.long)

        # Clean-prompt ids for ref model KL (KD-style). List[int]; empty when
        # parquet has no prompt_ref column.
        prompt_ref_str = self.prompts_ref[idx]
        if prompt_ref_str is None:
            raw_prompt_ref_ids = []
        else:
            raw_prompt_ref_ids = self._tokenize(prompt_ref_str).tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "position_ids": pos,
            "raw_prompt_ids": raw_ids.tolist(),
            "raw_prompt": prompt_str,
            "wm_seed": int(self.wm_seeds[idx]),
            "wm_fraction": float(self.wm_fracs[idx]),
            "task": str(self.tasks[idx]),
            "acrostic_target": str(self.acrostic_targets[idx]) if self.acrostic_targets[idx] else "",
            "raw_prompt_ref_ids": raw_prompt_ref_ids,
        }


class WatermarkRLPromptCollator:
    """Collate a list of items into a dict DataProto.from_single_dict can parse."""

    def __call__(self, items: list[dict]) -> dict:
        out = {}
        for k in WatermarkRLPromptDataset.TENSOR_KEYS:
            out[k] = torch.stack([it[k] for it in items])
        for k in WatermarkRLPromptDataset.NON_TENSOR_KEYS:
            out[k] = np.array([it[k] for it in items], dtype=object)
        return out
