"""
WatermarkKDDataset — dataset for watermark KD training.

Reads pre-formatted chat-template strings from parquet and tokenizes:
  - Actor inputs  : config.prompt_column           (default "prompt")
  - Ref inputs    : config.ref_prompt_column       (default "prompt_ref")

Both prompts are already apply_chat_template-formatted strings, so they are
tokenized directly without a second chat-template application.

``prompt_ref`` is per-sample and chosen by the parquet builder based on task:
  - green pos : clean prompt (no ICW)      — biased teacher = clean_ref + bias
  - neg       : clean prompt (no ICW)      — anchor to base distribution
  - initials  : ICW prompt (identical to ``prompt``) — biased teacher = ICW_ref + bias

Per-sample watermark seed/fraction are read from the "seed"/"fraction" columns.

Mixed pos+neg training: rows with ``z_score == NEG_SENTINEL`` (-99999) are
treated as negative samples (clean prompt, clean response). For these rows the
loss function routes training to clean-ref KL terms instead of biased-ref KL,
giving actor a safety anchor on non-watermarked inputs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


NEG_SENTINEL = -99999.0

# Per-sample task types for mixed-task training. Task_id = index in TASK_NAMES.
TASK_NAMES = ("green", "initials", "neg", "acrostics")
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}
ID_TO_TASK = {i: name for name, i in TASK_TO_ID.items()}


class WatermarkKDDataset(Dataset):
    """
    Dataset yielding both actor and ref tokenized sequences per sample.

    Each __getitem__ returns a dict with:
        input_ids           (L_actor,)  long  — actor: incontext wm prompt + response
        attention_mask      (L_actor,)  long
        position_ids        (L_actor,)  long
        loss_mask           (L_actor,)  float — 1 at response positions (SFTDataset convention)
        input_ids_ref       (L_ref,)    long  — ref: clean prompt + response
        attention_mask_ref  (L_ref,)    long
        position_ids_ref    (L_ref,)    long
        loss_mask_ref       (L_ref,)    float
        wm_seed             ()          long
        wm_fraction         ()          float
        is_negative         ()          bool  — True if this is a negative (clean) sample

        # Acrostic-only fields (zero/all-(-1) for non-acrostic samples):
        acrostic_bias_letter_idx_actor (L_actor,)  long — letter idx (-1=no bias,
                                                          0..25 = a..z) at each
                                                          actor position. After
                                                          worker's [:, 1:] shift,
                                                          slot at actor position
                                                          (P+n) → predicting
                                                          response[n] aligns with
                                                          flat position P-1+n.
        acrostic_bias_letter_idx_ref  (L_ref,)     long — same, for ref input
                                                          length (which differs
                                                          from L_actor since
                                                          prompt_ref is clean).
    """

    def __init__(self, parquet_files, tokenizer, config, max_samples: int = -1):
        import pandas as pd
        from verl.utils.fs import copy_to_local

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.max_length = int(config.get("max_length", 8192))
        self.truncation = config.get("truncation", "right")
        prompt_column = config.get("prompt_column", "prompt")
        ref_prompt_column = config.get("ref_prompt_column", "prompt_ref")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        dfs = []
        for pf in parquet_files:
            local = copy_to_local(pf, verbose=True)
            dfs.append(pd.read_parquet(local))
        df = pd.concat(dfs, ignore_index=True)

        for col in (prompt_column, ref_prompt_column, "response", "seed", "fraction"):
            if col not in df.columns:
                raise KeyError(
                    f"WatermarkKDDataset: required column '{col}' missing from parquet. "
                    f"Available columns: {list(df.columns)}"
                )

        total = len(df)
        if max_samples > 0 and max_samples < total:
            rng = np.random.default_rng(42)
            idx = rng.choice(total, size=max_samples, replace=False)
            df = df.iloc[idx.tolist()]
            print(f"WatermarkKDDataset: selected {max_samples} / {total} samples")

        self.prompts = df[prompt_column].tolist()
        self.prompts_ref = df[ref_prompt_column].tolist()
        self.responses = df["response"].tolist()
        self.wm_seeds = df["seed"].tolist()
        self.wm_fractions = df["fraction"].tolist()

        # Acrostic per-sample artifacts. ``acrostic_bias_letter_idx_response``
        # is the per-RESPONSE-token letter idx list pre-computed at parquet
        # build time (None / NaN for non-acrostic rows). At __getitem__ time we
        # lift it to (L_actor,) by placing entries at offsets [P, P+R-1].
        if "acrostic_bias_letter_idx_response" in df.columns:
            self.acrostic_bias_idx_response = df["acrostic_bias_letter_idx_response"].tolist()
        else:
            self.acrostic_bias_idx_response = [None] * len(df)
        if "acrostic_target" in df.columns:
            self.acrostic_targets = df["acrostic_target"].tolist()
        else:
            self.acrostic_targets = [None] * len(df)

        # Initials per-sample artifact. ``initials_active_idx_response`` is the
        # per-RESPONSE-token active-mask list (computed by
        # ``gptwm_initials.compute_initials_active_idx_response`` at parquet
        # build time): -1 = not active, 0 = active. Active means the next
        # response token is a leading-space-first-letter token, so the bias
        # should fire at this position. None / NaN for non-initials rows.
        # Lifted to (L_actor,) / (L_ref,) by the same helper as acrostic.
        if "initials_active_idx_response" in df.columns:
            self.initials_active_idx_response = df["initials_active_idx_response"].tolist()
        else:
            self.initials_active_idx_response = [None] * len(df)

        if "z_score" in df.columns:
            z_scores = df["z_score"].to_numpy()
            self.is_negatives = (z_scores == NEG_SENTINEL).tolist()
        else:
            self.is_negatives = [False] * len(df)

        # Per-sample task: read from "task" column if present; otherwise derive
        # from is_negative + mode (backward compat: all pos→"green", all neg→"neg").
        if "task" in df.columns:
            tasks = df["task"].astype(str).tolist()
        else:
            fallback_pos = config.get("watermark_mode", config.get("mode", "green"))
            tasks = [fallback_pos if not neg else "neg" for neg in self.is_negatives]
        unknown = {t for t in tasks if t not in TASK_TO_ID}
        if unknown:
            raise ValueError(
                f"Unknown task values in parquet: {unknown}. "
                f"Allowed: {list(TASK_TO_ID.keys())}"
            )
        self.task_names = tasks
        self.task_ids = [TASK_TO_ID[t] for t in tasks]
        n_neg = int(sum(self.is_negatives))
        print(
            f"WatermarkKDDataset: {len(df)} samples loaded "
            f"(pos={len(df) - n_neg}, neg={n_neg}, "
            f"prompt_column={prompt_column!r}, ref_prompt_column={ref_prompt_column!r})"
        )

    def __len__(self):
        return len(self.prompts)

    def _tokenize_sequence(self, prompt_str: str, response_str: str):
        """
        Tokenize a pre-formatted prompt string + response string.

        loss_mask follows SFTDataset convention:
          - 0 for positions 0 .. prompt_length-2
          - 1 for positions prompt_length-1 .. prompt_length+response_length-2
          - 0 for the last response token and all padding

        After the worker's [:, 1:] shift this means response-token predictions
        (including the first one, where the prompt's last position predicts r0)
        are counted in the loss.

        Returns:
            input_ids       (L,) long
            attention_mask  (L,) long
            position_ids    (L,) long
            loss_mask       (L,) float
        """
        tok = self.tokenizer

        prompt_ids = tok(
            prompt_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        response_ids = tok(
            response_str + tok.eos_token, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat([prompt_ids, response_ids])
        seq_len = input_ids.shape[0]

        # Truncate
        if seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
            else:
                raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}")
            seq_len = self.max_length

        attention_mask = torch.ones(seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long)

        # Build loss_mask (same convention as SFTDataset)
        loss_mask = attention_mask.clone().float()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, seq_len) - 1] = 0.0
        loss_mask[min(prompt_length + response_length, seq_len) - 1] = 0.0

        return input_ids, attention_mask, position_ids, loss_mask, prompt_length, response_length

    def _build_acrostic_bias_idx_tensor(
        self,
        per_step_bias_idx,
        prompt_length: int,
        response_length: int,
        seq_len: int,
        truncation_side: str,
    ) -> torch.Tensor:
        """Lift the per-response-step letter-idx list to a (seq_len,) tensor at
        offsets [P, P+R-1]. Returns all-(-1) tensor for non-acrostic samples
        (per_step_bias_idx None / empty).

        ``truncation_side`` tells us how the input_ids tensor was truncated
        (when prompt_length + response_length > max_length); we mirror that
        truncation so positions still match.
        """
        arr = torch.full((seq_len,), -1, dtype=torch.long)
        if per_step_bias_idx is None:
            return arr
        # Tolerate pyarrow-deserialized list-likes (numpy / pandas).
        try:
            n_steps = len(per_step_bias_idx)
        except TypeError:
            return arr
        if n_steps == 0:
            return arr

        # Untruncated layout: positions [0..P-1] = prompt, [P..P+R-1] = response.
        # If truncation_side == "left" and (P+R) > max_length, the head was
        # dropped, so the response shifts left by (P+R - max_length).
        # If "right", the tail was dropped — response may be partially clipped.
        full_len = prompt_length + response_length
        step_offset = 0
        if full_len <= seq_len:
            # No truncation
            response_start = prompt_length
            n_kept = response_length
        elif truncation_side == "left":
            drop = full_len - seq_len
            response_start = max(0, prompt_length - drop)
            # If drop > prompt_length, the head ate (drop - P) response steps.
            step_offset = max(0, drop - prompt_length)
            n_kept = response_length - step_offset
        else:  # right
            response_start = prompt_length
            n_kept = max(0, seq_len - prompt_length)
            n_kept = min(n_kept, response_length)

        # Place per_step_bias_idx[step_offset : step_offset + n_kept] into
        # arr[response_start : response_start + n_kept].
        for i in range(n_kept):
            src = step_offset + i
            dst = response_start + i
            if 0 <= dst < seq_len and 0 <= src < n_steps:
                v = int(per_step_bias_idx[src])
                arr[dst] = v
        return arr

    def __getitem__(self, item):
        prompt = self.prompts[item]
        prompt_ref = self.prompts_ref[item]
        response = self.responses[item]

        input_ids, attention_mask, position_ids, loss_mask, P_actor, R_actor = self._tokenize_sequence(
            prompt, response
        )
        input_ids_ref, attention_mask_ref, position_ids_ref, loss_mask_ref, P_ref, R_ref = self._tokenize_sequence(
            prompt_ref, response
        )

        acr_per_step = self.acrostic_bias_idx_response[item]
        acr_bias_idx_actor = self._build_acrostic_bias_idx_tensor(
            per_step_bias_idx=acr_per_step,
            prompt_length=P_actor,
            response_length=R_actor,
            seq_len=int(input_ids.shape[0]),
            truncation_side=self.truncation,
        )
        acr_bias_idx_ref = self._build_acrostic_bias_idx_tensor(
            per_step_bias_idx=acr_per_step,
            prompt_length=P_ref,
            response_length=R_ref,
            seq_len=int(input_ids_ref.shape[0]),
            truncation_side=self.truncation,
        )

        # Initials per-position active mask — same lifting helper (the helper
        # only places per-RESPONSE-step values into the response window of a
        # full-length array, which is task-agnostic). Non-initials samples
        # have per_step=None → returns all-(-1) tensor (inactive everywhere).
        init_per_step = self.initials_active_idx_response[item]
        init_active_idx_actor = self._build_acrostic_bias_idx_tensor(
            per_step_bias_idx=init_per_step,
            prompt_length=P_actor,
            response_length=R_actor,
            seq_len=int(input_ids.shape[0]),
            truncation_side=self.truncation,
        )
        init_active_idx_ref = self._build_acrostic_bias_idx_tensor(
            per_step_bias_idx=init_per_step,
            prompt_length=P_ref,
            response_length=R_ref,
            seq_len=int(input_ids_ref.shape[0]),
            truncation_side=self.truncation,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "input_ids_ref": input_ids_ref,
            "attention_mask_ref": attention_mask_ref,
            "position_ids_ref": position_ids_ref,
            "loss_mask_ref": loss_mask_ref,
            "wm_seed": torch.tensor(int(self.wm_seeds[item]), dtype=torch.long),
            "wm_fraction": torch.tensor(float(self.wm_fractions[item]), dtype=torch.float32),
            "is_negative": torch.tensor(bool(self.is_negatives[item]), dtype=torch.bool),
            "task_id": torch.tensor(int(self.task_ids[item]), dtype=torch.long),
            "acrostic_bias_letter_idx_actor": acr_bias_idx_actor,
            "acrostic_bias_letter_idx_ref": acr_bias_idx_ref,
            "initials_active_idx_actor": init_active_idx_actor,
            "initials_active_idx_ref": init_active_idx_ref,
        }
