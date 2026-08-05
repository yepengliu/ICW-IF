"""Acrostic logit-bias controller — per-request state machine for steering
generation toward an externally-supplied secret string.

Algorithm (decided 2026-04-29; markdown-aware refresh 2026-04-30):
  1. Initial state: target_idx=0, fail_streak=0, at sentence start.
  2. At each "pre-letter" position (text-so-far is in a slot that would precede
     a fresh sentence's first letter — see ``acrostics_icw.is_md_pre_letter``),
     apply +strength logit bias to all token ids whose decoded text (after
     stripping skippable chars + markdown emphasis) starts with
     ``secret[target_idx]``.
  3. After each token is sampled, advance internal text buffer. Use the
     markdown-aware extractor's diagnosis to find which acrostic positions
     have new captured letters.
  4. On each newly captured letter (status='valid' diag entry):
       * if letter == target_letter → target_idx += 1, fail_streak = 0
       * else → fail_streak += 1; if fail_streak == max_fail_streak,
         target_idx += 1, fail_streak = 0 (skip this letter).
  5. If we stay in the SAME pre-letter region for ``max_prefix_tokens``
     consecutive bias steps without any letter being sampled (model emits
     long emphasis / marker chains), give up and skip the current target.
  6. When target_idx >= len(secret), bias becomes a no-op for the rest.

Pre-letter detection (backward-only, no lookahead) is shared with the
extractor via ``acrostics_icw.is_md_pre_letter`` so bias fires at exactly
the positions where the extractor will count letters.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from watermark.acrostics_icw import (
    extract_first_letters_md_with_diagnosis,
    is_md_pre_letter,
)


class AcrosticBiasController:
    """Per-request stateful controller. One instance per generation request."""

    def __init__(
        self,
        secret: str,
        strength: float,
        max_fail_streak: int,
        letter_token_ids: Dict[str, List[int]],
        tokenizer,
        max_prefix_tokens: int = 8,
    ):
        self.secret = secret.lower()
        self.strength = float(strength)
        self.max_fail_streak = int(max_fail_streak)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.letter_token_ids = letter_token_ids
        self.tokenizer = tokenizer

        self.text = ""
        self.target_idx = 0
        self.fail_streak = 0
        self._processed_count = 0
        self._n_resolved = 0           # how many status='valid' diag entries already consumed
        self._consec_prefix_tokens = 0  # consecutive bias steps since last letter
        self.trace: List[dict] = []

    # ----- state update -----

    def update_with_past(self, output_ids: List[int]) -> None:
        """Absorb any tokens that appeared since the last call. ``output_ids``
        is the cumulative sampled-token list passed by vLLM each step."""
        n = len(output_ids)
        if n <= self._processed_count:
            return
        new_ids = list(output_ids[self._processed_count: n])
        self._processed_count = n
        if not new_ids:
            return
        # Decode new tokens together (preserves BPE merging across boundaries).
        new_text = self.tokenizer.decode(new_ids)
        self.text += new_text

        # Re-run extractor; resolve any new status='valid' diag entries beyond
        # what we've already consumed. Status='empty' entries (e.g., empty
        # markers like `## `) are skipped without advancing the resolved counter
        # so they can transition to 'valid' once the model emits content.
        diag = extract_first_letters_md_with_diagnosis(self.text)
        valid_entries = [d for d in diag if d["status"] == "valid"]
        for d in valid_entries[self._n_resolved:]:
            self._resolve(d)
        self._n_resolved = len(valid_entries)

    def _resolve(self, diag_entry: dict) -> None:
        if self.target_idx >= len(self.secret):
            return
        target = self.secret[self.target_idx]
        observed = diag_entry["letter"]   # md extractor populates `letter` for valid entries
        before = (self.target_idx, self.fail_streak)
        if observed == target:
            self.target_idx += 1
            self.fail_streak = 0
            outcome = "hit"
        else:
            self.fail_streak += 1
            if self.fail_streak >= self.max_fail_streak:
                self.target_idx += 1
                self.fail_streak = 0
                outcome = "skip"
            else:
                outcome = "miss"
        self.trace.append({
            "sent_pos": diag_entry["pos"],
            "target": target,
            "observed": observed,
            "status": diag_entry["status"],
            "outcome": outcome,
            "state_before": before,
            "state_after": (self.target_idx, self.fail_streak),
        })

    # ----- bias query -----

    def get_bias_letter(self) -> Optional[str]:
        if self.target_idx >= len(self.secret):
            return None
        if not is_md_pre_letter(self.text):
            self._consec_prefix_tokens = 0
            return None
        self._consec_prefix_tokens += 1
        if self._consec_prefix_tokens > self.max_prefix_tokens:
            # Stuck in pre-letter prefix without seeing a letter — skip target.
            self.trace.append({
                "target_idx": self.target_idx,
                "target": self.secret[self.target_idx],
                "outcome": "skip_max_prefix",
                "consec_tokens": self._consec_prefix_tokens,
                "fail_streak": self.fail_streak,
            })
            self.target_idx += 1
            self.fail_streak = 0
            self._consec_prefix_tokens = 1   # this step counts toward the new target
            if self.target_idx >= len(self.secret):
                return None
        return self.secret[self.target_idx]

    def get_bias_token_ids(self) -> List[int]:
        letter = self.get_bias_letter()
        if letter is None:
            return []
        return self.letter_token_ids.get(letter, [])
