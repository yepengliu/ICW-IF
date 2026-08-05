"""Initials ICW watermark: bias leading-space tokens whose first letter is in
the ``green`` set (13 of 26 letters selected per seed).

Detection: z-score on the fraction of leading-space English tokens whose first
letter is green, using an empirical per-seed γ computed from the token-count
distribution over A-Z.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

from watermark.gptwm import _get_english_token_ids, GPTWatermarkBase


LETTERS = list(string.ascii_uppercase)  # ['A', ..., 'Z']


# ---------- Helpers: first-letter classification ----------

def first_letter_of_token_string(tok_str: str) -> Optional[str]:
    """Return the uppercase first letter (A-Z) if the token string starts with
    a leading space followed by an ASCII letter; else None."""
    if not tok_str.startswith(" "):
        return None
    rest = tok_str[1:]
    if not rest:
        return None
    c = rest[0]
    if c.isalpha() and c.isascii():
        return c.upper()
    return None


def build_token_first_letter_map(
    tokenizer, vocab_size: int, english_token_ids: Optional[List[int]] = None
) -> Dict[int, str]:
    """Return {token_id: first_letter} for english+leading-space tokens."""
    if english_token_ids is None:
        english_token_ids = _get_english_token_ids(tokenizer, vocab_size)
    out: Dict[int, str] = {}
    for tid in english_token_ids:
        tok_str = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([tid]))
        letter = first_letter_of_token_string(tok_str)
        if letter is not None:
            out[tid] = letter
    return out


# ---------- Partition: 13 green / 13 red per seed ----------

def partition_letters(seed: int, n_green: int = 13) -> Tuple[List[str], List[str]]:
    """Deterministically split A-Z into green and red sets using ``seed``.
    Green letters are returned alphabetically; red likewise."""
    rng = np.random.default_rng(seed)
    idx = np.arange(26)
    rng.shuffle(idx)
    green_idx = sorted(idx[:n_green].tolist())
    red_idx = sorted(idx[n_green:].tolist())
    green = [LETTERS[i] for i in green_idx]
    red = [LETTERS[i] for i in red_idx]
    return green, red


# ---------- Mask building ----------

def build_initials_mask_numpy(
    seed: int,
    vocab_size: int,
    model_emb_length: int,
    tokenizer,
    english_token_ids: Optional[List[int]] = None,
    first_letter_map: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (mask, green_letters, red_letters). Mask is ``(model_emb_length,)``
    bool, True at token_ids whose first letter is in the green set.
    """
    assert model_emb_length > vocab_size, "model_emb_length must exceed vocab_size"
    if first_letter_map is None:
        first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_token_ids)
    green, red = partition_letters(seed)
    green_set = set(green)
    mask = np.zeros(model_emb_length, dtype=bool)
    for tid, letter in first_letter_map.items():
        if letter in green_set:
            mask[tid] = True
    return mask, green, red


def compute_gamma_from_stats(green_letters: List[str], stats_path: str) -> float:
    """Compute the null-hypothesis P(first-letter ∈ green) = sum of per-letter
    token fractions over green letters. Uses ``leading_space_first_letter_stats.json``.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    frac = stats["per_letter_fraction_letter_initial"]
    return float(sum(frac.get(ltr, 0.0) for ltr in green_letters))


# ---------- Base / Detector ----------

class InitialsWatermarkBase:
    """Holds the green/red partition and the corresponding token-id mask for a
    given seed and tokenizer."""

    def __init__(
        self,
        seed: int,
        strength: float,
        vocab_size: int,
        model_emb_length: int,
        tokenizer,
        english_token_ids: Optional[List[int]] = None,
        first_letter_map: Optional[Dict[int, str]] = None,
    ):
        self.seed = seed
        self.strength = strength
        self.vocab_size = vocab_size
        self.model_emb_length = model_emb_length
        self.tokenizer = tokenizer
        if first_letter_map is None:
            first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_token_ids)
        self.first_letter_map = first_letter_map
        mask_np, green, red = build_initials_mask_numpy(
            seed, vocab_size, model_emb_length, tokenizer,
            english_token_ids=english_token_ids, first_letter_map=first_letter_map,
        )
        self.mask = torch.tensor(mask_np, dtype=torch.float32)
        self.green_letters = green
        self.red_letters = red
        self.green_set = set(green)


class InitialsDetector(InitialsWatermarkBase):
    """Z-score detector for Initials ICW.

    For a decoded response, tokenize (``add_special_tokens=False``), filter to
    english+leading-space tokens, and compare the fraction of green-initial
    tokens to the per-seed null expectation γ.
    """

    def __init__(self, gamma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)

    @staticmethod
    def _z_score(num_green: int, total: int, gamma: float) -> float:
        if total == 0:
            return 0.0
        p = gamma
        return (num_green - p * total) / np.sqrt(p * (1.0 - p) * total)

    def hits(self, token_ids: List[int]) -> Tuple[int, int]:
        """Return (num_green_initial, num_leading_space_english) over the
        token_ids sequence (response only; do NOT pass prefix tokens in)."""
        n_total = 0
        n_green = 0
        for tid in token_ids:
            letter = self.first_letter_map.get(int(tid))
            if letter is None:
                continue
            n_total += 1
            if letter in self.green_set:
                n_green += 1
        return n_green, n_total

    def detect(self, token_ids: List[int]) -> float:
        n_green, n_total = self.hits(token_ids)
        return self._z_score(n_green, n_total, self.gamma)

    def unidetect(self, token_ids: List[int]) -> float:
        """Z-score using unique tokens only — analogous to the green-watermark
        ``unidetect`` (reduces autocorrelation)."""
        unique = list(set(int(t) for t in token_ids))
        return self.detect(unique)

    def hit_rate(self, token_ids: List[int]) -> float:
        n_green, n_total = self.hits(token_ids)
        return n_green / n_total if n_total > 0 else 0.0


# ---------- Stateful bias gating (mirrors acrostics_bias controller) ----------

_INITIAL_BOUNDARY_CHARS = frozenset(".!?,;:\"'()[]{}<>")


def is_initial_position(text: str) -> bool:
    """Return True if the running text-so-far ends at a word boundary, i.e.
    the next decoded token is expected to start a new English word.

    Rules (backward-only, no lookahead):
      * empty text  → True  (start of generation, treat as sentence start)
      * last char is whitespace (space / newline / tab) → True
      * last char is one of ``.!?,;:"'()[]{}<>`` → True
      * otherwise → False (mid-word continuation expected)
    """
    if not text:
        return True
    c = text[-1]
    if c.isspace():
        return True
    return c in _INITIAL_BOUNDARY_CHARS


def compute_initials_active_idx_response(
    response_ids: List[int],
    first_letter_map: Dict[int, str],
) -> List[int]:
    """Per-response-token active mask for KD-side per-position bias gating.

    For each position t (0-indexed within the response token sequence), the
    bias should fire iff the token sampled at the NEXT position (t+1) is a
    leading-space + first-letter-eng token (i.e. ``response_ids[t+1] in
    first_letter_map``). At training time this is ground truth — no
    backward-only state machine needed.

    Returned values (single-bucket; mirrors acrostic's per-letter idx schema
    so the loss-side gating logic can reuse ``>= 0`` checks):
      -1 : not active (don't apply bias at this position)
       0 : active     (apply +strength*green_mask at this position's logits)

    The last position is always -1 (no next token to predict).
    """
    L = len(response_ids)
    out = [-1] * L
    for t in range(L - 1):
        if response_ids[t + 1] in first_letter_map:
            out[t] = 0
    return out


class InitialsBiasController:
    """Per-request stateful controller for Initials ICW logit bias.

    Mirrors :class:`gptwm_acrostics_bias.AcrosticBiasController` in interface:
    one instance per generation request; ``update_with_past(output_ids)`` is
    called each step with the cumulative sampled-token list; ``is_active()``
    reports whether the bias should fire at the upcoming token.

    The bias mask itself is static (leading-space + green-initial tokens for
    the per-request seed); this controller only gates *when* to apply it.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text: str = ""
        self._processed_count: int = 0

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
        # Decode new tokens together (preserves BPE merging across boundaries
        # within this batch of new ids — matches AcrosticBiasController).
        new_text = self.tokenizer.decode(new_ids)
        self.text += new_text

    def is_active(self) -> bool:
        return is_initial_position(self.text)


# ---------- Smoke test ----------

