"""Replay AcrosticBiasController on response token IDs to produce a per-position
``bias_letter_idx`` array suitable for KD training.

Output convention (matches worker shift semantics):
    For a sample with prompt length P and response length R,
    bias_letter_idx[k] = letter idx (0..25) at position k in the FULL
    actor input_ids (length P+R), where:
      - bias_letter_idx[P + n] = the bias letter that vLLM step n would have
        applied (n = 0..R-1) when generating r_n.
      - bias_letter_idx[k] = -1 elsewhere (and at positions where vLLM step
        had no bias).

Worker's ``[:, 1:]`` shift then aligns: at selected flat position k = P-1+n
(predicting r_n), the shifted bias_letter_idx at flat k = bias_letter_idx[k+1]
= bias_letter_idx[P + n] = vLLM step n bias. Correct.

Modes:
  - "dense": every step in pre-letter zone gets active (matches vLLM bias path)
  - "sparse": only the LETTER-BEARING token (post-md-strip first char is alpha)
              gets active. Strips the prefix chain noise that would otherwise
              dilute the per-letter gradient under per_task normalization.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import List, Optional

from watermark.acrostics_icw import _md_strip_block_marker, _md_strip_skippable
from watermark.gptwm_acrostics_bias import AcrosticBiasController


def _is_letter_bearing_token(token_text: str) -> bool:
    """True iff after stripping leading whitespace + markdown block markers
    + emphasis + opening quotes, the token starts with an ASCII letter.

    This is the per-token equivalent of ``is_md_pre_letter`` flipping False:
    the controller's bias state ends when a letter is emitted, and that letter
    is the first ASCII alpha char after stripping markdown structure.
    """
    if not token_text:
        return False
    rest, _ = _md_strip_block_marker(token_text)
    rest, _ = _md_strip_skippable(rest)
    return bool(rest) and rest[0].isascii() and rest[0].isalpha()


def replay_controller_letter_idx(
    response_ids: List[int],
    secret: str,
    tokenizer,
    letter_token_ids: dict,
    max_fail_streak: int = 3,
    max_prefix_tokens: int = 8,
    mode: str = "dense",
) -> List[int]:
    """Replay the AcrosticBiasController on a fixed response token sequence.

    Returns a list of length ``len(response_ids)`` where entry n is:
      - 0..25 = letter index ('a'=0 .. 'z'=25) for vLLM step n's bias, OR
      - -1 if no bias at that step.

    ``mode``:
      - "dense" (default): every step where the controller would apply bias
        (i.e., we're in pre-letter zone) is marked. This includes the
        markdown prefix chain (``\\n\\n``, ``##``, ``**`` ...) where the
        target letter is the same as at the eventual letter-bearing token.
        Matches the actual vLLM bias path one-to-one.
      - "sparse": only the LETTER-BEARING token of each sentence-start is
        marked (the token whose decoded text starts with an ASCII letter
        after stripping leading whitespace + markdown block markers +
        emphasis + opening quotes). All prefix-chain steps are -1.
        Sentence-level retries (fail_streak < 3) are PRESERVED — each
        retry's letter-bearing token gets its own active position.
        Use this for V5a: per-letter-position gradient is no longer
        diluted by ~64% prefix chain noise positions in per_task mode.

    Caller is responsible for placing this into the full (prompt + response)
    bias_letter_idx array at offsets [P, P+R-1].
    """
    if mode not in ("dense", "sparse"):
        raise ValueError(f"mode must be 'dense' or 'sparse', got {mode!r}")
    secret_lower = secret.lower()
    ctrl = AcrosticBiasController(
        secret=secret_lower,
        strength=0.0,            # we don't apply bias in replay; only query letter
        max_fail_streak=max_fail_streak,
        letter_token_ids=letter_token_ids,
        tokenizer=tokenizer,
        max_prefix_tokens=max_prefix_tokens,
    )

    out: List[int] = []
    for n in range(len(response_ids)):
        # Step n: vLLM is about to predict response_ids[n] with output_ids[:n]
        ctrl.update_with_past(response_ids[:n])
        letter = ctrl.get_bias_letter()
        if letter is None:
            out.append(-1)
            continue
        if mode == "dense":
            out.append(ord(letter) - ord("a"))
        else:
            # Sparse: only mark if the about-to-be-emitted token actually
            # carries a letter (post-md-strip first char is alpha). Prefix-
            # chain tokens (``\\n\\n``, ``##``, ``**``, ...) strip to empty
            # or to a non-alpha char and get -1.
            token_text = tokenizer.decode([response_ids[n]])
            if _is_letter_bearing_token(token_text):
                out.append(ord(letter) - ord("a"))
            else:
                out.append(-1)
    return out


def build_full_bias_letter_idx(
    response_ids: List[int],
    secret: str,
    tokenizer,
    letter_token_ids: dict,
    prompt_length: int,
    actor_seq_length: int,
    max_fail_streak: int = 3,
    max_prefix_tokens: int = 8,
    mode: str = "dense",
) -> List[int]:
    """Build the (actor_seq_length,) bias_letter_idx array matching the dataset
    convention. Returns a Python list of int8-compatible ints (range [-1, 25]).

    For non-acrostic samples, pass ``secret=""`` to get an all-(-1) array.
    """
    arr = [-1] * actor_seq_length
    if not secret or not response_ids:
        return arr

    per_step = replay_controller_letter_idx(
        response_ids=response_ids,
        secret=secret,
        tokenizer=tokenizer,
        letter_token_ids=letter_token_ids,
        max_fail_streak=max_fail_streak,
        max_prefix_tokens=max_prefix_tokens,
        mode=mode,
    )
    # Place per_step[n] at full position prompt_length + n (vLLM step n predicts
    # r_n, which lives at full position prompt_length + n).
    for n, v in enumerate(per_step):
        full_pos = prompt_length + n
        if 0 <= full_pos < actor_seq_length:
            arr[full_pos] = v
    return arr
