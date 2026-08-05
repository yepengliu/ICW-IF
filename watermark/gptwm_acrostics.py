"""Acrostics watermark detector with ``unidetect(token_list) -> float`` API so
it plugs into ``WatermarkZScoreRewardFn`` the same way as ``GPTWatermarkDetector``
and ``InitialsDetector``.

Wraps ``acrostics_zstat.compute_lev_zstat`` (Levenshtein z-stat per ICW paper
Section 4.2.4). Decoding from token_ids → text uses the supplied tokenizer.

Usage:
    det = AcrosticsDetector(
        target="asdf",
        tokenizer=tokenizer,
        n_resample=200,   # lower for training hot path; 1000 for eval
    )
    z = det.unidetect(response_token_ids)

Note: the ``token_list`` argument name is kept for API parity with the green /
initials detectors, but Acrostics detection is text-level — there is no unique-
token deduplication semantics, so ``unidetect == detect``.
"""

from __future__ import annotations

from typing import List, Optional

from watermark.acrostics_zstat import compute_lev_zstat


class AcrosticsDetector:
    """Levenshtein z-stat detector for Acrostics ICW.

    Args:
        target: target secret string the response should encode as sentence
            initials (e.g., "asdf"). Stored at init; for per-sample targets,
            instantiate multiple detectors or extend the reward fn to override.
        tokenizer: HF tokenizer used to decode ``token_list`` back to text.
        n_resample: number of random permutations for the null distribution.
            Paper uses 1000; for training reward we can drop to 200 for speed.
        seed: RNG seed for null-distribution shuffling (reproducibility).
    """

    def __init__(
        self,
        target: str,
        tokenizer,
        n_resample: int = 200,
        seed: int = 0,
        strict: bool = True,
    ):
        assert tokenizer is not None, "tokenizer required"
        assert isinstance(target, str) and len(target) > 0, "target must be non-empty string"
        self.target = target
        self.tokenizer = tokenizer
        self.n_resample = int(n_resample)
        self.seed = int(seed)
        # strict=True (default for RL reward) blocks single-letter heading +
        # numbered-list reward-hacking patterns. Set strict=False to reproduce
        # paper-faithful detection (eval AUC numbers).
        self.strict = bool(strict)

    def _decode(self, token_list: List[int]) -> str:
        # Skip special tokens so e.g. <|im_end|>, pad tokens don't leak into text
        return self.tokenizer.decode(token_list, skip_special_tokens=True)

    def detect(self, token_list: List[int]) -> float:
        text = self._decode(token_list)
        stat = compute_lev_zstat(
            text=text, target=self.target,
            n_resample=self.n_resample, seed=self.seed,
            strict=self.strict,
        )
        return float(stat.z)

    def unidetect(self, token_list: List[int]) -> float:
        # Acrostics is text-level; unique-token dedup doesn't apply
        return self.detect(token_list)

