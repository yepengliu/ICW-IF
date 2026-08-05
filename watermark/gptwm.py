from typing import Any, List, Optional
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsProcessor


_english_token_ids_cache: dict = {}


def _get_english_token_ids(tokenizer, vocab_size: int):
    """Compute list of English token IDs once per (tokenizer, vocab_size); reused by mask building and GPTWatermarkBase."""
    cache_key = (id(tokenizer), vocab_size)
    if cache_key not in _english_token_ids_cache:
        vocab = tokenizer.get_vocab()
        # Some tokenizers (e.g. Gemma) scatter special tokens (`<unused42>`, `<eos>`, `<h4>`)
        # throughout the BPE id range, where the `tid < vocab_size` cutoff doesn't catch them
        # and they pass `is_english_token_v2` because they contain ASCII + alpha. Exclude
        # registered special tokens explicitly. No-op for Qwen3 / Llama (their specials are
        # all >= vocab_size).
        special_ids = set(getattr(tokenizer, 'added_tokens_decoder', {}).keys())
        english_token_ids = [
            tid for tok, tid in vocab.items()
            if GPTWatermarkBase.is_english_token_v2(tokenizer.convert_tokens_to_string([tok]))
            and tid < vocab_size
            and tid not in special_ids
        ]
        _english_token_ids_cache[cache_key] = sorted(english_token_ids)
    return _english_token_ids_cache[cache_key]


def _make_green_list_mask_numpy(
    watermark_key: int,
    fraction: float,
    vocab_size: int,
    model_emb_length: int,
    only_English: bool,
    tokenizer: Optional[object],
    english_token_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Build the green-list mask for a given seed as numpy bool array (for caching)."""
    assert watermark_key is not None, "watermark_key is None"
    assert model_emb_length > vocab_size, "model_emb_length is less than vocab_size"
    rng = np.random.default_rng(watermark_key)

    if only_English:
        if english_token_ids is None:
            english_token_ids = _get_english_token_ids(tokenizer, vocab_size)

        num_green_english = int(fraction * len(english_token_ids))
        english_mask = np.array([True] * num_green_english + [False] * (len(english_token_ids) - num_green_english))
        rng.shuffle(english_mask)
        mask = np.zeros(model_emb_length, dtype=bool)
        for i, token_id in enumerate(english_token_ids):
            mask[token_id] = english_mask[i]
    else:
        green_list_size = int(fraction * vocab_size)
        mask = np.array([True] * green_list_size + [False] * (vocab_size - green_list_size))
        rng.shuffle(mask)
        mask = np.concatenate([
            mask,
            np.zeros(model_emb_length - len(mask), dtype=bool),
        ])
    return mask


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        model_emb_length: The length of the model embedding.
        watermark_key: The random seed for the green-listing.
        only_English: If True, only English tokens will be considered for green-listing.
        tokenizer: The tokenizer instance (required if only_English=True).
    """

    @staticmethod
    def is_english_token(token: str) -> bool:
        """Check if a token is English (ASCII characters only, excluding first character)."""
        return all(ord(c) < 128 for c in token)

    @staticmethod
    def is_english_token_v2(token: str) -> bool:
        return all(ord(c) < 128 for c in token) and any(c.isalpha() for c in token)

    # def is_english_token_v2(token: str) -> bool:
        # return all(ord(c) < 128 and c.isalpha() for c in token)

    def __init__(
        self, 
        fraction: float = 0.5, 
        strength: float = 2.0, 
        vocab_size: int = None, 
        model_emb_length: int = None,
        watermark_key: int = 0,
        only_English: bool = False,
        tokenizer: Optional[object] = None
    ):
        self.tokenizer = tokenizer
        self.green_list_mask = torch.tensor(_make_green_list_mask_numpy(
            watermark_key, fraction, vocab_size, model_emb_length, only_English, tokenizer
        ), dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.only_English = only_English

        if only_English:
            english_token_ids = _get_english_token_ids(tokenizer, vocab_size)
            self.english_mask = torch.zeros(model_emb_length).long()
            self.english_mask[english_token_ids] = 1


class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsProcessor):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
        only_English: If True, only English tokens will be considered for green-listing.
        tokenizer: The tokenizer instance (required if only_English=True).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark = self.strength * self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        return new_logits


class BatchWatermarkLogitsProcessor(LogitsProcessor):
    """
    Logits processor that applies a per-sample green-list watermark using a seed per batch item.

    Set ``current_batch_seeds`` (list of int, length = batch size) before each generate call.
    Seeds are used to derive the green-list mask (same logic as GPTWatermarkLogitsWarper).
    Masks are cached by seed to avoid recomputation.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        strength: float = 2.0,
        vocab_size: int = None,
        model_emb_length: int = None,
        only_English: bool = False,
        tokenizer: Optional[object] = None,
    ):
        self.fraction = fraction
        self.strength = strength
        self.vocab_size = vocab_size
        self.model_emb_length = model_emb_length
        self.only_English = only_English
        self.tokenizer = tokenizer
        self._mask_cache = {}
        self.current_batch_seeds = None

    def _get_mask(self, seed: int) -> torch.Tensor:
        if seed not in self._mask_cache:
            self._mask_cache[seed] = torch.tensor(_make_green_list_mask_numpy(
                seed,
                self.fraction,
                self.vocab_size,
                self.model_emb_length,
                self.only_English,
                self.tokenizer,
            ), dtype=torch.float32)
        return self._mask_cache[seed]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        if self.current_batch_seeds is None:
            return scores
        batch_size = scores.shape[0]
        for i in range(batch_size):
            seed = self.current_batch_seeds[i]
            mask = self._get_mask(seed)
            scores[i] = scores[i] + self.strength * mask.to(scores.device)
        return scores


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        model_emb_length: The length of the model embedding.
        watermark_key: The random seed for the green-listing.
        only_English: If True, only English tokens will be considered for green-listing.
        tokenizer: The tokenizer instance (required if only_English=True).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(self.green_list_mask[sequence].sum())
        if self.only_English:
            total = int(self.english_mask[sequence].sum())
        else:
            total = len(sequence)
        if total < 2:
            return 0.0
        return self._z_score(green_tokens, total, self.fraction)

    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        green_tokens = int(self.green_list_mask[sequence].sum())
        if self.only_English:
            total = int(self.english_mask[sequence].sum())
        else:
            total = len(sequence)
        if total < 2:
            return 0.0
        return self._z_score(green_tokens, total, self.fraction)
    
    def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
        return z_score > tau, z_score

