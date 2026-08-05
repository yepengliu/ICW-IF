from typing import Optional

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from watermark.gptwm import _make_green_list_mask_numpy, _get_english_token_ids
from watermark.gptwm_initials import build_initials_mask_numpy, build_token_first_letter_map
from watermark.gptwm_acrostics_bias import AcrosticBiasController

_WATERMARK_CONFIG = None
_INITIALS_CONFIG = None
_ACROSTICS_BIAS_CONFIG = None


def set_watermark_config(
    fraction: float = 0.5,
    strength: float = 2.0,
    vocab_size: int = None,
    model_emb_length: int = None,
    only_English: bool = False,
    tokenizer: Optional[object] = None,
    default_watermark_key: Optional[int] = None,
):
    global _WATERMARK_CONFIG
    _WATERMARK_CONFIG = {
        "fraction": fraction,
        "strength": strength,
        "vocab_size": vocab_size,
        "model_emb_length": model_emb_length,
        "only_English": only_English,
        "tokenizer": tokenizer,
        "default_watermark_key": default_watermark_key,
    }
    if only_English:
        _WATERMARK_CONFIG["english_token_ids"] = _get_english_token_ids(
            tokenizer, vocab_size
        )


class GPTWatermarkAdapterLogitsProcessor(AdapterLogitsProcessor):
    """Per-request watermark logits processor for vLLM.

    Each request can specify its own watermark seed via
    ``SamplingParams(extra_args={"watermark_key": <int>})``.
    If no per-request seed is given, falls back to ``default_watermark_key``
    set in :func:`set_watermark_config`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        cfg = _WATERMARK_CONFIG
        assert cfg is not None, "call set_watermark_config() before building the LLM"

        self.fraction = cfg["fraction"]
        self.strength = cfg["strength"]
        self.vocab_size = cfg["vocab_size"]
        self.model_emb_length = cfg["model_emb_length"]
        self.only_English = cfg["only_English"]
        self.tokenizer = cfg["tokenizer"]
        self.english_token_ids = cfg.get("english_token_ids")
        self.default_watermark_key = cfg["default_watermark_key"]
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _get_mask(self, watermark_key: int) -> torch.Tensor:
        if watermark_key not in self._mask_cache:
            self._mask_cache[watermark_key] = torch.tensor(
                _make_green_list_mask_numpy(
                    watermark_key,
                    self.fraction,
                    self.vocab_size,
                    self.model_emb_length,
                    self.only_English,
                    self.tokenizer,
                    self.english_token_ids,
                ),
                dtype=torch.float32,
            )
        return self._mask_cache[watermark_key]

    def new_req_logits_processor(self, params: SamplingParams):
        watermark_key = self.default_watermark_key
        if params.extra_args and "watermark_key" in params.extra_args:
            watermark_key = params.extra_args["watermark_key"]
        if watermark_key is None:
            return None

        mask = self._get_mask(watermark_key)
        strength = self.strength

        def _apply_watermark(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            return logits + strength * mask.to(logits.device)

        return _apply_watermark

    def is_argmax_invariant(self) -> bool:
        return False


# ---------------- Initials ICW ----------------

def set_initials_config(
    strength: float,
    vocab_size: int,
    model_emb_length: int,
    tokenizer,
    default_seed: Optional[int] = None,
):
    """Configure the Initials adapter. Call before building the LLM."""
    global _INITIALS_CONFIG
    english_ids = _get_english_token_ids(tokenizer, vocab_size)
    first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_ids)
    _INITIALS_CONFIG = {
        "strength": strength,
        "vocab_size": vocab_size,
        "model_emb_length": model_emb_length,
        "tokenizer": tokenizer,
        "english_token_ids": english_ids,
        "first_letter_map": first_letter_map,
        "default_seed": default_seed,
    }


class InitialsAdapterLogitsProcessor(AdapterLogitsProcessor):
    """Per-request Initials ICW watermark for vLLM.

    Each request can specify its seed via
    ``SamplingParams(extra_args={"initials_seed": <int>})``.
    If no per-request seed is given, falls back to ``default_seed`` from
    :func:`set_initials_config`.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        cfg = _INITIALS_CONFIG
        assert cfg is not None, "call set_initials_config() before building the LLM"
        self.strength = cfg["strength"]
        self.vocab_size = cfg["vocab_size"]
        self.model_emb_length = cfg["model_emb_length"]
        self.tokenizer = cfg["tokenizer"]
        self.english_token_ids = cfg["english_token_ids"]
        self.first_letter_map = cfg["first_letter_map"]
        self.default_seed = cfg["default_seed"]
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _get_mask(self, seed: int) -> torch.Tensor:
        if seed not in self._mask_cache:
            mask_np, _, _ = build_initials_mask_numpy(
                seed, self.vocab_size, self.model_emb_length,
                self.tokenizer,
                english_token_ids=self.english_token_ids,
                first_letter_map=self.first_letter_map,
            )
            self._mask_cache[seed] = torch.tensor(mask_np, dtype=torch.float32)
        return self._mask_cache[seed]

    def new_req_logits_processor(self, params: SamplingParams):
        seed = self.default_seed
        if params.extra_args and "initials_seed" in params.extra_args:
            seed = params.extra_args["initials_seed"]
        if seed is None:
            return None
        mask = self._get_mask(seed)
        strength = self.strength

        # NOTE: We intentionally apply the bias at every decode step (no
        # stateful gating). BPE encodes word boundaries inside leading-space
        # tokens (e.g. " Hello"), so backward-only stateful gating can only
        # detect boundaries when punctuation appears as its own token — that
        # accounts for ~12% of leading-space-first-letter steps. A stateful
        # closure missed 88% of the steps the bias is supposed to steer
        # (verify: scripts/verify_stateful_gating_precision.py). The
        # InitialsBiasController is kept (gptwm_initials) for KD-side use,
        # where the full token sequence is known and per-position active
        # mask can be computed from ground truth.
        def _apply(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            return logits + strength * mask.to(logits.device)

        return _apply

    def is_argmax_invariant(self) -> bool:
        return False


# ---------------- Acrostics logit-bias (closed-loop steering) ----------------

def set_acrostics_bias_config(
    strength: float,
    vocab_size: int,
    model_emb_length: int,
    tokenizer,
    letter_token_ids_path: str,
    max_fail_streak: int = 3,
    max_prefix_tokens: int = 8,
):
    """Configure the acrostic logit-bias adapter. Loads pre-computed
    {letter -> [token_id, ...]} from JSON and builds per-letter bool masks once.
    """
    import json
    from pathlib import Path

    global _ACROSTICS_BIAS_CONFIG
    with open(Path(letter_token_ids_path)) as f:
        d = json.load(f)
    per_letter_token_ids = d["per_letter_token_ids"]   # {a..z: [int,...]}

    _ACROSTICS_BIAS_CONFIG = {
        "strength": float(strength),
        "vocab_size": vocab_size,
        "model_emb_length": model_emb_length,
        "tokenizer": tokenizer,
        "max_fail_streak": int(max_fail_streak),
        "max_prefix_tokens": int(max_prefix_tokens),
        "per_letter_token_ids": per_letter_token_ids,
    }


class AcrosticsBiasAdapterLogitsProcessor(AdapterLogitsProcessor):
    """Per-request acrostic steering for vLLM.

    Each request must specify its secret string via
    ``SamplingParams(extra_args={"acrostic_secret": <str>})``. The closure
    maintains an ``AcrosticBiasController`` per request that:
      * tracks generation state (target_idx / fail_streak / running text)
      * fires +strength bias on letter-X tokens at each "pre-letter" step
      * advances target on raw-first-letter match (cheat status NOT checked)
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        cfg = _ACROSTICS_BIAS_CONFIG
        assert cfg is not None, "call set_acrostics_bias_config() before building the LLM"
        self.strength = cfg["strength"]
        self.vocab_size = cfg["vocab_size"]
        self.model_emb_length = cfg["model_emb_length"]
        self.tokenizer = cfg["tokenizer"]
        self.max_fail_streak = cfg["max_fail_streak"]
        self.max_prefix_tokens = cfg.get("max_prefix_tokens", 8)
        self.per_letter_token_ids = cfg["per_letter_token_ids"]   # {letter: [int,...]}
        # Build per-letter mask lazily on first apply (storing torch.Tensor on
        # the instance during __init__ has been observed to interact badly with
        # vLLM v1's logits-processor loading path).
        self._letter_masks: dict[str, torch.Tensor] = {}

    def new_req_logits_processor(self, params: SamplingParams):
        secret = None
        if params.extra_args and "acrostic_secret" in params.extra_args:
            secret = params.extra_args["acrostic_secret"]
        if not secret:
            return None  # no-op for this request

        ctrl = AcrosticBiasController(
            secret=secret,
            strength=self.strength,
            max_fail_streak=self.max_fail_streak,
            letter_token_ids=self.per_letter_token_ids,
            tokenizer=self.tokenizer,
            max_prefix_tokens=self.max_prefix_tokens,
        )
        strength = self.strength
        per_letter_token_ids = self.per_letter_token_ids
        letter_masks = self._letter_masks
        model_emb_length = self.model_emb_length

        def _get_mask(letter: str, device: torch.device) -> torch.Tensor:
            key = (letter, str(device))
            cached = letter_masks.get(key)
            if cached is not None:
                return cached
            tids = per_letter_token_ids.get(letter, [])
            mask = torch.zeros(model_emb_length, dtype=torch.float32, device=device)
            if tids:
                mask[[t for t in tids if t < model_emb_length]] = 1.0
            letter_masks[key] = mask
            return mask

        def _apply(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            ctrl.update_with_past(output_ids)
            letter = ctrl.get_bias_letter()
            if letter is None:
                return logits
            mask = _get_mask(letter, logits.device)
            return logits + strength * mask

        return _apply

    def is_argmax_invariant(self) -> bool:
        return False
