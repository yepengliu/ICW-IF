"""Per-sample watermark z-score reward function for Stage 2 RL training.

Each sample carries its own (task, wm_seed, wm_fraction). For task=='acrostics',
the sample additionally carries its own ``acrostic_target`` (per-sample target
string) — this is critical so the model learns *generalizable* acrostic skill
instead of memorizing one fixed target. We build a detector per
(task, seed, fraction, target) on demand, cache it, and compute z-score on the
rollout response tokens. The scalar z is placed at the last response position
(token-level reward convention in verl/DAPO).

Input (DataProto):
  data.batch["responses"]                 (B, T)  long  — rollout response tokens
  data.non_tensor_batch["task"]           (B,)    object — per-sample task label
  data.non_tensor_batch["wm_seed"]        (B,)    object — per-sample seed
  data.non_tensor_batch["wm_fraction"]    (B,)    object — per-sample fraction/gamma
  data.non_tensor_batch["acrostic_target"] (B,)   object — per-sample target string
                                                            (only set when task=='acrostics';
                                                            falls back to config default)
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, Optional

import numpy as np
import torch


# Reward-hack signatures for acrostic task. Patterns observed in trained ckpts
# (n=477 audit 2026-05-06):
#   1. md_bold        — `**X**` markdown bold (Pure-RL hack: 99.4%; isolated
#                       single-letter bold = clearest signature)
#   2. emphasis_alt   — italic `_X_`, backtick `` `X` ``, paren `(X)` —
#                       documented variants (≥2 occurrences threshold to avoid
#                       false positives on lone legit usage like middle initials)
#   3. orphan_letter  — line containing just a capital letter (KD+RL r3: 0.4%)
#   4. letter_heading — line "X: <content>" (KD+RL r3: 3.1%)
#   5. secret_dump    — ≥6-char secret substring contiguous OR ≥4 space-separated
#                       capitals matching subseq (KD+RL r3: 0.8%)
# All violate ICW prompt rule 4 ("plain narrative prose; do not visually
# highlight first letters"). Detector LCS doesn't penalize (extracts letter
# regardless of decoration), so explicit gates are required. Force reward = 0
# on any hit. md_bold uses [A-Z] only (uppercase) to avoid false positives like
# the ABC Conjecture sample's `**a**/**b**/**c**`.
_ACROSTIC_MD_BOLD_RE = re.compile(r"\*\*[A-Z]\*\*")
_ACROSTIC_ITALIC_UNDERSCORE_RE = re.compile(r"(?<![A-Za-z0-9_])_[A-Z]_(?![A-Za-z0-9_])")
_ACROSTIC_BACKTICK_RE = re.compile(r"(?<![A-Za-z0-9])`[A-Z]`(?![A-Za-z0-9])")
_ACROSTIC_PAREN_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])\(\s*[A-Z]\s*\)(?![A-Za-z0-9])")
_ACROSTIC_ORPHAN_LETTER_RE = re.compile(r"(?m)^\s*[A-Z][.:]?\s*$")
_ACROSTIC_LETTER_HEADING_RE = re.compile(r"(?m)^\s*[A-Z]:\s+\S")
_ACROSTIC_SPACE_LETTERS_RE = re.compile(r"(?:[A-Z][\s\t]+){3,}[A-Z]")


def _detect_acrostic_hack(resp_text: str, target: Optional[str]) -> str:
    """Classify acrostic reward-hack pattern; return one of:
    'md_bold' / 'emphasis_alt' / 'orphan' / 'heading' / 'secret_dump' / 'clean'.

    md_bold fires on any `**X**` (uppercase only). Other emphasis variants
    require ≥2 occurrences total (italic + backtick + paren) to avoid false
    positives on lone legit usage. Orphan + heading require ≥2 occurrences
    (single 'Q.' could be Q&A label). Secret_dump: 6-char contiguous OR 4+
    space-separated capitals matching secret subseq.
    """
    if _ACROSTIC_MD_BOLD_RE.search(resp_text):
        return "md_bold"

    n_emphasis = (
        len(_ACROSTIC_ITALIC_UNDERSCORE_RE.findall(resp_text))
        + len(_ACROSTIC_BACKTICK_RE.findall(resp_text))
        + len(_ACROSTIC_PAREN_LETTER_RE.findall(resp_text))
    )
    if n_emphasis >= 2:
        return "emphasis_alt"

    if len(_ACROSTIC_ORPHAN_LETTER_RE.findall(resp_text)) >= 2:
        return "orphan"

    if len(_ACROSTIC_LETTER_HEADING_RE.findall(resp_text)) >= 2:
        return "heading"

    if target and len(target) >= 6:
        target_u = target.upper()
        resp_u = resp_text.upper()
        for i in range(len(target_u) - 5):
            if target_u[i:i + 6] in resp_u:
                return "secret_dump"
        if _ACROSTIC_SPACE_LETTERS_RE.search(resp_text):
            space_letters = re.findall(r"\b([A-Z])\b", resp_text)
            joined = "".join(space_letters)
            for i in range(len(joined) - 3):
                if joined[i:i + 4] in target_u:
                    return "secret_dump"

    return "clean"


# Ensure project root on sys.path for gptwm imports when invoked as Ray remote
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _round_frac(f: float, decimals: int = 6) -> float:
    return float(round(float(f), decimals))


class PerSampleWatermarkZScoreRewardFn:
    """Reward function that builds a detector per-sample from (task, seed, fraction).

    Caches detectors by (task, seed, rounded_fraction) to amortize construction
    cost. Assumes the tokenizer/model_config are the same across samples.
    """

    MIN_LEN = 200

    def __init__(
        self,
        tokenizer,
        model_config,
        strength: float = 2.0,
        only_english: bool = True,
        stats_file: str = "data/initials_icw/leading_space_first_letter_stats.json",
        active_tasks: Optional[list] = None,
        acrostics_n_resample: int = 1000,
        acrostics_detector_kind: str = "lcs",
    ):
        """Per-sample reward fn. Acrostic target MUST be in
        ``data.non_tensor_batch['acrostic_target']`` per sample — no fallback.

        Removed `acrostics_target` constructor param 2026-05-02 to prevent
        silent target mismatches: every secret string lives in the data only.
        """
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.model_config = model_config
        self.strength = float(strength)
        self.only_english = bool(only_english)
        self.stats_file = stats_file
        self.active_tasks = list(active_tasks) if active_tasks else ["green", "initials"]
        self.acrostics_n_resample = int(acrostics_n_resample)
        if acrostics_detector_kind not in ("hits", "lcs"):
            raise ValueError(f"acrostics_detector_kind must be 'hits' or 'lcs', got {acrostics_detector_kind!r}")
        self.acrostics_detector_kind = acrostics_detector_kind

        # Detector cache: {(task, seed, frac_key): detector}
        self._cache: Dict[tuple, object] = {}

    def _get_detector(self, task: str, seed: int, fraction: float, target: str = None):
        """Build (and cache) a detector for a single (task, seed, fraction, target) tuple.

        ``target`` is only used when ``task=='acrostics'``; for other tasks it is ignored
        in the cache key.
        """
        frac_key = _round_frac(fraction)
        if task == "acrostics":
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"task=acrostics requires non-empty per-sample target; got {target!r}. "
                    "Fix the parquet to populate acrostic_target."
                )
            key = (task, target)        # acrostics detector keyed on target only
        else:
            key = (task, int(seed), frac_key)
        if key in self._cache:
            return self._cache[key]

        if task == "green":
            from gptwm import GPTWatermarkDetector
            det = GPTWatermarkDetector(
                fraction=fraction,
                strength=self.strength,
                vocab_size=self.tokenizer.vocab_size,
                model_emb_length=self.model_config.vocab_size,
                watermark_key=int(seed),
                only_English=self.only_english,
                tokenizer=self.tokenizer,
            )
        elif task == "initials":
            from gptwm_initials import (
                InitialsDetector, partition_letters, compute_gamma_from_stats,
            )
            green, _ = partition_letters(int(seed))
            gamma = compute_gamma_from_stats(green, self.stats_file)
            det = InitialsDetector(
                gamma=gamma,
                seed=int(seed),
                strength=self.strength,
                vocab_size=self.tokenizer.vocab_size,
                model_emb_length=self.model_config.vocab_size,
                tokenizer=self.tokenizer,
            )
        elif task == "acrostics":
            # Reuse KD recipe's md-extractor + hits/lcs zstat adapter so RL
            # train/val and KD val all share the exact same detector pipeline
            # (extractor='md', no strict regex). Default kind='lcs' picks the
            # production detector validated on test_477 (AUC 0.94→0.99).
            from recipe.watermark_kd_ray.reward import _build_acrostics_detector
            # target was validated as non-empty above
            det = _build_acrostics_detector(
                target=target,
                tokenizer=self.tokenizer,
                n_resample=self.acrostics_n_resample,
                kind=self.acrostics_detector_kind,
            )
        else:
            raise ValueError(f"unknown task {task!r}")

        self._cache[key] = det
        return det

    def __call__(self, data, return_dict: bool = True):
        responses = data.batch["responses"]       # (B, T)
        B, T = responses.shape
        reward_tensor = torch.zeros(B, T, dtype=torch.float32)

        tasks       = data.non_tensor_batch["task"]
        wm_seeds    = data.non_tensor_batch["wm_seed"]
        wm_fracs    = data.non_tensor_batch["wm_fraction"]
        acr_targets = data.non_tensor_batch.get("acrostic_target", [None] * B)

        z_scores: list = []
        z_valid:  list = []
        resp_lens: list = []
        # Per-sample 0/1 indicator for ANY acrostic reward-hack hit (md_bold /
        # orphan / heading / secret_dump). Must be length B per-sample so verl's
        # DataProto.chunk can split alongside responses tensor across DP ranks.
        hack_indicator: list = []
        # Per-sample hack-kind label ('clean' / 'md_bold' / 'orphan' / 'heading'
        # / 'secret_dump') for diagnostic logging. Length B.
        hack_kind_per_sample: list = []
        per_task_z: Dict[str, list] = {t: [] for t in self.active_tasks}

        for i in range(B):
            task = str(tasks[i])
            seed = int(wm_seeds[i])
            frac = float(wm_fracs[i])
            target = acr_targets[i] if i < len(acr_targets) else None
            if isinstance(target, bytes):
                target = target.decode("utf-8")
            target = str(target) if (target is not None and str(target) not in ("None", "")) else None
            if task == "acrostics" and target is None:
                raise ValueError(
                    f"sample {i}: task=acrostics but acrostic_target missing in parquet"
                )

            ids = responses[i].tolist()
            token_list = [t for t in ids if t != self.pad_token_id]
            n = len(token_list)
            resp_lens.append(n)

            if n < self.MIN_LEN:
                z_scores.append(-1e6)
                z_valid.append(False)
                hack_indicator.append(0)
                hack_kind_per_sample.append("clean")
                for t in per_task_z:
                    per_task_z[t].append(float("nan"))
                continue

            # === Anti reward-hack gate (acrostic only) ===
            # Detect 4 visual-highlight patterns (md_bold / orphan / heading /
            # secret_dump) BEFORE invoking the LCS extractor. All violate ICW
            # prompt rule 4. n=477 audit 2026-05-06: Pure-RL=99.4% md_bold,
            # KD+RL r3=3.1% heading + 0.8% secret_dump. LCS detector doesn't
            # penalize (extracts the letter regardless of decoration) so explicit
            # gates are required.
            hack_kind = "clean"
            if task == "acrostics":
                resp_text = self.tokenizer.decode(token_list, skip_special_tokens=True)
                hack_kind = _detect_acrostic_hack(resp_text, target)
                if hack_kind != "clean":
                    z_scores.append(0.0)
                    z_valid.append(True)
                    hack_indicator.append(1)
                    hack_kind_per_sample.append(hack_kind)
                    reward_tensor[i, n - 1] = 0.0
                    for t in per_task_z:
                        per_task_z[t].append(0.0 if t == task else float("nan"))
                    continue
            hack_kind_per_sample.append(hack_kind)

            # Per-task z (fill only the one matching this sample's task)
            try:
                det = self._get_detector(task, seed, frac, target=target)
                z = float(det.unidetect(token_list))
            except Exception as e:
                print(f"[reward] detector error task={task} seed={seed} frac={frac} target={target}: {e}")
                z = 0.0

            z_scores.append(z)
            z_valid.append(True)
            hack_indicator.append(0)
            reward_tensor[i, n - 1] = z

            for t in per_task_z:
                per_task_z[t].append(z if t == task else float("nan"))

        extra = {
            "z_score": z_scores,
            "z_score_valid": z_valid,
            "response_len": resp_lens,
            # Backward-compat name; now means "any hack pattern", not just md bold.
            "acrostic_md_hack": hack_indicator,
            "acrostic_hack_any": hack_indicator,
        }
        # Per-kind 0/1 indicators (length B) — lets wandb track which gate fires.
        for kind in ("md_bold", "emphasis_alt", "orphan", "heading", "secret_dump"):
            extra[f"acrostic_hack_{kind}"] = [
                1 if k == kind else 0 for k in hack_kind_per_sample
            ]
        for t, arr in per_task_z.items():
            extra[f"z_score_{t}"] = arr

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
