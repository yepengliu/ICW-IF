"""
WatermarkZScoreRewardFn — val reward function for watermark KD Ray trainer.

Unified API: detectors are built from ``eval_tasks``. A single-element list
(e.g. ``[green]`` or ``[initials]``) acts as single-task val; multi-element
(``[green, initials]``) triggers per-task routing of the scalar reward based
on the val parquet's ``task`` column and emits per-detector z-scores so the
trainer can report per-task metrics.
"""

import sys
import os
from typing import Dict, List, Optional

import torch

# Ensure project root is on path for gptwm imports
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _build_green_detector(wm_seed, wm_fraction, strength, only_english, tokenizer, model_config):
    from gptwm import GPTWatermarkDetector
    return GPTWatermarkDetector(
        fraction=wm_fraction,
        strength=strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=wm_seed,
        only_English=only_english,
        tokenizer=tokenizer,
    )


def _build_initials_detector(wm_seed, strength, tokenizer, model_config, stats_file):
    from gptwm_initials import (
        InitialsDetector, partition_letters, compute_gamma_from_stats,
    )
    green, _ = partition_letters(wm_seed)
    gamma = compute_gamma_from_stats(green, stats_file)
    return InitialsDetector(
        gamma=gamma,
        seed=wm_seed,
        strength=strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        tokenizer=tokenizer,
    )


class _HitsZAcrosticDetector:
    """Adapter wrapping ``acrostics_zstat.compute_hits_zstat`` with a
    ``unidetect(token_list) -> float`` API for the reward fn.

    Uses the markdown-aware extractor + controller-walk hit count + shuffle-S
    permutation null (hits-z, the production detector as of 2026-04-29).
    """

    def __init__(self, target: str, tokenizer, n_resample: int = 200,
                 seed: int = 0, max_fail_streak: int = 3):
        assert tokenizer is not None
        assert isinstance(target, str) and len(target) > 0
        self.target = target
        self.tokenizer = tokenizer
        self.n_resample = int(n_resample)
        self.seed = int(seed)
        self.max_fail_streak = int(max_fail_streak)

    def _decode(self, token_list):
        return self.tokenizer.decode(token_list, skip_special_tokens=True)

    def detect(self, token_list):
        from acrostics_zstat import compute_hits_zstat
        text = self._decode(token_list)
        stat = compute_hits_zstat(
            text=text, target=self.target, extractor="md",
            n_resample=self.n_resample, seed=self.seed,
            max_fail_streak=self.max_fail_streak,
        )
        return float(stat.z)

    def unidetect(self, token_list):
        return self.detect(token_list)


class _LcsZAcrosticDetector:
    """Adapter wrapping ``acrostics_zstat.compute_lcs_zstat`` with a
    ``unidetect(token_list) -> float`` API for the reward fn.

    LCS-based detector with shuffle-S null. Drops the controller's
    fail_streak skip mechanism, so a few noise letters in fl don't kill
    real subsequent matches. Empirically: 12-char insertion attack drops
    AUC by 1pp (vs 6.5pp for hits-z) on filtered KD pilot data
    (2026-04-30 robustness analysis).
    """

    def __init__(self, target: str, tokenizer, n_resample: int = 1000,
                 seed: int = 0):
        assert tokenizer is not None
        assert isinstance(target, str) and len(target) > 0
        self.target = target
        self.tokenizer = tokenizer
        self.n_resample = int(n_resample)
        self.seed = int(seed)

    def _decode(self, token_list):
        return self.tokenizer.decode(token_list, skip_special_tokens=True)

    def detect(self, token_list):
        from acrostics_zstat import compute_lcs_zstat
        text = self._decode(token_list)
        stat = compute_lcs_zstat(
            text=text, target=self.target, extractor="md",
            n_resample=self.n_resample, seed=self.seed,
        )
        return float(stat.z)

    def unidetect(self, token_list):
        return self.detect(token_list)


def _build_acrostics_detector(target, tokenizer, n_resample: int = 200,
                              seed: int = 0, kind: str = "hits"):
    """Build an acrostic detector. ``kind`` selects the metric:
      - 'hits' (default, back-compat): controller-walk hits + shuffle-S null
      - 'lcs': LCS length + shuffle-S null (robust to insertion noise)
    """
    if kind == "lcs":
        return _LcsZAcrosticDetector(
            target=target, tokenizer=tokenizer,
            n_resample=n_resample, seed=seed,
        )
    if kind != "hits":
        raise ValueError(f"acrostic_detector kind must be 'hits' or 'lcs', got {kind!r}")
    return _HitsZAcrosticDetector(
        target=target, tokenizer=tokenizer,
        n_resample=n_resample, seed=seed,
    )


class WatermarkZScoreRewardFn:
    """Reward function that computes watermark z-scores as rewards.

    Builds one detector per entry in ``eval_tasks``. When multiple tasks are
    present, per-sample scalar z-score is routed by ``data.non_tensor_batch["task"]``
    and every detector's z is also included in ``reward_extra_info`` (under
    ``z_score_{task}``) so the trainer can report per-task metrics with a
    shared negative set.
    """

    MIN_LEN = 200

    def __init__(
        self,
        tokenizer,
        model_config,
        strength: float = 2.0,
        only_english: bool = True,
        stats_file: str = "data/initials_icw/leading_space_first_letter_stats.json",
        eval_tasks: Optional[List[str]] = None,
        eval_green_seed: int = 1,
        eval_green_fraction: float = 0.25,
        eval_initials_seed: int = 0,
        acrostics_n_resample: int = 200,
        acrostics_detector_kind: str = "hits",
    ):
        """Acrostic detector is built lazily per per-sample target string read
        from ``data.non_tensor_batch['acrostic_target']`` at __call__ time.

        Hard-fails (no fallback) if the val parquet does not provide a
        non-empty ``acrostic_target`` for samples whose detection routes
        through the acrostic detector. This forces the secret-string design
        to live in the data, never in code defaults — preventing silent
        target mismatches like the 2026-05-02 'asdf' bug.
        """
        assert tokenizer is not None, "tokenizer must be provided"
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.model_config = model_config

        self.eval_tasks = list(eval_tasks) if eval_tasks else ["green"]
        unknown = [t for t in self.eval_tasks if t not in ("green", "initials", "acrostics")]
        if unknown:
            raise ValueError(f"eval_tasks must be subset of {{green, initials, acrostics}}; got {unknown}")

        # green/initials use seeds (one fixed detector per task). Acrostic is
        # per-sample: lazy-built on first use, cached by target string.
        self.detectors: Dict[str, object] = {}
        if "green" in self.eval_tasks:
            self.detectors["green"] = _build_green_detector(
                wm_seed=eval_green_seed,
                wm_fraction=eval_green_fraction,
                strength=strength,
                only_english=only_english,
                tokenizer=tokenizer,
                model_config=model_config,
            )
        if "initials" in self.eval_tasks:
            self.detectors["initials"] = _build_initials_detector(
                wm_seed=eval_initials_seed,
                strength=strength,
                tokenizer=tokenizer,
                model_config=model_config,
                stats_file=stats_file,
            )

        # Acrostic params kept as instance state for lazy detector build
        self._acrostic_in_eval = "acrostics" in self.eval_tasks
        self._acrostic_n_resample = int(acrostics_n_resample)
        self._acrostic_detector_kind = acrostics_detector_kind
        # cache: target_str -> detector
        self._acrostic_detectors: Dict[str, object] = {}

        self.default_detector_key = self.eval_tasks[0]
        self.mode = "mixed" if len(self.eval_tasks) > 1 else self.eval_tasks[0]

    def _get_acrostic_detector(self, target: str):
        """Return cached or freshly-built acrostic detector for ``target``.
        Hard-fails on empty/None target (no fallback)."""
        if not isinstance(target, str) or not target:
            raise ValueError(
                f"acrostic_target must be a non-empty string; got {target!r}. "
                "Per-sample acrostic_target is required in data.non_tensor_batch — "
                "fix the val/train parquet to populate it."
            )
        det = self._acrostic_detectors.get(target)
        if det is None:
            det = _build_acrostics_detector(
                target=target,
                tokenizer=self.tokenizer,
                n_resample=self._acrostic_n_resample,
                kind=self._acrostic_detector_kind,
            )
            self._acrostic_detectors[target] = det
        return det

    # Back-compat: some callers may reference self.detector as a scalar.
    # For acrostic-only eval this is meaningless (per-sample), so raise.
    @property
    def detector(self):
        if self._acrostic_in_eval and self.default_detector_key == "acrostics":
            raise RuntimeError(
                "acrostic detector is per-sample; access via _get_acrostic_detector(target)"
            )
        return self.detectors[self.default_detector_key]

    def _detector_keys(self) -> List[str]:
        """Detector keys reported in extra metrics. Includes 'acrostics' as a
        virtual key when in eval_tasks (each sample uses its own target)."""
        keys = list(self.detectors.keys())
        if self._acrostic_in_eval:
            keys.append("acrostics")
        return keys

    def __call__(self, data, return_dict: bool = True):
        """
        Expected ``data.batch["responses"]`` shape (B, T). Per-sample task is
        read from ``data.non_tensor_batch["task"]``. For samples routed through
        the acrostic detector (task=='acrostics' or default-routed neg when
        default is acrostics), per-sample target is read from
        ``data.non_tensor_batch["acrostic_target"]`` and **must be a non-empty
        string** — otherwise we hard-fail with a clear error.
        """
        response_ids = data.batch["responses"]
        batch_size, resp_len = response_ids.shape
        reward_tensor = torch.zeros(batch_size, resp_len, dtype=torch.float32)

        tasks = None
        acr_targets = None
        if hasattr(data, "non_tensor_batch") and data.non_tensor_batch is not None:
            tasks = data.non_tensor_batch.get("task", None)
            acr_targets = data.non_tensor_batch.get("acrostic_target", None)

        keys = self._detector_keys()
        per_detector_z: Dict[str, List[float]] = {k: [] for k in keys}
        z_scores: List[float] = []
        z_score_valid: List[bool] = []

        for i in range(batch_size):
            token_ids = response_ids[i].tolist()
            token_list = [t for t in token_ids if t != self.pad_token_id]

            if len(token_list) < self.MIN_LEN:
                z_scores.append(-1_000_000.0)
                z_score_valid.append(False)
                for k in per_detector_z:
                    per_detector_z[k].append(-1_000_000.0)
                continue

            # Fixed-seed detectors (green/initials) run for every sample
            for k, det in self.detectors.items():
                per_detector_z[k].append(float(det.unidetect(token_list)))

            # Acrostic detector is per-sample (target read from data)
            if self._acrostic_in_eval:
                if acr_targets is None:
                    raise ValueError(
                        "acrostic in eval_tasks but data.non_tensor_batch has no "
                        "'acrostic_target' field — fix the val parquet."
                    )
                target_i = acr_targets[i]
                if isinstance(target_i, bytes):
                    target_i = target_i.decode("utf-8")
                # numpy object dtype may yield None, NaN, or 'None' string
                target_i = None if (target_i is None or (isinstance(target_i, float)) or str(target_i) == "None" or str(target_i) == "") else str(target_i)
                if target_i is None:
                    raise ValueError(
                        f"sample {i}: acrostic_target missing/empty in val parquet; "
                        "all samples (incl. neg) must carry per-sample target."
                    )
                acr_det = self._get_acrostic_detector(target_i)
                per_detector_z["acrostics"].append(float(acr_det.unidetect(token_list)))

            task_i = str(tasks[i]) if tasks is not None else None
            if task_i == "neg" or task_i is None or task_i not in per_detector_z:
                # neg / unknown / single-task path: use default detector
                z_sample = per_detector_z[self.default_detector_key][-1]
            else:
                z_sample = per_detector_z[task_i][-1]

            z_scores.append(z_sample)
            z_score_valid.append(True)
            last_pos = len(token_list) - 1
            reward_tensor[i, last_pos] = z_sample

        extra = {
            "z_score": z_scores,
            "z_score_valid": z_score_valid,
        }
        for k, arr in per_detector_z.items():
            extra[f"z_score_{k}"] = arr

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
