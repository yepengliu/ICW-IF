"""Acrostics watermark detectors: Levenshtein z-stat (paper-faithful) and
Smith-Waterman z-stat (new, larger dynamic range).

Both use shuffle-S permutation null:
  Lev:  D = (μ − d_obs) / σ   where d = Lev distance, lower = better
  SW:   z = (s_obs − μ) / σ   where s = SW score, higher = better

Both shuffle the observed first-letter sequence ℓ and recompute the metric
n_resample times to build the null. Lev follows ICW §4.2.4 verbatim; SW is
adapted from standard local-alignment scoring (+2 / -1 / -1).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from watermark.acrostics_icw import (
    _levenshtein,
    extract_first_letters,
    extract_first_letters_md,
    extract_first_letters_strict,
)


@dataclass
class LevZStat:
    fl: str
    target: str
    d_obs: int
    mu: float
    sigma: float
    z: float
    n_resample: int
    n_sentences: int


@dataclass
class HitsZStat:
    fl: str                      # extracted first-letter sequence (lowercased)
    secret: str                  # secret string (lowercased)
    hits: int                    # observed hit count via controller walk
    target_idx_final: int        # secret advancement at end of walk (hits + skips)
    skips: int                   # 3-fail-streak skip count
    mu: float                    # null mean (shuffle-S of fl multiset)
    sigma: float                 # null std
    z: float                     # empirical z = (hits - μ) / σ
    p: float                     # one-sided permutation p with Laplace smoothing
    n_resample: int
    n_sentences: int             # = len(fl)
    max_fail_streak: int


@dataclass
class SWZStat:
    fl: str                      # extracted first-letter sequence (lowercased)
    target: str                  # original target (lowercased)
    target_eff: str              # T truncated to len(fl) if shorter; else == target
    obs: int                     # SW score on (target_eff, fl)
    mu: float                    # null mean (shuffle-S)
    sigma: float                 # null std
    z: float                     # empirical zE = (obs - μ) / σ
    p: float                     # one-sided permutation p with Laplace smoothing
    n_resample: int
    n_sentences: int             # = len(fl)
    extractor: str               # "regex_strict" / "regex_loose" / "nltk"


@dataclass
class LcsZStat:
    fl: str                      # extracted first-letter sequence (lowercased)
    secret: str                  # secret string (lowercased)
    obs: int                     # |LCS(secret, fl)|
    mu: float                    # null mean (shuffle-S of fl multiset)
    sigma: float                 # null std
    z: float                     # empirical z = (obs - μ) / σ
    p: float                     # one-sided permutation p with Laplace smoothing
    n_resample: int
    n_sentences: int             # = len(fl)
    extractor: str               # extractor name for traceability


# ---------- LCS (mentor-recommended detector, robust to insertion noise) ----------

def lcs_length(s1: str, s2: str) -> int:
    """Length of the longest common subsequence of s1 and s2.

    No skip/fail_streak, no gap penalty: every character of s1 either matches
    a character in s2 in order (counted) or is silently skipped (cost 0).
    Equivalent to edit-distance with insert/delete/mismatch all costing 0
    except match, which costs -1 (i.e., max LCS = min editing cost).

    Time O(|s1|·|s2|), space O(min(|s1|,|s2|)) via 1D rolling DP.

    Robustness: a single noise char in s2 just contributes nothing to LCS,
    unlike controller-walk hits where 3 consecutive noise chars trigger
    skip and lose the next real match.
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    # Ensure n is the shorter dim for memory
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        s1_i = s1[i - 1]
        for j in range(1, n + 1):
            tmp = dp[j]
            if s1_i == s2[j - 1]:
                dp[j] = prev + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev = tmp
    return dp[n]


def compute_lcs_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
    extractor: str = "md",
) -> LcsZStat:
    """LCS-based z-stat with shuffle-S null.

    obs   = |LCS(secret, fl)| where fl = extractor(text), lowercase.
    null  = same metric on n_resample random permutations of fl
            (preserving char multiset).

    Compared with ``compute_hits_zstat``:
      - No fail_streak skip → robust to insertion noise (3-char attack
        AUC drop ≈ 1pp vs hits-z's ≈ 7pp on filtered KD data).
      - No gap penalty (vs Smith-Waterman) → no pathological cases where
        spread-out hits get drowned by gap costs.

    Args:
        text: model response.
        target: secret string (case-insensitive).
        n_resample: # shuffles for null. p-floor = 1/(n+1).
        seed: RNG seed for reproducibility.
        extractor: which first-letter extractor; default 'md'.

    Returns: LcsZStat with z as primary signal and Laplace-smoothed p.
    """
    extr_fn = _get_extractor(extractor)
    fl = extr_fn(text)
    tgt = target.lower()

    if not fl:
        return LcsZStat(
            fl="", secret=tgt, obs=0,
            mu=0.0, sigma=0.0, z=0.0, p=1.0,
            n_resample=0, n_sentences=0, extractor=extractor,
        )

    obs = lcs_length(tgt, fl)

    rng = random.Random(seed)
    fl_chars = list(fl)
    null_scores = []
    for _ in range(n_resample):
        rng.shuffle(fl_chars)
        null_scores.append(lcs_length(tgt, "".join(fl_chars)))

    mu = sum(null_scores) / len(null_scores)
    if len(null_scores) > 1:
        var = sum((x - mu) ** 2 for x in null_scores) / (len(null_scores) - 1)
        sigma = math.sqrt(var)
    else:
        sigma = 0.0

    z = (obs - mu) / sigma if sigma > 0 else 0.0
    k_ge = sum(1 for x in null_scores if x >= obs)
    p = (k_ge + 1) / (n_resample + 1)

    return LcsZStat(
        fl=fl, secret=tgt, obs=obs,
        mu=mu, sigma=sigma, z=z, p=p,
        n_resample=n_resample, n_sentences=len(fl),
        extractor=extractor,
    )


# ---------- Smith-Waterman ----------

def smith_waterman(T: str, S: str,
                   match: int = 2, mismatch: int = -1, gap: int = -1) -> int:
    """Standard Smith-Waterman local alignment max score.
    O(|T|*|S|) time and space (suitable for |T|, |S| ≤ ~100)."""
    m, n = len(T), len(S)
    if m == 0 or n == 0:
        return 0
    H = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if T[i - 1] == S[j - 1] else mismatch
            v = max(0,
                    H[i - 1][j - 1] + s,
                    H[i - 1][j] + gap,
                    H[i][j - 1] + gap)
            H[i][j] = v
            if v > best:
                best = v
    return best


_NLTK_READY = False


def _ensure_nltk_punkt() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass  # older NLTK doesn't need punkt_tab
    _NLTK_READY = True


def _get_extractor(name: str):
    """Return the extractor function by name.

    Names:
      * 'md' (default, production) — markdown-aware extractor; recognizes
        sentence-starts in plain prose and across markdown structure
        (heading / list-item / blockquote / paragraph).
      * 'regex_strict' — alias for 'md' (back-compat for callers that passed
        the strict-v3 name; the new md extractor replaces strict v3).
      * 'regex_loose' — legacy loose extractor (uppercase-only sentence start).
      * 'strict_v3' — pre-md heuristic-strict extractor (kept for archeology).
      * 'nltk' — ICW-paper-faithful Punkt-based extractor.
    """
    if name == "md" or name == "regex_strict":
        return extract_first_letters_md
    if name == "strict_v3":
        return extract_first_letters_strict
    if name == "regex_loose":
        return extract_first_letters
    if name == "nltk":
        # Lazy import to avoid hard nltk dep at module load
        from acrostics_icw import extract_first_letters_nltk  # noqa: F401
        _ensure_nltk_punkt()
        return extract_first_letters_nltk
    raise ValueError(f"unknown extractor {name!r}; "
                     "expected one of md / regex_strict / regex_loose / "
                     "strict_v3 / nltk")


def _controller_walk_hits(fl: str, secret: str, max_fail_streak: int = 3) -> tuple:
    """Walk ``fl`` with the same state machine as ``AcrosticBiasController``.

    For each char in fl: if it equals current target letter (secret[target_idx]),
    record hit and advance target. Otherwise record miss; on the max_fail_streak-th
    miss in a row, skip the current target (advance + reset fail). Stops when
    target_idx >= len(secret).

    Returns (hits, skips, target_idx_final).

    KNOWN LIMITATION (robustness, 2026-04-29 — TODO):
        The strict left-to-right + max_fail_streak walk is brittle under noise.
        Example: secret="ab", fl="acccb" → hits=1 (the 'b' at fl[4] is missed
        because fail_streak reaches 3 on 'ccc' and skips the 'b' target before
        we reach it). LCS-style extraction (allow non-consecutive matches
        anywhere in fl while preserving order) is more robust under
        substitution/deletion attacks: empirical 3-char attack drops
        hits-z AUC by ~9pp but SW(LCS, secret) only drops ~0.7pp on s=12 sweep.
        For the current detect-logit-bias-only use case (no adversarial
        attacks), this brittleness is acceptable. Revisit before adding
        deployment / paraphrase-attack scenarios.
    """
    target_idx = 0
    fail_streak = 0
    hits = 0
    skips = 0
    for c in fl:
        if target_idx >= len(secret):
            break
        if c == secret[target_idx]:
            hits += 1
            target_idx += 1
            fail_streak = 0
        else:
            fail_streak += 1
            if fail_streak >= max_fail_streak:
                skips += 1
                target_idx += 1
                fail_streak = 0
    return hits, skips, target_idx


def compute_hits_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
    max_fail_streak: int = 3,
    extractor: str = "md",
) -> HitsZStat:
    """Hits-based z-stat with shuffle-S null.

    obs = controller-walk hit count of ``fl`` against ``target``.
    null = same metric on n_resample random permutations of ``fl`` (preserving
    char multiset). Aligns the detection signal with what the bias controller
    actually does at generation time, so sparse-but-ordered hits (spaced over
    misses inside the 3-fail-streak budget) all count toward signal.

    Args:
        text: model response text.
        target: secret string (case-insensitive).
        n_resample: # shuffles for null distribution. p-floor = 1/(n+1).
        seed: RNG seed for shuffles.
        max_fail_streak: forced-skip threshold (matches controller default 3).
        extractor: which first-letter extractor (default 'md').

    Returns: HitsZStat with hits as the primary signal and a permutation
    p-value with Laplace smoothing.
    """
    extr_fn = _get_extractor(extractor)
    fl = extr_fn(text)
    tgt = target.lower()

    if not fl:
        return HitsZStat(
            fl="", secret=tgt, hits=0, target_idx_final=0, skips=0,
            mu=0.0, sigma=0.0, z=0.0, p=1.0,
            n_resample=0, n_sentences=0, max_fail_streak=max_fail_streak,
        )

    hits, skips, tidx_final = _controller_walk_hits(fl, tgt, max_fail_streak)

    rng = random.Random(seed)
    fl_chars = list(fl)
    null_scores = []
    for _ in range(n_resample):
        rng.shuffle(fl_chars)
        h, _, _ = _controller_walk_hits("".join(fl_chars), tgt, max_fail_streak)
        null_scores.append(h)

    mu = sum(null_scores) / len(null_scores)
    if len(null_scores) > 1:
        var = sum((x - mu) ** 2 for x in null_scores) / (len(null_scores) - 1)
        sigma = math.sqrt(var)
    else:
        sigma = 0.0

    z = (hits - mu) / sigma if sigma > 0 else 0.0
    k_ge = sum(1 for x in null_scores if x >= hits)
    p = (k_ge + 1) / (n_resample + 1)

    return HitsZStat(
        fl=fl, secret=tgt, hits=hits, target_idx_final=tidx_final, skips=skips,
        mu=mu, sigma=sigma, z=z, p=p,
        n_resample=n_resample, n_sentences=len(fl),
        max_fail_streak=max_fail_streak,
    )


def compute_sw_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
    strict: bool = True,
    truncate_target: bool = True,
    extractor: str = "regex_strict",
) -> SWZStat:
    """Smith-Waterman z-stat with shuffle-S null.

    Args:
        text: model response text.
        target: secret string (case-insensitive).
        n_resample: # shuffles for null distribution. p-floor = 1/(n+1).
        seed: RNG seed for shuffles (reproducibility).
        strict: kept for API parity; if extractor='regex_*' it picks
            strict-vs-loose. If extractor='nltk' this flag is ignored.
        truncate_target: if True and len(fl) < len(target), use target[:len(fl)]
            as the effective target. This matches the semantic "model wrote N
            sentences = signal compared against first N letters of secret".
        extractor: 'regex_strict' (default, RL-reward style) /
            'regex_loose' / 'nltk' (ICW paper style).

    Returns: SWZStat with empirical zE as the primary signal and a permutation
    p-value with Laplace smoothing.
    """
    # Resolve extractor (override `extractor` arg via legacy `strict` flag if
    # caller specified strict=False AND extractor default).
    if extractor == "regex_strict" and strict is False:
        extractor = "regex_loose"

    extr_fn = _get_extractor(extractor)
    fl = extr_fn(text)
    tgt = target.lower()

    # Edge case: no detectable sentences → zero signal
    if not fl:
        return SWZStat(
            fl="", target=tgt, target_eff="", obs=0,
            mu=0.0, sigma=0.0, z=0.0, p=1.0,
            n_resample=0, n_sentences=0, extractor=extractor,
        )

    # Truncate target if generation is shorter
    if truncate_target and len(fl) < len(tgt):
        tgt_eff = tgt[:len(fl)]
    else:
        tgt_eff = tgt

    obs = smith_waterman(tgt_eff, fl)

    # Null distribution: shuffle fl, recompute SW(tgt_eff, perm_fl)
    rng = random.Random(seed)
    fl_chars = list(fl)
    null_scores = []
    for _ in range(n_resample):
        rng.shuffle(fl_chars)
        null_scores.append(smith_waterman(tgt_eff, "".join(fl_chars)))

    mu = sum(null_scores) / len(null_scores)
    if len(null_scores) > 1:
        var = sum((x - mu) ** 2 for x in null_scores) / (len(null_scores) - 1)
        sigma = math.sqrt(var)
    else:
        sigma = 0.0

    z = (obs - mu) / sigma if sigma > 0 else 0.0
    k_ge = sum(1 for x in null_scores if x >= obs)
    p = (k_ge + 1) / (n_resample + 1)

    return SWZStat(
        fl=fl, target=tgt, target_eff=tgt_eff, obs=obs,
        mu=mu, sigma=sigma, z=z, p=p,
        n_resample=n_resample, n_sentences=len(fl),
        extractor=extractor,
    )


# ---------- Levenshtein z-stat (paper-faithful, kept for back-compat) ----------

def compute_lev_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
    strict: bool = False,
) -> LevZStat:
    """Compute Lev z-stat per paper Section 4.2.4.

    Null distribution: N random permutations of ℓ (preserves letter multiset).
    If ℓ is empty (no detectable sentences), returns z = 0.0.

    Args:
        strict: If True, use ``extract_first_letters_strict`` which filters
            single-letter / numbered-list heading cheats. Use this for RL
            reward to prevent reward hacking. Default False keeps paper-
            faithful detection behavior for evaluation comparisons.
    """
    extractor = extract_first_letters_strict if strict else extract_first_letters
    fl = extractor(text)
    tgt = target.lower()
    d_obs = _levenshtein(fl, tgt)

    if not fl:
        return LevZStat(fl="", target=tgt, d_obs=d_obs, mu=float(d_obs),
                        sigma=0.0, z=0.0, n_resample=0, n_sentences=0)

    rng = random.Random(seed)
    fl_list = list(fl)
    dists = []
    for _ in range(n_resample):
        rng.shuffle(fl_list)
        dists.append(_levenshtein("".join(fl_list), tgt))
    mu = sum(dists) / len(dists)
    if len(dists) > 1:
        var = sum((d - mu) ** 2 for d in dists) / (len(dists) - 1)
    else:
        var = 0.0
    sigma = math.sqrt(var)
    z = (mu - d_obs) / sigma if sigma > 0 else 0.0
    return LevZStat(fl=fl, target=tgt, d_obs=d_obs, mu=mu, sigma=sigma, z=z,
                    n_resample=n_resample, n_sentences=len(fl))
