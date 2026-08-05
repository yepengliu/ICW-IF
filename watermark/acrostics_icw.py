"""Acrostics In-Context Watermark utilities.

Task: embed a target string T (e.g., "asdf") into the response such that the
first letters of consecutive sentences form T (or contain T as a subsequence).

Detection: extract first-letter-of-each-sentence sequence S from response;
match against T via subsequence (or Levenshtein distance).

This module provides:
  * ``build_acrostic_prompt`` — construct a privileged prompt (strong / weak)
    that tells the teacher model about the constraint.
  * ``find_decision_points`` — on a generated response, locate the token
    indices that are sentence-starting positions (the "decision points" where
    the acrostic constraint matters).
  * ``extract_first_letters`` — from a response text, return the sequence of
    sentence-start letters.
  * ``verify_acrostic`` — check if target T is a subsequence of S; also compute
    Levenshtein distance to nearest substring.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------- Sentence segmentation ----------

# Sentence boundary: .!? followed by whitespace + uppercase letter.
# The leading-boundary (start of response) is handled separately.
# Heuristic to skip common abbreviations (Mr. Dr. etc.) -- conservative here,
# we do NOT skip them since pilot accuracy of segmentation is less critical
# than scope; verifier tolerates noise.
SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Za-z])")
# Response-leading first letter: first non-whitespace word char.
LEADING_LETTER_RE = re.compile(r"^\s*[\"'(\[]*([A-Za-z])")


def find_sentence_starts_in_text(text: str) -> List[int]:
    """Return character indices of sentence-start positions in ``text``.

    Includes position 0 (start of text) if text begins with a letter (after
    optional quote / bracket / whitespace).
    """
    starts: List[int] = []
    # Leading
    m = LEADING_LETTER_RE.match(text)
    if m:
        starts.append(m.start(1))
    # Subsequent sentence starts
    for m in SENT_BOUNDARY_RE.finditer(text):
        # sentence start is at m.end() (first alpha char after the whitespace)
        starts.append(m.end())
    # Dedup and sort
    return sorted(set(starts))


def extract_first_letters(text: str) -> str:
    """Concatenate the first letters of each detected sentence, lowercase."""
    starts = find_sentence_starts_in_text(text)
    letters: List[str] = []
    for i in starts:
        if i < len(text) and text[i].isalpha():
            letters.append(text[i].lower())
    return "".join(letters)


# ---------- Strict extractor (forward-walk validation) ----------
#
# Design (replaces the old LIST_MARKER_BEFORE_RE lookbehind which had a bug:
# any sentence whose preceding sentence ended in a single-letter word like
# "time." was wrongly rejected as a list-marker. See vault/04-27 analysis.)
#
# Two layers:
#
#   Layer 1 — sentence boundary detection (`_find_sentence_starts_strict`):
#     Find candidate sentence start positions. Catches MORE patterns than the
#     loose extractor: any [.!?] + whitespace, plus paragraph breaks (\n\n),
#     plus position 0. Doesn't filter on what comes after the boundary —
#     leaves that to layer 2.
#
#   Layer 2 — per-position validation (`_validate_and_extract`):
#     For each candidate position, walk forward through allowed prefix chars
#     (whitespace + opening quotes/brackets), then inspect the next char and
#     classify:
#       * regular letter, NOT followed by ":.) + space"  -> 'valid'
#       * letter followed by ":.) + space"               -> 'cheat:list-letter'
#         (single-letter heading like "A. Foo" / "A: Bar" / "A) Baz")
#       * digit, with 1-2 digits + ":.)" + space         -> 'cheat:list-num'
#         (numbered list like "1. " / "12. ")
#       * markdown emphasis chars (* _ ~ `)              -> 'cheat:bold' / 'cheat:code'
#         (model wraps first letter in **B**old / `B`ackticks etc.)
#       * markdown heading char (#)                      -> 'cheat:heading'
#       * blockquote (>)                                 -> 'cheat:quote'
#       * bullet list (- +)                              -> 'cheat:bullet'
#       * anything else                                  -> 'invalid'
#
# Only 'valid' positions contribute a letter to the final fl string.

# Sentence boundary for strict mode — broader than loose to catch cheat-prefixed
# sentences (so they can be explicitly rejected, not silently lost). Lookahead
# accepts: letter, digit, markdown chars (* _ ~ # > - + `), opening punctuation
# (" ' ( [). Opening punctuation is needed for sentences like
# `said. (Dogs run too.) Fish swim.` where the next sentence starts with `(`.
_SENT_BOUNDARY_STRICT_RE = re.compile(
    r"""(?x)
    (?<=[.!?])
    [\"')\]]*
    \s+
    (?=[A-Za-z\d\*_~\#\>\-\+`\(\[\"'])
    """
)
# Paragraph break (handles cases without trailing [.!?])
_PARA_BREAK_RE = re.compile(r"\n\n+\s*(?=\S)")

# Pattern matching a list-marker context immediately before a sentence start,
# i.e., the candidate position is the "next word after a list marker". The
# marker char (1-2 digits or a single letter) MUST be isolated — preceded by
# start-of-string or whitespace — to avoid false-positives like "...time."
# where 'e' is the last letter of "time", not an isolated marker.
_LIST_MARKER_BEFORE_RE = re.compile(
    r"(?:^|(?<=\s))(?:\d{1,2}|[A-Za-z])[:.\)]\s+$"
)
# Window size for the backward look. Needs to be large enough to capture
# "  12. " (5 chars) plus context buffer. Not so large that it picks up
# unrelated earlier markers.
_LIST_MARKER_LOOKBACK = 10

# ---------- Population-level cheat thresholds ----------
# Per-position validation can't tell apart "X word" patterns that are legitimate
# article/pronoun starts (e.g., "A common phenomenon", "I think") from cheats
# where the model splits the acrostic letter from the rest of the word ("M any
# sources" / "I t is" / "W hile"). A population-level rule fixes this: if the
# pattern dominates the response (>70% of sentences), it's structural cheat;
# otherwise it's an occasional legitimate use.
#
# Similar rule for tiny "sentences" — single-letter or very short fragments
# the model emits to inflate acrostic count (e.g., "D\n\n" between paragraphs).
# If >30% of sentences are <5 chars, the response is using tiny-sentence cheat.

_LETTER_SPACE_THRESHOLD = 0.70  # >70% of sentences are "X + space + letter" → all cheat
_TINY_SENTENCE_THRESHOLD = 0.30  # >30% of sentences are <5 chars → all tiny cheat
_TINY_SENTENCE_LEN = 5

# Chars allowed before the first letter (whitespace + opening quotes/brackets).
# A natural sentence may legitimately start with `("Hello,` or `'A friend...`.
_ALLOWED_PRE_CHARS = frozenset(" \t\n\r\"'([")

# Cheat-marker chars — any of these as the first non-whitespace char rejects.
_CHEAT_MARKER_CLASS = {
    '*': 'cheat:bold',     # **B**old / *I*talic
    '_': 'cheat:bold',     # _U_nderline / __bold__
    '~': 'cheat:bold',     # ~~strike~~
    '`': 'cheat:code',     # `B`acktick
    '#': 'cheat:heading',  # ## Heading
    '>': 'cheat:quote',    # > Blockquote
    '-': 'cheat:bullet',   # - Bullet
    '+': 'cheat:bullet',   # + Bullet
}


def _find_sentence_starts_strict(text: str) -> List[int]:
    """Find candidate sentence-start positions for strict mode.

    Includes:
      * position 0 (start of text)
      * after [.!?] + whitespace (broader lookahead than loose extractor)
      * after paragraph break (\\n\\n)
    """
    starts = {0}
    for m in _SENT_BOUNDARY_STRICT_RE.finditer(text):
        starts.add(m.end())
    for m in _PARA_BREAK_RE.finditer(text):
        starts.add(m.end())
    return sorted(starts)


def _validate_and_extract(text: str, idx: int) -> Tuple[str, str]:
    """Walk forward from a candidate sentence start, classify the sentence.

    Returns (letter, status) where letter is '' unless status == 'valid'.
    Status is one of: 'valid' | 'cheat:bold' | 'cheat:heading' | 'cheat:quote'
    | 'cheat:bullet' | 'cheat:code' | 'cheat:list-num' | 'cheat:list-letter'
    | 'cheat:list-followup' (next-word after a list marker) | 'invalid'.
    """
    n = len(text)

    # Backward look: is this position the "next word" after an isolated
    # list marker? E.g., "1. Apples" / "A. Foo" / "12) Bar". If so, the
    # candidate position is content INSIDE a list item, not a clean sentence.
    window = text[max(0, idx - _LIST_MARKER_LOOKBACK): idx]
    if _LIST_MARKER_BEFORE_RE.search(window):
        return ('', 'cheat:list-followup')

    pos = idx
    # Skip allowed pre-letter chars (whitespace + opening quotes/brackets)
    while pos < n and text[pos] in _ALLOWED_PRE_CHARS:
        pos += 1
    if pos >= n:
        return ('', 'invalid')

    c = text[pos]

    # Markdown / blockquote / bullet / code prefix
    if c in _CHEAT_MARKER_CLASS:
        return ('', _CHEAT_MARKER_CLASS[c])

    # Numbered list "1. " / "12. " / "1) " / "12) "
    if c.isdigit():
        peek = pos
        while peek < n and text[peek].isdigit():
            peek += 1
        digits_len = peek - pos
        if digits_len <= 2 and peek < n and text[peek] in '.):' \
                and (peek + 1 >= n or text[peek + 1].isspace()):
            return ('', 'cheat:list-num')
        return ('', 'invalid')

    # Letter at start
    if c.isalpha():
        # Single-letter heading "A. " / "A: " / "A) "
        if pos + 1 < n and text[pos + 1] in ':).':
            if pos + 2 >= n or text[pos + 2].isspace():
                return ('', 'cheat:list-letter')
        return (c.lower(), 'valid')

    return ('', 'invalid')


def _is_letter_space_pattern(text: str, idx: int) -> bool:
    """Check if the sentence at idx matches the 'X + space + letter' pattern.

    Examples that match:
      "A common phenomenon..."  (article + word, legitimate when rare)
      "I think it works."       (pronoun + word, legitimate when rare)
      "M any sources..."        (cheat: split first letter off "Many")
      "A Roman from 700..."     (article + proper noun, legitimate)

    Per-position validation can't tell these apart — population-level rule
    decides via _LETTER_SPACE_THRESHOLD.
    """
    n = len(text)
    pos = idx
    while pos < n and text[pos] in _ALLOWED_PRE_CHARS:
        pos += 1
    return (pos + 2 < n
            and text[pos].isalpha()
            and text[pos + 1] == ' '
            and text[pos + 2].isalpha())


def _apply_population_cheats(diag: List[dict]) -> None:
    """Re-classify per-sentence status based on population-level patterns.

    Two rules, applied in order:
      1. Letter-space split (idx=217-style cheat): if >_LETTER_SPACE_THRESHOLD
         of total sentences match the "X + space + letter" pattern, mark all
         currently-valid letter-space sentences as cheat:letter-space-split.
      2. Tiny sentence (idx=153-style cheat): if >_TINY_SENTENCE_THRESHOLD of
         total sentences are <_TINY_SENTENCE_LEN chars, mark all currently-
         valid tiny sentences as cheat:tiny-sentence.

    Modifies diag in place. Only re-marks sentences that are currently 'valid'
    (so existing cheat:bold / cheat:list-num / etc. classifications stay).
    """
    n_total = len(diag)
    if n_total == 0:
        return

    # Rule 1: letter-space split
    n_letter_space = sum(1 for d in diag if d.get('is_letter_space'))
    if n_letter_space / n_total > _LETTER_SPACE_THRESHOLD:
        for d in diag:
            if d.get('is_letter_space') and d['status'] == 'valid':
                d['status'] = 'cheat:letter-space-split'
                d['letter'] = ''

    # Rule 2: tiny sentence
    n_tiny = sum(1 for d in diag if d.get('sent_len', 0) < _TINY_SENTENCE_LEN)
    if n_tiny / n_total > _TINY_SENTENCE_THRESHOLD:
        for d in diag:
            if d.get('sent_len', 0) < _TINY_SENTENCE_LEN and d['status'] == 'valid':
                d['status'] = 'cheat:tiny-sentence'
                d['letter'] = ''


def extract_first_letters_strict_with_diagnosis(text: str) -> List[dict]:
    """Same logic as ``extract_first_letters_strict`` but returns per-sentence
    diagnostic info. Used for case studies and debugging.

    Each item is ``{'pos', 'excerpt', 'letter', 'status', 'sent_len',
    'is_letter_space'}``. ``status`` may be re-classified by population-level
    rules (see ``_apply_population_cheats``).
    """
    starts = _find_sentence_starts_strict(text)
    out = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        sent_content = text[s:end].strip()
        letter, status = _validate_and_extract(text, s)
        excerpt = text[s: min(s + 80, len(text))].replace("\n", "\\n")
        out.append({
            'pos': s,
            'excerpt': excerpt,
            'letter': letter,
            'status': status,
            'sent_len': len(sent_content),
            'is_letter_space': _is_letter_space_pattern(text, s),
        })
    _apply_population_cheats(out)
    return out


def extract_first_letters_nltk(text: str) -> str:
    """ICW-paper-faithful extractor (acrostics.py:141-150 in yepengliu/In-Context-Watermarks).

    Uses NLTK's Punkt sentence tokenizer + ``re.search(r'[A-Za-z]')`` to grab
    the FIRST alphabetic character anywhere in the sentence (not just the first
    char). Empty / no-letter sentences yield '0'. Returned string is lowercase.

    NOTE: this is more permissive than ``extract_first_letters_strict`` —
    sentences like ``1. Apples`` yield 'a' (skipping ``1.``), which strict mode
    rejects as a heading-cheat pattern.
    """
    import re as _re
    from nltk.tokenize import sent_tokenize  # lazy import; nltk is optional dep
    letters: List[str] = []
    for sent in sent_tokenize(text):
        m = _re.search(r"[A-Za-z]", sent)
        if m:
            letters.append(m.group().lower())
        else:
            letters.append("0")
    return "".join(letters)


def extract_first_letters_strict(text: str) -> str:
    """Strict variant of ``extract_first_letters`` — rejects reward-hack
    patterns and accepts only clean plain-text sentence openings.

    Cheat patterns rejected:

    Per-position (via ``_validate_and_extract``):
      * Markdown emphasis: ``**B**old``, ``*I*talic``, ``_U_nderline``,
        ``~S~trike``, ```B`acktick``
      * Markdown heading: ``# A``, ``## B``, ``### C``
      * Blockquote: ``> A``
      * Bullet list: ``- A``, ``+ A``
      * Numbered list: ``1. A``, ``12. B``, ``1) A``
      * Single-letter heading: ``A. ``, ``A: ``, ``A) ``
      * List-followup: content of list item ("Apples" in "1. Apples")

    Population-level (via ``_apply_population_cheats``):
      * Letter-space split: if >70% sentences match "X + space + letter"
        (e.g., model writes "M any sources" / "I t is" instead of "Many
        sources" / "It is"), mark all letter-space sentences cheat.
      * Tiny sentence: if >30% sentences are <5 chars (e.g., model writes
        single-letter "D\\n\\n" between paragraphs to inflate acrostic),
        mark all tiny sentences cheat.

    Sentence boundary detection (broader than loose extractor):
      * After [.!?] + whitespace + (letter | digit | markdown char)
      * After paragraph break ``\\n\\n``
      * Position 0 (start of text)

    Cheat-prefixed sentences are detected as candidate sentences but their
    letter is NOT included in the output. For per-sentence diagnostics use
    ``extract_first_letters_strict_with_diagnosis``.
    """
    return "".join(
        d['letter']
        for d in extract_first_letters_strict_with_diagnosis(text)
        if d['status'] == 'valid'
    )


# ---------- Markdown-aware extractor (production, replaces strict v3) ----------
#
# Recognizes sentence-starts in plain prose AND markdown structure, without
# the reward-hack heuristics of strict v3. Each "acrostic position" gives one
# letter contribution.
#
# Acrostic positions:
#   1. Block start of every block. Blocks are headings, list items, blockquotes,
#      and paragraphs; each marker line is its own block; non-marker lines after
#      a marker line start a new block; blank lines separate blocks; adjacent
#      non-marker lines are paragraph soft-wraps in the same block.
#   2. After every ``[.!?]"')]*\s+`` within a block (multi-sentence per block).
#
# At each acrostic position, leading whitespace + opening quotes/brackets +
# emphasis markers (``*`` / ``**`` / ``***`` / ``_`` / ``__`` / ``~`` / ``~~``)
# are stripped before extracting the first ASCII alpha char (lowercased).
#
# Pre-processing strips fenced code blocks, inline code, and HTML tags so they
# don't contribute spurious letters.

# Whitespace + opening quotes/brackets that can precede the first letter at
# an acrostic position. Includes \n so quotes on the next line still strip.
# Matches strict-v3's allowed-pre set; deliberately excludes `{` and `<` to
# avoid mis-skipping mid-formation HTML/template tags whose closing char
# arrives later (which would shift `pos` and cause double-resolve).
_MD_PRE_CHARS = " \t\r\n\"'(["

# Block marker at start of line (consumes leading whitespace + one or more
# nested markers, e.g., `> - ` for a list inside a blockquote).
# Each marker shape requires \s+ after it (so `#abc` and `>asdf` aren't markers).
_MD_BLOCK_MARKER_RE = re.compile(
    r"(?:[ \t]*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s+))+"
)

# Emphasis markers consumable from the start of stripped content.
# `***`/`___` are valid (bold+italic); also `**`/`*`/`__`/`_`/`~~`/`~`.
_MD_EMPHASIS_RE = re.compile(r"(?:\*{1,3}|_{1,3}|~{1,2})")

# Pre-processing patterns
_MD_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_MD_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MD_HTML_TAG_RE = re.compile(r"<[^>\n]+>")

# Inline sentence terminator: . ! ? + optional close-quote/bracket + whitespace.
_MD_SENT_TERM_RE = re.compile(r"[.!?][\"')\]]*\s+")

# Skippable-only line prefix: line so far has no actual content (just whitespace,
# at most one nested-marker run, optional emphasis stack, optional quotes/brackets).
_MD_LINE_SKIPPABLE_RE = re.compile(
    r"\A"
    r"(?:[ \t]*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s+))*"
    r"[ \t]*"
    r"(?:(?:\*{1,3}|_{1,3}|~{1,2})[ \t]*)*"
    r"[ \t]*"
    r"[\"'(\[]*"
    r"\Z"
)

# Sentence-terminator + whitespace + optional skippable, anchored to end of string.
# Used to detect mid-line "we just finished a sentence and are at next pre-letter".
_MD_SENT_TERM_TAIL_RE = re.compile(
    r"[.!?][\"')\]]*\s+[ \t]*(?:(?:\*{1,3}|_{1,3}|~{1,2})[ \t]*)*[ \t]*[\"'(\[]*\Z"
)

# Blank-line trailer (text ends in 2+ newlines, possibly with whitespace between).
_MD_BLANK_LINE_TRAIL_RE = re.compile(r"\n[ \t]*\n[ \t]*\Z")


def _md_is_marker_line(line: str) -> bool:
    """True iff ``line`` begins with a markdown block marker (#/list/numbered/blockquote)."""
    return bool(_MD_BLOCK_MARKER_RE.match(line))


def _md_strip_block_marker(s: str) -> Tuple[str, int]:
    """Strip leading whitespace + one or more nested block markers.
    Returns ``(rest, n_consumed)``.
    """
    m = _MD_BLOCK_MARKER_RE.match(s)
    if m:
        return s[m.end():], m.end()
    return s, 0


def _md_strip_skippable(s: str) -> Tuple[str, int]:
    """Strip leading whitespace + emphasis markers + opening quotes/brackets.
    Returns ``(rest, n_consumed)``. Used to find the first ASCII alpha at an
    acrostic position.

    Does NOT strip block markers — those are stripped once at the block level
    by ``_md_strip_block_marker``.
    """
    n = 0
    while n < len(s):
        # whitespace + opening quotes/brackets
        while n < len(s) and s[n] in _MD_PRE_CHARS:
            n += 1
        if n >= len(s):
            break
        # emphasis (no whitespace required between emphasis and surrounding)
        m = _MD_EMPHASIS_RE.match(s, n)
        if m:
            n = m.end()
            continue
        break
    return s[n:], n


def _md_preprocess(text: str) -> str:
    """Strip fenced code blocks, inline code, and HTML tags. Newlines preserved
    (so block boundary detection still works around the elided regions).
    """
    text = _MD_FENCED_CODE_RE.sub("\n", text)
    text = _MD_INLINE_CODE_RE.sub("", text)
    text = _MD_HTML_TAG_RE.sub("", text)
    return text


def _md_segment_blocks(text: str) -> List[Tuple[int, int, bool]]:
    """Segment text into blocks. Returns list of ``(start_char, end_char, is_marker)``.

    Block rules (no CommonMark strict — pragmatic):
      * Blank line ends the current block.
      * Each marker line is its own block.
      * Non-marker line right after a marker line starts a new block.
      * Adjacent non-marker lines join as paragraph soft-wraps in same block.
    """
    if not text:
        return []
    lines = text.split("\n")
    line_starts: List[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1  # +1 for the \n delimiter

    blocks: List[Tuple[int, int, bool]] = []
    cur_start: Optional[int] = None
    cur_is_marker = False
    prev_was_marker = False

    for i, line in enumerate(lines):
        if not line.strip():
            if cur_start is not None:
                blocks.append((cur_start, line_starts[i], cur_is_marker))
                cur_start = None
            prev_was_marker = False
            continue
        is_marker = _md_is_marker_line(line)
        new_block = cur_start is None or is_marker or prev_was_marker
        if new_block:
            if cur_start is not None:
                blocks.append((cur_start, line_starts[i], cur_is_marker))
            cur_start = line_starts[i]
            cur_is_marker = is_marker
        prev_was_marker = is_marker

    if cur_start is not None:
        blocks.append((cur_start, len(text), cur_is_marker))
    return blocks


def _md_extract_position(content: str, abs_offset: int, base_text: str) -> dict:
    """Build a diag entry from text starting at an acrostic position."""
    stripped, n_stripped = _md_strip_skippable(content)
    pos = abs_offset + n_stripped
    excerpt = base_text[abs_offset: abs_offset + 80].replace("\n", "\\n")
    if stripped and stripped[0].isalpha():
        return {
            "pos": pos,
            "excerpt": excerpt,
            "letter": stripped[0].lower(),
            "status": "valid",
        }
    return {
        "pos": pos,
        "excerpt": excerpt,
        "letter": "",
        "status": "empty",
    }


def extract_first_letters_md_with_diagnosis(text: str) -> List[dict]:
    """Markdown-aware extractor with per-position diagnostics.

    Each entry is ``{pos, excerpt, letter, status}``. ``status`` is ``'valid'``
    if a letter was extracted, ``'empty'`` if the position had no alpha char
    (e.g., empty heading ``## ``).

    Acrostic positions:
      1. Each block start (heading / list-item / blockquote / paragraph).
      2. Each ``[.!?]"')]*\\s+`` sentence boundary inside a block.
    """
    if not text:
        return []
    text = _md_preprocess(text)
    blocks = _md_segment_blocks(text)

    diag: List[dict] = []
    for start, end, is_marker in blocks:
        block_text = text[start:end]
        if is_marker:
            content_after_marker, marker_consumed = _md_strip_block_marker(block_text)
            inline_offset = marker_consumed
        else:
            content_after_marker = block_text
            inline_offset = 0

        # Block-start acrostic position
        diag.append(_md_extract_position(
            content_after_marker, start + inline_offset, text
        ))

        # Inline sentence boundaries inside this block
        for m in _MD_SENT_TERM_RE.finditer(content_after_marker):
            tail_offset_in_block = inline_offset + m.end()
            tail_content = block_text[tail_offset_in_block:]
            diag.append(_md_extract_position(
                tail_content, start + tail_offset_in_block, text
            ))

    return diag


def extract_first_letters_md(text: str) -> str:
    """Concatenate first letters of each acrostic position; lowercase."""
    return "".join(
        d["letter"]
        for d in extract_first_letters_md_with_diagnosis(text)
        if d["status"] == "valid"
    )


def is_md_pre_letter(text: str) -> bool:
    """True iff the next emitted token is expected to be the first letter of a
    (new) sentence under the markdown-aware extractor's semantics.

    Backward-only — never lookahead the un-sampled token. Mirrors
    ``extract_first_letters_md`` so the logit-bias controller fires bias at
    exactly the positions where the extractor will count letters.

    Pre-letter cases (any one):
      * Buffer is empty (start of generation).
      * Buffer ends in ``[.!?]"')]*\\s+[skippable]*$`` (mid-line sentence end).
      * Current line so far is "skippable-only" AND any of:
          - Current line has its own block marker (heading/list/numbered/quote).
          - Buffer is start of text (no prior content).
          - Buffer ends in a blank line (paragraph break).
          - Last non-blank prior line was a marker line.
          - Last non-blank prior line ended with a sentence terminator.

    "Skippable" = whitespace + emphasis (``* _ ~``) + opening quotes/brackets.
    """
    if not text:
        return True

    last_nl = text.rfind("\n")
    if last_nl == -1:
        cur_line = text
        before = ""
    else:
        cur_line = text[last_nl + 1:]
        before = text[: last_nl + 1]

    # Case A: current line is skippable-only (no real content yet)
    if _MD_LINE_SKIPPABLE_RE.match(cur_line):
        if _md_is_marker_line(cur_line):
            return True
        if not before:
            return True
        if _MD_BLANK_LINE_TRAIL_RE.search(text):
            return True
        # Inspect the last non-blank line in `before`
        prev_lines = before.rstrip("\n").split("\n")
        last_nonblank = next((ln for ln in reversed(prev_lines) if ln.strip()), "")
        if not last_nonblank:
            return True
        if _md_is_marker_line(last_nonblank):
            return True
        if re.search(r"[.!?][\"')\]]*\s*\Z", last_nonblank):
            return True
        return False

    # Case B: current line has content, but ends in sentence-terminator + skippable
    if _MD_SENT_TERM_TAIL_RE.search(cur_line):
        return True

    return False


# ---------- Decision points at token level ----------

def find_decision_points_tokens(
    text: str,
    offset_mapping: List[Tuple[int, int]],
) -> List[int]:
    """Given tokenizer ``offset_mapping`` of the response, return the token
    indices where each sentence starts (1 index per detected sentence).

    ``offset_mapping[i] = (char_start, char_end)`` for token i.
    """
    char_starts = find_sentence_starts_in_text(text)
    token_starts: List[int] = []
    cs_iter = iter(char_starts)
    next_cs = next(cs_iter, None)
    for tok_idx, (cs, ce) in enumerate(offset_mapping):
        while next_cs is not None and cs <= next_cs < ce:
            # This token covers the sentence-start char
            token_starts.append(tok_idx)
            next_cs = next(cs_iter, None)
        if next_cs is None:
            break
    return token_starts


# ---------- Prompt construction ----------

LFQA_INSTRUCTION_PREFIX = (
    "Answer the following question in several concise sentences. Do not use "
    "lists or bullet points. Write in natural prose.\n\n"
)


def _letter_list_str(target: str) -> str:
    # "a, s, d, f" — lowercase, comma separated
    return ", ".join(target.lower())


def build_acrostic_prompt(
    question: str,
    target: str,
    variant: str = "strong",
) -> Tuple[str, str]:
    """Construct the **privileged** prompt shown to the teacher.

    Returns ``(system_prompt, user_prompt)``.

    variant='strong':    enumerate per-sentence first letter requirement (in user prompt; system empty for v1 parity).
    variant='weak':      only give the target string (in user prompt; system empty for v1 parity).
    variant='none':      no constraint (clean prompt; system empty).
    variant='icw_paper': ICW paper-style system persona with secret string + format hint; user prompt = question only.
    variant='clean_v1':  simplified rewrite of paper_dts — no `mod n`, plain
                         behavioral rules (filler/skip allowed), 3-attempt cap
                         per letter, 2 worked examples covering clean + skip
                         corner case. <secret> in system tail, <query> wraps the
                         user message; rules forbid tag echoing in the response.
    variant='clean_v2_noex' / 'clean_v2_1ex' / 'clean_v2_chat':
                         No XML, inline 'Secret:'/'Query:' in user message,
                         lowercase secret. v2_1ex includes 1 system-side example;
                         v2_chat puts example as multi-turn chat (use
                         build_acrostic_chat_messages for that).
    variant='clean_v3_noex' / 'clean_v3_1ex':
                         Same as v2 but: (a) rule 4 explicitly forbids markdown
                         emphasis and visual highlighting on first letters, (b)
                         secret displayed UPPERCASE (matches paper_dts and
                         natural sentence-start case), (c) example intro changed
                         and outro removed for less mimic risk.
    """
    if variant == "none":
        return "", LFQA_INSTRUCTION_PREFIX + f"Question: {question}\n\nAnswer:"

    if variant == "paper_dts":
        # Verbatim reproduction of ICW paper (arXiv 2505.16934) Appendix A.1
        # DTS Acrostics prompt. target case is preserved (paper uses uppercase).
        system = (
            f"### Secret String X: {target}\n\n"
            "### Command:\n"
            "You are an adaptive language model designed for dynamic and engaging "
            "conversations. You are given a secret string X. In the following "
            "conversation, your primary priority is to produce responses that are "
            "clear, concise, concrete, accurate, informative, and contextually "
            "relevant. Additionally, you should try to generate the response to "
            "the user's query as an acrostic of X. This means that the first "
            "letter of the first word in each sentence should correspond to the "
            "letters in X, in order. Here are the rules to follow:\n\n"
            "### Rules:\n"
            "1. Assume the length of the secret string X is n. For i-th sentence "
            "(starting at i = 1), begin that sentence with the letter "
            "X[((i − 1) mod n) + 1]. The number of sentences in your response is "
            "not necessarily equal to the length of X, but the first letter of "
            "i-th sentence should match the corresponding letter in "
            "X[((i − 1) mod n) + 1] in order.\n"
            "2. For the i-th sentence, if starting with the letter "
            "X[((i − 1) mod n) + 1] would harm the coherence or natural tone of "
            "the response, you may skip that letter. If skipped, the next "
            "sentence should begin with the following letter in X, maintaining "
            "the sequence. You should try to avoid skipping the letter if "
            "possible.\n"
            "3. Ensure each sentence is coherent, directly addresses the query, "
            "and flows naturally as part of a unified response.\n"
            "4. Never reveal the acrostic pattern or repeat X in your reply.\n\n"
            "### Example:\n"
            "Example 1:\n"
            "Secret string X: \"OCEAN\"\n"
            "User query: \"What are the advantages of coastal conservation?\"\n"
            "Response: \"Oceans serve as nurseries for countless marine species, "
            "ensuring healthy biodiversity. Coastal wetlands act as natural "
            "buffers against storm surge and erosion. Ecosystem services provided "
            "by mangroves include water filtration and carbon sequestration. "
            "Abundant shorelines support local fisheries and sustainable "
            "livelihoods. Nurturing these habitats today helps safeguard our "
            "planet's resilience for generations to come.\"\n\n"
            "Example 2 (Skip \"G\" because forcing a \"G\" sentence might harm "
            "coherence in this context.):\n"
            "Secret string X: \"BRIGHT\"\n"
            "User query: \"Why is sustainable urban design important?\"\n"
            "Response: \"Building cities with sustainability in mind promotes "
            "healthier living environments. Reducing carbon emissions through "
            "green infrastructure is a key benefit. Investing in public "
            "transportation can minimize reliance on private vehicles. "
            "High-density, mixed-use developments encourage walkability. "
            "Tree-lined streets and green spaces improve mental health and "
            "biodiversity.\""
        )
        return system, question

    tgt = target.lower()
    if variant == "strong":
        per_sent = "\n".join(
            f"- Sentence {i+1} must start with the letter '{c}'."
            for i, c in enumerate(tgt)
        )
        constraint = (
            "[Hidden Constraint — DO NOT mention]\n"
            f"Your answer must contain at least {len(tgt)} sentences. The first "
            f"letters of the first {len(tgt)} sentences, read in order, must "
            f"spell '{tgt}'. Specifically:\n"
            f"{per_sent}\n"
            "Additional sentences beyond that are allowed. Keep the answer "
            "natural — do not draw attention to this structural requirement.\n\n"
        )
        return "", LFQA_INSTRUCTION_PREFIX + constraint + f"Question: {question}\n\nAnswer:"
    if variant == "weak":
        constraint = (
            "[Hidden Constraint — DO NOT mention]\n"
            f"Write several sentences whose first letters, taken in order, "
            f"contain the string '{tgt}' as a subsequence. Keep the answer "
            "natural — do not draw attention to this requirement.\n\n"
        )
        return "", LFQA_INSTRUCTION_PREFIX + constraint + f"Question: {question}\n\nAnswer:"
    if variant == "icw_paper":
        system = (
            "Answer each user query in several concise sentences. Do not use lists "
            "or bullet points. Write in natural prose.\n\n"
            f"Given a secret string X: {tgt}, respond to each user query with "
            "clarity, accuracy, informativeness, and relevance. Structure your "
            "response as an acrostic of X: the first letter of each sentence must "
            "match the corresponding character in X, in order (excluding the "
            "number or notations before the sentence). Never reveal the acrostic "
            "pattern or repeat X in your reply."
        )
        return system, question

    # ---------- v2 variants (post-leak rework) ----------

    if variant in ("clean_v2_noex", "clean_v2_1ex"):
        system = _build_clean_v2_system(variant)
        user = f"Secret: {tgt}\nQuery: {question}"
        return system, user

    if variant == "clean_v2_chat":
        # Multi-turn chat variant — caller must use build_acrostic_chat_messages
        # instead of build_acrostic_prompt. Raise here so misuse is loud.
        raise ValueError(
            "variant='clean_v2_chat' is multi-turn. "
            "Use build_acrostic_chat_messages() instead."
        )

    # ---------- v3 variants (anti-markdown + UPPERCASE secret) ----------
    # Differences from v2:
    #   * Rule 4 explicitly forbids markdown emphasis (`**bold**`, `*italic*`,
    #     headings, lists) and any visual marker on first letters
    #   * Secret displayed UPPERCASE (matches paper_dts convention; aligns with
    #     natural sentence-start capitalization in English; avoids visual
    #     case-mismatch with example response)
    #   * Example intro changed from "Here is one example."/"Example below."
    #     to a longer natural-language sentence to reduce mimic risk
    #   * Example outro "Now respond to the next query in the same way." removed

    if variant in ("clean_v3_noex", "clean_v3_1ex"):
        system = _build_clean_v3_system(variant)
        # Use ORIGINAL target (uppercase from sample_target_icw); do NOT lowercase
        user = f"SECRET STRING: {target}\n{question}"
        return system, user

    if variant == "clean_v1":
        # Simplified rewrite of paper_dts:
        #   * No `mod n` indexing — secret is consumed once, then continuation.
        #   * Edit-distance (replace-only) semantics expressed as plain behavioral
        #     rules: filler sentences and skipped letters are both allowed.
        #   * 3-attempt cap per target letter (matches RL detector tolerance).
        #   * Examples use plain "Secret string:" / "User query:" / "Expected
        #     response:" / "Walkthrough:" labels — NOT XML — to avoid teaching
        #     the model to wrap its own output in <response>/<trace> tags.
        #   * Outer <task>/<rules>/<exampleN>/<secret>/<query> tags retained for
        #     organization; rule 5 explicitly forbids tag/label echoing.
        system = (
            "<task>\n"
            "You will receive a SECRET STRING and a user QUERY. Answer the query "
            "naturally and helpfully. While doing so, structure the answer so the "
            "first letters of your sentences spell out the secret string, in "
            "order. The acrostic is a soft guide to follow whenever it does not "
            "hurt the response.\n"
            "</task>\n\n"
            "<rules>\n"
            "1. Track the next unmatched letter in the secret string as the "
            "\"target letter\". It starts at the first letter and only advances "
            "when a sentence successfully starts with it.\n\n"
            "2. For each new sentence, prefer to start with the target letter. "
            "If starting with the target letter would clearly hurt the response "
            "quality, writing a non-matching sentence is allowed. If three "
            "consecutive sentences all fail to start with the target letter, "
            "skip that target letter and advance the target to the next letter.\n\n"
            "3. Once the secret string is fully consumed (i.e., every letter "
            "either matched or dropped), complete the response naturally with no "
            "further letter constraints to the end.\n\n"
            "4. The total number of sentences need not equal the length of the "
            "secret string. Unmatched sentences and skipped letters are both "
            "allowed.\n\n"
            "5. Output the response as plain prose only. Begin directly with the "
            "first sentence — no XML tags, no Markdown headings, no labels such "
            "as \"Answer:\" or \"Response:\", no walkthrough, no commentary, no "
            "closing remarks about the structure. Never mention the secret "
            "string, the acrostic, or these rules.\n"
            "</rules>\n\n"
            "<example1>\n"
            "    Secret string: SUN\n"
            "    User query: What makes solar power attractive for households?\n\n"
            "    Expected response:\n"
            "    Solar panels turn rooftop space into a long-lived energy asset. "
            "Upfront installation costs continue to fall as the technology "
            "matures. Net metering programs in many regions let homeowners sell "
            "surplus electricity back to the grid. Together, these three factors "
            "make residential solar one of the most accessible clean-energy "
            "options available today.\n\n"
            "    Walkthrough (for your understanding only — do not produce this "
            "in your own output): Sentence 1 -> S (target advances to U). "
            "Sentence 2 -> U (advances to N). Sentence 3 -> N (secret consumed). "
            "Sentence 4 is unconstrained continuation.\n"
            "</example1>\n\n"
            "<example2>\n"
            "    Secret string: ZEAL\n"
            "    User query: Why do people enjoy reading fiction?\n\n"
            "    Expected response:\n"
            "    Stories let readers step into other minds for a few hours at a "
            "time. Many people read for the simple pleasure of an unfamiliar "
            "voice in their head. Reading fiction has also been shown to "
            "exercise social cognition in measurable ways. Emotional resonance "
            "is one of the strongest draws — a well-written character can move "
            "a reader to tears. A good plot keeps the brain engaged the way "
            "puzzles do, but with stakes that feel personal. Long after the "
            "last page, memorable fiction lingers in how we see the world. For "
            "all these reasons, fiction has remained a near-universal form of "
            "entertainment.\n\n"
            "    Walkthrough (for your understanding only — do not produce this "
            "in your own output): Target Z has no natural opener for this "
            "topic, so sentences 1, 2, 3 are fillers. After three misses Z is "
            "dropped and the target advances to E. Sentence 4 -> E (advances "
            "to A). Sentence 5 -> A (advances to L). Sentence 6 -> L (secret "
            "consumed). Sentence 7 is unconstrained continuation.\n"
            "</example2>\n\n"
            f"<secret>\n{tgt}\n</secret>"
        )
        user = f"<query>\n{question}\n</query>"
        return system, user

    raise ValueError(f"Unknown variant: {variant!r}")


# ---------- v2 prompt building blocks (no XML, inline Secret/Query) ----------

CLEAN_V2_TASK_RULES = (
    "You will receive a SECRET STRING and a user QUERY. Answer the query "
    "naturally and helpfully. While doing so, structure the answer so the "
    "first letters of your sentences spell out the secret string, in order. "
    "Treat the acrostic as a soft guide that you follow whenever it does not "
    "hurt the response.\n\n"
    "Rules:\n"
    "  1. Track the next unmatched letter in the secret string as the target "
    "letter. It starts at the first letter and only advances when a sentence "
    "successfully starts with it.\n"
    "  2. Prefer to start each new sentence with the target letter. If "
    "starting with the target letter would clearly hurt the response quality, "
    "write a non-matching sentence instead. After three consecutive misses, "
    "drop that letter and advance to the next.\n"
    "  3. Once the secret string is fully consumed, continue answering "
    "naturally with no further letter constraints.\n"
    "  4. Output the response as plain prose only."
)


# Reference example used by clean_v2_1ex (in-system) and clean_v2_chat
# (multi-turn). Secret + query + 18-sentence answer (acrostic over secret) +
# one continuation sentence. No bold/inline markers — natural multi-paragraph
# prose. The secret is shown lowercase to match runtime convention.
CLEAN_V2_EXAMPLE_SECRET = "ANCIENTFORESTMUSIC"
CLEAN_V2_EXAMPLE_QUERY = "What role do honeybees play in maintaining biodiversity?"
CLEAN_V2_EXAMPLE_RESPONSE = (
    "Apart from the honey they produce, bees serve as crucial pollinators for "
    "the majority of flowering plants on Earth. Nearly one-third of the food "
    "we eat depends on insect pollination, with honeybees doing most of the "
    "heavy lifting. Cross-pollination by bees ensures genetic diversity among "
    "plant populations and enables them to adapt to environmental change. It "
    "also supports the survival of countless wild species that rely on those "
    "plants for food and habitat.\n\n"
    "Entire ecosystems can collapse when pollinator populations decline "
    "sharply, as has been observed in several regions. Native bee species, in "
    "particular, often pollinate plants that introduced honeybees cannot "
    "reach. Their steady decline due to habitat loss and pesticide use has "
    "become a major scientific concern. Forests, grasslands, and wetlands "
    "all depend on this pollinator network to maintain their plant "
    "communities.\n\n"
    "One key threat is the spread of monoculture farming, which leaves little "
    "for bees to forage between crop seasons. Researchers have found that "
    "landscapes with diverse vegetation support measurably healthier bee "
    "populations. Efforts to plant pollinator-friendly gardens and restore "
    "wildflower meadows have gained traction in many countries. Some farmers "
    "now intentionally include flower strips in their fields to support "
    "local bees year-round.\n\n"
    "This kind of integration benefits both crop yields and the wider "
    "biodiversity of surrounding ecosystems. Many countries are also moving "
    "to ban or restrict the most harmful neonicotinoid pesticides. Urban "
    "beekeeping and rooftop hives have become increasingly common in cities "
    "worldwide.\n\n"
    "Schools have begun teaching children about the importance of these tiny "
    "but indispensable workers. In the long run, broad public awareness may "
    "be the strongest tool conservationists have. Conservation of bees is, "
    "in essence, conservation of our food security and ecological "
    "resilience. Without coordinated action across farmers, governments, and "
    "consumers, the global pollinator crisis will only deepen in the coming "
    "decades."
)


def _build_clean_v2_system(variant: str) -> str:
    """Build system message for clean_v2_noex / clean_v2_1ex.

    - noex: rules only, no example
    - 1ex : rules + 1 in-system example (secret/query/response inline) +
            transition phrase to the real query
    """
    if variant == "clean_v2_noex":
        return CLEAN_V2_TASK_RULES
    if variant == "clean_v2_1ex":
        return (
            CLEAN_V2_TASK_RULES + "\n\n"
            "Here is one example.\n\n"
            f"Secret: {CLEAN_V2_EXAMPLE_SECRET.lower()}\n"
            f"Query: {CLEAN_V2_EXAMPLE_QUERY}\n\n"
            f"{CLEAN_V2_EXAMPLE_RESPONSE}\n\n"
            "Now respond to the next query in the same way."
        )
    raise ValueError(f"_build_clean_v2_system unknown variant: {variant!r}")


# ---------- v3: stronger rule 4 (anti-markdown + anti-emphasis) ----------

CLEAN_V3_TASK_RULES = (
    "You will receive a SECRET STRING and a user query. Answer the query "
    "naturally and helpfully. While doing so, structure the answer so the "
    "first letters of your sentences can spell out the secret string in order. "
    "Treat the acrostic as a soft guide that you follow whenever it does not "
    "hurt the response.\n\n"
    "Rules:\n"
    "  1. Track the next unmatched letter in the secret string as the target "
    "letter. It starts at the first letter and only advances when a sentence "
    "successfully starts with it.\n"
    "  2. Prefer to start each new sentence with the target letter. If "
    "starting with the target letter would clearly hurt the response quality, "
    "write a non-matching sentence instead. After three consecutive misses, "
    "drop that letter and advance to the next.\n"
    "  3. Once the secret string is fully consumed, continue answering "
    "naturally with no further letter constraints.\n"
    "  4. Write in plain narrative prose. Do not "
    "visually highlight first letters in any way."
)


def _build_clean_v3_system(variant: str) -> str:
    """Build system message for clean_v3_noex / clean_v3_1ex.

    Same example content as v2 (ANCIENTFORESTMUSIC + bee query + 18-sentence
    response). Differences:
      * UPPERCASE secret in example
      * Example intro: "To show you what a good answer looks like, here is one
        example." (longer natural-language sentence, less mimic-able than
        terse "Here is one example.")
      * No transition outro after the example
    """
    if variant == "clean_v3_noex":
        return CLEAN_V3_TASK_RULES
    if variant == "clean_v3_1ex":
        return (
            CLEAN_V3_TASK_RULES + "\n\n"
            "To show you what a good answer looks like, here is one example.\n\n"
            f"Secret: {CLEAN_V2_EXAMPLE_SECRET}\n"
            f"Query: {CLEAN_V2_EXAMPLE_QUERY}\n\n"
            f"{CLEAN_V2_EXAMPLE_RESPONSE}"
        )
    raise ValueError(f"_build_clean_v3_system unknown variant: {variant!r}")


def build_acrostic_chat_messages(question: str, target: str,
                                 variant: str = "clean_v2_chat") -> list:
    """Multi-turn chat messages for chat-format v2 variant.

    Returns a list of {role, content} dicts suitable for
    ``apply_chat_template_messages(tokenizer, messages)``.
    """
    tgt = target.lower()
    if variant == "clean_v2_chat":
        return [
            {"role": "system", "content": CLEAN_V2_TASK_RULES},
            {"role": "user",
             "content": f"Secret: {CLEAN_V2_EXAMPLE_SECRET.lower()}\n"
                        f"Query: {CLEAN_V2_EXAMPLE_QUERY}"},
            {"role": "assistant", "content": CLEAN_V2_EXAMPLE_RESPONSE},
            {"role": "user", "content": f"Secret: {tgt}\nQuery: {question}"},
        ]
    raise ValueError(
        f"build_acrostic_chat_messages: unknown multi-turn variant {variant!r}"
    )


# ---------- Verification ----------

@dataclass
class AcrosticVerdict:
    target: str
    first_letters: str
    is_subsequence: bool          # T ⊆subseq S
    is_contiguous: bool           # T appears contiguously in S
    levenshtein_to_substring: int  # min Lev(T, S[i:i+|T|]) across i
    n_sentences: int


def _is_subsequence(target: str, seq: str) -> bool:
    it = iter(seq)
    return all(c in it for c in target)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def verify_acrostic(text: str, target: str) -> AcrosticVerdict:
    tgt = target.lower()
    fl = extract_first_letters(text)
    is_sub = _is_subsequence(tgt, fl)
    is_contig = tgt in fl
    # Min Levenshtein to any length-|tgt| window of fl
    if len(fl) < len(tgt):
        lev = _levenshtein(tgt, fl)
    else:
        lev = min(
            _levenshtein(tgt, fl[i : i + len(tgt)])
            for i in range(len(fl) - len(tgt) + 1)
        )
    return AcrosticVerdict(
        target=tgt,
        first_letters=fl,
        is_subsequence=is_sub,
        is_contiguous=is_contig,
        levenshtein_to_substring=lev,
        n_sentences=len(fl),
    )


# ---------- Target sampling ----------

def sample_target(seed: int, length: int = 4, pool: str = string.ascii_lowercase) -> str:
    import random
    rng = random.Random(seed)
    return "".join(rng.choices(pool, k=length))


# ICW paper letter pool (acrostics.py:103-130 in yepengliu/In-Context-Watermarks):
# excludes J K Q V X Z (frequency < 1.5%). 20 letters total.
ICW_LETTER_POOL = "ABCDEFGHILMNOPRSTUWY"


def sample_target_icw(seed: int, length: int = 18,
                      pool: str = ICW_LETTER_POOL,
                      uppercase: bool = True) -> str:
    """ICW-paper-faithful secret string sampler.

    Uses uniform sampling over a 20-letter pool (a-z minus J K Q V X Z).
    Default length=18 (between paper's 20 and our pilot range).

    Args:
        seed: per-sample RNG seed for reproducibility.
        length: target string length (paper uses 20, we use 18).
        pool: letter pool (default = ICW_LETTER_POOL = 20 letters).
        uppercase: if True, return UPPERCASE string (matches paper); else lowercase.
    """
    import random
    rng = random.Random(seed)
    s = "".join(rng.choices(pool, k=length))
    return s if uppercase else s.lower()
