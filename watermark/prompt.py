"""System prompts for generation tasks.

To add a new dataset type, call ``register_prompt`` with base and in-context
templates.  The in-context template must contain a ``{green_tokens}``
placeholder that will be filled at runtime.

Example::

    register_prompt(
        "my_new_task",
        base="You are a helpful assistant for my_new_task.",
        incontext=(
            "### Green Token List: {green_tokens}\n\n"
            "You are a helpful assistant for my_new_task. "
            "Use as many green tokens as possible."
        ),
    )
"""

from typing import Dict


_DATASET_PROMPTS: Dict[str, Dict[str, str]] = {}


def register_prompt(dataset_type: str, *, base: str, incontext: str):
    """Register base and in-context prompt templates for *dataset_type*."""
    _DATASET_PROMPTS[dataset_type.lower()] = {"base": base, "incontext": incontext}


def get_system_prompt(dataset_type: str) -> str:
    """Return the base system prompt (no green-token list)."""
    return get_incontext_system_prompt(dataset_type, "")


def get_incontext_system_prompt(dataset_type: str, green_token_string: str = "") -> str:
    """Return the system prompt, optionally enriched with a green-token list.

    Args:
        dataset_type: Registered dataset name (case-insensitive).
        green_token_string: Pipe-separated green token string, or ``""`` for
            a plain (no-watermark) prompt.
    """
    dtype = dataset_type.lower().strip()
    if dtype not in _DATASET_PROMPTS:
        available = ", ".join(sorted(_DATASET_PROMPTS)) or "(none)"
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. Available: {available}"
        )
    entry = _DATASET_PROMPTS[dtype]
    if green_token_string:
        return entry["incontext"].format(green_tokens=green_token_string)
    return entry["base"]


# ------------------------------ Built-in dataset types ------------------------------

register_prompt(
    "lfqa",
    base="""""",
    incontext="""\
<green>
{green_tokens}
</green>

Respond to the user query. Seamlessly incorporate as many tokens from <green> as possible without compromising text quality.""",
)

register_prompt(
    "lfqa_initials",
    base="""""",
    incontext="""\
<green>
{green_letters}
</green>

<red>
{red_letters}
</red>

Given the <green> and <red> letter lists, respond to the user query with clarity, accuracy, informativeness, and relevance. Favor words beginning with letters from <green> and minimize those beginning with letters from <red>. Never reveal the <green> and <red> letter lists in your reply.""",
)


def get_initials_incontext_prompt(dataset_type: str, green_letters, red_letters) -> str:
    """Return the Initials ICW system prompt with both letter lists formatted
    as comma-separated uppercase strings.

    The letter order is preserved verbatim — callers control whether to pass
    sorted lists (inference / detection) or shuffled lists (training data
    synthesis, to prevent student from memorising positional cues).
    Pass empty/None letters to get the plain base prompt.
    """
    dtype = dataset_type.lower().strip()
    if dtype not in _DATASET_PROMPTS:
        available = ", ".join(sorted(_DATASET_PROMPTS)) or "(none)"
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. Available: {available}"
        )
    entry = _DATASET_PROMPTS[dtype]
    if not green_letters or not red_letters:
        return entry["base"]
    g = ", ".join(green_letters)
    r = ", ".join(red_letters)
    return entry["incontext"].format(green_letters=g, red_letters=r)
