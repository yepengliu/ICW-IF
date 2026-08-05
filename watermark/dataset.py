import json

from datasets import Dataset
import torch

from watermark.prompt import get_system_prompt


# ------------------------------ Utility functions ------------------------------
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for d in data:
            file.write(json.dumps(d) + '\n')


def load_generation_dataset(jsonl_path, num_test=None):
    rows = load_jsonl(jsonl_path)
    if num_test is not None:
        rows = rows[:num_test]
    return Dataset.from_list(rows)


def apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Format a single (system, user) turn using the tokenizer's chat template.

    Falls back to a plain-text format when no chat template is available.
    """
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


def apply_chat_template_messages(tokenizer, messages: list) -> str:
    """Format a multi-turn message list using the tokenizer's chat template.

    Each item must be a {"role": ..., "content": ...} dict. Roles can be
    'system' / 'user' / 'assistant'. Falls back to plain-text rendering
    when no chat template is available.
    """
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg.get('content', '')}")
    return "\n\n".join(parts) + "\n\nAssistant:"

# ------------------------------ Map functions ------------------------------
def make_prompt_mapper(tokenizer, system_prompt: str, *, tokenize: bool = False):
    """Create a HuggingFace Dataset ``.map()`` function that applies a chat
    template with *system_prompt*.

    Args:
        tokenizer: The tokenizer instance.
        system_prompt: System prompt to prepend.
        tokenize: If ``True``, also tokenize the formatted prompt (for HF
            Transformers).  Call with ``batched=False``.
            If ``False``, return formatted strings only (for vLLM).  Call with
            ``batched=True``.

    Returns:
        A map function.  Depending on *tokenize*:

        * ``False`` -- adds ``input_prompt`` (list[str]).
        * ``True``  -- adds ``input_prompt`` (str), ``input_ids``,
          ``attention_mask``.
    """
    if tokenize:
        def _fn(example):
            prompt_text = apply_chat_template(
                tokenizer, system_prompt, example["prefix"],
            )
            encoded = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                padding=False,
            )
            example.update({
                "input_prompt": prompt_text,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            })
            return example
    else:
        def _fn(example):
            example["input_prompt"] = [
                apply_chat_template(tokenizer, system_prompt, p)
                for p in example["prefix"]
            ]
            return example

    return _fn


def make_multitask_prompt_mapper(tokenizer):
    """Create a ``.map()`` function that resolves the system prompt per example
    from ``example["dataset_type"]``.

    Must be called with ``batched=False, with_indices=True``.
    Adds ``input_prompt`` (str) and ``idx`` (int) to each example.
    """
    def _fn(example, idx):
        dataset_type = example.get("dataset_type", "lfqa")
        system_prompt = get_system_prompt(dataset_type)
        prompt_text = apply_chat_template(
            tokenizer, system_prompt, example["prefix"],
        )
        example.update({
            "input_prompt": prompt_text,
            "idx": idx,
        })
        return example

    return _fn


def make_tokenize_mapper(tokenizer):
    """Create a ``.map()`` function that tokenizes the raw prefix (no chat
    template).  Call with ``batched=True``.

    Adds ``input_ids``, ``attention_mask``, and ``gold_completion_ids``.
    """
    def _fn(example):
        prefix_enc = tokenizer(
            example["prefix"],
            truncation=True,
            padding=False,
        )
        gold_enc = tokenizer(
            example["gold_completion"],
            truncation=True,
            padding=False,
        )
        example.update({
            **prefix_enc,
            "gold_completion_ids": gold_enc["input_ids"],
        })
        return example

    return _fn


# ------------------------------ Collate functions ------------------------------
def collate_fn(batch, tokenizer):
    """Collate a list of examples into a padded batch dict.

    Tokenizes ``input_prompt`` (chat-template-formatted) if present, otherwise
    falls back to ``prefix``.  Optional fields (``dataset_type``, ``idx``) are
    included when available.
    """
    prompt_key = "input_prompt" if "input_prompt" in batch[0] else "prefix"
    encoded = tokenizer(
        [x[prompt_key] for x in batch],
        padding=True,
        return_tensors="pt",
    )

    batch_dict = {
        "prefix": [x["prefix"] for x in batch],
        "gold_completion": [x["gold_completion"] for x in batch],
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    if "input_prompt" in batch[0]:
        batch_dict["input_prompt"] = [x["input_prompt"] for x in batch]
    if "dataset_type" in batch[0]:
        batch_dict["dataset_type"] = [x["dataset_type"] for x in batch]
    if "idx" in batch[0]:
        batch_dict["idx"] = [x["idx"] for x in batch]

    return batch_dict
