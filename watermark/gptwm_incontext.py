"""
In-context watermark implementation: generates the green token string for use in system prompts.
"""
from watermark.gptwm import GPTWatermarkBase
import torch
import random

class InContextWatermarkGenerator(GPTWatermarkBase):
    """
    Generate green word list for in-context watermarking.
    The green words will be used as part of the system prompt via prompt.get_incontext_system_prompt.
    """

    def __init__(self, *args, **kwargs):
        if 'tokenizer' not in kwargs or kwargs['tokenizer'] is None:
            raise ValueError("tokenizer must be provided for InContextWatermarkGenerator")

        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)

        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = kwargs['tokenizer']

    def get_green_token_string(self, shuffle: bool = True) -> str:
        """Return pipe-separated string of green tokens for use in system prompts.

        Args:
            shuffle: If True, randomize token order (for training diversity).
                     Set to False for val/test to enable vLLM prefix caching.
        """
        sep = "|"
        green_token_ids = torch.nonzero(self.green_list_mask, as_tuple=True)[0].tolist()
        green_tokens = self.tokenizer.convert_ids_to_tokens(green_token_ids)
        green_token_list = []
        for token in green_tokens:
            s = self.tokenizer.convert_tokens_to_string([token])
            if s:
                green_token_list.append(s)
        if shuffle:
            random.shuffle(green_token_list)

        return sep.join(green_token_list)
