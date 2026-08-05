"""
RawPromptIdsAgentLoop — single-turn agent loop that uses pre-tokenized
`raw_prompt_ids` directly instead of re-applying apply_chat_template.

Use this for val datasets where the prompt is already a fully-formatted
chat-template string (e.g. with green token list + <think></think>
no-think suppression already baked in).

Registered as "raw_prompt_ids_agent".
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("raw_prompt_ids_agent")
class RawPromptIdsAgentLoop(AgentLoopBase):
    """Single-turn agent loop using pre-tokenized raw_prompt_ids.

    Instead of re-applying apply_chat_template (which loses the green token
    list, no-think suppression, etc.), this loop takes the `raw_prompt_ids`
    field from the non-tensor batch — which RLHFDataset populates from the
    already-formatted prompt string — and passes them directly to vLLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids: list[int] = list(kwargs["raw_prompt_ids"])

        validate = bool(kwargs.get("validate", False))
        uid = kwargs.get("uid")
        request_id = uuid4().hex

        if validate:
            decoded = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(prompt_ids[-200:], skip_special_tokens=False),
            )
            logger.warning(
                "[raw_prompt_ids_agent] uid=%s prompt_tokens=%d last_200_decoded=%r",
                uid, len(prompt_ids), decoded,
            )

        request_sampling_params = dict(sampling_params)
        # Always cap max_tokens to response_length so that negative validation
        # samples (short prompts ~500 tokens) don't cause vLLM to generate
        # ~82,000 tokens instead of 600. Without this, vllm_async_server
        # computes max_tokens = max_model_len - len(prompt_ids), which is
        # enormous for short prompts.
        request_sampling_params["max_tokens"] = self.response_length
        if validate:
            request_sampling_params["_verl_debug_validate"] = True
            request_sampling_params["_verl_debug_uid"] = uid

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=request_sampling_params,
                image_data=None,
            )

        response_mask = [1] * len(output.token_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
        )
