# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_DEBUG_VALIDATE_PROMPT_PRINT_COUNT = 0


def _format_debug_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n...<omitted {omitted} chars>...\n{text[-tail:]}"


def _format_debug_ids(token_ids: list[int], edge: int = 32) -> str:
    if len(token_ids) <= edge * 2:
        return str(token_ids)
    return f"{token_ids[:edge]} ... {token_ids[-edge:]}"


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global _DEBUG_VALIDATE_PROMPT_PRINT_COUNT

        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))
        validate = bool(kwargs.get("validate", False))
        uid = kwargs.get("uid")

        metrics = {}
        request_id = uuid4().hex

        # Use processor if available for multimodal support
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )

        if validate and _DEBUG_VALIDATE_PROMPT_PRINT_COUNT < 2:
            raw_prompt_text = json.dumps(messages, ensure_ascii=False, indent=2)
            decoded_prompt = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            )
            print(
                "[WMKD DEBUG][single_turn_agent] "
                f"uid={uid} request_id={request_id} prompt_tokens={len(prompt_ids)}",
                flush=True,
            )
            print(
                "[WMKD DEBUG][single_turn_agent] "
                f"prompt_token_ids={_format_debug_ids(prompt_ids)}",
                flush=True,
            )
            print(
                "[WMKD DEBUG][single_turn_agent] "
                f"raw_prompt_messages=\n{_format_debug_text(raw_prompt_text)}",
                flush=True,
            )
            print(
                "[WMKD DEBUG][single_turn_agent] "
                f"rebuilt_prompt=\n{_format_debug_text(decoded_prompt)}",
                flush=True,
            )
            _DEBUG_VALIDATE_PROMPT_PRINT_COUNT += 1

        request_sampling_params = dict(sampling_params)
        if validate:
            request_sampling_params["_verl_debug_validate"] = True
            request_sampling_params["_verl_debug_uid"] = uid

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=request_sampling_params,
                image_data=image_data,
            )
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
        )
        return output
