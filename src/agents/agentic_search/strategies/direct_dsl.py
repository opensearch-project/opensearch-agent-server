"""Direct free-form DSL generation: the model authors the full ``_search`` body
from the mapping + question in one forced structured-tool call (the port of
ml-commons' ``QueryPlanningTool``)."""

from __future__ import annotations

from typing import Any

from strands import Agent

from agents.agentic_search.prompts import SYSTEM_BLOCKS, USER_PROMPT, EmitSearch
from agents.agentic_search.strategies.base import GenerationRequest
from utils.logging_helpers import get_logger, log_info_event

logger = get_logger(__name__)


class DirectDslStrategy:
    name = "direct_dsl"

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        # Fresh Agent per call = stateless requests; the reused model carries the
        # connection. structured_output_model (not the deprecated
        # structured_output()) keeps the system-prompt cache point intact.
        agent = Agent(
            model=request.model,
            system_prompt=SYSTEM_BLOCKS,
            tools=[],
            callback_handler=None,
        )
        user_msg = USER_PROMPT.format(
            question=request.question,
            index_name=request.index_name,
            mapping=request.mapping,
        )
        result = agent(user_msg, structured_output_model=EmitSearch).structured_output
        log_info_event(
            logger,
            "Generated DSL via direct strategy.",
            "agentic_search.direct_dsl_generated",
            index_name=request.index_name,
            reason=result.reason,
        )
        return result.dsl
