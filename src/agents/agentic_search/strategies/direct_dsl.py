"""Direct-DSL generation: the model authors the whole ``_search`` body.

The default generation strategy. It gives the model the index mapping and one
sample document, then has it author a complete OpenSearch query body:

- On Bedrock, a single forced tool call (see :func:`forced_tool_fill`) keeps the
  model's ``EmitSearch`` rationale inside the tool input rather than as leading
  free text. Other providers (e.g. Ollama) use the portable strands
  ``structured_output`` path, which does not support forcing ``toolChoice``.
- The sample document supplies real field values (e.g. exact keyword/enum values),
  which the mapping alone does not, helping the model choose the right field and term.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import Agent

from agents.agentic_search.prompts.direct_dsl import (
    SYSTEM_BLOCKS,
    USER_PROMPT,
    EmitSearch,
)
from agents.agentic_search.strategies.base import GenerationRequest
from agents.agentic_search.strategies.forced_tool import (
    forced_tool_fill,
    supports_forced_tool,
)

logger = logging.getLogger(__name__)

# Cap each sample-doc field value to this many chars so a large text field can't
# dominate the prompt.
_MAX_TRUNCATE_CHARS = 250


def _fetch_sample_document(client: Any, index_name: str) -> str:
    """Fetch one indexed document with field values truncated.

    Returns a JSON string, or ``""`` on any failure. The sample document is
    best-effort enrichment, so an empty index or a search error is non-fatal.
    """
    try:
        resp = client.search(
            index=index_name,
            body={"size": 1, "query": {"match_all": {}}, "sort": ["_doc"]},
            _source=True,
        )
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return ""
        source = hits[0].get("_source", {})
        truncated = {
            k: (v[:_MAX_TRUNCATE_CHARS] if isinstance(v, str) else v)
            for k, v in source.items()
        }
        return json.dumps(truncated)
    except Exception:  # noqa: BLE001 - best-effort; proceed without a sample
        logger.warning(
            "Sample document fetch failed for index=%s; proceeding without", index_name
        )
        return ""


class DirectDslStrategy:
    """Generate a ``_search`` body directly from the NLQ, index mapping, and a sample doc."""

    name = "direct_dsl"

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        """Return the OpenSearch ``_search`` body as a dict for the NLQ.

        ``SYSTEM_BLOCKS`` carries the rules and examples behind a cache point, so on
        warm calls only the per-request tail (mapping, sample document, question) is
        billed. Uses the forced-tool path on Bedrock and the portable strands
        ``structured_output`` path elsewhere.
        """
        sample = _fetch_sample_document(request.client, request.index_name)
        user_msg = USER_PROMPT.format(
            question=request.question,
            index_name=request.index_name,
            mapping=request.mapping,
            sample_document=sample or "(none available)",
        )
        result = self._emit(request.model, user_msg)
        logger.info(
            "Generated DSL for index=%s (reason=%s)", request.index_name, result.reason
        )
        return result.dsl

    @staticmethod
    def _emit(model: Any, user_msg: str) -> EmitSearch:
        """Produce the ``EmitSearch`` result, using the best path for the provider.

        Bedrock forces the tool call (no leading free text); other providers fall
        back to the portable strands ``structured_output`` path, which handles the
        provider differences but cannot force ``toolChoice``.
        """
        if supports_forced_tool(model):
            return forced_tool_fill(
                model=model,
                schema_model=EmitSearch,
                system_blocks=SYSTEM_BLOCKS,
                user_message=user_msg,
            )
        agent = Agent(
            model=model,
            system_prompt=SYSTEM_BLOCKS,
            tools=[],
            callback_handler=None,
        )
        return agent(user_msg, structured_output_model=EmitSearch).structured_output
