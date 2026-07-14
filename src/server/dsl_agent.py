"""The ``dsl_generator`` agent: NLQ -> OpenSearch DSL, reachable via ``POST /invoke``.

Agentic search calls the agent server synchronously through ``/invoke`` (RFC #140).
This module registers the DSL-generation capability as an ``/invoke`` agent rather
than a bespoke route: :class:`DslInvokeAgent` adapts the pluggable generation
engine to the orchestrator's context-aware calling convention, and
:func:`wrap_inference_results` shapes the reply as the ml-commons
``inference_results`` envelope so a connector's built-in passthrough lands the
DSL string in ``ModelTensor.result`` (where neural-search reads it).

The engine is injected via :func:`set_dsl_generator`; see
:class:`server.dsl_generator.BedrockDslGenerator` for the default.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from server.types import InferenceResultsResponse

logger = logging.getLogger(__name__)

# Returned when generation fails so the search degrades to matching everything
# rather than erroring.
FALLBACK_DSL = '{"size":10,"query":{"match_all":{}}}'


class DslGenerator(Protocol):
    """Pluggable engine that returns an OpenSearch ``_search`` body as a JSON string."""

    def generate(
        self, question: str, index_name: str, auth_token: str | None = None
    ) -> str: ...


_generator: DslGenerator | None = None


def set_dsl_generator(generator: DslGenerator) -> None:
    """Register the engine backing the ``dsl_generator`` agent (call at startup)."""
    global _generator
    _generator = generator


def wrap_inference_results(dsl_query: str) -> InferenceResultsResponse:
    """Wrap a DSL string in the ml-commons inference_results passthrough envelope."""
    return {
        "inference_results": [
            {
                "output": [{"name": "response", "result": dsl_query}],
                "status_code": 200,
            }
        ]
    }


def generate_dsl_or_fallback(
    question: str, index_name: str, auth_token: str | None = None
) -> str:
    """Generate a DSL string for an NLQ, degrading to ``match_all`` on any failure.

    Returns a bare DSL string (not the envelope); the caller wraps it. On a
    missing generator, a generation error, or output that isn't a JSON object,
    returns :data:`FALLBACK_DSL` so the upstream search degrades instead of erroring.
    """
    if _generator is None:
        logger.error("DSL generation requested but no DslGenerator is registered")
        return FALLBACK_DSL

    try:
        dsl_query = _generator.generate(question, index_name, auth_token=auth_token)
        # Fall back unless the generated DSL is a JSON object (a _search body).
        # A bare primitive/array (e.g. "ok", []) parses but is not valid DSL.
        if not isinstance(json.loads(dsl_query), dict):
            raise ValueError("generated DSL is not a JSON object")
    except Exception:  # noqa: BLE001 - any generation failure degrades to fallback
        logger.exception(
            "DSL generation failed for index=%s; returning fallback", index_name
        )
        return FALLBACK_DSL

    return dsl_query


class DslInvokeAgent:
    """The ``dsl_generator`` agent: adapts the DSL engine to ``/invoke``.

    The orchestrator calls a context-aware agent as
    ``agent(prompt, context=..., auth_token=...)``. This maps that onto
    ``generate_dsl_or_fallback(question, index_name, auth_token)``: the NLQ is the
    ``prompt`` (``query`` in the ``/invoke`` body), and ``index_name`` comes from
    the structured ``context``. Returns a bare DSL string; the ``/invoke`` handler
    wraps it in the envelope when ``response_format=inference_results`` is set.
    """

    # Tells the orchestrator to pass context + auth_token, not just the prompt
    # (see AgentOrchestrator.invoke).
    accepts_invoke_context = True

    def __call__(
        self,
        prompt: str | list[dict],
        context: dict | None = None,
        auth_token: str | None = None,
    ) -> str:
        context = context or {}
        index_name = context.get("index_name")
        if not isinstance(prompt, str) or not prompt.strip():
            logger.error("dsl_generator invoked without a string query; returning fallback")
            return FALLBACK_DSL
        if not index_name:
            logger.error(
                "dsl_generator invoked without context.index_name; returning fallback"
            )
            return FALLBACK_DSL
        return generate_dsl_or_fallback(prompt, index_name, auth_token=auth_token)
