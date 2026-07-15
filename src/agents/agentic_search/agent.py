"""The ``agentic_search`` agent: NLQ -> OpenSearch DSL, reached via ``POST /invoke``.

Given a natural-language query and a target index in the structured ``context``,
the agent fetches the index mapping with the caller's forwarded credentials,
delegates generation to the strategy named by ``context["strategy"]`` (default
``direct_dsl``), and returns the ``_search`` body as a JSON string. Any failure
degrades to a broad ``match_all`` so the upstream search returns a safe result
set instead of erroring.
"""

from __future__ import annotations

import json
from typing import Any

from opensearchpy import OpenSearch

from agents.agentic_search.prompts import FALLBACK_DSL
from agents.agentic_search.strategies import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    GenerationRequest,
)
from utils.logging_helpers import get_logger, log_error_event, log_info_event
from utils.model_factory import create_model

logger = get_logger(__name__)


class AgenticSearchAgent:
    """Adapts NLQ->DSL generation to the orchestrator's ``/invoke`` convention."""

    # Tells the orchestrator to pass context + auth_token, not just the prompt.
    accepts_invoke_context = True

    def __init__(self, opensearch_url: str, *, verify_certs: bool = False) -> None:
        self._opensearch_url = opensearch_url
        self._verify_certs = verify_certs
        self._model = create_model()

    def __call__(
        self,
        prompt: str | list[dict],
        context: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> str:
        try:
            return json.dumps(self._generate(prompt, context or {}, auth_token))
        except Exception:
            log_error_event(
                logger,
                "Agentic-search generation failed; returning fallback.",
                "agentic_search.generation_failed",
            )
            return FALLBACK_DSL

    def _generate(
        self,
        prompt: str | list[dict],
        context: dict[str, Any],
        auth_token: str | None,
    ) -> dict[str, Any]:
        if not isinstance(context, dict):
            raise ValueError("context must be a JSON object")
        index_name = context.get("index_name")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("query must be a non-empty string")
        if not index_name:
            raise ValueError("context.index_name is required")

        strategy_name = context.get("strategy", DEFAULT_STRATEGY)
        strategy = STRATEGIES.get(strategy_name)
        if strategy is None:
            raise ValueError(f"unknown strategy '{strategy_name}'")

        client = self._client(auth_token)
        dsl = strategy.generate(
            GenerationRequest(
                question=prompt,
                index_name=index_name,
                mapping=json.dumps(client.indices.get_mapping(index=index_name)),
                context=context,
                model=self._model,
                client=client,
            )
        )
        if not isinstance(dsl, dict):
            raise ValueError("strategy did not return a JSON object")
        log_info_event(
            logger,
            "Agentic-search generation succeeded.",
            "agentic_search.generated",
            index_name=index_name,
            strategy=strategy_name,
        )
        return dsl

    def _client(self, auth_token: str | None) -> OpenSearch:
        # Fresh client per request on purpose: it carries the caller's bearer
        # token. Caching it on the instance would leak credentials across callers.
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        return OpenSearch(
            hosts=[self._opensearch_url],
            headers=headers,
            verify_certs=self._verify_certs,
            timeout=30,
        )
