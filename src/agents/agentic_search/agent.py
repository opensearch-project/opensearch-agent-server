# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""The ``agentic_search`` agent: NLQ -> OpenSearch DSL, reached via ``POST /invoke``.

Given a natural-language ``query`` and a target ``index_name`` (in the structured
``context``), the agent fetches the index mapping with the caller's forwarded
credentials, dispatches to the generation strategy named by ``context.strategy``
(default ``direct_dsl``), and returns the ``_search`` body as a JSON string. Any
failure degrades to a broad ``match_all`` so the upstream search returns a safe
result set instead of erroring.

Registered via :func:`create_agentic_search_agent`, following the same factory
convention as the other agents in ``src/agents``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opensearchpy import OpenSearch

from agents.agentic_search.prompts import FALLBACK_DSL
from agents.agentic_search.strategies import (
    DEFAULT_STRATEGY,
    MULTI_TEMPLATE_STRATEGY,
    SINGLE_TEMPLATE_STRATEGY,
    STRATEGIES,
)
from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.template_base import distinct_template_ids
from utils.model_factory import create_model

logger = logging.getLogger(__name__)


def _template_strategy_for(context: dict[str, Any]) -> str | None:
    """Name the template strategy this request needs, or ``None`` for free-DSL."""
    n = len(distinct_template_ids(context))
    if n > 1:
        return MULTI_TEMPLATE_STRATEGY
    if n == 1:
        return SINGLE_TEMPLATE_STRATEGY
    return None


class AgenticSearchAgent:
    """Adapts NLQ->DSL generation to the orchestrator's ``/invoke`` convention.

    The orchestrator calls a context-aware agent as
    ``agent(prompt, context=..., auth_token=...)``: ``prompt`` is the NLQ (the
    ``query`` field), ``context`` carries ``index_name`` and an optional
    ``strategy``, and ``auth_token`` is the caller's forwarded bearer token used
    to reach the cluster. Returns the ``_search`` body as a JSON string.
    """

    # Signals the orchestrator to pass context + auth_token, not just the prompt.
    accepts_invoke_context = True

    def __init__(self, opensearch_url: str) -> None:
        self._opensearch_url = opensearch_url
        # Built once and reused across calls; the provider/credentials are fixed
        # for the process, while each request gets a fresh stateless Agent.
        self._model = create_model()

    def __call__(
        self,
        prompt: str | list[dict],
        context: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> str:
        try:
            return json.dumps(self._generate(prompt, context, auth_token))
        except Exception:  # noqa: BLE001 - any failure degrades to the fallback
            logger.exception("Agentic-search generation failed; returning fallback")
            return FALLBACK_DSL

    def _generate(
        self,
        prompt: str | list[dict],
        context: dict[str, Any] | None,
        auth_token: str | None,
    ) -> dict[str, Any]:
        context = context if isinstance(context, dict) else {}
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("query must be a non-empty string")
        index_name = context.get("index_name")
        if not index_name:
            raise ValueError("context.index_name is required")

        strategy = self._select_strategy(context)

        client = self._client(auth_token)
        # Skip the per-query mapping fetch when the strategy doesn't need it
        # (template fill fills typed params and uses no mapping). Strategies that
        # omit `needs_mapping` are treated as needing it, so existing behavior holds.
        mapping = ""
        if getattr(strategy, "needs_mapping", True):
            mapping = json.dumps(client.indices.get_mapping(index=index_name))
        dsl = strategy.generate(
            GenerationRequest(
                question=prompt,
                index_name=index_name,
                mapping=mapping,
                context=context,
                model=self._model,
                client=client,
            )
        )

        # Guard the contract: a strategy must return a JSON object (a _search
        # body). Anything else degrades to the fallback rather than reaching the
        # cluster as an invalid query.
        if not isinstance(dsl, dict):
            raise ValueError("strategy did not return a _search body object")
        return dsl

    @staticmethod
    def _select_strategy(context: dict[str, Any]) -> GenerationStrategy:
        """Pick the generation strategy for this request.

        An explicit ``context.strategy`` always wins. Otherwise the template keys are
        the mode switch, and how many candidates a request carries decides which
        template path runs:

        - two or more distinct ids in ``template_ids`` -> ``multi_template_fill``,
          which picks one and fills it in a single call;
        - exactly one id (in either ``template_ids`` or ``template_id``) ->
          ``template_fill``, the single-template path;
        - neither -> the free-DSL default.

        A single-candidate request does not go through the multi-template path: with
        no choice to make it uses the dedicated single-template prompt. Template mode
        stays opt-in and additive — callers that send no template keys are unaffected,
        and an existing caller sending a scalar ``template_id`` keeps its current
        behavior exactly.
        """
        strategy_name = context.get("strategy")
        if strategy_name is None:
            strategy_name = _template_strategy_for(context) or DEFAULT_STRATEGY
        strategy: GenerationStrategy | None = STRATEGIES.get(strategy_name)
        if strategy is None:
            raise ValueError(f"unknown strategy '{strategy_name}'")
        return strategy

    def _client(self, auth_token: str | None) -> OpenSearch:
        # Forward the caller's bearer token per request when present; otherwise
        # the client falls back to its environment configuration.
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        return OpenSearch(hosts=[self._opensearch_url], headers=headers, timeout=30)


def create_agentic_search_agent(opensearch_url: str) -> AgenticSearchAgent:
    """Create the ``agentic_search`` agent (factory used at startup registration)."""
    return AgenticSearchAgent(opensearch_url)
