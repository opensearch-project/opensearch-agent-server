"""Page-Context Router for the OpenSearch Agent Server.

Routes incoming requests to the appropriate sub-agent based on page_context.
Implements a two-tier routing strategy:

  Tier 1: Deterministic page_context match (fast, no LLM needed)
  Tier 2: LLM-based intent detection (future — stubbed for POC)
  Tier 3: Fallback to default agent
"""

from __future__ import annotations

from orchestrator.registry import AgentRegistration, AgentRegistry
from utils.logging_helpers import get_logger, log_debug_event, log_info_event

logger = get_logger(__name__)


class PageContextRouter:
    """Routes requests to sub-agents by page context."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def route(self, page_context: str | None) -> AgentRegistration:
        """Route a request to the appropriate agent.

        Args:
            page_context: The page context from the request's forwardedProps.

        Returns:
            The AgentRegistration that should handle this request.

        Raises:
            RuntimeError: If no default agent is registered and no match is found.
        """
        # Tier 1: Deterministic page_context match
        if page_context:
            agent = self._registry.get_agent_for_context(page_context)
            if agent:
                log_debug_event(
                    logger,
                    f"Tier 1 match: page_context='{page_context}' -> agent='{agent.name}'",
                    "router.tier1_match",
                    page_context=page_context,
                    agent_name=agent.name,
                )
                return agent

        # Tier 2: LLM-based intent detection (future)
        # TODO: Analyze user message content to determine intent and route accordingly

        # Tier 3: Default agent
        default = self._registry.get_default()
        if default is None:
            raise RuntimeError(
                "No default agent registered. Register an agent with is_default=True."
            )

        log_info_event(
            logger,
            f"Default: page_context='{page_context}' -> agent='{default.name}'",
            "router.default",
            page_context=page_context,
            agent_name=default.name,
        )
        return default
