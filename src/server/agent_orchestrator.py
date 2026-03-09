"""Agent Orchestrator — routes requests to AG-UI Strands agent wrappers.

Holds a dictionary of ``ag_ui_strands.StrandsAgent`` wrappers keyed by agent
name.  The outer shell (routing, auth, persistence, SSE encoding, cancellation)
remains custom; this module is the thin glue between the router and the
off-the-shelf AG-UI event conversion layer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui_strands import StrandsAgent as AGUIStrandsAgent
from ag_ui_strands.config import StrandsAgentConfig
from strands import Agent as StrandsAgentCore

from orchestrator.router import PageContextRouter
from utils.logging_helpers import get_logger, log_debug_event, log_info_event

logger = get_logger(__name__)


def _extract_app_id_from_context(context: list) -> str | None:
    """Extract appId from the AG-UI context array.

    OpenSearch Dashboards sends page context as a Context entry with a JSON
    value containing ``appId`` (e.g. "discover", "explore", "home").
    This function finds the first entry whose value contains an appId.

    Args:
        context: List of AG-UI Context objects (description + value).

    Returns:
        The appId string, or None if not found.
    """
    for ctx in context:
        try:
            value = ctx.value if isinstance(ctx.value, dict) else json.loads(ctx.value)
            if isinstance(value, dict) and "appId" in value:
                app_id = value["appId"]
                log_debug_event(
                    logger,
                    f"Extracted appId='{app_id}' from AG-UI context",
                    "orchestrator.context_app_id",
                    app_id=app_id,
                )
                return app_id
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return None


def _extract_page_context(input_data: RunAgentInput) -> str | None:
    """Extract page_context from RunAgentInput.

    Strategy:
      1. Check forwardedProps.page_context (direct override, useful for curl testing)
      2. Check AG-UI context array for page context with appId (sent by Dashboard)

    Args:
        input_data: AG-UI RunAgentInput.

    Returns:
        page_context string or None.
    """
    page_context = None
    if hasattr(input_data, "forwarded_props") and input_data.forwarded_props:
        page_context = input_data.forwarded_props.get("page_context")

    if not page_context and hasattr(input_data, "context") and input_data.context:
        page_context = _extract_app_id_from_context(input_data.context)

    return page_context


class AgentOrchestrator:
    """Routes AG-UI requests to the appropriate ``ag_ui_strands.StrandsAgent``.

    Each sub-agent type (ART, fallback, …) is registered as a named wrapper.
    ``run()`` resolves the agent name via :class:`PageContextRouter` and
    delegates to the wrapper's ``.run()``, yielding AG-UI events.
    """

    def __init__(self, router: PageContextRouter) -> None:
        self._agents: dict[str, AGUIStrandsAgent] = {}
        self._router = router

    def register_agent(
        self,
        name: str,
        strands_agent: StrandsAgentCore,
        description: str = "",
        config: StrandsAgentConfig | None = None,
    ) -> None:
        """Wrap *strands_agent* in ``ag_ui_strands.StrandsAgent`` and store it.

        Args:
            name: Unique agent name (must match registry name).
            strands_agent: Pre-initialized Strands Agent.
            description: Human-readable description.
            config: Optional tool-behavior configuration.
        """
        agui_agent = AGUIStrandsAgent(
            agent=strands_agent,
            name=name,
            description=description,
            config=config,
        )
        self._agents[name] = agui_agent
        log_info_event(
            logger,
            f"Registered agent '{name}' in orchestrator",
            "orchestrator.agent_registered",
            agent_name=name,
        )

    async def run(
        self, input_data: RunAgentInput, agent_name: str | None = None
    ) -> AsyncIterator[Any]:
        """Yield AG-UI events for *input_data*.

        If *agent_name* is ``None`` the orchestrator extracts ``page_context``
        from *input_data* and uses :class:`PageContextRouter` to resolve the
        target agent.

        Args:
            input_data: AG-UI ``RunAgentInput``.
            agent_name: Explicit agent name (skips routing).

        Yields:
            AG-UI protocol events.
        """
        if agent_name is None:
            page_context = _extract_page_context(input_data)
            registration = self._router.route(page_context)
            agent_name = registration.name
            log_debug_event(
                logger,
                f"Routed page_context='{page_context}' -> agent='{agent_name}'",
                "orchestrator.routed",
                page_context=page_context,
                agent_name=agent_name,
            )

        agent = self._agents.get(agent_name)
        if agent is None:
            raise RuntimeError(
                f"No agent registered with name '{agent_name}'. "
                f"Available: {list(self._agents)}"
            )

        async for event in agent.run(input_data):
            yield event
