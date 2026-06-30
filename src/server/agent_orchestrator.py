"""Agent Orchestrator — routes requests to AG-UI Strands agent wrappers.

``ag_ui_strands.StrandsAgent`` instances are created once per agent name and
then cached so that the per-thread ``StrandsAgentCore`` (and its
``ConversationManager``) survives across requests, giving the agent persistent
conversation memory.  Authentication is handled by :class:`~utils.obo_context.OboAuth`
instances stored on each agent's httpx client — the orchestrator calls
``set_token()`` before each run to inject fresh credentials.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui_strands import StrandsAgent as AGUIStrandsAgent
from ag_ui_strands.config import StrandsAgentConfig
from strands import Agent as StrandsAgentCore

from orchestrator.router import PageContextRouter
from utils.logging_helpers import get_logger, log_debug_event, log_info_event
from utils.obo_context import OboAuth

logger = get_logger(__name__)

# A factory callable that returns a pre-configured Strands Agent.
# Headers are no longer passed to the factory — OboAuth handles auth.
AgentFactory = Callable[[], StrandsAgentCore]


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


def _extract_bearer_token(headers: dict[str, str] | None) -> str | None:
    """Extract the Bearer token from an Authorization header dict.

    Args:
        headers: HTTP headers dict (may contain "authorization" key).

    Returns:
        The raw JWT token string, or None.
    """
    if not headers:
        return None
    auth = headers.get("authorization") or headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:]
    return auth  # non-Bearer value — pass through as-is


class AgentOrchestrator:
    """Routes AG-UI requests to the appropriate ``ag_ui_strands.StrandsAgent``.

    Instead of holding pre-created agents, the orchestrator stores *factory*
    functions.  Each factory returns a ``StrandsAgentCore`` with an httpx
    client configured with :class:`~utils.obo_context.OboAuth`.

    Before each ``run()`` the orchestrator calls ``OboAuth.set_token()`` on
    the agent's auth instance.  The token is stored behind a
    ``threading.Lock``, so it is visible to the MCP client's background
    thread where httpx requests are actually executed.
    """

    def __init__(self, router: PageContextRouter) -> None:
        self._agent_factories: dict[str, dict[str, Any]] = {}
        self._cached_agui_agents: dict[str, AGUIStrandsAgent] = {}
        self._router = router

    def register_agent_factory(
        self,
        name: str,
        factory: AgentFactory,
        description: str = "",
        config: StrandsAgentConfig | None = None,
    ) -> None:
        """Register an agent factory for on-demand agent creation.

        Args:
            name: Unique agent name (must match registry name).
            factory: Callable that returns a pre-configured Strands Agent.
            description: Human-readable description.
            config: Optional tool-behavior configuration.
        """
        self._agent_factories[name] = {
            "factory": factory,
            "description": description,
            "config": config,
        }
        log_info_event(
            logger,
            f"Registered agent factory '{name}' in orchestrator",
            "orchestrator.agent_factory_registered",
            agent_name=name,
        )

    async def run(
        self,
        input_data: RunAgentInput,
        agent_name: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Yield AG-UI events for *input_data*.

        If *agent_name* is ``None`` the orchestrator extracts ``page_context``
        from *input_data* and uses :class:`PageContextRouter` to resolve the
        target agent.

        Before yielding events the OBO token from *headers* is set on the
        agent's :class:`~utils.obo_context.OboAuth` instance via
        ``set_token()``.  The token is stored behind a ``threading.Lock`` so
        it is accessible from the MCP client's background thread where httpx
        requests are executed.

        Args:
            input_data: AG-UI ``RunAgentInput``.
            agent_name: Explicit agent name (skips routing).
            headers: Optional HTTP headers forwarded from the Dashboards
                request (e.g. ``Authorization: Bearer <obo-token>``).

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

        factory_info = self._agent_factories.get(agent_name)
        if factory_info is None:
            raise RuntimeError(
                f"No agent factory registered with name '{agent_name}'. "
                f"Available: {list(self._agent_factories)}"
            )

        # Reuse a cached AGUIStrandsAgent so that its _agents_by_thread dict
        # (and the Strands ConversationManager inside each per-thread agent)
        # persists across requests — giving the agent conversation memory.
        agui_agent = self._cached_agui_agents.get(agent_name)
        if agui_agent is None:
            strands_agent = factory_info["factory"]()
            agui_agent = AGUIStrandsAgent(
                agent=strands_agent,
                name=agent_name,
                description=factory_info["description"],
                config=factory_info["config"],
            )
            # Keep the MCP client reference on the wrapper to prevent GC
            # from closing the MCP session.
            mcp_ref = getattr(strands_agent, "_mcp_client", None)
            if mcp_ref is not None:
                agui_agent._mcp_client = mcp_ref
            # Keep the OboAuth instance so we can call set_token() on
            # subsequent requests.
            obo_auth = getattr(strands_agent, "_obo_auth", None)
            if obo_auth is not None:
                agui_agent._obo_auth = obo_auth
            self._cached_agui_agents[agent_name] = agui_agent
            log_debug_event(
                logger,
                f"Created and cached agent '{agent_name}'",
                "orchestrator.agent_created",
                agent_name=agent_name,
            )
        else:
            log_debug_event(
                logger,
                f"Reusing cached agent '{agent_name}'",
                "orchestrator.agent_reused",
                agent_name=agent_name,
            )

        # Set the OBO token on the agent's OboAuth instance.  This is
        # thread-safe (lock-protected) and visible to the MCP client's
        # background thread where httpx requests are actually executed.
        token = _extract_bearer_token(headers)
        obo_auth = getattr(agui_agent, "_obo_auth", None)
        if obo_auth is not None:
            obo_auth.set_token(token)

        async for event in agui_agent.run(input_data):
            yield event
