"""Agent Registry for the OpenSearch Agent Server.

Maps page contexts to agent factories. The orchestrator uses this registry
to determine which sub-agent should handle a given request.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRegistration:
    """Registration entry for a sub-agent."""

    name: str
    description: str
    page_contexts: list[str] = field(default_factory=list)
    is_default: bool = False


class AgentRegistry:
    """Registry mapping page contexts to agent factories.

    Agents register themselves with one or more page_contexts. The router
    queries this registry to find the right agent for a given context.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentRegistration] = {}
        self._page_context_map: dict[str, str] = {}  # page_context -> agent_name
        self._default_name: str | None = None

    def register(self, registration: AgentRegistration) -> None:
        """Register a sub-agent.

        Args:
            registration: Agent registration with name, factory, and page contexts.

        Raises:
            ValueError: If agent name is already registered or page_context conflicts.
        """
        if registration.name in self._agents:
            raise ValueError(f"Agent '{registration.name}' is already registered.")

        for ctx in registration.page_contexts:
            if ctx in self._page_context_map:
                existing = self._page_context_map[ctx]
                raise ValueError(
                    f"Page context '{ctx}' is already mapped to agent '{existing}'."
                )

        self._agents[registration.name] = registration
        for ctx in registration.page_contexts:
            self._page_context_map[ctx] = registration.name

        if registration.is_default:
            self._default_name = registration.name

    def get_agent_for_context(self, page_context: str) -> AgentRegistration | None:
        """Look up the agent registered for a page context.

        Args:
            page_context: The page context string (e.g. "search-relevance").

        Returns:
            AgentRegistration if found, None otherwise.
        """
        agent_name = self._page_context_map.get(page_context)
        if agent_name:
            return self._agents[agent_name]
        return None

    def get_default(self) -> AgentRegistration | None:
        """Get the default agent registration.

        Returns:
            The default AgentRegistration, or None if not registered.
        """
        if self._default_name:
            return self._agents.get(self._default_name)
        return None

    def list_agents(self) -> list[AgentRegistration]:
        """List all registered agents.

        Returns:
            List of all AgentRegistration entries.
        """
        return list(self._agents.values())
