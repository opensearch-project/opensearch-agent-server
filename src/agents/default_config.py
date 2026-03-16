"""Fallback agent configuration.

Controls which MCP tool categories are available to the fallback agent.
The setting can be overridden via the ``FALLBACK_AGENT_TOOLS`` environment
variable.

By default the fallback agent has access to **all** tools exposed by the MCP
server (``FALLBACK_AGENT_TOOLS`` is empty).  Set it to a comma-separated list
of category names or individual tool names to restrict access::

    # Give the fallback agent only core search tools
    FALLBACK_AGENT_TOOLS=core_tools

    # Give the fallback agent core tools plus one SRW tool
    FALLBACK_AGENT_TOOLS=core_tools,GetExperimentTool

See :data:`~agents.tool_filter.TOOL_GROUPS` for valid category names.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FallbackAgentConfig(BaseSettings):
    """Configuration for the fallback agent's MCP tool access.

    Set via a comma-separated environment variable, e.g.::

        FALLBACK_AGENT_TOOLS=core_tools,search_relevance

    An empty value (the default) means all MCP tools are available.
    """

    model_config = SettingsConfigDict(
        env_prefix="FALLBACK_",
        case_sensitive=False,
        extra="ignore",
    )

    agent_tools: list[str] | str = []
    """Tool filter for the fallback agent.

    Defaults to an empty list, which passes all MCP tools through unchanged.
    """

    @field_validator("agent_tools", mode="before")
    @classmethod
    def _normalise_list(cls, v: object) -> list[str]:
        """Normalise to ``list[str]``, trimming whitespace and dropping empty items."""
        if isinstance(v, str):
            items: list[str] = v.split(",")
        elif isinstance(v, list):
            items = [str(i) for i in v]
        else:
            return v  # type: ignore[return-value]
        return [item.strip() for item in items if item.strip()]
