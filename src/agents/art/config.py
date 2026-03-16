"""ART agent configuration.

Controls which MCP tool categories are available to each specialized agent.
All settings can be overridden via environment variables (``ART_`` prefix).

**Category names** (see :data:`~agents.art.specialized_agents.TOOL_GROUPS`):

- ``core_tools`` — general search and index operations (SearchIndexTool, etc.)
- ``search_relevance`` — all Search Relevance Workbench tools (experiments,
  judgment lists, search configurations, query sets)
- ``experiment`` — experiment lifecycle only
- ``judgment`` — judgment list management only
- ``search_config`` — search configuration management only
- ``query_set`` — query set management only

Individual tool names (e.g. ``GetExperimentTool``) can also be mixed in.

**Examples** (via environment variables)::

    # Give the UBI agent access to data-distribution as well as core tools
    ART_UBI_AGENT_TOOLS=core_tools,DataDistributionTool

    # Give the evaluation agent every search-relevance tool
    ART_EVALUATION_AGENT_TOOLS=core_tools,search_relevance
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ARTAgentConfig(BaseSettings):
    """Configuration for the ART specialized agents' MCP tool access.

    Each field is a list of category names and/or individual tool names.
    Set via a comma-separated environment variable, e.g.::

        ART_HYPOTHESIS_AGENT_TOOLS=core_tools,experiment,search_config,query_set

    Implementation note: the fields are typed ``list[str] | str`` so that
    pydantic-settings treats failed JSON parsing as a non-fatal parse failure
    and passes the raw string through to the ``_normalise_list`` validator,
    which converts it to ``list[str]``.  At runtime the value is always a
    ``list[str]``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ART_",
        case_sensitive=False,
        extra="ignore",
    )

    hypothesis_agent_tools: list[str] | str = [
        "core_tools",
        "experiment",
        "search_config",
        "query_set",
        "somTool"
    ]
    """Tool categories for the hypothesis agent.

    Defaults to search + experiment management + search configs + query sets.
    Judgment lists are excluded — the hypothesis agent only does pairwise
    sanity checks, not full offline evaluation.
    """

    evaluation_agent_tools: list[str] | str = [
        "core_tools",
        "search_relevance",
    ]
    """Tool categories for the evaluation agent.

    Defaults to search + all Search Relevance Workbench tools, giving the
    agent access to experiments, judgment lists, configs, and query sets.
    """

    ubi_agent_tools: list[str] | str = [
        "core_tools",
    ]
    """Tool categories for the user-behavior-analysis agent.

    Defaults to core search tools only — UBI analysis is read-only queries
    against the ubi_queries and ubi_events indices.
    """

    @field_validator(
        "hypothesis_agent_tools",
        "evaluation_agent_tools",
        "ubi_agent_tools",
        mode="before",
    )
    @classmethod
    def _normalise_list(cls, v: object) -> list[str]:
        """Normalise to ``list[str]``, trimming whitespace and dropping empty items.

        Handles both a raw comma-separated string (from an env var) and a
        plain list (from direct instantiation or a JSON-formatted env var).
        """
        if isinstance(v, str):
            items: list[str] = v.split(",")
        elif isinstance(v, list):
            items = [str(i) for i in v]
        else:
            return v  # type: ignore[return-value]
        return [item.strip() for item in items if item.strip()]
