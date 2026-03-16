"""Shared MCP tool filtering utilities.

Provides :data:`TOOL_GROUPS` (the category registry) and :func:`_select_tools`
(the runtime filter).  Both the ART specialized agents and the fallback agent
import from here so tool-name definitions live in a single place.

**Category names**:

- ``core_tools`` — general search and index operations (SearchIndexTool, etc.)
- ``search_relevance`` — all Search Relevance Workbench tools (experiments,
  judgment lists, search configurations, query sets)
- ``experiment`` — experiment lifecycle only
- ``judgment`` — judgment list management only
- ``search_config`` — search configuration management only
- ``query_set`` — query set management only

Individual tool names (e.g. ``GetExperimentTool``) can also be used directly.
"""

from __future__ import annotations

from typing import Any

from utils.logging_helpers import get_logger, log_info_event, log_warning_event

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool group definitions
# Category names mirror the OpenSearch MCP Server's logical groupings.
# Each entry maps a category name to the exact tool names exposed by the server.
# ---------------------------------------------------------------------------
TOOL_GROUPS: dict[str, frozenset[str]] = {
    # General search and index operations (OpenSearch MCP Server: "core_tools")
    "core_tools": frozenset({
        "SearchIndexTool",
        "ListIndexTool",
        "CountTool",
        "ExplainTool",
        "DataDistributionTool",
        "LogPatternAnalysisTool",
        "GenericOpenSearchApiTool",
    }),
    # All Search Relevance Workbench tools (OpenSearch MCP Server: "search_relevance")
    # Convenience meta-category — equivalent to experiment + judgment +
    # search_config + query_set combined.
    "search_relevance": frozenset({
        # experiment lifecycle
        "CreateExperimentTool",
        "GetExperimentTool",
        "DeleteExperimentTool",
        # judgment list management
        "CreateJudgmentListTool",
        "CreateLLMJudgmentListTool",
        "CreateUBIJudgmentListTool",
        "GetJudgmentListTool",
        "DeleteJudgmentListTool",
        # search configuration (query DSL templates)
        "CreateSearchConfigurationTool",
        "GetSearchConfigurationTool",
        "DeleteSearchConfigurationTool",
        # query set management
        "CreateQuerySetTool",
        "GetQuerySetTool",
        "DeleteQuerySetTool",
        "SampleQuerySetTool",
    }),
    # Fine-grained sub-categories for selective access
    "experiment": frozenset({
        "CreateExperimentTool",
        "GetExperimentTool",
        "DeleteExperimentTool",
    }),
    "judgment": frozenset({
        "CreateJudgmentListTool",
        "CreateLLMJudgmentListTool",
        "CreateUBIJudgmentListTool",
        "GetJudgmentListTool",
        "DeleteJudgmentListTool",
    }),
    "search_config": frozenset({
        "CreateSearchConfigurationTool",
        "GetSearchConfigurationTool",
        "DeleteSearchConfigurationTool",
    }),
    "query_set": frozenset({
        "CreateQuerySetTool",
        "GetQuerySetTool",
        "DeleteQuerySetTool",
        "SampleQuerySetTool",
    }),
}


def _select_tools(
    tools: list[Any],
    filters: list[str] | None = None,
) -> list[Any]:
    """Filter MCP tools by category name and/or explicit tool name.

    Each item in *filters* is first looked up as a category key in
    :data:`TOOL_GROUPS`.  If no matching category is found the item is treated
    as a direct tool name.  This means category names and individual tool names
    can be freely mixed in the same list — which is exactly what the
    ``ART_*_AGENT_TOOLS`` and ``FALLBACK_AGENT_TOOLS`` environment variables
    accept.

    If *filters* is ``None`` or empty the full list is returned unchanged.

    Args:
        tools: Full list of Strands MCP tool objects.
        filters: Category names (from :data:`TOOL_GROUPS`) and/or explicit
            ``tool_name`` strings to include.

    Returns:
        Filtered list containing only the tools whose ``tool_name`` is in the
        resolved allow-list.
    """
    if not filters:
        return tools

    allowed: set[str] = set()
    for f in filters:
        group = TOOL_GROUPS.get(f)
        if group is not None:
            allowed |= group
        else:
            # Not a known category — treat as a direct tool name.
            allowed.add(f)

    selected = [t for t in tools if getattr(t, "tool_name", None) in allowed]
    log_info_event(
        logger,
        f"[Agents] Tool filter applied: {len(selected)}/{len(tools)} tools selected "
        f"(filters={filters})",
        "agents.tools_filtered",
        selected=len(selected),
        total=len(tools),
    )

    actual_names = {getattr(t, "tool_name", None) for t in tools} - {None}
    unmatched = allowed - actual_names
    if unmatched:
        log_warning_event(
            logger,
            f"[Agents] Configured tool(s) not available from MCP server: "
            f"{sorted(unmatched)}. Check your tool configuration.",
            "agents.tools_not_available",
            unmatched=sorted(unmatched),
        )

    return selected
