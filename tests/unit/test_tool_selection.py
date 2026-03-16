"""
Unit tests for TOOL_GROUPS and _select_tools.

Covers:
- TOOL_GROUPS category membership and completeness
- search_relevance meta-category is the union of the four sub-categories
- _select_tools: no filter, single category, multiple categories
- _select_tools: direct tool name, mixed category + tool name
- _select_tools: empty tools list, tools without tool_name attribute
- Config-driven filtering via ARTAgentConfig values
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.tool_filter import TOOL_GROUPS, _select_tools

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str) -> MagicMock:
    """Create a mock MCP tool object with the given tool_name."""
    t = MagicMock()
    t.tool_name = name
    return t


def _tool_names(tools: list) -> set[str]:
    return {t.tool_name for t in tools}


# ---------------------------------------------------------------------------
# TOOL_GROUPS structure
# ---------------------------------------------------------------------------

class TestToolGroups:
    """TOOL_GROUPS has the expected categories and contents."""

    def test_expected_categories_present(self):
        assert set(TOOL_GROUPS) >= {
            "core_tools",
            "search_relevance",
            "experiment",
            "judgment",
            "search_config",
            "query_set",
        }

    def test_core_tools_contains_search_tools(self):
        assert {"SearchIndexTool", "ListIndexTool", "CountTool"} <= TOOL_GROUPS["core_tools"]

    def test_search_relevance_is_union_of_sub_categories(self):
        expected = (
            TOOL_GROUPS["experiment"]
            | TOOL_GROUPS["judgment"]
            | TOOL_GROUPS["search_config"]
            | TOOL_GROUPS["query_set"]
        )
        assert TOOL_GROUPS["search_relevance"] == expected

    def test_search_relevance_does_not_overlap_core_tools(self):
        assert TOOL_GROUPS["search_relevance"].isdisjoint(TOOL_GROUPS["core_tools"])

    def test_sub_categories_are_subsets_of_search_relevance(self):
        for cat in ("experiment", "judgment", "search_config", "query_set"):
            assert TOOL_GROUPS[cat] <= TOOL_GROUPS["search_relevance"], (
                f"'{cat}' tools should all appear in 'search_relevance'"
            )

    def test_all_entries_are_frozensets(self):
        for name, group in TOOL_GROUPS.items():
            assert isinstance(group, frozenset), f"TOOL_GROUPS['{name}'] should be a frozenset"


# ---------------------------------------------------------------------------
# _select_tools behaviour
# ---------------------------------------------------------------------------

class TestSelectTools:
    """_select_tools filters the tool list correctly."""

    # --- no-op cases ---

    def test_none_filter_returns_all(self):
        tools = [_make_tool("SearchIndexTool"), _make_tool("CreateExperimentTool")]
        assert _select_tools(tools, None) == tools

    def test_empty_filter_returns_all(self):
        tools = [_make_tool("SearchIndexTool"), _make_tool("CreateExperimentTool")]
        assert _select_tools(tools, []) == tools

    def test_empty_tools_list_returns_empty(self):
        assert _select_tools([], ["core_tools"]) == []

    # --- category filtering ---

    def test_single_category(self):
        core = [_make_tool(n) for n in TOOL_GROUPS["core_tools"]]
        srw = [_make_tool("CreateExperimentTool")]
        result = _select_tools(core + srw, ["core_tools"])
        assert _tool_names(result) == TOOL_GROUPS["core_tools"]

    def test_multiple_categories_returns_union(self):
        all_tools = [
            _make_tool(n)
            for n in TOOL_GROUPS["core_tools"] | TOOL_GROUPS["experiment"] | TOOL_GROUPS["judgment"]
        ]
        result = _select_tools(all_tools, ["experiment", "judgment"])
        expected = TOOL_GROUPS["experiment"] | TOOL_GROUPS["judgment"]
        assert _tool_names(result) == expected

    def test_search_relevance_meta_category(self):
        srw_names = TOOL_GROUPS["search_relevance"]
        all_tools = [_make_tool(n) for n in srw_names | TOOL_GROUPS["core_tools"]]
        result = _select_tools(all_tools, ["search_relevance"])
        assert _tool_names(result) == srw_names

    # --- direct tool name as filter ---

    def test_direct_tool_name_not_in_any_category(self):
        """An item that is not a category key is treated as a direct tool name."""
        tools = [_make_tool("SearchIndexTool"), _make_tool("MyCustomTool")]
        result = _select_tools(tools, ["MyCustomTool"])
        assert _tool_names(result) == {"MyCustomTool"}

    def test_direct_tool_name_in_existing_category(self):
        """A single tool name that happens to be inside a category is also matched."""
        tools = [_make_tool("SearchIndexTool"), _make_tool("CountTool")]
        result = _select_tools(tools, ["SearchIndexTool"])
        assert _tool_names(result) == {"SearchIndexTool"}

    # --- mixed category + direct tool name (the config use-case) ---

    def test_mixed_category_and_tool_name(self):
        """Category names and individual tool names can be freely mixed."""
        tools = [
            _make_tool("SearchIndexTool"),   # in core_tools
            _make_tool("CountTool"),          # in core_tools
            _make_tool("DataDistributionTool"),  # in core_tools
            _make_tool("CreateExperimentTool"),  # in experiment / search_relevance
        ]
        # Give only CountTool from core_tools plus CreateExperimentTool by name
        result = _select_tools(tools, ["CountTool", "CreateExperimentTool"])
        assert _tool_names(result) == {"CountTool", "CreateExperimentTool"}

    def test_category_and_extra_tool_name_combined(self):
        """Category expands normally; extra tool name adds on top."""
        core_tools = [_make_tool(n) for n in TOOL_GROUPS["core_tools"]]
        extra = _make_tool("CreateExperimentTool")
        result = _select_tools(core_tools + [extra], ["core_tools", "CreateExperimentTool"])
        assert _tool_names(result) == TOOL_GROUPS["core_tools"] | {"CreateExperimentTool"}

    # --- tools without tool_name ---

    def test_tool_without_tool_name_is_excluded(self):
        """Tools that lack a tool_name attribute are silently excluded."""
        good = _make_tool("SearchIndexTool")
        bad = MagicMock(spec=[])  # no tool_name attribute
        result = _select_tools([good, bad], ["core_tools"])
        assert good in result
        assert bad not in result

    def test_tool_with_none_tool_name_is_excluded(self):
        t = MagicMock()
        t.tool_name = None
        result = _select_tools([t], ["core_tools"])
        assert result == []

    # --- warning for unavailable tools ---

    def test_warns_when_configured_tool_not_in_mcp_server(self):
        """A warning is emitted when a configured tool name has no match in the tool list."""
        tools = [_make_tool("SearchIndexTool")]
        with patch("agents.tool_filter.log_warning_event") as mock_warn:
            _select_tools(tools, ["SearchIndexTool", "NonExistentTool"])
        mock_warn.assert_called_once()
        args = mock_warn.call_args
        assert "NonExistentTool" in str(args)

    def test_no_warning_when_all_configured_tools_available(self):
        """No warning is emitted when every configured tool is present in the tool list."""
        tools = [_make_tool(n) for n in TOOL_GROUPS["core_tools"]]
        with patch("agents.tool_filter.log_warning_event") as mock_warn:
            _select_tools(tools, ["core_tools"])
        mock_warn.assert_not_called()

    def test_warns_for_category_tools_not_in_mcp_server(self):
        """Warning fires when a category expands to names absent from the tool list."""
        # Only provide one of the core tools
        tools = [_make_tool("SearchIndexTool")]
        with patch("agents.tool_filter.log_warning_event") as mock_warn:
            _select_tools(tools, ["core_tools"])
        mock_warn.assert_called_once()
        # The warning should mention the missing names
        unmatched_arg = mock_warn.call_args.kwargs.get("unmatched") or mock_warn.call_args[1].get("unmatched")
        missing = TOOL_GROUPS["core_tools"] - {"SearchIndexTool"}
        assert set(unmatched_arg) == missing

    # --- ordering is preserved ---

    def test_order_is_preserved(self):
        names = ["SearchIndexTool", "CountTool", "ListIndexTool"]
        tools = [_make_tool(n) for n in names]
        result = _select_tools(tools, ["core_tools"])
        assert [t.tool_name for t in result] == names


# ---------------------------------------------------------------------------
# Config-driven filtering (integration between ARTAgentConfig and _select_tools)
# ---------------------------------------------------------------------------

class TestConfigDrivenFiltering:
    """_select_tools works correctly when driven by ARTAgentConfig values."""

    def test_default_hypothesis_tools_excludes_judgment(self):
        from agents.art.config import ARTAgentConfig
        config = ARTAgentConfig()
        all_tools = [_make_tool(n) for n in TOOL_GROUPS["search_relevance"] | TOOL_GROUPS["core_tools"]]
        result = _select_tools(all_tools, config.hypothesis_agent_tools)
        result_names = _tool_names(result)
        # No judgment tools should appear
        assert result_names.isdisjoint(TOOL_GROUPS["judgment"])

    def test_default_evaluation_tools_includes_all_srw(self):
        from agents.art.config import ARTAgentConfig
        config = ARTAgentConfig()
        all_tools = [_make_tool(n) for n in TOOL_GROUPS["search_relevance"] | TOOL_GROUPS["core_tools"]]
        result = _select_tools(all_tools, config.evaluation_agent_tools)
        result_names = _tool_names(result)
        assert TOOL_GROUPS["search_relevance"] <= result_names

    def test_default_ubi_tools_excludes_srw(self):
        from agents.art.config import ARTAgentConfig
        config = ARTAgentConfig()
        all_tools = [_make_tool(n) for n in TOOL_GROUPS["search_relevance"] | TOOL_GROUPS["core_tools"]]
        result = _select_tools(all_tools, config.ubi_agent_tools)
        result_names = _tool_names(result)
        assert result_names.isdisjoint(TOOL_GROUPS["search_relevance"])

    def test_env_var_override_adds_individual_tool(self, monkeypatch):
        """ART_UBI_AGENT_TOOLS=core_tools,DataDistributionTool works end-to-end."""
        from agents.art.config import ARTAgentConfig
        monkeypatch.setenv("ART_UBI_AGENT_TOOLS", "core_tools,DataDistributionTool")
        config = ARTAgentConfig()
        # DataDistributionTool is already in core_tools, so this is a no-op in practice,
        # but the filter should still match it without error.
        tools = [_make_tool("DataDistributionTool"), _make_tool("CreateExperimentTool")]
        result = _select_tools(tools, config.ubi_agent_tools)
        assert _tool_names(result) == {"DataDistributionTool"}

    def test_env_var_override_restricts_evaluation_agent(self, monkeypatch):
        from agents.art.config import ARTAgentConfig
        monkeypatch.setenv("ART_EVALUATION_AGENT_TOOLS", "experiment")
        config = ARTAgentConfig()
        all_tools = [_make_tool(n) for n in TOOL_GROUPS["search_relevance"] | TOOL_GROUPS["core_tools"]]
        result = _select_tools(all_tools, config.evaluation_agent_tools)
        assert _tool_names(result) == TOOL_GROUPS["experiment"]
