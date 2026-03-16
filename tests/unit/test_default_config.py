"""
Unit tests for FallbackAgentConfig.

Covers:
- Default value (empty list — all tools allowed)
- Comma-separated string parsing from environment variables
- Whitespace trimming
- Empty-value handling
- Direct instantiation overrides env vars
"""

import pytest

from agents.default_config import FallbackAgentConfig

pytestmark = pytest.mark.unit


class TestFallbackAgentConfigDefaults:
    """Default value is an empty list (all tools pass through)."""

    def test_default_is_empty_list(self):
        config = FallbackAgentConfig()
        assert config.agent_tools == []

    def test_empty_list_means_all_tools_pass_through(self):
        """Verify that _select_tools returns all tools when config is empty."""
        from unittest.mock import MagicMock

        from agents.tool_filter import _select_tools

        tools = [MagicMock(), MagicMock()]
        config = FallbackAgentConfig()
        assert _select_tools(tools, config.agent_tools) == tools


class TestFallbackAgentConfigEnvVarParsing:
    """Comma-separated env vars are parsed into lists."""

    def test_single_category(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_AGENT_TOOLS", "core_tools")
        config = FallbackAgentConfig()
        assert config.agent_tools == ["core_tools"]

    def test_comma_separated_categories(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_AGENT_TOOLS", "core_tools,search_relevance")
        config = FallbackAgentConfig()
        assert config.agent_tools == ["core_tools", "search_relevance"]

    def test_whitespace_is_trimmed(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_AGENT_TOOLS", "core_tools , search_relevance")
        config = FallbackAgentConfig()
        assert config.agent_tools == ["core_tools", "search_relevance"]

    def test_empty_string_produces_empty_list(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_AGENT_TOOLS", "")
        config = FallbackAgentConfig()
        assert config.agent_tools == []

    def test_individual_tool_name_accepted(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_AGENT_TOOLS", "core_tools,GetExperimentTool")
        config = FallbackAgentConfig()
        assert config.agent_tools == ["core_tools", "GetExperimentTool"]

    def test_direct_instantiation_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_AGENT_TOOLS", "search_relevance")
        config = FallbackAgentConfig(agent_tools=["core_tools"])
        assert config.agent_tools == ["core_tools"]
