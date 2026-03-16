"""
Unit tests for ARTAgentConfig.

Covers:
- Default tool category values for each agent
- Comma-separated string parsing from environment variables
- Whitespace trimming
- Empty-value handling
"""

import pytest

from agents.art.config import ARTAgentConfig

pytestmark = pytest.mark.unit


class TestARTAgentConfigDefaults:
    """Default values are sane and use the correct category names."""

    def test_hypothesis_agent_defaults(self):
        config = ARTAgentConfig()
        assert config.hypothesis_agent_tools == [
            "core_tools",
            "experiment",
            "search_config",
            "query_set",
        ]

    def test_evaluation_agent_defaults(self):
        config = ARTAgentConfig()
        assert config.evaluation_agent_tools == ["core_tools", "search_relevance"]

    def test_ubi_agent_defaults(self):
        config = ARTAgentConfig()
        assert config.ubi_agent_tools == ["core_tools"]

    def test_judgment_excluded_from_hypothesis_defaults(self):
        """The hypothesis agent must not have judgment tools by default."""
        config = ARTAgentConfig()
        assert "judgment" not in config.hypothesis_agent_tools

    def test_search_relevance_covers_evaluation_defaults(self):
        """Evaluation agent uses the broad search_relevance category."""
        config = ARTAgentConfig()
        assert "search_relevance" in config.evaluation_agent_tools


class TestARTAgentConfigEnvVarParsing:
    """Comma-separated env vars are parsed into lists."""

    def test_comma_separated_string(self, monkeypatch):
        monkeypatch.setenv("ART_HYPOTHESIS_AGENT_TOOLS", "core_tools,experiment")
        config = ARTAgentConfig()
        assert config.hypothesis_agent_tools == ["core_tools", "experiment"]

    def test_single_item_string(self, monkeypatch):
        monkeypatch.setenv("ART_UBI_AGENT_TOOLS", "core_tools")
        config = ARTAgentConfig()
        assert config.ubi_agent_tools == ["core_tools"]

    def test_whitespace_is_trimmed(self, monkeypatch):
        monkeypatch.setenv(
            "ART_EVALUATION_AGENT_TOOLS", "core_tools , search_relevance"
        )
        config = ARTAgentConfig()
        assert config.evaluation_agent_tools == ["core_tools", "search_relevance"]

    def test_empty_string_produces_empty_list(self, monkeypatch):
        monkeypatch.setenv("ART_UBI_AGENT_TOOLS", "")
        config = ARTAgentConfig()
        assert config.ubi_agent_tools == []

    def test_individual_tool_name_accepted(self, monkeypatch):
        """A direct tool name (not a category) is accepted as-is."""
        monkeypatch.setenv("ART_UBI_AGENT_TOOLS", "core_tools,DataDistributionTool")
        config = ARTAgentConfig()
        assert config.ubi_agent_tools == ["core_tools", "DataDistributionTool"]

    def test_all_three_fields_independent(self, monkeypatch):
        monkeypatch.setenv("ART_HYPOTHESIS_AGENT_TOOLS", "core_tools")
        monkeypatch.setenv("ART_EVALUATION_AGENT_TOOLS", "search_relevance")
        monkeypatch.setenv("ART_UBI_AGENT_TOOLS", "core_tools,CountTool")
        config = ARTAgentConfig()
        assert config.hypothesis_agent_tools == ["core_tools"]
        assert config.evaluation_agent_tools == ["search_relevance"]
        assert config.ubi_agent_tools == ["core_tools", "CountTool"]

    def test_direct_instantiation_overrides_env(self, monkeypatch):
        """Directly passed values take precedence over env vars."""
        monkeypatch.setenv("ART_UBI_AGENT_TOOLS", "search_relevance")
        config = ARTAgentConfig(ubi_agent_tools=["core_tools"])
        assert config.ubi_agent_tools == ["core_tools"]
