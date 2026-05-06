"""Unit tests for the default agent — skill loading and agent construction.

Verifies that:
1. The bundled ``skills/`` directory is auto-discovered by ``load_all_skills``.
2. ``create_default_agent()`` wires MCP tools and skills into the strands Agent.
3. ``LoggingAgentSkills`` emits an INFO-level log on skill activation and
   still delegates state tracking to the parent class.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from strands import Skill

from agents.default_agent import (
    LoggingAgentSkills,
    create_default_agent,
)
from agents.skill_loader import load_all_skills

pytestmark = pytest.mark.unit

# Skills expected to ship in the repo's ``skills/`` directory.
# Append a new entry here when adding a new skill — both the real-repo
# discovery test and the plugin-wiring test will cover it automatically.
EXPECTED_REPO_SKILLS = ["ppl-reference"]


class TestLoadAllSkills:
    """Group 1 — skill auto-discovery via ``load_all_skills()``."""

    @pytest.mark.parametrize("expected_name", EXPECTED_REPO_SKILLS)
    def test_discovers_expected_skill(self, expected_name: str) -> None:
        """Bundled ``src/agents/bundled_skills/`` yields every expected skill."""
        skills = load_all_skills()

        names = [s.name for s in skills]
        assert expected_name in names, (
            f"expected {expected_name} in loaded skills, got {names}"
        )
        skill = next(s for s in skills if s.name == expected_name)
        assert skill.description, f"{expected_name} has empty description"


@pytest.fixture
def mock_mcp_tools() -> list[MagicMock]:
    """Two synthetic MCP tools."""
    tool_a = MagicMock()
    tool_a.tool_name = "list_indices"
    tool_b = MagicMock()
    tool_b.tool_name = "search_index"
    return [tool_a, tool_b]


@pytest.fixture
def patch_mcp(mock_mcp_tools: list[MagicMock]):
    """Patch MCPClient + streamable_http_client + httpx.AsyncClient.

    Yields the MCPClient class mock so tests can inspect calls if needed.
    """
    with (
        patch("agents.default_agent.MCPClient") as mock_mcp_client_cls,
        patch("agents.default_agent.streamable_http_client"),
        patch("agents.default_agent.httpx.AsyncClient"),
    ):
        mock_client = MagicMock()
        mock_client.list_tools_sync.return_value = mock_mcp_tools
        mock_mcp_client_cls.return_value = mock_client
        yield mock_mcp_client_cls


@pytest.mark.usefixtures("patch_mcp")
class TestCreateDefaultAgent:
    """Group 2 — agent construction via ``create_default_agent()``."""

    def test_registers_mcp_tools(self, mock_mcp_tools: list[MagicMock]) -> None:
        """MCP tools from ``list_tools_sync()`` are forwarded to the strands Agent."""
        with (
            patch("agents.default_agent.load_all_skills", return_value=[]),
            patch("agents.default_agent.Agent") as mock_agent_cls,
        ):
            create_default_agent("http://localhost:9200")

        mock_agent_cls.assert_called_once()
        tools_kwarg = mock_agent_cls.call_args.kwargs["tools"]
        assert tools_kwarg == mock_mcp_tools

    def test_attaches_logging_agent_skills_plugin(self) -> None:
        """When skills are discovered, a ``LoggingAgentSkills`` plugin is attached."""
        fake_skill = Skill(name="fake-skill", description="a fake")
        with patch("agents.default_agent.load_all_skills", return_value=[fake_skill]):
            agent = create_default_agent("http://localhost:9200")

        plugins = agent._plugin_registry._plugins
        assert "agent_skills" in plugins
        skills_plugin = plugins["agent_skills"]
        assert isinstance(skills_plugin, LoggingAgentSkills)
        assert "fake-skill" in skills_plugin._skills

    def test_no_skills_plugin_when_skills_dir_empty(self) -> None:
        """Zero skills discovered → no ``agent_skills`` plugin attached."""
        with patch("agents.default_agent.load_all_skills", return_value=[]):
            agent = create_default_agent("http://localhost:9200")

        assert "agent_skills" not in agent._plugin_registry._plugins

    @pytest.mark.parametrize("expected_name", EXPECTED_REPO_SKILLS)
    def test_real_skill_registered_in_plugin(self, expected_name: str) -> None:
        """End-to-end: each expected skill lands inside the ``agent_skills`` plugin."""
        # Do NOT patch load_all_skills — let the real function run against the repo.
        agent = create_default_agent("http://localhost:9200")

        plugins = agent._plugin_registry._plugins
        assert "agent_skills" in plugins
        assert expected_name in plugins["agent_skills"]._skills


class TestLoggingAgentSkills:
    """Group 3 — ``LoggingAgentSkills._track_activated_skill`` behavior."""

    def test_activation_logs_at_info_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Activation emits exactly one INFO-level record containing the skill name."""
        plugin = LoggingAgentSkills(
            skills=[Skill(name="my-skill", description="a skill")]
        )
        mock_agent = MagicMock()
        mock_agent.state.get.return_value = None

        with caplog.at_level(logging.INFO, logger="agents.default_agent"):
            plugin._track_activated_skill(mock_agent, "my-skill")

        activation_records = [
            r for r in caplog.records if "Skill activated by agent" in r.message
        ]
        assert len(activation_records) == 1
        record = activation_records[0]
        assert record.levelno == logging.INFO
        assert "my-skill" in record.message

    def test_activation_delegates_to_parent(self) -> None:
        """Parent ``_track_activated_skill`` still runs — state is updated on the agent."""
        plugin = LoggingAgentSkills(
            skills=[Skill(name="my-skill", description="a skill")]
        )
        mock_agent = MagicMock()
        # Simulate empty state so parent initializes it.
        mock_agent.state.get.return_value = None

        plugin._track_activated_skill(mock_agent, "my-skill")

        # Parent calls agent.state.set(state_key, {"activated_skills": [...]})
        mock_agent.state.set.assert_called_once()
        call_args = mock_agent.state.set.call_args
        assert call_args.args[0] == "agent_skills"
        assert call_args.args[1] == {"activated_skills": ["my-skill"]}
