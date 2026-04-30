"""Unit tests for the default agent — skill loading and agent construction.

Verifies that:
1. ``_load_all_skills()`` auto-discovers ``skills/`` directories correctly.
2. ``create_default_agent()`` wires MCP tools and skills into the strands Agent.
3. ``LoggingAgentSkills`` emits an INFO-level log on skill activation and
   still delegates state tracking to the parent class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from strands import Skill

from agents import default_agent
from agents.default_agent import (
    _DEFAULT_BEDROCK_MODEL_ID,
    LoggingAgentSkills,
    _load_all_skills,
    create_default_agent,
)

pytestmark = pytest.mark.unit

# Skills expected to ship in the repo's ``skills/`` directory.
# Append a new entry here when adding a new skill — both the real-repo
# discovery test and the plugin-wiring test will cover it automatically.
EXPECTED_REPO_SKILLS = ["ppl-reference"]


def _write_skill(skill_dir: Path, name: str, description: str) -> None:
    """Create a minimal valid SKILL.md under ``skill_dir``."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# Body\nContent\n"
    )


@pytest.fixture
def fake_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``_load_all_skills`` to a tmp project root.

    ``_load_all_skills`` resolves ``skills/`` relative to
    ``default_agent.__file__``. We simulate a project root at ``tmp_path`` by
    patching that module attribute to a synthetic path three levels deep.
    """
    fake_file = tmp_path / "src" / "agents" / "default_agent.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.touch()
    monkeypatch.setattr(default_agent, "__file__", str(fake_file))
    return tmp_path


class TestLoadAllSkills:
    """Group 1 — skill auto-discovery via ``_load_all_skills()``."""

    @pytest.mark.parametrize("expected_name", EXPECTED_REPO_SKILLS)
    def test_discovers_expected_skill(self, expected_name: str) -> None:
        """Real repo ``skills/`` dir yields every skill in ``EXPECTED_REPO_SKILLS``."""
        skills = _load_all_skills()

        names = [s.name for s in skills]
        assert expected_name in names, (
            f"expected {expected_name} in loaded skills, got {names}"
        )
        skill = next(s for s in skills if s.name == expected_name)
        assert skill.description, f"{expected_name} has empty description"

    def test_discovers_skills_from_custom_dir(self, fake_project_root: Path) -> None:
        """Two fake SKILL.md files under ``skills/`` → both are returned."""
        skills_dir = fake_project_root / "skills"
        _write_skill(skills_dir / "alpha-skill", "alpha-skill", "First skill")
        _write_skill(skills_dir / "beta-skill", "beta-skill", "Second skill")

        skills = _load_all_skills()

        names = sorted(s.name for s in skills)
        assert names == ["alpha-skill", "beta-skill"]

    def test_returns_empty_when_skills_dir_missing(
        self, fake_project_root: Path
    ) -> None:
        """Missing ``skills/`` directory → returns [] without raising."""
        # fake_project_root has no skills/ subdirectory
        assert not (fake_project_root / "skills").exists()

        skills = _load_all_skills()

        assert skills == []

    def test_skips_entries_without_skill_md(self, fake_project_root: Path) -> None:
        """Non-skill entries (loose files, dirs without SKILL.md) are skipped."""
        skills_dir = fake_project_root / "skills"
        _write_skill(skills_dir / "valid-skill", "valid-skill", "Real skill")
        # Directory without SKILL.md
        (skills_dir / "empty-dir").mkdir()
        # Stray file directly under skills/
        (skills_dir / "README.md").write_text("not a skill")

        skills = _load_all_skills()

        assert [s.name for s in skills] == ["valid-skill"]


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
            patch("agents.default_agent._load_all_skills", return_value=[]),
            patch("agents.default_agent.Agent") as mock_agent_cls,
        ):
            create_default_agent("http://localhost:9200")

        mock_agent_cls.assert_called_once()
        tools_kwarg = mock_agent_cls.call_args.kwargs["tools"]
        assert tools_kwarg == mock_mcp_tools

    def test_attaches_logging_agent_skills_plugin(self) -> None:
        """When skills are discovered, a ``LoggingAgentSkills`` plugin is attached."""
        fake_skill = Skill(name="fake-skill", description="a fake")
        with patch("agents.default_agent._load_all_skills", return_value=[fake_skill]):
            agent = create_default_agent("http://localhost:9200")

        plugins = agent._plugin_registry._plugins
        assert "agent_skills" in plugins
        skills_plugin = plugins["agent_skills"]
        assert isinstance(skills_plugin, LoggingAgentSkills)
        assert "fake-skill" in skills_plugin._skills

    def test_no_skills_plugin_when_skills_dir_empty(self) -> None:
        """Zero skills discovered → no ``agent_skills`` plugin attached."""
        with patch("agents.default_agent._load_all_skills", return_value=[]):
            agent = create_default_agent("http://localhost:9200")

        assert "agent_skills" not in agent._plugin_registry._plugins

    @pytest.mark.parametrize("expected_name", EXPECTED_REPO_SKILLS)
    def test_real_skill_registered_in_plugin(self, expected_name: str) -> None:
        """End-to-end: each expected skill lands inside the ``agent_skills`` plugin."""
        # Do NOT patch _load_all_skills — let the real function run against the repo.
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


@pytest.mark.usefixtures("patch_mcp")
class TestDefaultAgentModelSelection:
    """Group 4 — Bedrock model selection via ``BEDROCK_INFERENCE_PROFILE_ARN``.

    Regression coverage for issue #94: the default agent must respect the
    ``BEDROCK_INFERENCE_PROFILE_ARN`` env var and fall back to the documented
    Strands default when it is unset.
    """

    def test_uses_inference_profile_arn_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``BEDROCK_INFERENCE_PROFILE_ARN`` is set, the agent uses that ARN."""
        test_arn = (
            "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/"
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        monkeypatch.setenv("BEDROCK_INFERENCE_PROFILE_ARN", test_arn)

        with (
            patch("agents.default_agent._load_all_skills", return_value=[]),
            patch("agents.default_agent.BedrockModel") as mock_bedrock_cls,
            patch("agents.default_agent._get_aws_session") as mock_session_fn,
        ):
            create_default_agent("http://localhost:9200")

        mock_bedrock_cls.assert_called_once_with(
            model_id=test_arn,
            boto_session=mock_session_fn.return_value,
            streaming=True,
        )

    def test_falls_back_to_default_when_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``BEDROCK_INFERENCE_PROFILE_ARN`` is unset, falls back to default."""
        monkeypatch.delenv("BEDROCK_INFERENCE_PROFILE_ARN", raising=False)

        with (
            patch("agents.default_agent._load_all_skills", return_value=[]),
            patch("agents.default_agent.BedrockModel") as mock_bedrock_cls,
            patch("agents.default_agent._get_aws_session") as mock_session_fn,
        ):
            create_default_agent("http://localhost:9200")

        mock_bedrock_cls.assert_called_once_with(
            model_id=_DEFAULT_BEDROCK_MODEL_ID,
            boto_session=mock_session_fn.return_value,
            streaming=True,
        )
