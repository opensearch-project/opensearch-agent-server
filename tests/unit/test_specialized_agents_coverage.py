# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for agents.art.specialized_agents.

Covers set_mcp_client plus the success and error/rate-limit branches of the
three specialized agent tools. Agent/model creation is fully mocked; no MCP,
model, or network calls are made.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from helpers.specialized_agents_helpers import (
    patch_evaluation_agent_dependencies,
    patch_hypothesis_agent_dependencies,
    patch_ubi_agent_dependencies,
)

from agents.art import specialized_agents

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_mcp_state() -> Generator[None, None, None]:
    orig_client = specialized_agents._mcp_client
    orig_tools = specialized_agents._mcp_tools
    specialized_agents._mcp_client = None
    specialized_agents._mcp_tools = None
    yield
    specialized_agents._mcp_client = orig_client
    specialized_agents._mcp_tools = orig_tools


@pytest.fixture
def patch_plugins():
    with (
        patch(
            "agents.art.specialized_agents.context_management_plugins", return_value=[]
        ),
        patch("agents.art.specialized_agents.create_conversation_manager"),
    ):
        yield


class TestSetMcpClient:
    def test_resolves_and_stores_tools(self):
        client = MagicMock()
        client.list_tools_sync.return_value = ["tool-a", "tool-b"]
        specialized_agents.set_mcp_client(client)
        assert specialized_agents._mcp_client is client
        assert specialized_agents._mcp_tools == ["tool-a", "tool-b"]


class TestMissingTools:
    async def test_hypothesis_missing_tools(self):
        result = await specialized_agents.hypothesis_agent("q")
        assert "MCP tools not configured" in result

    async def test_evaluation_missing_tools(self):
        result = await specialized_agents.evaluation_agent("q")
        assert "MCP tools not configured" in result

    async def test_ubi_missing_tools(self):
        result = await specialized_agents.user_behavior_analysis_agent("q")
        assert "MCP tools not configured" in result


class TestHypothesisAgent:
    async def test_success(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value="hypothesis-result")
        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent("analyze")
        assert "hypothesis-result" in result
        mock_agent.invoke_async.assert_awaited_once()

    async def test_rate_limit(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("Rate limit reached"))
        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent("q")
        assert "Rate limit reached" in result

    async def test_generic_error(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("db down"))
        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent("q")
        assert "Error in hypothesis generation" in result


class TestEvaluationAgent:
    async def test_success(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value="eval-result")
        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("evaluate")
        assert "eval-result" in result

    async def test_429_error(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("HTTP 429"))
        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("q")
        assert "Rate limit reached" in result

    async def test_generic_error(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("kaboom"))
        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("q")
        assert "Error in evaluation" in result


class TestUserBehaviorAgent:
    async def test_success(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value="ubi-result")
        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent("ctr")
        assert "ubi-result" in result

    async def test_generic_error(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("boom"))
        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent("q")
        assert "Error in user behavior analysis" in result

    async def test_rate_limit(self, patch_plugins):
        specialized_agents._mcp_tools = [MagicMock()]
        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("rate limit hit"))
        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent("q")
        assert "Rate limit reached" in result
