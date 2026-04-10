"""
Unit tests for specialized agents error scenarios.

Tests error paths and edge cases for specialized agent operations including:
- Tool call failures
- Timeout handling
- Rate limit handling
- Missing data scenarios
- Missing tools initialization
"""

import asyncio
from collections.abc import Generator
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.art import specialized_agents
from helpers.specialized_agents_helpers import (
    patch_evaluation_agent_dependencies,
    patch_hypothesis_agent_dependencies,
    patch_ubi_agent_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_mcp_client() -> Generator[None, None, None]:
    """Reset _mcp_client and _mcp_tools before and after each test to ensure isolation."""
    # Save original state
    original_client = specialized_agents._mcp_client
    original_tools = specialized_agents._mcp_tools

    # Reset before test
    specialized_agents._mcp_client = None
    specialized_agents._mcp_tools = None

    yield

    # Restore original state after test
    specialized_agents._mcp_client = original_client
    specialized_agents._mcp_tools = original_tools


@pytest.fixture(autouse=True)
def mock_monitor() -> Generator[None, None, None]:
    """Mock emitter to avoid dependencies."""
    with patch("agents.art.specialized_agents.monitored_tool", side_effect=lambda x: x):
        yield


class TestSpecializedAgentsErrors:
    """Test specialized agent error scenarios."""

    @pytest.mark.asyncio
    async def test_hypothesis_agent_tool_failure(self):
        """Test hypothesis agent when tool call fails."""
        # Should handle tool errors gracefully
        specialized_agents._mcp_client = MagicMock()
        specialized_agents._mcp_tools = [MagicMock()]

        mock_agent = MagicMock()
        # Simulate tool failure during agent invocation
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception("Tool execution failed: create_experiment")
        )

        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent(
                "Analyze search relevance issue"
            )

            # Should return error message, not crash
            assert "Error in hypothesis generation" in result
            assert "Tool execution failed" in result
            mock_agent.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_agent_timeout(self):
        """Test evaluation agent timeout handling."""
        # Should timeout gracefully, emit error event
        specialized_agents._mcp_client = MagicMock()
        specialized_agents._mcp_tools = [MagicMock()]

        mock_agent = MagicMock()
        # Simulate timeout during agent invocation
        mock_agent.invoke_async = AsyncMock(
            side_effect=asyncio.TimeoutError(
                "Agent execution timed out after 30 minutes"
            )
        )

        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent(
                "Evaluate search configuration"
            )

            # Should return error message with timeout indication
            assert "Error in evaluation" in result
            assert "timed out" in result.lower() or "timeout" in result.lower()
            mock_agent.invoke_async.assert_called_once()


    @pytest.mark.asyncio
    async def test_user_behavior_agent_missing_data(self):
        """Test user behavior agent with missing UBI data."""
        # Should handle missing data gracefully
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()
        specialized_agents._mcp_tools = [MagicMock()]

        mock_agent = MagicMock()
        # Simulate missing data scenario - agent should handle this gracefully
        # by returning a message about missing data rather than crashing
        mock_agent.invoke_async = AsyncMock(
            return_value="No UBI data available for the specified time range. Please check that the ubi_queries and ubi_events indices contain data."
        )

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent(
                "Analyze user engagement for query 'laptop'"
            )

            # Should return a response (even if indicating missing data), not crash
            assert result is not None
            assert isinstance(result, str)
            # The agent should handle missing data and return a helpful message
            mock_agent.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_behavior_agent_tool_error_missing_data(self):
        """Test user behavior agent when UBI tools fail due to missing data."""
        # Should handle tool errors due to missing data gracefully
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()
        specialized_agents._mcp_tools = [MagicMock()]

        mock_agent = MagicMock()
        # Simulate tool error when data is missing
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception("Index ubi_queries not found or empty")
        )

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent(
                "Analyze user engagement"
            )

            # Should return error message, not crash
            assert "Error in user behavior analysis" in result
            assert "ubi_queries" in result or "not found" in result.lower()
            mock_agent.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_initialization_missing_tools(self):
        """Test agent initialization when tools are not available."""
        # Should initialize with available tools, log warnings
        # This tests the scenario where _opensearch_tools is empty
        # The agent should return an error message, not crash
        from agents.art import specialized_agents

        # Ensure client and tools are unset (fixture handles this)
        assert specialized_agents._mcp_client is None
        assert specialized_agents._mcp_tools is None

        # Test hypothesis agent without tools
        result = await specialized_agents.hypothesis_agent("test query")
        assert "MCP tools not configured" in result

        # Test evaluation agent without tools
        result = await specialized_agents.evaluation_agent("test query")
        assert "MCP tools not configured" in result

        # Test user behavior agent without tools
        result = await specialized_agents.user_behavior_analysis_agent("test query")
        assert "MCP tools not configured" in result

    @pytest.mark.asyncio
    async def test_hypothesis_agent_agent_creation_error(self):
        """Test hypothesis agent when Agent creation fails."""
        # Should handle agent creation errors gracefully
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()
        specialized_agents._mcp_tools = [MagicMock()]

        # Simulate Agent creation failure using ExitStack to handle many patches
        stack = ExitStack()
        mock_agent_class = stack.enter_context(
            patch("agents.art.specialized_agents.Agent")
        )
        stack.enter_context(patch("agents.art.specialized_agents.BedrockModel"))
        stack.enter_context(patch("agents.art.specialized_agents.bedrock_session"))
        stack.enter_context(
            patch("tools.art.experiment_tools.aggregate_experiment_results")
        )

        try:
            # Simulate Agent creation failure
            mock_agent_class.side_effect = Exception(
                "Failed to create agent: missing dependencies"
            )

            result = await specialized_agents.hypothesis_agent("test query")

            # Should return error message
            assert "Error in hypothesis generation" in result
        finally:
            stack.close()

    @pytest.mark.asyncio
    async def test_evaluation_agent_connection_error(self):
        """Test evaluation agent when connection to Bedrock fails."""
        # Should handle connection errors gracefully
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()
        specialized_agents._mcp_tools = [MagicMock()]

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(
            side_effect=ConnectionError("Failed to connect to Bedrock service")
        )

        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("test query")

            # Should return error message
            assert "Error in evaluation" in result
            assert "connect" in result.lower() or "bedrock" in result.lower()
            mock_agent.invoke_async.assert_called_once()
