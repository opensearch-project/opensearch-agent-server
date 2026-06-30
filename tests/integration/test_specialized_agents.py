"""
Integration tests for Specialized Agents.

Tests agent creation, tool configuration, and error handling for specialized agents.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.art import specialized_agents
from helpers.specialized_agents_helpers import (
    patch_evaluation_agent_dependencies,
    patch_hypothesis_agent_dependencies,
    patch_ubi_agent_dependencies,
)

pytestmark = pytest.mark.integration


# Mock emitter to avoid missing dependencies in tests
@pytest.fixture(autouse=True)
def mock_get_emitter() -> Generator[None, None, None]:
    """Mock emitter so agent wrappers do not require external context."""
    with patch("agents.art.specialized_agents.monitored_tool", side_effect=lambda x: x):
        yield


@pytest.fixture(autouse=True)
def reset_mcp_client() -> Generator[None, None, None]:
    """Reset _mcp_client before and after each test to ensure isolation."""
    # Save original state
    original_client = specialized_agents._mcp_client

    # Reset before test
    specialized_agents._mcp_client = None

    yield

    # Restore original state after test
    specialized_agents._mcp_client = original_client


@pytest.mark.integration
class TestSetOpenSearchTools:
    """Tests for set_opensearch_tools function."""

    @pytest.mark.asyncio
    async def test_set_opensearch_tools(self):
        """Test that set_mcp_client sets the global mcp client variable."""
        from agents.art import specialized_agents

        mock_client = MagicMock()
        specialized_agents.set_mcp_client(mock_client)

        assert specialized_agents._mcp_client == mock_client


@pytest.mark.integration
class TestHypothesisAgent:
    """Tests for hypothesis_agent function."""

    @pytest.mark.asyncio
    async def test_hypothesis_agent_no_tools_configured(self):
        """Test that hypothesis_agent returns error when tools not configured."""
        # autouse fixture ensures tools are empty
        result = await specialized_agents.hypothesis_agent("test query")

        assert "Error: MCPClient not configured" in result

    @pytest.mark.asyncio
    async def test_hypothesis_agent_success(self):
        """Test that hypothesis_agent creates agent and invokes successfully."""
        # Set tools
        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value="Test hypothesis response")

        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent("test query")

            assert result == "Test hypothesis response"
            mock_agent.invoke_async.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_hypothesis_agent_rate_limit_error(self):
        """Test that hypothesis_agent handles rate limit errors."""
        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )

        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent("test query")

            assert "Rate limit" in result or "429" in result

    @pytest.mark.asyncio
    async def test_hypothesis_agent_general_error(self):
        """Test that hypothesis_agent handles general errors."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("General error"))

        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent("test query")

            assert "Error in hypothesis generation" in result
            assert "General error" in result


@pytest.mark.integration
class TestEvaluationAgent:
    """Tests for evaluation_agent function."""

    @pytest.mark.asyncio
    async def test_evaluation_agent_no_tools_configured(self):
        """Test that evaluation_agent returns error when tools not configured."""
        from agents.art import specialized_agents

        # autouse fixture ensures tools are empty
        result = await specialized_agents.evaluation_agent("test query")

        assert "Error: MCPClient not configured" in result

    @pytest.mark.asyncio
    async def test_evaluation_agent_success(self):
        """Test that evaluation_agent creates agent and invokes successfully."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value="Test evaluation response")

        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("test query")

            assert result == "Test evaluation response"
            mock_agent.invoke_async.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_evaluation_agent_rate_limit_error(self):
        """Test that evaluation_agent handles rate limit errors."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception("429 Too Many Requests")
        )

        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("test query")

            assert "Rate limit" in result or "429" in result

    @pytest.mark.asyncio
    async def test_evaluation_agent_general_error(self):
        """Test that evaluation_agent handles general errors (non-rate-limit)."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("General error"))

        with patch_evaluation_agent_dependencies(mock_agent):
            result = await specialized_agents.evaluation_agent("test query")

            assert "Error in evaluation" in result
            assert "General error" in result


@pytest.mark.integration
class TestUserBehaviorAnalysisAgent:
    """Tests for user_behavior_analysis_agent function."""

    @pytest.mark.asyncio
    async def test_user_behavior_analysis_agent_no_tools_configured(self):
        """Test that user_behavior_analysis_agent returns error when tools not configured."""
        from agents.art import specialized_agents

        # autouse fixture ensures tools are empty
        result = await specialized_agents.user_behavior_analysis_agent("test query")

        assert "Error: MCPClient not configured" in result

    @pytest.mark.asyncio
    async def test_user_behavior_analysis_agent_success(self):
        """Test that user_behavior_analysis_agent creates agent and invokes successfully."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(return_value="Test UBI analysis response")

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent("test query")

            assert result == "Test UBI analysis response"
            mock_agent.invoke_async.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_user_behavior_analysis_agent_error_handling(self):
        """Test that user_behavior_analysis_agent handles errors."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(side_effect=Exception("Test error"))

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent("test query")

            assert "Error in user behavior analysis" in result
            assert "Test error" in result

    @pytest.mark.asyncio
    async def test_user_behavior_analysis_agent_rate_limit_error(self):
        """Test that user_behavior_analysis_agent handles rate limit errors."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception("429 Too Many Requests")
        )

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent("test query")

            assert "Rate limit" in result or "429" in result




@pytest.mark.integration
class TestSpecializedAgentsErrorHandling:
    """Additional error handling tests for specialized agents."""

    @pytest.mark.asyncio
    async def test_hypothesis_agent_tool_failure(self):
        """Test hypothesis agent when tool call fails during execution."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        # Simulate tool failure during agent invocation
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception(
                "Tool execution failed: create_experiment returned error"
            )
        )

        with patch_hypothesis_agent_dependencies(mock_agent):
            result = await specialized_agents.hypothesis_agent(
                "Analyze search relevance issue"
            )

            # Should handle tool errors gracefully
            assert "Error in hypothesis generation" in result
            assert "Tool execution failed" in result
            mock_agent.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_agent_timeout(self):
        """Test evaluation agent timeout handling."""
        import asyncio

        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

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

            # Should timeout gracefully, emit error event
            assert "Error in evaluation" in result
            assert "timed out" in result.lower() or "timeout" in result.lower()
            mock_agent.invoke_async.assert_called_once()


    @pytest.mark.asyncio
    async def test_user_behavior_agent_missing_data(self):
        """Test user behavior agent with missing UBI data."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        # Simulate missing data scenario - agent should handle this gracefully
        mock_agent.invoke_async = AsyncMock(
            return_value="No UBI data available for the specified time range. Please check that the ubi_queries and ubi_events indices contain data."
        )

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent(
                "Analyze user engagement for query 'laptop'"
            )

            # Should handle missing data gracefully
            assert result is not None
            assert isinstance(result, str)
            # The agent should return a helpful message about missing data
            assert "UBI" in result or "data" in result.lower()

    @pytest.mark.asyncio
    async def test_user_behavior_agent_tool_error_missing_data(self):
        """Test user behavior agent when UBI tools fail due to missing data."""
        from agents.art import specialized_agents

        specialized_agents._mcp_client = MagicMock()

        mock_agent = MagicMock()
        # Simulate tool error when data is missing
        mock_agent.invoke_async = AsyncMock(
            side_effect=Exception("Index ubi_queries not found or empty")
        )

        with patch_ubi_agent_dependencies(mock_agent):
            result = await specialized_agents.user_behavior_analysis_agent(
                "Analyze user engagement"
            )

            # Should handle missing data errors gracefully
            assert "Error in user behavior analysis" in result
            assert "ubi_queries" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_agent_initialization_missing_tools(self):
        """Test agent initialization when tools are not available."""
        from agents.art import specialized_agents

        # Ensure client is unset (fixture handles this)
        assert specialized_agents._mcp_client is None

        # Test all agents without tools - should initialize with available tools, log warnings
        # Hypothesis agent
        result = await specialized_agents.hypothesis_agent("test query")
        assert "Error: MCPClient not configured" in result

        # Evaluation agent
        result = await specialized_agents.evaluation_agent("test query")
        assert "Error: MCPClient not configured" in result

        # User behavior agent
        result = await specialized_agents.user_behavior_analysis_agent("test query")
        assert "Error: MCPClient not configured" in result
