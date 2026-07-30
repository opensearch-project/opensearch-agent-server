"""
Helper functions for patching dependencies in specialized agent tests.

These functions reduce code duplication across test files by providing
reusable context managers for patching all dependencies needed for testing
specialized agents.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch


def patch_hypothesis_agent_dependencies(mock_agent: MagicMock) -> ExitStack:
    """Helper function to patch all dependencies for hypothesis_agent.

    Args:
        mock_agent: Mock agent instance to be returned by Agent constructor

    Returns:
        ExitStack context manager with all patches applied
    """
    stack = ExitStack()
    stack.enter_context(
        patch("agents.art.specialized_agents.Agent", return_value=mock_agent)
    )
    stack.enter_context(patch("agents.art.specialized_agents.create_model"))
    stack.enter_context(
        patch("tools.art.experiment_tools.aggregate_experiment_results")
    )
    return stack


def patch_evaluation_agent_dependencies(mock_agent: MagicMock) -> ExitStack:
    """Helper function to patch all dependencies for evaluation_agent.

    Args:
        mock_agent: Mock agent instance to be returned by Agent constructor

    Returns:
        ExitStack context manager with all patches applied
    """
    stack = ExitStack()
    stack.enter_context(
        patch("agents.art.specialized_agents.Agent", return_value=mock_agent)
    )
    stack.enter_context(patch("agents.art.specialized_agents.create_model"))
    stack.enter_context(
        patch("tools.art.experiment_tools.aggregate_experiment_results")
    )
    return stack


def patch_ubi_agent_dependencies(mock_agent: MagicMock) -> ExitStack:
    """Helper function to patch all dependencies for user_behavior_analysis_agent.

    Args:
        mock_agent: Mock agent instance to be returned by Agent constructor

    Returns:
        ExitStack context manager with all patches applied
    """
    stack = ExitStack()
    stack.enter_context(
        patch("agents.art.specialized_agents.Agent", return_value=mock_agent)
    )
    stack.enter_context(patch("agents.art.specialized_agents.create_model"))
    return stack
