# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Pytest configuration and shared fixtures for opensearch-agent-server tests.
"""

import os
from collections.abc import Callable, Generator
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Environment variables — set before any imports that read config at module level
# ---------------------------------------------------------------------------
os.environ["OPENSEARCH_URL"] = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

# Mock AWS credentials for tests (real calls are patched out in unit tests)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test-key")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test-secret")
os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_env(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, str]]:
    """Patch environment variables for the duration of a test.

    Usage:
        def test_something(patch_env):
            patch_env(OPENSEARCH_URL="http://other:9200")
    """

    def _patch(clear: bool = False, **kwargs: str) -> dict[str, str]:
        if clear:
            for key in dict(os.environ):
                monkeypatch.delenv(key, raising=False)
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
        return kwargs

    return _patch


@pytest.fixture
def test_opensearch_url() -> str:
    """Returns the test OpenSearch URL (TEST_OPENSEARCH_URL env var, default localhost:9200)."""
    return os.getenv("TEST_OPENSEARCH_URL", "http://localhost:9200")


# ---------------------------------------------------------------------------
# MCP & Specialized Agents Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    """Return a mock MCPClient configured with resolved tools."""
    client = MagicMock()
    client.list_tools_sync.return_value = [MagicMock()]
    return client


@pytest.fixture
def configured_specialized_agents(
    mock_mcp_client: MagicMock,
) -> Generator[MagicMock, None, None]:
    """Configure specialized_agents with a mock MCPClient and clean up on teardown."""
    from agents.art import specialized_agents

    specialized_agents.set_mcp_client(mock_mcp_client)
    yield mock_mcp_client

    specialized_agents._mcp_client = None
    specialized_agents._mcp_tools = None
