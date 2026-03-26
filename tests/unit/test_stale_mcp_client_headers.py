"""Tests for dynamic headers resolution in MCP client transport.

Validates that the MCP transport lambda calls headers_getter() dynamically to
fetch current headers on each connection, allowing cached agents to use fresh
per-request auth credentials (e.g., rotated tokens) without requiring agent
recreation.

This test prevents regression of the stale headers bug where cached agents
would continue using headers from the first request instead of fetching
fresh headers on subsequent requests.
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import MagicMock, patch

from agents.art.art_agent import create_art_agent

_TEST_URL = "http://opensearch-test:9200"

_PATCHES = {
    "MCPClient": "agents.art.art_agent.MCPClient",
    "streamablehttp_client": "agents.art.art_agent.streamablehttp_client",
    "set_opensearch_tools": "agents.art.art_agent.set_opensearch_tools",
    "Agent": "agents.art.art_agent.Agent",
    "BedrockModel": "agents.art.art_agent.BedrockModel",
    "boto3_session": "agents.art.art_agent.boto3.Session",
}


def test_transport_lambda_fetches_dynamic_headers():
    """Verify transport lambda calls headers_getter() dynamically on each invocation.

    This test validates the fix for the stale headers bug. The transport lambda
    must call headers_getter() each time it's invoked, not capture a static
    snapshot of headers at creation time.

    Steps:
      1. Create a headers_getter that returns {"Authorization": "Bearer token-v1"}
      2. Call create_art_agent(url, headers_getter)
      3. Capture the transport lambda passed to MCPClient
      4. Invoke the transport lambda and verify it uses token-v1
      5. Update headers_getter to return {"Authorization": "Bearer token-v2"}
      6. Invoke the transport lambda again and verify it now uses token-v2

    This ensures cached agents can use fresh headers on each request.
    """
    captured_transport_lambda = None
    current_headers = {"Authorization": "Bearer token-v1"}

    def headers_getter():
        return current_headers

    def capture_mcp_client(transport_fn):
        nonlocal captured_transport_lambda
        captured_transport_lambda = transport_fn
        mock_client = MagicMock()
        mock_client.list_tools_sync.return_value = []
        return mock_client

    with (
        patch(_PATCHES["MCPClient"], side_effect=capture_mcp_client),
        patch(_PATCHES["streamablehttp_client"]) as mock_streamable,
        patch(_PATCHES["set_opensearch_tools"]),
        patch(_PATCHES["Agent"]),
        patch(_PATCHES["BedrockModel"]),
        patch(_PATCHES["boto3_session"]),
    ):
        create_art_agent(_TEST_URL, headers_getter)

        assert captured_transport_lambda is not None, (
            "MCPClient was not called with a transport lambda"
        )

        # First invocation with token-v1
        captured_transport_lambda()
        first_call_headers = mock_streamable.call_args[1].get("headers")
        assert first_call_headers == {"Authorization": "Bearer token-v1"}, (
            f"First call should use token-v1, got {first_call_headers}"
        )

        # Update headers to token-v2
        current_headers["Authorization"] = "Bearer token-v2"

        # Second invocation should get fresh headers
        captured_transport_lambda()
        second_call_headers = mock_streamable.call_args[1].get("headers")
        assert second_call_headers == {"Authorization": "Bearer token-v2"}, (
            "Bug: transport lambda did not fetch fresh headers via headers_getter(). "
            f"Expected token-v2, got {second_call_headers}. "
            "The cached agent should use current headers on each request."
        )