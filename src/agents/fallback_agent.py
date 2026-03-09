"""Fallback Agent — General OpenSearch Assistant.

A simple Strands agent with all OpenSearch MCP Server tools.
Handles general queries when no specialized sub-agent matches the page context.
"""

from __future__ import annotations

import os

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

from utils.logging_helpers import get_logger, log_info_event

logger = get_logger(__name__)

# Default URL for the OpenSearch MCP server (Streamable HTTP).
DEFAULT_MCP_SERVER_URL = "http://localhost:3001/mcp"

FALLBACK_SYSTEM_PROMPT = """You are a helpful OpenSearch assistant. You help users understand
and manage their OpenSearch clusters.

You have access to OpenSearch tools via the MCP Server. Use them to answer questions about:
- Cluster health and status
- Index management (list, create, delete, mappings)
- Searching and querying indices
- Cluster settings and configuration
- Node and shard information

When answering:
- Use the available tools to fetch real data from OpenSearch
- Present results clearly and concisely
- If a tool call fails, explain what went wrong and suggest alternatives
- If you don't have the right tool for a request, explain what's available
"""


def create_fallback_agent(opensearch_url: str) -> Agent:
    """Create the fallback agent with all OpenSearch MCP tools.

    Connects to the OpenSearch MCP server via Streamable HTTP transport.
    The server URL defaults to ``http://localhost:3001/mcp`` and can be
    overridden with the ``MCP_SERVER_URL`` environment variable.

    Args:
        opensearch_url: OpenSearch cluster URL (informational — the MCP
            server is assumed to already be configured for this cluster).

    Returns:
        Configured Strands Agent with MCP tools.
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)

    mcp_client = MCPClient(lambda: streamablehttp_client(mcp_server_url))

    agent = Agent(
        system_prompt=FALLBACK_SYSTEM_PROMPT,
        tools=[mcp_client],
    )

    tool_count = len(agent.tool_registry.registry)
    log_info_event(
        logger,
        f"Fallback agent initialized with {tool_count} MCP tools "
        f"(server={mcp_server_url}).",
        "fallback_agent.initialized",
        tool_count=tool_count,
        mcp_server_url=mcp_server_url,
        opensearch_url=opensearch_url,
    )

    return agent
