"""Fallback Agent — General OpenSearch Assistant.

A simple Strands agent with all OpenSearch MCP Server tools.
Handles general queries when no specialized sub-agent matches the page context.
"""

from __future__ import annotations

import os

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

from agents.default_config import FallbackAgentConfig
from agents.tool_filter import _select_tools
from server.constants import DEFAULT_MCP_SERVER_URL
from utils.logging_helpers import get_logger, log_info_event

logger = get_logger(__name__)

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

# Agent tool configuration — reads FALLBACK_* env vars once at import time.
_fallback_config = FallbackAgentConfig()


def create_fallback_agent(
    opensearch_url: str, headers: dict[str, str] | None = None
) -> Agent:
    """Create the fallback agent with OpenSearch MCP tools.

    Connects to the OpenSearch MCP server via Streamable HTTP transport.
    The server URL defaults to ``http://localhost:3001/mcp`` and can be
    overridden with the ``MCP_SERVER_URL`` environment variable.

    The set of tools available to the agent is controlled by the
    ``FALLBACK_AGENT_TOOLS`` environment variable (comma-separated category
    names or individual tool names).  When the variable is unset or empty
    all MCP tools are available.

    Args:
        opensearch_url: OpenSearch cluster URL (informational — the MCP
            server is assumed to already be configured for this cluster).
        headers: Optional HTTP headers to forward to the MCP server
            (e.g. Authorization for OpenSearch authentication).

    Returns:
        Configured Strands Agent with MCP tools.
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)

    mcp_client = MCPClient(lambda: streamablehttp_client(mcp_server_url, headers=headers))
    mcp_client.start()

    all_tools = list(mcp_client.list_tools_sync())
    tools = _select_tools(all_tools, _fallback_config.agent_tools)

    agent = Agent(
        system_prompt=FALLBACK_SYSTEM_PROMPT,
        tools=tools,
    )

    log_info_event(
        logger,
        f"Fallback agent initialized with {len(tools)}/{len(all_tools)} MCP tools "
        f"(server={mcp_server_url}).",
        "fallback_agent.initialized",
        tool_count=len(tools),
        total_tool_count=len(all_tools),
        mcp_server_url=mcp_server_url,
        opensearch_url=opensearch_url,
    )

    return agent
