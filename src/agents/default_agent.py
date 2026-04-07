"""Default Agent — General OpenSearch Assistant.

A simple Strands agent with all OpenSearch MCP Server tools.
Handles general queries when no specialized sub-agent matches the page context.
"""

from __future__ import annotations

import os

import httpx
from mcp.client.streamable_http import streamable_http_client
from strands import Agent
from strands.tools.mcp import MCPClient

from server.constants import DEFAULT_MCP_SERVER_URL
from utils.logging_helpers import get_logger, log_info_event
from utils.obo_context import OboAuth

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful OpenSearch assistant. You help users understand
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


def create_default_agent(opensearch_url: str) -> Agent:
    """Create the default agent with all OpenSearch MCP tools.

    Connects to the OpenSearch MCP server via Streamable HTTP transport.
    The server URL defaults to ``http://localhost:3001/mcp`` and can be
    overridden with the ``MCP_SERVER_URL`` environment variable.

    Authentication is handled by :class:`~utils.obo_context.OboAuth`.
    The orchestrator calls ``obo_auth.set_token()`` before each run to
    inject the OBO token.  The token is stored behind a threading lock
    so it is accessible from the MCP client's background thread.

    Args:
        opensearch_url: OpenSearch cluster URL (informational — the MCP
            server is assumed to already be configured for this cluster).

    Returns:
        Configured Strands Agent with MCP tools.
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)

    # OboAuth injects the OBO token into every outgoing httpx request.
    # The token is set by the orchestrator before each agent run via
    # set_token() and stored behind a threading.Lock — so the MCP
    # client's background thread can read it safely.
    obo_auth = OboAuth()
    http_client = httpx.AsyncClient(
        auth=obo_auth,
        timeout=httpx.Timeout(30, read=300),
        verify=False,
        follow_redirects=True,
    )

    mcp_client = MCPClient(
        lambda: streamable_http_client(mcp_server_url, http_client=http_client)
    )
    mcp_client.start()

    tools = list(mcp_client.list_tools_sync())

    agent = Agent(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=tools,
    )

    # Keep references to prevent GC from closing the MCP session and
    # to allow the orchestrator to set tokens on subsequent requests.
    agent._mcp_client = mcp_client  # prevent GC
    agent._obo_auth = obo_auth  # expose for token refresh

    tool_count = len(agent.tool_registry.registry)
    log_info_event(
        logger,
        f"Default agent initialized with {tool_count} MCP tools "
        f"(server={mcp_server_url}).",
        "default_agent.initialized",
        tool_count=tool_count,
        mcp_server_url=mcp_server_url,
        opensearch_url=opensearch_url,
    )

    return agent
