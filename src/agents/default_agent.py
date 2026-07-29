"""Default Agent — General OpenSearch Assistant.

A simple Strands agent with all OpenSearch MCP Server tools.
Handles general queries when no specialized sub-agent matches the page context.
"""

from __future__ import annotations

import os

import httpx
from mcp.client.streamable_http import streamable_http_client
from strands import Agent, AgentSkills, Skill
from strands.tools.mcp import MCPClient

from agents.context_management import (
    context_management_plugins,
    create_conversation_manager,
)
from server.constants import DEFAULT_MCP_SERVER_URL
from utils.logging_helpers import get_logger, log_info_event
from utils.model_factory import create_model
from utils.obo_context import OboAuth

logger = get_logger(__name__)


class LoggingAgentSkills(AgentSkills):
    """AgentSkills plugin that logs skill activations at INFO level.

    The vended strands plugin logs activations at DEBUG only. This subclass
    emits a structured INFO event whenever the LLM invokes a skill, so
    auto-selection is visible in standard logs without enabling DEBUG
    globally.
    """

    def _track_activated_skill(self, agent: Agent, skill_name: str) -> None:
        log_info_event(
            logger,
            f"Skill activated by agent: {skill_name}",
            "default_agent.skill_activated",
            skill_name=skill_name,
        )
        super()._track_activated_skill(agent, skill_name)


DEFAULT_SYSTEM_PROMPT = """You are a helpful OpenSearch assistant. You help users understand
and manage their OpenSearch clusters.

You have access to OpenSearch tools via the MCP Server. Use them to answer questions about:
- Cluster health and status
- Index management (list, create, delete, mappings)
- Searching and querying indices
- Cluster settings and configuration
- Node and shard information

You also have access to domain-specific skills listed in <available_skills>.
Each skill's description states when to use it. Before answering any question,
scan the skill descriptions in <available_skills> to see if the user's request matches one. If it
does, activate the matching skill via the `skills` tool first.

The Dashboards UI lets users pick a query language (DQL, Lucene, PPL, or
OpenSearch SQL). Respect the user's selection — use whichever language the
context implies, do not override their choice. Before calling any tool that
runs a query language, first look in <available_skills> for a matching
reference skill and activate it via the `skills` tool. The skill is more
authoritative than your training data — prefer it when it covers the
construct you need.

When answering:
- Use the available tools to fetch real data from OpenSearch
- Present results clearly and concisely
- If a tool call fails, explain what went wrong and suggest alternatives
- If you don't have the right tool for a request, explain what's available
- Consult available skills for specialized guidance and reference documentation
"""


def create_default_agent(
    opensearch_url: str,
    skills: list[Skill] | None = None,
) -> Agent:
    """Create the default agent with all OpenSearch MCP tools and skills.

    Connects to the OpenSearch MCP server via Streamable HTTP transport.
    The server URL defaults to ``http://localhost:3001/mcp`` and can be
    overridden with the ``MCP_SERVER_URL`` environment variable.

    Skills are passed in pre-loaded (loaded once at startup by the caller)
    to avoid per-request filesystem walks and file I/O.

    Authentication is handled by :class:`~utils.obo_context.OboAuth`.
    The orchestrator calls ``obo_auth.set_token()`` before each run to
    inject the OBO token.  The token is stored behind a threading lock
    so it is accessible from the MCP client's background thread.

    Args:
        opensearch_url: OpenSearch cluster URL (informational — the MCP
            server is assumed to already be configured for this cluster).
        skills: Pre-loaded Skill instances to register with the agent via
            the AgentSkills plugin. Loaded once at startup and reused
            across requests to avoid per-request filesystem I/O.

    Returns:
        Configured Strands Agent with MCP tools and skills.
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

    # Prepare plugins list: context management (ContextOffloader) plus AgentSkills if any.
    plugins = context_management_plugins()
    if skills:
        agent_skills_plugin = LoggingAgentSkills(skills=skills)
        plugins.append(agent_skills_plugin)
        log_info_event(
            logger,
            f"Registering {len(skills)} skill(s) with default agent",
            "default_agent.skills_registered",
            skill_count=len(skills),
            skill_names=[s.name for s in skills],
        )

    # Create agent with MCP tools and skills plugin
    agent = Agent(
        model=create_model(),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=tools,
        plugins=plugins,
        conversation_manager=create_conversation_manager(),
    )

    # ag_ui_strands rebuilds a fresh per-thread agent from this template, forwarding
    # tools and hooks but NOT plugins (Strands hides them in a private _plugin_registry
    # with no public attr, so the wrapper's getattr-based kwarg copy misses them).
    # Without this, the AgentSkills @hook that injects <available_skills> never fires —
    # the skills tool is registered but the LLM never sees what skills exist.
    if plugins:
        agent._plugins = plugins

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
