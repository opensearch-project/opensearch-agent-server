"""ART Agent — Search Relevance Testing Sub-Agent.

Wraps the ART (Automated Relevance Testing) orchestrator agent from the
os-art package. This agent handles search relevance tasks when the user
is on the search-relevance page in OpenSearch Dashboards.

The ART orchestrator coordinates 4 specialized agents:
  - hypothesis_agent: Generate and test search improvement hypotheses
  - evaluation_agent: Offline relevance evaluation (NDCG, MAP, MRR)
  - user_behavior_analysis_agent: Analyze UBI click data
  - online_testing_agent: Run interleaved A/B tests
"""

from __future__ import annotations

import os

from strands import Agent

from utils.logging_helpers import get_logger, log_info_event
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp import MCPClient

logger = get_logger(__name__)

# Default Bedrock model — same as Strands SDK default.
# Used when BEDROCK_INFERENCE_PROFILE_ARN is not explicitly set.
_DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

DEFAULT_MCP_SERVER_URL = "http://localhost:3001/mcp"

ART_SYSTEM_PROMPT="You are an expert search relevance tuning system."

def create_art_agent(opensearch_url: str) -> Agent:
    """Create the ART orchestrator agent.

    Reuses the same LLM model as the fallback agent (Strands default Bedrock
    model with credentials from ~/.aws/credentials).  If BEDROCK_INFERENCE_PROFILE_ARN
    is not set, we default to the standard Strands model ID so the ART specialized
    agents can create their own BedrockModel instances without error.

    Args:
        opensearch_url: OpenSearch cluster URL.

    Returns:
        Configured Strands Agent for ART orchestration.
    """
    # Ensure BEDROCK_INFERENCE_PROFILE_ARN is set before importing os_art,
    # because os_art.agents.specialized_agents reads it at module level.
    if not os.getenv("BEDROCK_INFERENCE_PROFILE_ARN"):
        os.environ["BEDROCK_INFERENCE_PROFILE_ARN"] = _DEFAULT_BEDROCK_MODEL_ID
        log_info_event(
            logger,
            f"BEDROCK_INFERENCE_PROFILE_ARN not set, defaulting to {_DEFAULT_BEDROCK_MODEL_ID}",
            "art_agent.default_model",
            model_id=_DEFAULT_BEDROCK_MODEL_ID,
        )

    # Also default BEDROCK_HAIKU_INFERENCE_PROFILE_ARN (used by user_behavior_analysis_agent)
    if not os.getenv("BEDROCK_HAIKU_INFERENCE_PROFILE_ARN"):
        os.environ["BEDROCK_HAIKU_INFERENCE_PROFILE_ARN"] = _DEFAULT_BEDROCK_MODEL_ID

    log_info_event(
        logger,
        f"Initializing ART agent with OpenSearch at {opensearch_url}",
        "art_agent.initializing",
        opensearch_url=opensearch_url,
    )

    mcp_server_url = os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)

    mcp_client = MCPClient(lambda: streamablehttp_client(mcp_server_url))

    agent = Agent(
        system_prompt=ART_SYSTEM_PROMPT,
        tools=[mcp_client],
    )

    log_info_event(
        logger,
        "ART agent initialized successfully",
        "art_agent.initialized",
    )

    return agent
