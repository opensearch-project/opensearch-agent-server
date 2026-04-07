"""ART Agent — Search Relevance Testing Sub-Agent.

Orchestrates 3 specialized agents for search relevance tuning:
  - hypothesis_agent: Generate and test search improvement hypotheses
  - evaluation_agent: Offline relevance evaluation (NDCG, MAP, Precision)
  - user_behavior_analysis_agent: Analyze UBI click data
"""

from __future__ import annotations

import os

import boto3
import httpx
from mcp.client.streamable_http import streamable_http_client
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient

from agents.art.specialized_agents import (
    evaluation_agent,
    hypothesis_agent,
    set_mcp_client,
    user_behavior_analysis_agent,
)
from utils.logging_helpers import get_logger, log_info_event
from utils.obo_context import OboAuth

logger = get_logger(__name__)

# Default Bedrock model — same as Strands SDK default.
# Used when BEDROCK_INFERENCE_PROFILE_ARN is not explicitly set.
_DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

DEFAULT_MCP_SERVER_URL = "http://localhost:3001/mcp"

ORCHESTRATOR_SYSTEM_PROMPT = """You are a search relevance tuning orchestrator agent.

Your role is to help users solve search quality issues in OpenSearch by coordinating with specialized agents:

1. **user_behavior_analysis_agent**: Use this agent when users need to understand actual user engagement
   and behavior patterns. This agent analyzes UBI data including CTR metrics, click patterns, zero-click
   rates, and can identify poorly performing queries or high-engagement content.

2. **hypothesis_agent**: Use this agent when users report search relevance issues or need help
   understanding why search results are poor. This agent will analyze queries, examine results,
   check user behavior data, and generate hypotheses about how to resolve the search relevance issues
   and improve search result quality. Use this agent to run pairwise comparison experiments that
   show the quantitative difference between search configurations through change based metrics.
   A quantitative change on its own does not automatically mean improved quality.

3. **evaluation_agent**: Use this agent when users need to evaluate search relevance offline,
   compare search configurations, calculate metrics to measure search relevance
   or summarize past experiments. No other metrics than NDCG, MAP, Precision are available. Stick to these.
   Use this agent to run pointwise experiments that show the qualitative difference between search configurations
   through search quality metrics.


Your process:
1. Understand the user's request
2. Determine which specialized agent(s) to use
3. Delegate tasks to the appropriate agent(s)
4. Synthesize and present the results to the user

Apply the search relevance tuning process where possible:
1. Identify the relevance issue
2. Generate hypotheses including a first sanity check and smoke test using pairwise experiments.
3. Test the hypothesis using:
   a) Offline evaluation with pointwise experiments (judgment-based metrics)
   Choose based on what the user needs.
   Prefer offline if no offline evaluation has been done yet for the current process.
4. Go back to generating hypotheses if validation fails.

Always ask the user for confirmation before executing a step, unless the user gives you the permission
to proceed without their confirmation.

Be helpful, clear, and ensure the user gets complete answers by leveraging the right specialists.
"""


def _get_aws_session() -> boto3.Session:
    """Create a boto3 session using the default AWS credential provider chain.

    Supports environment variables, ~/.aws/credentials, IAM roles,
    EC2 instance profiles, ECS task roles, and temporary credentials.
    """
    return boto3.Session()


def _create_orchestrator_model(inference_profile_arn: str) -> BedrockModel:
    """Create a BedrockModel for the orchestrator."""
    return BedrockModel(
        model_id=inference_profile_arn,
        boto_session=_get_aws_session(),
        streaming=True,
    )


def create_art_agent(opensearch_url: str) -> Agent:
    """Create the ART orchestrator agent.

    Initializes the MCP connection to OpenSearch via MCPClient, configures the
    specialized sub-agents with the resulting tools, and returns the orchestrator Agent.

    Authentication is handled by :class:`~utils.obo_context.OboAuth`.
    The orchestrator calls ``obo_auth.set_token()`` before each run to
    inject the OBO token.  The token is stored behind a threading lock
    so it is accessible from the MCP client's background thread.

    Args:
        opensearch_url: OpenSearch cluster URL.

    Returns:
        A Strands Agent configured as the ART orchestrator.
    """
    # Default BEDROCK_INFERENCE_PROFILE_ARN if not set so specialized agents
    # can create their own BedrockModel instances without error.
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

    inference_profile_arn = os.environ["BEDROCK_INFERENCE_PROFILE_ARN"]

    log_info_event(
        logger,
        f"Initializing ART agent with OpenSearch at {opensearch_url}",
        "art_agent.initializing",
        opensearch_url=opensearch_url,
    )

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

    log_info_event(
        logger,
        f"MCP client created for {mcp_server_url}",
        "art_agent.mcp_created",
        mcp_server_url=mcp_server_url,
    )

    # Share the authenticated MCPClient with specialized agents so they
    # use the same transport (and auth headers) when calling MCP tools.
    set_mcp_client(mcp_client)

    # Build the orchestrator agent
    orchestrator = Agent(
        model=_create_orchestrator_model(inference_profile_arn),
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=[
            user_behavior_analysis_agent,
            hypothesis_agent,
            evaluation_agent,
        ],
    )

    # Keep references to prevent GC from closing the MCP session and
    # to allow the orchestrator to set tokens on subsequent requests.
    orchestrator._mcp_client = mcp_client  # prevent GC
    orchestrator._obo_auth = obo_auth  # expose for token refresh

    log_info_event(
        logger,
        "ART agent initialized successfully",
        "art_agent.initialized",
    )

    return orchestrator
