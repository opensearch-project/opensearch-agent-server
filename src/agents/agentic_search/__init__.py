"""Agentic-search agent: natural-language query -> OpenSearch DSL, via ``/invoke``.

Exposes :func:`create_agentic_search_agent`, the factory registered with the
orchestrator (mirrors ``create_default_agent`` / ``create_art_agent``).
"""

from __future__ import annotations

from agents.agentic_search.agent import AgenticSearchAgent
from server.config import get_config


def create_agentic_search_agent(opensearch_url: str) -> AgenticSearchAgent:
    """Create the ``agentic_search`` agent.

    TLS verification for the mapping-fetch client follows the server config
    (``OPENSEARCH_VERIFY_CERTS``), defaulting to disabled to match the MCP path
    against the security plugin's self-signed demo certificates.
    """
    config = get_config()
    return AgenticSearchAgent(
        opensearch_url, verify_certs=config.opensearch_verify_certs
    )


__all__ = ["AgenticSearchAgent", "create_agentic_search_agent"]
