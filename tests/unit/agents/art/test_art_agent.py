# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for ``agents.art.art_agent.create_art_agent``.

The MCP transport, OBO auth, model factory, conversation manager and the
Strands ``Agent`` constructor are all mocked, so the factory path is exercised
without a network, an LLM, or a live MCP server.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from agents.art import art_agent

pytestmark = pytest.mark.unit


@contextmanager
def _patched_factory():
    """Patch every external collaborator of ``create_art_agent``.

    Yields a namespace of the individual mocks so a test can assert on the
    wiring (Agent kwargs, ``set_mcp_client`` call, ``mcp_client.start()``).
    """
    mcp_client = MagicMock(name="mcp_client")
    agent_obj = MagicMock(name="orchestrator")
    with (
        patch.object(art_agent, "OboAuth") as obo_cls,
        patch.object(art_agent.httpx, "AsyncClient") as http_cls,
        patch.object(art_agent, "MCPClient", return_value=mcp_client) as mcp_cls,
        patch.object(art_agent, "streamable_http_client") as stream_fn,
        patch.object(art_agent, "set_mcp_client") as set_client,
        patch.object(art_agent, "Agent", return_value=agent_obj) as agent_cls,
        patch.object(art_agent, "create_model", return_value=MagicMock()) as model_fn,
        patch.object(
            art_agent, "create_conversation_manager", return_value=MagicMock()
        ) as cm_fn,
        patch.object(
            art_agent, "context_management_plugins", return_value=["plugin"]
        ) as plugins_fn,
    ):
        ns = MagicMock()
        ns.mcp_client = mcp_client
        ns.agent_obj = agent_obj
        ns.obo_cls = obo_cls
        ns.http_cls = http_cls
        ns.mcp_cls = mcp_cls
        ns.stream_fn = stream_fn
        ns.set_client = set_client
        ns.agent_cls = agent_cls
        ns.model_fn = model_fn
        ns.cm_fn = cm_fn
        ns.plugins_fn = plugins_fn
        yield ns


def test_create_art_agent_builds_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: returns the constructed Agent with MCP + specialized tools."""
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    with _patched_factory() as ns:
        result = art_agent.create_art_agent("http://localhost:9200")

    assert result is ns.agent_obj
    # MCP client started and shared with the specialized agents.
    ns.mcp_client.start.assert_called_once()
    ns.set_client.assert_called_once_with(ns.mcp_client)
    # Agent constructed with the three specialized agents as tools.
    ns.agent_cls.assert_called_once()
    kwargs = ns.agent_cls.call_args.kwargs
    assert kwargs["system_prompt"] == art_agent.ORCHESTRATOR_SYSTEM_PROMPT
    assert kwargs["tools"] == [
        art_agent.user_behavior_analysis_agent,
        art_agent.hypothesis_agent,
        art_agent.evaluation_agent,
    ]
    assert kwargs["model"] is ns.model_fn.return_value
    assert kwargs["conversation_manager"] is ns.cm_fn.return_value
    assert kwargs["plugins"] == ["plugin"]
    # GC-guard references attached to the returned orchestrator.
    assert result._mcp_client is ns.mcp_client
    assert result._obo_auth is ns.obo_cls.return_value


def test_uses_default_mcp_url_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``MCP_SERVER_URL`` env var, the bundled default URL is used."""
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    with _patched_factory() as ns:
        art_agent.create_art_agent("http://localhost:9200")
        # streamable_http_client is invoked lazily inside the MCPClient lambda;
        # invoke it while the patches are still active.
        factory_lambda = ns.mcp_cls.call_args.args[0]
        factory_lambda()
        assert ns.stream_fn.call_args.args[0] == art_agent.DEFAULT_MCP_SERVER_URL


def test_uses_env_mcp_url_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured ``MCP_SERVER_URL`` overrides the default endpoint."""
    monkeypatch.setenv("MCP_SERVER_URL", "http://mcp.example:9999/mcp")
    with _patched_factory() as ns:
        art_agent.create_art_agent("http://localhost:9200")
        factory_lambda = ns.mcp_cls.call_args.args[0]
        factory_lambda()
        assert ns.stream_fn.call_args.args[0] == "http://mcp.example:9999/mcp"
        # http_client kwarg is the patched AsyncClient instance.
        assert ns.stream_fn.call_args.kwargs["http_client"] is ns.http_cls.return_value
