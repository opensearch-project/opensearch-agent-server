# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.agent_orchestrator focused on the uncovered
extraction helpers, routing seams, invoke context/timeout paths, and the
context-managed per-thread agent cache.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import server.agent_orchestrator as ao
from server.agent_orchestrator import (
    AgentOrchestrator,
    _ContextManagedThreadAgents,
    _extract_app_id_from_context,
    _extract_bearer_token,
    _extract_page_context,
)

pytestmark = pytest.mark.unit


class TestExtractHelpers:
    def test_extract_app_id_from_dict_value(self):
        ctx = SimpleNamespace(value={"appId": "discover"})
        assert _extract_app_id_from_context([ctx]) == "discover"

    def test_extract_app_id_from_json_string_value(self):
        ctx = SimpleNamespace(value='{"appId": "explore"}')
        assert _extract_app_id_from_context([ctx]) == "explore"

    def test_extract_app_id_skips_invalid_and_returns_none(self):
        bad = SimpleNamespace(value="not-json")
        no_app = SimpleNamespace(value={"other": 1})
        assert _extract_app_id_from_context([bad, no_app]) is None

    def test_extract_page_context_from_forwarded_props(self):
        data = SimpleNamespace(forwarded_props={"page_context": "home"}, context=[])
        assert _extract_page_context(data) == "home"

    def test_extract_page_context_from_context_array(self):
        data = SimpleNamespace(
            forwarded_props={}, context=[SimpleNamespace(value={"appId": "dashboards"})]
        )
        assert _extract_page_context(data) == "dashboards"

    def test_extract_page_context_none(self):
        data = SimpleNamespace(forwarded_props=None, context=[])
        assert _extract_page_context(data) is None

    def test_extract_bearer_token_none_headers(self):
        assert _extract_bearer_token(None) is None

    def test_extract_bearer_token_strips_bearer_prefix(self):
        assert _extract_bearer_token({"authorization": "Bearer abc123"}) == "abc123"

    def test_extract_bearer_token_capitalized_header(self):
        assert _extract_bearer_token({"Authorization": "Bearer xyz"}) == "xyz"

    def test_extract_bearer_token_non_bearer_passthrough(self):
        assert _extract_bearer_token({"authorization": "Basic zzz"}) == "Basic zzz"


class TestContextManagedThreadAgents:
    def test_setitem_applies_context_management(self, monkeypatch):
        applied = []
        monkeypatch.setattr(ao, "apply_context_management", lambda a: applied.append(a))
        d = _ContextManagedThreadAgents()
        agent = MagicMock()
        d["thread-1"] = agent
        assert applied == [agent]
        assert d["thread-1"] is agent


@pytest.fixture
def orchestrator():
    router = MagicMock()
    router.route.return_value = SimpleNamespace(name="default")
    orch = AgentOrchestrator(router)
    orch._agents_created = []

    def factory():
        agent = MagicMock()
        agent._obo_auth = MagicMock()
        agent._mcp_client = MagicMock()
        orch._agents_created.append(agent)
        return agent

    orch.register_agent_factory(name="default", factory=factory, description="d")
    return orch


class TestRun:
    @pytest.mark.asyncio
    async def test_run_routes_via_router_when_no_agent_name(self, orchestrator):
        """With agent_name=None the router resolves the target agent."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run
            mock_agui.return_value._agents_by_thread = {}

            data = SimpleNamespace(forwarded_props={}, context=[])
            events = [e async for e in orchestrator.run(data, headers=None)]

        assert len(events) == 1
        orchestrator._router.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_unknown_agent_raises_runtime_error(self, orchestrator):
        data = SimpleNamespace(forwarded_props={}, context=[])
        with pytest.raises(RuntimeError, match="No agent factory registered"):
            async for _ in orchestrator.run(data, agent_name="ghost"):
                pass


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_routes_when_no_agent_name(self, orchestrator):
        orchestrator._agent_factories["default"]["factory"] = lambda: _plain_agent(
            orchestrator, return_value="routed"
        )
        result = await orchestrator.invoke(prompt="hi", headers=None)
        assert result == "routed"
        orchestrator._router.route.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_invoke_unknown_agent_raises(self, orchestrator):
        with pytest.raises(RuntimeError, match="No agent factory registered"):
            await orchestrator.invoke(prompt="hi", agent_name="ghost")

    @pytest.mark.asyncio
    async def test_invoke_context_aware_agent(self, orchestrator):
        captured = {}

        def factory():
            agent = MagicMock()
            agent._obo_auth = MagicMock()
            agent._mcp_client = MagicMock()
            agent.accepts_invoke_context = True

            def call(prompt, context=None, auth_token=None):
                captured["prompt"] = prompt
                captured["context"] = context
                captured["auth_token"] = auth_token
                return "ctx-response"

            agent.side_effect = call
            return agent

        orchestrator._agent_factories["default"]["factory"] = factory
        result = await orchestrator.invoke(
            prompt="q",
            agent_name="default",
            headers={"authorization": "Bearer tok"},
            context={"index_name": "idx"},
        )
        assert result == "ctx-response"
        assert captured["context"] == {"index_name": "idx"}
        assert captured["auth_token"] == "tok"

    @pytest.mark.asyncio
    async def test_invoke_timeout_cancels_and_raises(self, orchestrator):
        agent_holder = {}

        def factory():
            agent = MagicMock()
            agent._obo_auth = MagicMock()
            agent._mcp_client = MagicMock()
            agent.accepts_invoke_context = False
            agent.side_effect = lambda prompt: time.sleep(0.3)
            agent_holder["agent"] = agent
            return agent

        orchestrator._agent_factories["default"]["factory"] = factory
        with pytest.raises(TimeoutError):
            await orchestrator.invoke(prompt="slow", agent_name="default", timeout=0.02)
        agent_holder["agent"].cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_none_result_returns_empty_string(self, orchestrator):
        orchestrator._agent_factories["default"]["factory"] = lambda: _plain_agent(
            orchestrator, return_value=None
        )
        result = await orchestrator.invoke(prompt="hi", agent_name="default")
        assert result == ""


def _plain_agent(orchestrator, return_value):
    agent = MagicMock()
    agent._obo_auth = MagicMock()
    agent._mcp_client = MagicMock()
    agent.accepts_invoke_context = False
    agent.side_effect = lambda prompt: return_value
    orchestrator._agents_created.append(agent)
    return agent
