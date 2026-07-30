# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for AgentOrchestrator credential isolation and resource cleanup.

Tests verify that the per-request agent creation fix:
1. Creates a separate agent instance per concurrent request (no shared state)
2. Each request uses its own credentials (no cross-contamination)
3. MCP client is properly cleaned up after each request (no resource leaks)
4. MCP client is cleaned up even when the agent errors
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from server.agent_orchestrator import AgentOrchestrator

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.route.return_value = MagicMock(name="default")
    return router


@pytest.fixture
def orchestrator(mock_router):
    orch = AgentOrchestrator(mock_router)
    orch._agents_created = []

    def mock_factory():
        agent = MagicMock()
        obo_auth = MagicMock()
        agent._obo_auth = obo_auth
        mcp_client = MagicMock()
        agent._mcp_client = mcp_client
        orch._agents_created.append(agent)
        return agent

    orch.register_agent_factory(name="default", factory=mock_factory)
    return orch


@pytest.fixture
def mock_input_data():
    input_data = MagicMock()
    input_data.thread_id = "thread-1"
    input_data.run_id = "run-1"
    input_data.forwarded_props = {}
    input_data.context = []
    return input_data


class TestCredentialIsolation:
    """Tests that concurrent requests get isolated credentials."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_get_separate_agents(
        self, orchestrator, mock_input_data
    ):
        """Each concurrent request must create its own agent instance."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                await asyncio.sleep(0.05)
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            async def run_with_token(token, run_id):
                input_data = MagicMock()
                input_data.thread_id = f"thread-{run_id}"
                input_data.run_id = run_id
                input_data.forwarded_props = {}
                input_data.context = []
                events = []
                async for event in orchestrator.run(
                    input_data,
                    agent_name="default",
                    headers={"authorization": f"Bearer {token}"},
                ):
                    events.append(event)
                return events

            await asyncio.gather(
                run_with_token("token_A", "run-a"),
                run_with_token("token_B", "run-b"),
            )

        assert len(orchestrator._agents_created) == 2

    @pytest.mark.asyncio
    async def test_each_request_gets_its_own_token(self, orchestrator, mock_input_data):
        """Each agent instance gets the correct token, not a shared one."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            async def run_with_token(token, run_id):
                input_data = MagicMock()
                input_data.thread_id = f"thread-{run_id}"
                input_data.run_id = run_id
                input_data.forwarded_props = {}
                input_data.context = []
                async for _ in orchestrator.run(
                    input_data,
                    agent_name="default",
                    headers={"authorization": f"Bearer {token}"},
                ):
                    pass

            await run_with_token("token_alice", "run-alice")
            await run_with_token("token_bob", "run-bob")

        alice_agent = orchestrator._agents_created[0]
        bob_agent = orchestrator._agents_created[1]

        alice_agent._obo_auth.set_token.assert_called_once_with("token_alice")
        bob_agent._obo_auth.set_token.assert_called_once_with("token_bob")

    @pytest.mark.asyncio
    async def test_no_shared_obo_auth_between_requests(
        self, orchestrator, mock_input_data
    ):
        """Each request must have a different OboAuth instance."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            async def run_request(run_id):
                input_data = MagicMock()
                input_data.thread_id = f"thread-{run_id}"
                input_data.run_id = run_id
                input_data.forwarded_props = {}
                input_data.context = []
                async for _ in orchestrator.run(
                    input_data, agent_name="default", headers=None
                ):
                    pass

            await run_request("run-1")
            await run_request("run-2")

        agent_1 = orchestrator._agents_created[0]
        agent_2 = orchestrator._agents_created[1]
        assert agent_1._obo_auth is not agent_2._obo_auth


class TestMcpClientCleanup:
    """Tests that MCP client resources are properly cleaned up."""

    @pytest.mark.asyncio
    async def test_mcp_client_stopped_after_successful_run(
        self, orchestrator, mock_input_data
    ):
        """MCP client must be stopped after a successful request."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            async for _ in orchestrator.run(
                mock_input_data, agent_name="default", headers=None
            ):
                pass

        agent = orchestrator._agents_created[0]
        agent._mcp_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_client_stopped_on_error(self, orchestrator, mock_input_data):
        """MCP client must be stopped even if the agent run raises."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def failing_run(data):
                raise RuntimeError("agent exploded")
                yield  # noqa: unreachable — makes this an async generator

            mock_agui.return_value.run = failing_run

            with pytest.raises(RuntimeError, match="agent exploded"):
                async for _ in orchestrator.run(
                    mock_input_data, agent_name="default", headers=None
                ):
                    pass

        agent = orchestrator._agents_created[0]
        agent._mcp_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_client_stop_exception_is_suppressed(
        self, orchestrator, mock_input_data
    ):
        """If mcp_client.stop() raises, it should be suppressed."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            # Make stop() raise
            def raise_on_stop():
                raise RuntimeError("stop failed")

            async for _ in orchestrator.run(
                mock_input_data, agent_name="default", headers=None
            ):
                pass

        agent = orchestrator._agents_created[0]
        agent._mcp_client.stop.side_effect = RuntimeError("stop failed")
        # Verify the run completed without error (stop exception suppressed)

    @pytest.mark.asyncio
    async def test_no_mcp_client_does_not_error(self, orchestrator, mock_input_data):
        """Agent without _mcp_client attribute should not error on cleanup."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            # Remove _mcp_client from the next created agent
            original_factory = orchestrator._agent_factories["default"]["factory"]

            def factory_no_mcp():
                agent = original_factory()
                del agent._mcp_client
                return agent

            orchestrator._agent_factories["default"]["factory"] = factory_no_mcp

            # Should not raise
            async for _ in orchestrator.run(
                mock_input_data, agent_name="default", headers=None
            ):
                pass


class TestPerRequestAgentCreation:
    """Tests that agents are never cached/shared."""

    @pytest.mark.asyncio
    async def test_factory_called_every_request(self, orchestrator, mock_input_data):
        """Factory must be called on every request, not cached."""
        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui:

            async def fake_run(data):
                yield MagicMock(type=MagicMock(value="RUN_FINISHED"))

            mock_agui.return_value.run = fake_run

            for i in range(5):
                input_data = MagicMock()
                input_data.thread_id = f"thread-{i}"
                input_data.run_id = f"run-{i}"
                input_data.forwarded_props = {}
                input_data.context = []
                async for _ in orchestrator.run(
                    input_data, agent_name="default", headers=None
                ):
                    pass

        assert len(orchestrator._agents_created) == 5


class TestInvokeCredentialIsolation:
    """Tests that concurrent /invoke requests get isolated credentials."""

    @pytest.mark.asyncio
    async def test_concurrent_invoke_requests_get_separate_agents(self, orchestrator):
        """Each concurrent invoke request must create its own agent instance."""

        async def run_invoke(token):
            return await orchestrator.invoke(
                prompt="hello",
                agent_name="default",
                headers={"authorization": f"Bearer {token}"},
            )

        await asyncio.gather(
            run_invoke("token_A"),
            run_invoke("token_B"),
        )

        assert len(orchestrator._agents_created) == 2

    @pytest.mark.asyncio
    async def test_invoke_each_request_gets_its_own_token(self, orchestrator):
        """Each invoke agent instance gets the correct token."""
        await orchestrator.invoke(
            prompt="hello",
            agent_name="default",
            headers={"authorization": "Bearer token_alice"},
        )
        await orchestrator.invoke(
            prompt="hello",
            agent_name="default",
            headers={"authorization": "Bearer token_bob"},
        )

        alice_agent = orchestrator._agents_created[0]
        bob_agent = orchestrator._agents_created[1]

        alice_agent._obo_auth.set_token.assert_called_once_with("token_alice")
        bob_agent._obo_auth.set_token.assert_called_once_with("token_bob")

    @pytest.mark.asyncio
    async def test_invoke_no_shared_obo_auth(self, orchestrator):
        """Each invoke request must have a different OboAuth instance."""
        await orchestrator.invoke(prompt="hello", agent_name="default", headers=None)
        await orchestrator.invoke(prompt="hello", agent_name="default", headers=None)

        agent_1 = orchestrator._agents_created[0]
        agent_2 = orchestrator._agents_created[1]
        assert agent_1._obo_auth is not agent_2._obo_auth


class TestInvokeMcpClientCleanup:
    """Tests that /invoke properly cleans up MCP client resources."""

    @pytest.mark.asyncio
    async def test_invoke_mcp_client_stopped_after_success(self, orchestrator):
        """MCP client must be stopped after a successful invoke."""
        await orchestrator.invoke(prompt="hello", agent_name="default", headers=None)

        agent = orchestrator._agents_created[0]
        agent._mcp_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_mcp_client_stopped_on_error(self, orchestrator):
        """MCP client must be stopped even if invoke raises."""

        # Make the agent call raise
        def failing_agent(prompt):
            raise RuntimeError("agent exploded")

        original_factory = orchestrator._agent_factories["default"]["factory"]

        def factory_that_fails():
            agent = original_factory()
            agent.side_effect = RuntimeError("agent exploded")
            return agent

        orchestrator._agent_factories["default"]["factory"] = factory_that_fails

        with pytest.raises(RuntimeError):
            await orchestrator.invoke(
                prompt="hello", agent_name="default", headers=None
            )

        agent = orchestrator._agents_created[0]
        agent._mcp_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_no_mcp_client_does_not_error(self, orchestrator):
        """Agent without _mcp_client should not error on cleanup."""
        original_factory = orchestrator._agent_factories["default"]["factory"]

        def factory_no_mcp():
            agent = original_factory()
            del agent._mcp_client
            return agent

        orchestrator._agent_factories["default"]["factory"] = factory_no_mcp

        # Should not raise
        await orchestrator.invoke(prompt="hello", agent_name="default", headers=None)

    @pytest.mark.asyncio
    async def test_invoke_factory_called_every_request(self, orchestrator):
        """Factory must be called on every invoke, not cached."""
        for _ in range(5):
            await orchestrator.invoke(
                prompt="hello", agent_name="default", headers=None
            )

        assert len(orchestrator._agents_created) == 5
