# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for context management primitives (issue #138).

Context management is built on two Strands vended primitives:
- ``SummarizingConversationManager`` — proactive conversation-length compression.
- ``ContextOffloader`` plugin — offloads oversized tool results to external storage,
  keeping a preview plus a retrievable reference in context.
"""

from __future__ import annotations

import json

import pytest
from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.hooks.events import AfterToolCallEvent, BeforeModelCallEvent
from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage

from agents.context_management import (
    COMPRESSION_THRESHOLD,
    MAX_RESULT_TOKENS,
    PRESERVE_RECENT_MESSAGES,
    PREVIEW_TOKENS,
    SUMMARY_RATIO,
    apply_context_management,
    context_management_plugins,
    create_context_offloader,
    create_conversation_manager,
)

pytestmark = pytest.mark.unit

_OFFLOADER_PLUGIN_NAME = "context_offloader"
_RETRIEVE_TOOL = "retrieve_offloaded_content"


class _FakeModel:
    """Minimal model for constructing a real strands Agent without a provider.

    Implements the ``count_tokens`` the ``ContextOffloader`` awaits per tool result, using
    the same chars/4 heuristic Strands falls back to when tiktoken/native counting is off.
    """

    stateful = False

    def __init__(self, context_window_limit: int = 1_000_000) -> None:
        self.context_window_limit = context_window_limit
        self.config: dict = {}

    def get_config(self) -> dict:
        return self.config

    async def count_tokens(
        self, messages, tool_specs=None, system_prompt=None, system_prompt_content=None
    ) -> int:
        return len(json.dumps(messages, default=str)) // 4


def _callback_count(agent: Agent, event_type: type) -> int:
    """Number of hook callbacks registered on ``agent`` for ``event_type``."""
    return len(agent.hooks._registered_callbacks.get(event_type, []))


def _offloader_of(agent: Agent) -> ContextOffloader:
    """The ``ContextOffloader`` plugin currently registered on ``agent``."""
    return agent._plugin_registry._plugins[_OFFLOADER_PLUGIN_NAME]


class TestCreateConversationManager:
    """Tests for conversation manager construction."""

    def test_builds_summarizing_manager(self) -> None:
        manager = create_conversation_manager()
        assert isinstance(manager, SummarizingConversationManager)
        assert manager.preserve_recent_messages == PRESERVE_RECENT_MESSAGES
        assert manager.summary_ratio == pytest.approx(SUMMARY_RATIO)

    def test_proactive_compression_threshold(self) -> None:
        """Proactive compression fires at the 0.85 ratio of the context window."""
        manager = create_conversation_manager()
        assert manager._compression_threshold == pytest.approx(COMPRESSION_THRESHOLD)


class TestCreateContextOffloader:
    """Tests for the ContextOffloader plugin factory."""

    def test_builds_offloader_with_configured_thresholds(self) -> None:
        offloader = create_context_offloader()
        assert isinstance(offloader, ContextOffloader)
        assert offloader._max_result_tokens == MAX_RESULT_TOKENS
        assert offloader._preview_tokens == PREVIEW_TOKENS
        assert isinstance(offloader._storage, InMemoryStorage)

    def test_fresh_storage_per_call(self) -> None:
        """Each offloader gets its own store — content must not leak across sessions."""
        a = create_context_offloader()
        b = create_context_offloader()
        assert a._storage is not b._storage

    def test_context_management_plugins_returns_offloader(self) -> None:
        plugins = context_management_plugins()
        assert len(plugins) == 1
        assert isinstance(plugins[0], ContextOffloader)


class TestApplyContextManagement:
    """Tests for re-applying context management onto an already-built agent."""

    def test_attaches_manager_offloader_and_hooks(self) -> None:
        """A bare agent gains the summarizing manager, the offloader plugin, and its tool."""
        agent = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        assert not isinstance(
            agent.conversation_manager, SummarizingConversationManager
        )
        assert _RETRIEVE_TOOL not in agent.tool_registry.registry
        assert _callback_count(agent, AfterToolCallEvent) == 0
        # A bare agent carries its default manager's BeforeModelCallEvent callbacks;
        # apply_context_management must swap that manager out, not stack on top of it.
        assert _callback_count(agent, BeforeModelCallEvent) > 0

        apply_context_management(agent)

        assert isinstance(agent.conversation_manager, SummarizingConversationManager)
        assert agent.conversation_manager.preserve_recent_messages == (
            PRESERVE_RECENT_MESSAGES
        )
        assert _OFFLOADER_PLUGIN_NAME in agent._plugin_registry._plugins
        assert _RETRIEVE_TOOL in agent.tool_registry.registry
        # The offloader registers one AfterToolCallEvent hook; the fresh manager owns the
        # single remaining BeforeModelCallEvent hook (the prior manager's were detached).
        assert _callback_count(agent, AfterToolCallEvent) == 1
        assert _callback_count(agent, BeforeModelCallEvent) == 1

    def test_is_idempotent(self) -> None:
        """A second call is a no-op — no duplicate-plugin error, no extra hooks."""
        agent = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        apply_context_management(agent)
        after_first = _offloader_of(agent)
        bmc = _callback_count(agent, BeforeModelCallEvent)
        atc = _callback_count(agent, AfterToolCallEvent)

        apply_context_management(agent)  # must not raise

        assert _offloader_of(agent) is after_first
        assert _callback_count(agent, BeforeModelCallEvent) == bmc
        assert _callback_count(agent, AfterToolCallEvent) == atc

    def test_uses_configured_compression_threshold(self) -> None:
        """The re-applied manager uses the 0.85 ratio regardless of window size."""
        agent = Agent(
            model=_FakeModel(context_window_limit=1_000_000),
            system_prompt="x",
            tools=[],
        )
        apply_context_management(agent)
        assert agent.conversation_manager._compression_threshold == pytest.approx(
            COMPRESSION_THRESHOLD
        )

    def test_each_agent_gets_distinct_manager_and_storage(self) -> None:
        """Manager (running summary) and offloader storage are per-agent, never shared."""
        a = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        b = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        apply_context_management(a)
        apply_context_management(b)
        assert a.conversation_manager is not b.conversation_manager
        assert _offloader_of(a)._storage is not _offloader_of(b)._storage


class TestSurvivesAgUiStrandsRebuild:
    """Regression: context management must survive the ``ag_ui_strands`` per-thread rebuild.

    The wrapper (0.1.1) rebuilds a per-thread ``Agent`` and drops ``conversation_manager``,
    ``hooks``, and ``plugins``; the orchestrator's ``_ContextManagedThreadAgents`` cache
    re-applies them. This exercises that real path.
    """

    def _build_per_thread_agent(self, wrapper, thread_id: str) -> Agent:
        """Reproduce the wrapper's lazy per-thread construction (agent.py:85-91)."""
        agent = Agent(
            model=wrapper._model,
            system_prompt=wrapper._system_prompt,
            tools=wrapper._tools,
            **wrapper._agent_kwargs,
        )
        wrapper._agents_by_thread[thread_id] = agent
        return wrapper._agents_by_thread[thread_id]

    def _wrapped_template(self):
        from ag_ui_strands import StrandsAgent as AGUIStrandsAgent

        from server.agent_orchestrator import _ContextManagedThreadAgents

        # Template as produced by create_default_agent: manager + offloader plugin set.
        template = Agent(
            model=_FakeModel(),
            system_prompt="sp",
            tools=[],
            conversation_manager=create_conversation_manager(),
            plugins=context_management_plugins(),
        )
        wrapper = AGUIStrandsAgent(agent=template, name="default")
        wrapper._agents_by_thread = _ContextManagedThreadAgents(
            wrapper._agents_by_thread
        )
        return wrapper

    def test_per_thread_agent_has_context_management(self) -> None:
        wrapper = self._wrapped_template()

        agent = self._build_per_thread_agent(wrapper, "thread-A")

        assert isinstance(agent.conversation_manager, SummarizingConversationManager)
        assert _OFFLOADER_PLUGIN_NAME in agent._plugin_registry._plugins
        assert _RETRIEVE_TOOL in agent.tool_registry.registry
        assert _callback_count(agent, AfterToolCallEvent) == 1
        assert _callback_count(agent, BeforeModelCallEvent) == 1

    def test_distinct_threads_get_distinct_managers_and_storage(self) -> None:
        wrapper = self._wrapped_template()

        a = self._build_per_thread_agent(wrapper, "thread-A")
        b = self._build_per_thread_agent(wrapper, "thread-B")

        assert a.conversation_manager is not b.conversation_manager
        assert _offloader_of(a)._storage is not _offloader_of(b)._storage


class TestProactiveCompressionFires:
    """Behavioral: proactive compression fires above the 0.85 boundary and stays silent below.

    Drives a real ``BeforeModelCallEvent`` through the agent's hook registry with
    ``reduce_context`` spied so no model/summarizer call is made (fully offline).
    """

    def _managed_agent(self, limit: int) -> tuple[Agent, list]:
        """A real Agent with our manager + offloader; returns (agent, reduce_context calls)."""
        agent = Agent(
            model=_FakeModel(context_window_limit=limit),
            system_prompt="x",
            tools=[],
            conversation_manager=create_conversation_manager(),
            plugins=context_management_plugins(),
        )
        calls: list = []
        # Spy on reduce_context so the trigger is observed without a real summarizer call.
        agent.conversation_manager.reduce_context = (  # type: ignore[method-assign]
            lambda agent, **kwargs: calls.append(agent)
        )
        return agent, calls

    def _fire(self, agent: Agent, projected_tokens: int) -> None:
        """Dispatch a real BeforeModelCallEvent through the agent's own hook registry."""
        event = BeforeModelCallEvent(
            agent=agent,
            invocation_state={},
            projected_input_tokens=projected_tokens,
        )
        agent.hooks.invoke_callbacks(event)

    def test_fires_just_over_threshold(self) -> None:
        """Just over 0.85 * 200k tokens, compression triggers."""
        agent, calls = self._managed_agent(200_000)
        self._fire(agent, int(200_000 * COMPRESSION_THRESHOLD) + 1)
        assert calls == [agent]

    def test_silent_just_under_threshold(self) -> None:
        """Just under 0.85 * 200k tokens, compression does not trigger."""
        agent, calls = self._managed_agent(200_000)
        self._fire(agent, int(200_000 * COMPRESSION_THRESHOLD) - 1)
        assert calls == []

    def test_ratio_holds_on_large_window(self) -> None:
        """On a 1M window the 0.85 ratio fires at ~850k, not at a 200k absolute."""
        agent, calls = self._managed_agent(1_000_000)
        self._fire(agent, 170_000)  # far below 0.85 * 1M
        assert calls == []
        self._fire(agent, int(1_000_000 * COMPRESSION_THRESHOLD) + 1)
        assert calls == [agent]


class TestOffloaderOffloadsLargeResult:
    """Behavioral: an oversized tool result is offloaded and stays retrievable."""

    def _managed_agent(self) -> Agent:
        return Agent(
            model=_FakeModel(context_window_limit=200_000),
            system_prompt="x",
            tools=[],
            conversation_manager=create_conversation_manager(),
            plugins=context_management_plugins(),
        )

    async def _fire_after_tool(
        self, agent: Agent, tool_use_id: str, result: dict
    ) -> AfterToolCallEvent:
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": tool_use_id, "name": "SearchIndexTool", "input": {}},
            invocation_state={},
            result=result,
        )
        # The offloader's AfterToolCallEvent handler is async; it replaces event.result.
        await agent.hooks.invoke_callbacks_async(event)
        return event

    async def test_large_result_offloaded_and_retrievable(self) -> None:
        agent = self._managed_agent()
        # Well over MAX_RESULT_TOKENS (1500) under the chars/4 estimate.
        original = "y" * (MAX_RESULT_TOKENS * 4 * 4)
        result = {
            "toolUseId": "t1",
            "status": "success",
            "content": [{"text": original}],
        }

        event = await self._fire_after_tool(agent, "t1", result)

        # The in-context result is now a compact preview, not the full payload.
        new_text = event.result["content"][0]["text"]
        assert len(new_text) < len(original)
        assert "[Offloaded:" in new_text

        # The full content is still retrievable from the offloader's own store.
        offloader = _offloader_of(agent)
        reference = "mem_1_t1_0"  # InMemoryStorage ref scheme: mem_<counter>_<toolUseId>_<block>
        content_bytes, content_type = offloader._storage.retrieve(reference)
        assert content_bytes.decode("utf-8") == original
        assert content_type == "text/plain"

    async def test_small_result_left_inline(self) -> None:
        agent = self._managed_agent()
        result = {
            "toolUseId": "t2",
            "status": "success",
            "content": [{"text": "small result"}],
        }

        event = await self._fire_after_tool(agent, "t2", result)

        assert event.result["content"][0]["text"] == "small result"
