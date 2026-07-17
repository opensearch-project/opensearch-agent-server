"""Unit tests for context management primitives (issue #138)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.hooks.events import AfterToolCallEvent, BeforeModelCallEvent

from agents.context_management import (
    CONTEXT_WINDOW_LIMIT,
    MAX_TOOL_OUTPUT_CHARS,
    PRESERVE_RECENT_MESSAGES,
    SUMMARIZATION_COMPRESSION_THRESHOLD,
    SUMMARIZATION_TRIGGER_TOKENS,
    ToolOutputTruncationHook,
    apply_context_management,
    create_conversation_manager,
)

pytestmark = pytest.mark.unit


class _FakeModel:
    """Minimal model for constructing a real strands Agent without a provider."""

    stateful = False

    def __init__(self, context_window_limit: int = 1_000_000) -> None:
        self.context_window_limit = context_window_limit
        self.config: dict = {}


def _callback_count(agent: Agent, event_type: type) -> int:
    """Number of hook callbacks registered on ``agent`` for ``event_type``."""
    return len(agent.hooks._registered_callbacks.get(event_type, []))


class TestCompressionThreshold:
    """Tests for the fixed proactive-compression ratio (matches the reference agent)."""

    def test_threshold_is_170k_over_200k(self) -> None:
        """The fixed ratio is 170k / 200k = 0.85, firing near Sonnet's 200k window."""
        assert SUMMARIZATION_COMPRESSION_THRESHOLD == pytest.approx(0.85)
        assert SUMMARIZATION_TRIGGER_TOKENS / CONTEXT_WINDOW_LIMIT == pytest.approx(
            SUMMARIZATION_COMPRESSION_THRESHOLD
        )


class TestCreateConversationManager:
    """Tests for conversation manager construction."""

    def test_builds_summarizing_manager(self) -> None:
        manager = create_conversation_manager()
        assert isinstance(manager, SummarizingConversationManager)
        assert manager.preserve_recent_messages == PRESERVE_RECENT_MESSAGES

    def test_proactive_compression_enabled(self) -> None:
        """Proactive compression threshold is the fixed 0.85 ratio."""
        manager = create_conversation_manager()
        assert manager._compression_threshold == pytest.approx(0.85)


class TestToolOutputTruncationHook:
    """Tests for POST_TOOL output truncation."""

    def _event(self, result: object) -> AfterToolCallEvent:
        """Build a minimal AfterToolCallEvent carrying ``result``."""
        event = MagicMock(spec=AfterToolCallEvent)
        event.result = result
        return event

    def test_registers_after_tool_call_callback(self) -> None:
        registry = MagicMock()
        ToolOutputTruncationHook().register_hooks(registry)
        registry.add_callback.assert_called_once()
        assert registry.add_callback.call_args.args[0] is AfterToolCallEvent

    def test_truncates_oversized_text(self) -> None:
        hook = ToolOutputTruncationHook(max_chars=10)
        result = {"content": [{"text": "x" * 50}]}
        hook._truncate(self._event(result))
        text = result["content"][0]["text"]
        assert text.startswith("x" * 10)
        assert "output truncated to 10 characters" in text

    def test_leaves_small_text_untouched(self) -> None:
        hook = ToolOutputTruncationHook(max_chars=MAX_TOOL_OUTPUT_CHARS)
        result = {"content": [{"text": "small"}]}
        hook._truncate(self._event(result))
        assert result["content"][0]["text"] == "small"

    def test_ignores_non_text_blocks(self) -> None:
        hook = ToolOutputTruncationHook(max_chars=10)
        result = {"content": [{"json": {"a": 1}}, "not-a-dict"]}
        # Should not raise and should not modify non-text content.
        hook._truncate(self._event(result))
        assert result["content"][0] == {"json": {"a": 1}}

    def test_ignores_non_dict_result(self) -> None:
        hook = ToolOutputTruncationHook(max_chars=10)
        # e.g. the tool raised and result is an Exception — must be a no-op.
        hook._truncate(self._event(ValueError("boom")))


class TestApplyContextManagement:
    """Tests for re-applying context management onto an already-built agent."""

    def test_attaches_manager_and_both_hooks(self) -> None:
        """A bare agent gains the summarizing manager and both lifecycle hooks."""
        agent = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        # A bare agent already carries default BeforeModelCallEvent callbacks; capture the
        # count so we can assert the proactive-compression hook is *added* on top.
        assert not isinstance(
            agent.conversation_manager, SummarizingConversationManager
        )
        assert _callback_count(agent, AfterToolCallEvent) == 0
        before_bmc = _callback_count(agent, BeforeModelCallEvent)

        apply_context_management(agent)

        assert isinstance(agent.conversation_manager, SummarizingConversationManager)
        assert agent.conversation_manager.preserve_recent_messages == (
            PRESERVE_RECENT_MESSAGES
        )
        # Assert BeforeModelCallEvent increments (before + 1), not >= 1: a bare agent already
        # has such callbacks, so >= 1 would pass even if the proactive hook were dropped.
        assert _callback_count(agent, AfterToolCallEvent) == 1
        assert _callback_count(agent, BeforeModelCallEvent) == before_bmc + 1

    def test_uses_fixed_compression_threshold(self) -> None:
        """The re-applied manager uses the fixed 0.85 ratio regardless of window size."""
        agent = Agent(
            model=_FakeModel(context_window_limit=1_000_000),
            system_prompt="x",
            tools=[],
        )
        apply_context_management(agent)
        assert agent.conversation_manager._compression_threshold == pytest.approx(0.85)

    def test_each_agent_gets_a_distinct_manager(self) -> None:
        """Managers are stateful, so per-agent instances must not be shared."""
        a = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        b = Agent(model=_FakeModel(), system_prompt="x", tools=[])
        apply_context_management(a)
        apply_context_management(b)
        assert a.conversation_manager is not b.conversation_manager


class TestSurvivesAgUiStrandsRebuild:
    """Regression: context management must survive the ``ag_ui_strands`` per-thread rebuild.

    The wrapper (0.1.1) rebuilds a per-thread ``Agent`` and drops ``conversation_manager`` +
    ``hooks``; the orchestrator's ``_ContextManagedThreadAgents`` cache re-applies them. This
    exercises that real path.
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

    def test_per_thread_agent_has_context_management(self) -> None:
        from ag_ui_strands import StrandsAgent as AGUIStrandsAgent

        from server.agent_orchestrator import _ContextManagedThreadAgents

        # Template as produced by create_default_agent (conversation_manager + hook set).
        template = Agent(
            model=_FakeModel(),
            system_prompt="sp",
            tools=[],
            conversation_manager=create_conversation_manager(),
            hooks=[ToolOutputTruncationHook()],
        )
        wrapper = AGUIStrandsAgent(agent=template, name="default")
        # Apply the orchestrator's cache swap.
        wrapper._agents_by_thread = _ContextManagedThreadAgents(
            wrapper._agents_by_thread
        )

        # Baseline BeforeModelCallEvent count so we assert the proactive hook is *added*.
        bare = Agent(model=_FakeModel(), system_prompt="sp", tools=[])
        before_bmc = _callback_count(bare, BeforeModelCallEvent)

        agent = self._build_per_thread_agent(wrapper, "thread-A")

        assert isinstance(agent.conversation_manager, SummarizingConversationManager)
        assert _callback_count(agent, AfterToolCallEvent) == 1
        assert _callback_count(agent, BeforeModelCallEvent) == before_bmc + 1

    def test_distinct_threads_get_distinct_managers(self) -> None:
        from ag_ui_strands import StrandsAgent as AGUIStrandsAgent

        from server.agent_orchestrator import _ContextManagedThreadAgents

        template = Agent(model=_FakeModel(), system_prompt="sp", tools=[])
        wrapper = AGUIStrandsAgent(agent=template, name="default")
        wrapper._agents_by_thread = _ContextManagedThreadAgents(
            wrapper._agents_by_thread
        )

        a = self._build_per_thread_agent(wrapper, "thread-A")
        b = self._build_per_thread_agent(wrapper, "thread-B")

        assert a.conversation_manager is not b.conversation_manager


class TestProactiveCompressionFires:
    """Behavioral: proactive compression fires above the 0.85 boundary and stays silent below.

    Drives a real ``BeforeModelCallEvent`` through the agent's hook registry with
    ``reduce_context`` spied so no model/summarizer call is made (fully offline).
    """

    def _managed_agent(self, limit: int) -> tuple[Agent, list]:
        """A real Agent with our fixed-0.85 manager; returns (agent, reduce_context calls)."""
        agent = Agent(
            model=_FakeModel(context_window_limit=limit),
            system_prompt="x",
            tools=[],
            conversation_manager=create_conversation_manager(),
            hooks=[ToolOutputTruncationHook()],
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
        """At 170_001 tokens on a 200k window (ratio > 0.85) compression triggers."""
        agent, calls = self._managed_agent(200_000)
        self._fire(agent, SUMMARIZATION_TRIGGER_TOKENS + 1)
        assert calls == [agent]

    def test_silent_just_under_threshold(self) -> None:
        """At 169_999 tokens on a 200k window (ratio < 0.85) compression does not trigger."""
        agent, calls = self._managed_agent(200_000)
        self._fire(agent, SUMMARIZATION_TRIGGER_TOKENS - 1)
        assert calls == []

    def test_fixed_ratio_holds_on_large_window(self) -> None:
        """On a 1M window the fixed 0.85 ratio fires at ~850k, not at the old 170k absolute."""
        agent, calls = self._managed_agent(1_000_000)
        self._fire(agent, SUMMARIZATION_TRIGGER_TOKENS)  # 170k — far below 0.85 * 1M
        assert calls == []
        self._fire(agent, 850_001)  # just over 0.85 * 1M
        assert calls == [agent]

    def test_truncation_hook_mutates_real_oversized_result(self) -> None:
        """A real ``AfterToolCallEvent`` through the agent's hook registry caps an oversized result."""
        agent, _ = self._managed_agent(200_000)
        oversized = "y" * (MAX_TOOL_OUTPUT_CHARS + 5_000)
        result = {
            "toolUseId": "t1",
            "status": "success",
            "content": [{"text": oversized}],
        }
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t1", "name": "SearchIndexTool", "input": {}},
            invocation_state={},
            result=result,
        )
        agent.hooks.invoke_callbacks(event)
        text = result["content"][0]["text"]
        assert len(text) < len(oversized)
        assert text.startswith("y" * MAX_TOOL_OUTPUT_CHARS)
        assert f"output truncated to {MAX_TOOL_OUTPUT_CHARS} characters" in text
