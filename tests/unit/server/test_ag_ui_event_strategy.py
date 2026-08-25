# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Hermetic coverage tests for server.ag_ui_event_strategy.

All AG-UI events are synthetic ``SimpleNamespace`` stand-ins carrying only a
``type`` attribute (read by ``is_event_type``) plus whatever fields the handler
under test inspects. Persistence and activity-monitor collaborators are mocks,
so no real model calls, I/O, or network occur.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from ag_ui.core import EventType

from server import ag_ui_event_strategy as strat
from server.ag_ui_event_strategy import (
    AGUIEventContext,
    AGUIEventHandler,
    AGUIEventHandlerChain,
    TextMessageContentHandler,
    TextMessageEndHandler,
    TextMessageStartHandler,
    ToolCallEndActivityHandler,
    ToolCallStartActivityHandler,
    _get_tool_call_error_message,
    _is_tool_call_error,
    create_agui_event_handler_chain,
)

pytestmark = pytest.mark.unit


def _event(event_type: EventType, **attrs) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **attrs)


def _ctx(
    event, *, persistence=None, activity_monitor=None, message_id=None, content=None
):
    return AGUIEventContext(
        event=event,
        run_id="run-1",
        thread_id="thread-1",
        current_message_id=message_id,
        current_message_content=content if content is not None else [],
        persistence=persistence,
        activity_monitor=activity_monitor,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestToolCallErrorHelpers:
    def test_is_error_flag_true(self):
        assert (
            _is_tool_call_error(_event(EventType.TOOL_CALL_END, is_error=True)) is True
        )

    def test_error_attribute_non_empty(self):
        assert (
            _is_tool_call_error(_event(EventType.TOOL_CALL_END, error="boom")) is True
        )

    def test_error_empty_string_is_not_error(self):
        assert _is_tool_call_error(_event(EventType.TOOL_CALL_END, error="")) is False

    def test_no_error_indicators(self):
        assert _is_tool_call_error(_event(EventType.TOOL_CALL_END)) is False

    def test_error_message_prefers_error_attr(self):
        assert (
            _get_tool_call_error_message(
                _event(EventType.TOOL_CALL_END, error="e1", message="m1")
            )
            == "e1"
        )

    def test_error_message_falls_back_to_message(self):
        assert (
            _get_tool_call_error_message(
                _event(EventType.TOOL_CALL_END, error="", message="m1")
            )
            == "m1"
        )

    def test_error_message_none_when_absent(self):
        assert _get_tool_call_error_message(_event(EventType.TOOL_CALL_END)) is None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TestAbstractBase:
    def test_base_methods_raise_not_implemented(self):
        class _Concrete(AGUIEventHandler):
            def can_handle(self, event):
                return super().can_handle(event)

            def handle(self, context):
                return super().handle(context)

        handler = _Concrete()
        with pytest.raises(NotImplementedError):
            handler.can_handle(_event(EventType.TEXT_MESSAGE_START))
        with pytest.raises(NotImplementedError):
            handler.handle(_ctx(_event(EventType.TEXT_MESSAGE_START)))


# ---------------------------------------------------------------------------
# Text message handlers
# ---------------------------------------------------------------------------


class TestTextMessageStartHandler:
    def test_can_handle(self):
        h = TextMessageStartHandler()
        assert h.can_handle(_event(EventType.TEXT_MESSAGE_START)) is True
        assert h.can_handle(_event(EventType.TEXT_MESSAGE_END)) is False

    def test_uses_event_message_id(self):
        h = TextMessageStartHandler()
        mid, content = h.handle(
            _ctx(_event(EventType.TEXT_MESSAGE_START, message_id="m9"))
        )
        assert mid == "m9"
        assert content == []

    def test_generates_message_id_when_missing(self):
        h = TextMessageStartHandler()
        mid, content = h.handle(
            _ctx(_event(EventType.TEXT_MESSAGE_START, message_id=None))
        )
        assert mid is not None and mid != ""
        assert content == []


class TestTextMessageContentHandler:
    def test_can_handle(self):
        assert TextMessageContentHandler().can_handle(
            _event(EventType.TEXT_MESSAGE_CONTENT)
        )

    def test_appends_delta(self):
        h = TextMessageContentHandler()
        ctx = _ctx(
            _event(EventType.TEXT_MESSAGE_CONTENT, delta="hello"), message_id="m1"
        )
        _, content = h.handle(ctx)
        assert content == ["hello"]

    def test_falls_back_to_content_attr(self):
        h = TextMessageContentHandler()
        ctx = _ctx(_event(EventType.TEXT_MESSAGE_CONTENT, content="fromcontent"))
        _, content = h.handle(ctx)
        assert content == ["fromcontent"]

    def test_empty_delta_appends_nothing(self):
        h = TextMessageContentHandler()
        ctx = _ctx(_event(EventType.TEXT_MESSAGE_CONTENT, delta=""))
        _, content = h.handle(ctx)
        assert content == []


class TestTextMessageEndHandler:
    def test_can_handle(self):
        assert TextMessageEndHandler().can_handle(_event(EventType.TEXT_MESSAGE_END))

    def test_no_persistence_logs_warning_and_resets(self):
        h = TextMessageEndHandler()
        ctx = _ctx(
            _event(EventType.TEXT_MESSAGE_END), persistence=None, message_id="m1"
        )
        mid, content = h.handle(ctx)
        assert mid is None
        assert content == []

    def test_saves_message_with_content(self):
        persistence = MagicMock()
        h = TextMessageEndHandler()
        ctx = _ctx(
            _event(EventType.TEXT_MESSAGE_END),
            persistence=persistence,
            message_id="m1",
            content=["ab", "cd"],
        )
        mid, content = h.handle(ctx)
        assert mid is None and content == []
        persistence.save_message.assert_called_once()
        kwargs = persistence.save_message.call_args.kwargs
        assert kwargs["message_id"] == "m1"
        assert kwargs["content"] == "abcd"
        assert kwargs["role"] == "assistant"

    def test_skips_save_when_no_message_id(self):
        persistence = MagicMock()
        h = TextMessageEndHandler()
        ctx = _ctx(
            _event(EventType.TEXT_MESSAGE_END, message_id=None),
            persistence=persistence,
            message_id=None,
        )
        mid, content = h.handle(ctx)
        assert mid is None and content == []
        persistence.save_message.assert_not_called()

    def test_message_id_from_event_and_content_fallback(self):
        persistence = MagicMock()
        h = TextMessageEndHandler()
        # No context message_id, but event carries one; empty content list -> fall
        # back to event.message text.
        ctx = _ctx(
            _event(
                EventType.TEXT_MESSAGE_END, message_id="evt-mid", message="final text"
            ),
            persistence=persistence,
            message_id=None,
            content=[],
        )
        h.handle(ctx)
        kwargs = persistence.save_message.call_args.kwargs
        assert kwargs["message_id"] == "evt-mid"
        assert kwargs["content"] == "final text"

    def test_save_failure_is_logged(self, monkeypatch):
        persistence = MagicMock()

        def _boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(strat, "safe_persistence_operation", _boom)
        h = TextMessageEndHandler()
        ctx = _ctx(
            _event(EventType.TEXT_MESSAGE_END),
            persistence=persistence,
            message_id="m1",
            content=["x"],
        )
        # Exception is caught and logged; handler still resets state.
        mid, content = h.handle(ctx)
        assert mid is None and content == []


# ---------------------------------------------------------------------------
# Tool call activity handlers
# ---------------------------------------------------------------------------


class TestToolCallStartActivityHandler:
    def test_can_handle(self):
        assert ToolCallStartActivityHandler().can_handle(
            _event(EventType.TOOL_CALL_START)
        )

    def test_no_activity_monitor_noop(self):
        h = ToolCallStartActivityHandler()
        ctx = _ctx(
            _event(EventType.TOOL_CALL_START, tool_call_id="tc1"), activity_monitor=None
        )
        assert h.handle(ctx) == (None, [])

    def test_tracks_start(self):
        monitor = MagicMock()
        h = ToolCallStartActivityHandler()
        ctx = _ctx(
            _event(
                EventType.TOOL_CALL_START,
                tool_call_id="tc1",
                tool_call_name="search",
                arguments={"q": 1},
            ),
            activity_monitor=monitor,
        )
        h.handle(ctx)
        monitor.track_tool_call_start.assert_called_once_with(
            tool_call_id="tc1", tool_name="search", arguments={"q": 1}
        )

    def test_missing_tool_call_id_logs_error(self):
        monitor = MagicMock()
        h = ToolCallStartActivityHandler()
        ctx = _ctx(
            _event(EventType.TOOL_CALL_START, tool_call_id=None),
            activity_monitor=monitor,
        )
        h.handle(ctx)
        monitor.track_tool_call_start.assert_not_called()


class TestToolCallEndActivityHandler:
    def test_can_handle(self):
        assert ToolCallEndActivityHandler().can_handle(_event(EventType.TOOL_CALL_END))

    def test_no_activity_monitor_noop(self):
        h = ToolCallEndActivityHandler()
        ctx = _ctx(
            _event(EventType.TOOL_CALL_END, tool_call_id="tc1"), activity_monitor=None
        )
        assert h.handle(ctx) == (None, [])

    def test_tracks_end_with_error(self):
        monitor = MagicMock()
        h = ToolCallEndActivityHandler()
        ctx = _ctx(
            _event(EventType.TOOL_CALL_END, tool_call_id="tc1", error="oops"),
            activity_monitor=monitor,
        )
        h.handle(ctx)
        monitor.track_tool_call_end.assert_called_once_with(
            tool_call_id="tc1", success=False, error="oops"
        )

    def test_tracks_end_success(self):
        monitor = MagicMock()
        h = ToolCallEndActivityHandler()
        ctx = _ctx(
            _event(EventType.TOOL_CALL_END, tool_call_id="tc2"),
            activity_monitor=monitor,
        )
        h.handle(ctx)
        monitor.track_tool_call_end.assert_called_once_with(
            tool_call_id="tc2", success=True, error=None
        )

    def test_missing_tool_call_id_logs_error(self):
        monitor = MagicMock()
        h = ToolCallEndActivityHandler()
        ctx = _ctx(
            _event(EventType.TOOL_CALL_END, tool_call_id=None), activity_monitor=monitor
        )
        h.handle(ctx)
        monitor.track_tool_call_end.assert_not_called()


# ---------------------------------------------------------------------------
# Handler chain + factory
# ---------------------------------------------------------------------------


class TestHandlerChain:
    def test_all_matching_handlers_run(self):
        persistence = MagicMock()
        monitor = MagicMock()
        chain = AGUIEventHandlerChain(
            [
                TextMessageStartHandler(),
                ToolCallStartActivityHandler(),  # should not match a START text event
            ]
        )
        ctx = _ctx(
            _event(EventType.TEXT_MESSAGE_START, message_id="m1"),
            persistence=persistence,
            activity_monitor=monitor,
        )
        mid, content = chain.process_event(ctx)
        assert mid == "m1"
        assert content == []

    def test_no_matching_handler_returns_state(self):
        chain = AGUIEventHandlerChain([TextMessageStartHandler()])
        ctx = _ctx(_event(EventType.TOOL_CALL_END), message_id="keep")
        mid, content = chain.process_event(ctx)
        assert mid == "keep"


class TestCreateChain:
    def test_persistence_only(self):
        chain = create_agui_event_handler_chain(
            persistence=MagicMock(), activity_monitor=None
        )
        assert len(chain.handlers) == 3

    def test_activity_only(self):
        chain = create_agui_event_handler_chain(
            persistence=None, activity_monitor=MagicMock()
        )
        assert len(chain.handlers) == 2

    def test_both(self):
        chain = create_agui_event_handler_chain(
            persistence=MagicMock(), activity_monitor=MagicMock()
        )
        assert len(chain.handlers) == 5

    def test_neither(self):
        chain = create_agui_event_handler_chain(persistence=None, activity_monitor=None)
        assert chain.handlers == []
