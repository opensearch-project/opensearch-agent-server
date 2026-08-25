# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Hermetic coverage tests for server.ag_ui_event_processor.

The Strands agent event stream is mocked as an async generator feeding
synthetic AG-UI events; the encoder, persistence, and activity monitor are all
mocks. No real model calls or network I/O.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from ag_ui.core import EventType

from server.ag_ui_event_processor import (
    AGUIEventProcessor,
    _complete_run,
    _handle_run_error,
    _process_event_stream,
    generate_events,
)

pytestmark = pytest.mark.unit


def _event(event_type: EventType, **attrs) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **attrs)


def _encoder(return_value="data: enc\n\n"):
    enc = MagicMock()
    enc.encode.return_value = return_value
    return enc


async def _aiter(events):
    for e in events:
        yield e


# ---------------------------------------------------------------------------
# AGUIEventProcessor
# ---------------------------------------------------------------------------


class TestProcessorInit:
    def test_defaults(self):
        p = AGUIEventProcessor(encoder=_encoder())
        assert p.persistence is None
        assert p.activity_monitor is None
        assert p._handler_chain.handlers == []

    def test_with_collaborators(self):
        p = AGUIEventProcessor(
            encoder=_encoder(), persistence=MagicMock(), activity_monitor=MagicMock()
        )
        assert len(p._handler_chain.handlers) == 5


class TestProcessEvent:
    def test_happy_path_with_persistence(self):
        persistence = MagicMock()
        encoder = _encoder()
        p = AGUIEventProcessor(encoder=encoder, persistence=persistence)
        event = _event(EventType.TEXT_MESSAGE_START, message_id="m1")
        mid, content, encoded = p.process_event(event, "run", "thread", None, [])
        assert encoded == "data: enc\n\n"
        assert mid == "m1"
        # save_event was invoked through persistence path
        persistence.save_event.assert_called_once()

    def test_encode_failure_falls_back_to_error_event(self):
        encoder = MagicMock()
        # First encode (real event) raises; second encode (error event) succeeds.
        encoder.encode.side_effect = [RuntimeError("bad"), "data: error\n\n"]
        p = AGUIEventProcessor(encoder=encoder)
        _, _, encoded = p.process_event(
            _event(EventType.TEXT_MESSAGE_CONTENT, delta="x"), "run", "thread", None, []
        )
        assert encoded == "data: error\n\n"
        assert encoder.encode.call_count == 2

    def test_encode_and_fallback_failure_uses_json_string(self):
        encoder = MagicMock()
        encoder.encode.side_effect = RuntimeError("always broken")
        p = AGUIEventProcessor(encoder=encoder)
        _, _, encoded = p.process_event(
            _event(EventType.TEXT_MESSAGE_CONTENT, delta="x"), "run", "thread", None, []
        )
        assert "ENCODING_ERROR" in encoded
        assert encoded.startswith("data: ")


class TestSaveEventToPersistence:
    def test_model_dump_path(self):
        persistence = MagicMock()
        p = AGUIEventProcessor(encoder=_encoder(), persistence=persistence)
        event = MagicMock()
        event.type = EventType.TEXT_MESSAGE_START
        event.model_dump.return_value = {"type": "TEXT_MESSAGE_START"}
        p._save_event_to_persistence(event, "run", "thread")
        kwargs = persistence.save_event.call_args.kwargs
        assert kwargs["event_type"] == "TEXT_MESSAGE_START"
        assert kwargs["event_data"] == {"type": "TEXT_MESSAGE_START"}

    def test_dict_path(self):
        persistence = MagicMock()
        p = AGUIEventProcessor(encoder=_encoder(), persistence=persistence)

        class _DictEvent:
            type = EventType.TOOL_CALL_START

            def dict(self, exclude_none=False):
                return {"type": "TOOL_CALL_START"}

        p._save_event_to_persistence(_DictEvent(), "run", "thread")
        kwargs = persistence.save_event.call_args.kwargs
        assert kwargs["event_data"] == {"type": "TOOL_CALL_START"}

    def test_neither_serializer(self):
        persistence = MagicMock()
        p = AGUIEventProcessor(encoder=_encoder(), persistence=persistence)
        p._save_event_to_persistence(_event(EventType.TOOL_CALL_END), "run", "thread")
        kwargs = persistence.save_event.call_args.kwargs
        assert kwargs["event_data"] == {"type": "TOOL_CALL_END"}

    def test_unknown_event_type(self):
        persistence = MagicMock()
        p = AGUIEventProcessor(encoder=_encoder(), persistence=persistence)
        p._save_event_to_persistence(SimpleNamespace(type=None), "run", "thread")
        kwargs = persistence.save_event.call_args.kwargs
        assert kwargs["event_type"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# _process_event_stream
# ---------------------------------------------------------------------------


class TestProcessEventStream:
    async def test_yields_encoded_events(self):
        orchestrator = MagicMock()
        orchestrator.run.return_value = _aiter(
            [
                _event(EventType.TEXT_MESSAGE_START, message_id="m1"),
                _event(EventType.TEXT_MESSAGE_CONTENT, delta="hi"),
            ]
        )
        p = AGUIEventProcessor(encoder=_encoder("data: e\n\n"))
        out = [
            chunk
            async for chunk in _process_event_stream(
                orchestrator, MagicMock(), p, "run", "thread", headers=None
            )
        ]
        assert out == ["data: e\n\n", "data: e\n\n"]
        orchestrator.run.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_run_error
# ---------------------------------------------------------------------------


class TestHandleRunError:
    def test_encodes_error_event(self):
        p = AGUIEventProcessor(encoder=_encoder("data: err\n\n"))
        out = _handle_run_error(p, "run", "thread", "user", ValueError("boom"))
        assert out == "data: err\n\n"

    def test_encode_failure_json_fallback(self):
        encoder = MagicMock()
        encoder.encode.side_effect = RuntimeError("no encode")
        p = AGUIEventProcessor(encoder=encoder)
        out = _handle_run_error(p, "run", "thread", "user", ValueError("boom"))
        assert "RUN_ERROR" in out
        assert "boom" in out


# ---------------------------------------------------------------------------
# _complete_run
# ---------------------------------------------------------------------------


class TestCompleteRun:
    def test_remaining_tool_calls_completed(self):
        monitor = MagicMock()
        monitor.get_remaining_tool_calls.return_value = ["tc1", "tc2"]
        persistence = MagicMock()
        p = AGUIEventProcessor(
            encoder=_encoder(), persistence=persistence, activity_monitor=monitor
        )
        _complete_run(p, "run", "thread", "user", 3, datetime.now())
        monitor.complete_remaining_tool_calls.assert_called_once()
        monitor.log_summary.assert_called_once()
        persistence.save_run_finish.assert_called_once()

    def test_no_remaining_tool_calls(self):
        monitor = MagicMock()
        monitor.get_remaining_tool_calls.return_value = []
        p = AGUIEventProcessor(encoder=_encoder(), activity_monitor=monitor)
        _complete_run(p, "run", "thread", "user", 0, datetime.now())
        monitor.complete_remaining_tool_calls.assert_not_called()
        monitor.log_summary.assert_called_once()

    def test_no_monitor_no_persistence(self):
        p = AGUIEventProcessor(encoder=_encoder())
        # Should not raise with neither collaborator present.
        _complete_run(p, "run", "thread", "user", 1, datetime.now())


# ---------------------------------------------------------------------------
# generate_events
# ---------------------------------------------------------------------------


class TestGenerateEvents:
    async def test_normal_flow_completes(self):
        orchestrator = MagicMock()
        orchestrator.run.return_value = _aiter(
            [_event(EventType.TEXT_MESSAGE_CONTENT, delta="a")]
        )
        monitor = MagicMock()
        monitor.get_remaining_tool_calls.return_value = []
        p = AGUIEventProcessor(
            encoder=_encoder("data: g\n\n"), activity_monitor=monitor
        )
        out = [
            chunk
            async for chunk in generate_events(
                orchestrator,
                MagicMock(),
                p,
                "run",
                "thread",
                "user",
                datetime.now(),
            )
        ]
        assert out == ["data: g\n\n"]
        monitor.log_summary.assert_called_once()

    async def test_error_flow_emits_error_event(self):
        async def _boom(*args, **kwargs):
            raise RuntimeError("stream failed")
            yield  # pragma: no cover - makes this an async generator

        orchestrator = MagicMock()
        orchestrator.run.return_value = _boom()
        monitor = MagicMock()
        monitor.get_remaining_tool_calls.return_value = []
        p = AGUIEventProcessor(
            encoder=_encoder("data: errored\n\n"), activity_monitor=monitor
        )
        out = [
            chunk
            async for chunk in generate_events(
                orchestrator,
                MagicMock(),
                p,
                "run",
                "thread",
                "user",
                datetime.now(),
            )
        ]
        assert out == ["data: errored\n\n"]
        # finally block still ran cleanup
        monitor.log_summary.assert_called_once()
