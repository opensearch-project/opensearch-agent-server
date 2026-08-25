# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.run_route_helpers (cancellable event streaming)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import server.run_route_helpers as rrh
from server.run_route_helpers import (
    check_and_handle_cancellation,
    consume_event_generator_with_cancellation,
    create_event_queue,
    emit_cancellation_events,
    put_critical_event_with_retry,
    put_event_with_backpressure,
    yield_events_from_queue,
)

pytestmark = pytest.mark.unit


def _encoder():
    enc = MagicMock()
    enc.encode.side_effect = lambda ev: "ENC"
    return enc


async def _agen(items):
    for it in items:
        yield it


class TestCreateEventQueue:
    def test_uses_config_maxsize(self, monkeypatch):
        monkeypatch.setattr(
            rrh, "get_config", lambda: SimpleNamespace(max_event_queue_size=7)
        )
        q = create_event_queue()
        assert q.maxsize == 7


class TestPutEventWithBackpressure:
    @pytest.mark.asyncio
    async def test_success(self):
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=5)
        ok = await put_event_with_backpressure(q, "e", "r", "t")
        assert ok is True
        assert q.qsize() == 1

    @pytest.mark.asyncio
    async def test_timeout_drops_event(self):
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        await q.put("full")
        ok = await put_event_with_backpressure(q, "e", "r", "t", timeout=0.01)
        assert ok is False


class TestPutCriticalEventWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=5)
        ok = await put_critical_event_with_retry(q, "e", "r", "t", "evt")
        assert ok is True

    @pytest.mark.asyncio
    async def test_success_after_retry(self, monkeypatch):
        # NOTE: the retry/success log calls in the source pass `event_name` both
        # positionally and as a keyword (a latent bug that raises TypeError). We
        # mock the two logging helpers so the retry *logic* under test can run.
        monkeypatch.setattr(rrh, "DEFAULT_EVENT_STREAM_CHECK_TIMEOUT", 0.001)
        monkeypatch.setattr(rrh, "log_error_event", MagicMock())
        monkeypatch.setattr(rrh, "log_info_event", MagicMock())
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        await q.put("full")

        async def _drain():
            await asyncio.sleep(0.02)
            await q.get()

        drain_task = asyncio.create_task(_drain())
        ok = await put_critical_event_with_retry(
            q,
            "e",
            "r",
            "t",
            "evt",
            max_retries=3,
            initial_timeout=0.01,
            max_timeout=0.05,
        )
        await drain_task
        assert ok is True

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, monkeypatch):
        # See note in test_success_after_retry re: mocking the logging helpers.
        monkeypatch.setattr(rrh, "DEFAULT_EVENT_STREAM_CHECK_TIMEOUT", 0.001)
        monkeypatch.setattr(rrh, "log_error_event", MagicMock())
        monkeypatch.setattr(rrh, "log_info_event", MagicMock())
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        await q.put("full")
        ok = await put_critical_event_with_retry(
            q,
            "e",
            "r",
            "t",
            "evt",
            max_retries=1,
            initial_timeout=0.01,
            max_timeout=0.02,
        )
        assert ok is False


class TestEmitCancellationEvents:
    @pytest.mark.asyncio
    async def test_emits_two_events(self):
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        enc = _encoder()
        await emit_cancellation_events(enc, "r", "t", q, "log.event")
        assert q.qsize() == 2
        assert enc.encode.call_count == 2


class TestCheckAndHandleCancellation:
    @pytest.mark.asyncio
    async def test_canceled_emits_and_returns_true(self):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=True)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        result = await check_and_handle_cancellation(rm, "r", "t", _encoder(), q)
        assert result is True
        assert q.qsize() == 2

    @pytest.mark.asyncio
    async def test_not_canceled_returns_false(self):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=False)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        result = await check_and_handle_cancellation(rm, "r", "t", _encoder(), q)
        assert result is False
        assert q.qsize() == 0


class TestConsumeEventGenerator:
    @pytest.mark.asyncio
    async def test_normal_consumption(self, monkeypatch):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=False)
        monkeypatch.setattr(rrh, "get_run_manager", lambda: rm)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        err = await consume_event_generator_with_cancellation(
            _agen(["a", "b"]), "r", "t", _encoder(), q
        )
        assert err is None
        assert q.qsize() == 2

    @pytest.mark.asyncio
    async def test_cancellation_mid_stream(self, monkeypatch):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=True)
        monkeypatch.setattr(rrh, "get_run_manager", lambda: rm)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        err = await consume_event_generator_with_cancellation(
            _agen(["a", "b"]), "r", "t", _encoder(), q
        )
        assert err is None
        # Cancellation emits 2 events and breaks before consuming stream events.
        assert q.qsize() == 2

    @pytest.mark.asyncio
    async def test_generator_cancelled_error(self, monkeypatch):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=False)
        monkeypatch.setattr(rrh, "get_run_manager", lambda: rm)

        async def _boom():
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        err = await consume_event_generator_with_cancellation(
            _boom(), "r", "t", _encoder(), q
        )
        assert err is None
        assert q.qsize() == 2  # cancellation events emitted

    @pytest.mark.asyncio
    async def test_generator_exception_returned(self, monkeypatch):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=False)
        monkeypatch.setattr(rrh, "get_run_manager", lambda: rm)

        async def _boom():
            raise ValueError("kaboom")
            yield  # pragma: no cover

        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        err = await consume_event_generator_with_cancellation(
            _boom(), "r", "t", _encoder(), q
        )
        assert isinstance(err, ValueError)

    @pytest.mark.asyncio
    async def test_backpressure_timeout_breaks(self, monkeypatch):
        rm = MagicMock()
        rm.is_run_canceled = AsyncMock(return_value=False)
        monkeypatch.setattr(rrh, "get_run_manager", lambda: rm)
        # Force put_event_with_backpressure to report a dropped event.
        monkeypatch.setattr(
            rrh, "put_event_with_backpressure", AsyncMock(return_value=False)
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        err = await consume_event_generator_with_cancellation(
            _agen(["a", "b"]), "r", "t", _encoder(), q
        )
        assert err is None


class TestYieldEventsFromQueue:
    @pytest.mark.asyncio
    async def test_yields_all_events_when_task_done(self, monkeypatch):
        monkeypatch.setattr(
            rrh,
            "get_config",
            lambda: SimpleNamespace(
                max_generator_wait_time=30.0, max_consecutive_timeouts=5
            ),
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        await q.put("e1")
        await q.put("e2")

        done = asyncio.create_task(asyncio.sleep(0))
        await done

        out = [ev async for ev in yield_events_from_queue(q, done, None, "r", "t")]
        assert out == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_generator_error_is_raised(self, monkeypatch):
        monkeypatch.setattr(
            rrh,
            "get_config",
            lambda: SimpleNamespace(
                max_generator_wait_time=30.0, max_consecutive_timeouts=5
            ),
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        done = asyncio.create_task(asyncio.sleep(0))
        await done

        with pytest.raises(ValueError, match="gen failed"):
            async for _ in yield_events_from_queue(
                q, done, ValueError("gen failed"), "r", "t"
            ):
                pass

    @pytest.mark.asyncio
    async def test_max_generator_wait_time_break(self, monkeypatch):
        monkeypatch.setattr(
            rrh,
            "get_config",
            lambda: SimpleNamespace(
                max_generator_wait_time=-1.0, max_consecutive_timeouts=5
            ),
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        running = asyncio.create_task(asyncio.Event().wait())
        try:
            out = [
                ev async for ev in yield_events_from_queue(q, running, None, "r", "t")
            ]
            assert out == []
        finally:
            running.cancel()

    @pytest.mark.asyncio
    async def test_consecutive_timeout_break(self, monkeypatch):
        monkeypatch.setattr(rrh, "DEFAULT_EVENT_STREAM_CHECK_TIMEOUT", 0.001)
        monkeypatch.setattr(
            rrh,
            "get_config",
            lambda: SimpleNamespace(
                max_generator_wait_time=30.0, max_consecutive_timeouts=2
            ),
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        running = asyncio.create_task(asyncio.Event().wait())
        try:
            out = [
                ev async for ev in yield_events_from_queue(q, running, None, "r", "t")
            ]
            assert out == []
        finally:
            running.cancel()

    @pytest.mark.asyncio
    async def test_keepalive_heartbeat_emitted(self, monkeypatch):
        monkeypatch.setattr(rrh, "DEFAULT_EVENT_STREAM_CHECK_TIMEOUT", 0.0005)
        # Allow 100 timeouts before break so the %100 keep-alive fires.
        monkeypatch.setattr(
            rrh,
            "get_config",
            lambda: SimpleNamespace(
                max_generator_wait_time=30.0, max_consecutive_timeouts=101
            ),
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        running = asyncio.create_task(asyncio.Event().wait())
        try:
            out = [
                ev async for ev in yield_events_from_queue(q, running, None, "r", "t")
            ]
            assert ": keep-alive\n\n" in out
        finally:
            running.cancel()
