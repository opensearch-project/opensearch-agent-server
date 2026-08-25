# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.run_routes handlers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse

import server.run_routes as run_routes
from server.exceptions import ConflictError, NotFoundError
from server.run_routes import (
    _extract_auth_headers,
    cancel_run_route,
    create_run_route,
    get_run_events_route,
    get_run_route,
)

pytestmark = pytest.mark.unit


class _Headers:
    def __init__(self, data):
        self._data = {k.lower(): v for k, v in (data or {}).items()}

    def get(self, key, default=None):
        return self._data.get(key.lower(), default)


def _make_request(headers=None, auth_enabled=False, client_host="127.0.0.1"):
    config = SimpleNamespace(auth_enabled=auth_enabled)
    app = SimpleNamespace(state=SimpleNamespace(config=config))
    client = SimpleNamespace(host=client_host) if client_host else None
    return SimpleNamespace(
        headers=_Headers(headers),
        state=SimpleNamespace(),
        client=client,
        app=app,
        url=SimpleNamespace(path="/runs"),
    )


class TestExtractAuthHeaders:
    def test_returns_authorization_header(self):
        req = _make_request(headers={"Authorization": "Bearer tok"})
        assert _extract_auth_headers(req) == {"authorization": "Bearer tok"}

    def test_returns_none_when_absent(self):
        req = _make_request(headers={})
        assert _extract_auth_headers(req) is None


class TestCreateRunRoute:
    def _input(self):
        data = MagicMock()
        data.thread_id = "t1"
        data.run_id = "r1"
        data.messages = [{"role": "user", "content": "hi"}]
        data.to_run_agent_input.return_value = SimpleNamespace(messages=[])
        return data

    @pytest.mark.asyncio
    async def test_streams_events_without_persistence(self):
        async def fake_generate_events(**kwargs):
            yield "data: one\n\n"
            yield "data: two\n\n"

        with (
            patch.object(run_routes, "generate_events", fake_generate_events),
            patch.object(run_routes, "AGUIEventProcessor", MagicMock()),
            patch.object(run_routes, "AGUIActivityMonitor", MagicMock()),
        ):
            resp = create_run_route(MagicMock(), None, self._input(), _make_request())
            assert isinstance(resp, StreamingResponse)
            chunks = [c async for c in resp.body_iterator]

        assert "data: one\n\n" in chunks
        assert "data: two\n\n" in chunks

    @pytest.mark.asyncio
    async def test_duplicate_running_run_raises_conflict(self):
        persistence = MagicMock()
        persistence.get_run.return_value = {"status": "running"}
        with pytest.raises(ConflictError):
            create_run_route(MagicMock(), persistence, self._input(), _make_request())

    @pytest.mark.asyncio
    async def test_persists_thread_and_run_start(self):
        persistence = MagicMock()
        persistence.get_run.return_value = None
        persistence.get_thread.return_value = None

        async def fake_generate_events(**kwargs):
            yield "data: x\n\n"

        with (
            patch.object(run_routes, "generate_events", fake_generate_events),
            patch.object(run_routes, "AGUIEventProcessor", MagicMock()),
            patch.object(run_routes, "AGUIActivityMonitor", MagicMock()),
        ):
            resp = create_run_route(
                MagicMock(), persistence, self._input(), _make_request()
            )
            _ = [c async for c in resp.body_iterator]

        persistence.save_thread.assert_called()
        persistence.save_run_start.assert_called()


class TestGetRunRoute:
    def test_no_persistence_returns_fallback(self):
        result = get_run_route(None, "r1", request=_make_request())
        assert result["status"] == "unknown"
        assert result["metadata"]["fallback"] is True

    def test_returns_run_from_persistence(self):
        persistence = MagicMock()
        run = {"id": "r1", "status": "completed"}
        persistence.get_run.return_value = run
        result = get_run_route(persistence, "r1", request=_make_request())
        assert result == run

    def test_missing_run_raises_not_found(self):
        persistence = MagicMock()
        persistence.get_run.return_value = None
        with pytest.raises(NotFoundError):
            get_run_route(persistence, "r1", request=_make_request())

    def test_persistence_error_uses_fallback(self):
        persistence = MagicMock()
        persistence.get_run.side_effect = RuntimeError("db down")
        result = get_run_route(persistence, "r1", request=_make_request())
        assert result["metadata"]["fallback"] is True


class TestGetRunEventsRoute:
    def test_no_persistence_returns_fallback(self):
        result = get_run_events_route(None, "r1", request=_make_request())
        assert result["events"] == []
        assert result["runId"] == "r1"

    def test_returns_wrapped_events_list(self):
        persistence = MagicMock()
        persistence.get_run.return_value = {"id": "r1"}
        persistence.get_events.return_value = [{"e": 1}, {"e": 2}]
        result = get_run_events_route(persistence, "r1", request=_make_request())
        assert result["count"] == 2
        assert result["events"] == [{"e": 1}, {"e": 2}]

    def test_missing_run_raises_not_found(self):
        persistence = MagicMock()
        persistence.get_run.return_value = None
        with pytest.raises(NotFoundError):
            get_run_events_route(persistence, "r1", request=_make_request())

    def test_run_check_error_uses_fallback(self):
        persistence = MagicMock()
        persistence.get_run.side_effect = RuntimeError("boom")
        result = get_run_events_route(persistence, "r1", request=_make_request())
        assert result["events"] == []

    def test_events_error_returns_fallback_dict(self):
        persistence = MagicMock()
        persistence.get_run.return_value = {"id": "r1"}
        persistence.get_events.side_effect = RuntimeError("events down")
        result = get_run_events_route(persistence, "r1", request=_make_request())
        # handle_read_operation_with_fallback returns the fallback dict.
        assert result["events"] == []
        assert result["runId"] == "r1"


class TestCancelRunRoute:
    @pytest.mark.asyncio
    async def test_not_active_already_canceled(self):
        rm = MagicMock()
        rm.is_run_active = AsyncMock(return_value=False)
        rm.is_run_canceled = AsyncMock(return_value=True)
        with patch.object(run_routes, "get_run_manager", lambda: rm):
            result = await cancel_run_route(MagicMock(), "r1", request=_make_request())
        assert result["canceled"] is True
        assert "already canceled" in result["message"]

    @pytest.mark.asyncio
    async def test_not_active_not_canceled(self):
        rm = MagicMock()
        rm.is_run_active = AsyncMock(return_value=False)
        rm.is_run_canceled = AsyncMock(return_value=False)
        with patch.object(run_routes, "get_run_manager", lambda: rm):
            result = await cancel_run_route(MagicMock(), "r1", request=_make_request())
        assert result["canceled"] is False
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_active_cancel_success(self):
        rm = MagicMock()
        rm.is_run_active = AsyncMock(return_value=True)
        rm.cancel_run = AsyncMock(return_value=True)
        with patch.object(run_routes, "get_run_manager", lambda: rm):
            result = await cancel_run_route(MagicMock(), "r1", request=_make_request())
        assert result["canceled"] is True
        assert "successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_active_cancel_failed(self):
        rm = MagicMock()
        rm.is_run_active = AsyncMock(return_value=True)
        rm.cancel_run = AsyncMock(return_value=False)
        with patch.object(run_routes, "get_run_manager", lambda: rm):
            result = await cancel_run_route(MagicMock(), "r1", request=_make_request())
        assert result["canceled"] is False
        assert "already completed" in result["message"]
