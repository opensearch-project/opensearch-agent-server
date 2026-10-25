# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Integration tests for the /runs AG-UI SSE endpoints.

Validates:
- POST /runs SSE streaming lifecycle (RUN_STARTED -> TEXT_MESSAGE_* -> RUN_FINISHED)
- POST /runs schema validation (422 on invalid input)
- POST /runs error handling (RUN_ERROR event emitted, not HTTP 500)
- POST /runs/{run_id}/cancel cancellation handling (active, duplicate, not found)
- GET /runs/{run_id} status retrieval (with persistence, fallback, 404)
- GET /runs/{run_id}/events event retrieval (with persistence, fallback, 404)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Generator
from unittest.mock import MagicMock, patch

import pytest
from ag_ui.core import (
    EventType,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from fastapi.testclient import TestClient

from orchestrator.router import PageContextRouter
from server import ag_ui_app
from server.ag_ui_app import app, get_orchestrator
from server.agent_orchestrator import AgentOrchestrator
from server.constants import DEFAULT_EVENT_LIMIT
from server.run_manager import get_run_manager

pytestmark = pytest.mark.integration


def _parse_sse_events(response_text: str) -> list[dict]:
    """Parse SSE data lines into JSON dicts."""
    events = []
    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    """Mock MCP client that tracks stop() calls."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_strands_agent(mock_mcp_client: MagicMock) -> MagicMock:
    """Mock StrandsAgentCore with attached mock MCP client."""
    agent = MagicMock()
    agent._mcp_client = mock_mcp_client
    return agent


@pytest.fixture
def sample_events() -> list:
    """Standard sequence of AG-UI events representing an agent execution."""
    return [
        RunStartedEvent(
            type=EventType.RUN_STARTED, thread_id="thread-test-1", run_id="run-test-1"
        ),
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START, message_id="msg-resp-1", role="assistant"
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id="msg-resp-1",
            delta="Hello from OpenSearch agent!",
        ),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="msg-resp-1"),
        RunFinishedEvent(
            type=EventType.RUN_FINISHED, thread_id="thread-test-1", run_id="run-test-1"
        ),
    ]


@pytest.fixture
def orchestrator_with_mock_agent(mock_strands_agent: MagicMock) -> AgentOrchestrator:
    """Create an AgentOrchestrator configured with a mock agent factory."""
    router = MagicMock(spec=PageContextRouter)
    router.route.return_value.name = "default"

    orch = AgentOrchestrator(router)
    orch.register_agent_factory(
        name="default",
        factory=lambda: mock_strands_agent,
        description="Test agent factory",
    )
    return orch


@pytest.fixture
def client(
    orchestrator_with_mock_agent: AgentOrchestrator,
) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with rate limiting disabled and orchestrator dependency overridden."""
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator_with_mock_agent
    with patch("server.ag_ui_app.rate_limit", lambda f: f):
        yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_persistence() -> MagicMock:
    """Mock persistence provider implementing PersistenceProtocol."""
    persistence = MagicMock()
    persistence.get_run.return_value = None
    persistence.get_thread.return_value = None
    persistence.get_events.return_value = []
    return persistence


# ---------------------------------------------------------------------------
# POST /runs: Streaming and Validation Tests
# ---------------------------------------------------------------------------


class TestCreateRunEndpoint:
    """Integration tests for POST /runs SSE streaming endpoint."""

    def test_create_run_streams_valid_sse_events(
        self,
        client: TestClient,
        mock_mcp_client: MagicMock,
        sample_events: list,
    ):
        """Valid request returns SSE stream with RUN_STARTED, message lifecycle, and RUN_FINISHED."""

        async def fake_agui_run(input_data) -> AsyncIterator:
            for event in sample_events:
                yield event

        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui_cls:
            mock_agui_instance = MagicMock()
            mock_agui_instance.run = fake_agui_run
            mock_agui_cls.return_value = mock_agui_instance

            payload = {
                "threadId": "thread-test-1",
                "runId": "run-test-1",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "Hello"}],
            }

            resp = client.post("/runs", json=payload)

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = _parse_sse_events(resp.text)
        assert len(events) == 5

        # 1. RUN_STARTED event
        assert events[0]["type"] == "RUN_STARTED"
        assert events[0]["threadId"] == "thread-test-1"
        assert events[0]["runId"] == "run-test-1"

        # 2. TEXT_MESSAGE_START event
        assert events[1]["type"] == "TEXT_MESSAGE_START"
        assert events[1]["messageId"] == "msg-resp-1"
        assert events[1]["role"] == "assistant"

        # 3. TEXT_MESSAGE_CONTENT event
        assert events[2]["type"] == "TEXT_MESSAGE_CONTENT"
        assert events[2]["messageId"] == "msg-resp-1"
        assert events[2]["delta"] == "Hello from OpenSearch agent!"

        # 4. TEXT_MESSAGE_END event
        assert events[3]["type"] == "TEXT_MESSAGE_END"
        assert events[3]["messageId"] == "msg-resp-1"

        # 5. RUN_FINISHED event
        assert events[4]["type"] == "RUN_FINISHED"
        assert events[4]["threadId"] == "thread-test-1"
        assert events[4]["runId"] == "run-test-1"

        # Verify MCP client cleanup occurred on completion
        mock_mcp_client.stop.assert_called_once()

    def test_create_run_with_stream_iterator(
        self,
        client: TestClient,
        sample_events: list,
    ):
        """Valid request consumes chunked stream line-by-line via client.stream."""

        async def fake_agui_run(input_data) -> AsyncIterator:
            for event in sample_events:
                yield event

        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui_cls:
            mock_agui_instance = MagicMock()
            mock_agui_instance.run = fake_agui_run
            mock_agui_cls.return_value = mock_agui_instance

            payload = {
                "threadId": "thread-test-1",
                "runId": "run-test-1",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "Hello"}],
            }

            with client.stream("POST", "/runs", json=payload) as stream_resp:
                assert stream_resp.status_code == 200
                data_lines = [
                    line.strip()
                    for line in stream_resp.iter_lines()
                    if line.strip().startswith("data: ")
                ]

        assert len(data_lines) == 5
        first_event = json.loads(data_lines[0][6:])
        last_event = json.loads(data_lines[-1][6:])
        assert first_event["type"] == "RUN_STARTED"
        assert last_event["type"] == "RUN_FINISHED"

    def test_create_run_duplicate_returns_409(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When a run with status='running' already exists in persistence, returns HTTP 409."""
        mock_persistence.get_run.return_value = {"id": "r1", "status": "running"}
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)
        payload = {
            "threadId": "t1",
            "runId": "r1",
            "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "hi"}],
        }
        resp = client.post("/runs", json=payload)
        assert resp.status_code == 409

    def test_create_run_with_persistence_enabled(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        sample_events: list,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Valid request with persistence saves thread, run start, and run finish."""
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)

        async def fake_agui_run(input_data) -> AsyncIterator:
            for event in sample_events:
                yield event

        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui_cls:
            mock_agui_instance = MagicMock()
            mock_agui_instance.run = fake_agui_run
            mock_agui_cls.return_value = mock_agui_instance

            payload = {
                "threadId": "thread-persisted-1",
                "runId": "run-persisted-1",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "Hello"}],
            }

            resp = client.post("/runs", json=payload)

        assert resp.status_code == 200
        assert mock_persistence.save_thread.call_count >= 1
        mock_persistence.save_run_start.assert_called_once_with(
            run_id="run-persisted-1",
            thread_id="thread-persisted-1",
        )
        mock_persistence.save_run_finish.assert_called_once_with(
            run_id="run-persisted-1",
            status="completed",
        )

    @pytest.mark.parametrize(
        "invalid_payload",
        [
            {},  # empty body
            {
                "threadId": "t1",
                "runId": "r1",
                "messages": [{"role": "user", "content": "hi"}],
            },  # missing state
            {
                "state": {},
                "runId": "r1",
                "messages": [{"role": "user", "content": "hi"}],
            },  # missing threadId
            {
                "threadId": "t1",
                "state": {},
                "messages": [{"role": "user", "content": "hi"}],
            },  # missing runId
            {
                "threadId": "t1",
                "runId": "r1",
                "state": {},
                "messages": [],
            },  # empty messages
            {
                "threadId": "t1",
                "runId": "r1",
                "state": {},
                "messages": [{"role": "user"}],
            },  # missing content
        ],
    )
    def test_create_run_invalid_body_returns_422(
        self,
        client: TestClient,
        invalid_payload: dict,
    ):
        """Invalid request body returns HTTP 422 Unprocessable Entity."""
        resp = client.post("/runs", json=invalid_payload)
        assert resp.status_code == 422

    def test_agent_error_emits_run_error_event_not_500(
        self,
        client: TestClient,
    ):
        """Error during agent execution emits RUN_ERROR SSE event instead of HTTP 500."""

        async def failing_agui_run(input_data) -> AsyncIterator:
            yield RunStartedEvent(
                type=EventType.RUN_STARTED, thread_id="t-err", run_id="r-err"
            )
            raise RuntimeError("Underlying LLM service unreachable")

        with patch("server.agent_orchestrator.AGUIStrandsAgent") as mock_agui_cls:
            mock_agui_instance = MagicMock()
            mock_agui_instance.run = failing_agui_run
            mock_agui_cls.return_value = mock_agui_instance

            payload = {
                "threadId": "t-err",
                "runId": "r-err",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "hi"}],
            }

            resp = client.post("/runs", json=payload)

        # Status must be 200 (streaming initiated), not 500
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        assert len(events) >= 2

        assert events[0]["type"] == "RUN_STARTED"
        assert events[1]["type"] == "RUN_ERROR"
        assert "Underlying LLM service unreachable" in events[1]["message"]
        assert events[1]["code"] == "RUN_ERROR"


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/cancel: Cancellation Tests
# ---------------------------------------------------------------------------


class TestCancelRunEndpoint:
    """Integration tests for POST /runs/{run_id}/cancel endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_active_run(self, client: TestClient):
        """Canceling an active run cancels the task and returns canceled: True."""
        run_manager = get_run_manager()
        run_id = "run-active-cancel-1"

        async def dummy_long_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(dummy_long_task())
        await run_manager.register_run(run_id, task)

        try:
            resp = client.post(f"/runs/{run_id}/cancel")
            assert resp.status_code == 200
            data = resp.json()
            assert data["runId"] == run_id
            assert data["canceled"] is True
            assert "successfully" in data["message"]
            assert task.cancelled() or task.cancelling()
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await run_manager.unregister_run(run_id)

    @pytest.mark.asyncio
    async def test_cancel_active_but_already_completed_run(self, client: TestClient):
        """When an active run's underlying task has already completed, returns canceled: False."""
        run_manager = get_run_manager()
        run_id = "run-active-completed-1"

        async def dummy_completed_task():
            return "done"

        task = asyncio.create_task(dummy_completed_task())
        await task
        await run_manager.register_run(run_id, task)

        try:
            resp = client.post(f"/runs/{run_id}/cancel")
            assert resp.status_code == 200
            data = resp.json()
            assert data["runId"] == run_id
            assert data["canceled"] is False
            assert "already completed" in data["message"]
        finally:
            await run_manager.unregister_run(run_id)

    def test_cancel_already_canceled_run(self, client: TestClient):
        """Canceling an already canceled run returns canceled: True with appropriate message."""
        run_manager = get_run_manager()
        run_id = "run-already-canceled-1"

        with (
            patch.object(run_manager, "is_run_active", return_value=False),
            patch.object(run_manager, "is_run_canceled", return_value=True),
        ):
            resp = client.post(f"/runs/{run_id}/cancel")

        assert resp.status_code == 200
        data = resp.json()
        assert data["runId"] == run_id
        assert data["canceled"] is True
        assert "already canceled" in data["message"]

    def test_cancel_non_existent_run(self, client: TestClient):
        """Canceling a non-existent or already completed run returns canceled: False."""
        run_manager = get_run_manager()
        run_id = "run-non-existent-99"

        with (
            patch.object(run_manager, "is_run_active", return_value=False),
            patch.object(run_manager, "is_run_canceled", return_value=False),
        ):
            resp = client.post(f"/runs/{run_id}/cancel")

        assert resp.status_code == 200
        data = resp.json()
        assert data["runId"] == run_id
        assert data["canceled"] is False
        assert "not found or already completed" in data["message"]


# ---------------------------------------------------------------------------
# GET /runs/{run_id}: Status Retrieval Tests
# ---------------------------------------------------------------------------


class TestGetRunEndpoint:
    """Integration tests for GET /runs/{run_id} endpoint."""

    def test_get_run_with_persistence(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Returns run details from persistence store."""
        run_record = {
            "id": "run-status-1",
            "thread_id": "thread-status-1",
            "status": "completed",
            "created_at": "2026-09-23T12:00:00Z",
            "finished_at": "2026-09-23T12:01:00Z",
        }
        mock_persistence.get_run.return_value = run_record
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)

        resp = client.get("/runs/run-status-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "run-status-1"
        assert data["status"] == "completed"
        assert data["thread_id"] == "thread-status-1"
        mock_persistence.get_run.assert_called_once_with(run_id="run-status-1")

    def test_get_run_without_persistence_returns_fallback(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When persistence is disabled, returns fallback response with unknown status."""
        monkeypatch.setattr(ag_ui_app, "persistence", None)

        resp = client.get("/runs/run-no-persistence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "run-no-persistence"
        assert data["status"] == "unknown"
        assert data.get("metadata", {}).get("fallback") is True

    def test_get_run_not_found_returns_404(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When persistence is enabled but run does not exist, returns HTTP 404."""
        mock_persistence.get_run.return_value = None
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)

        resp = client.get("/runs/run-missing-99")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/events: Event Retrieval Tests
# ---------------------------------------------------------------------------


class TestGetRunEventsEndpoint:
    """Integration tests for GET /runs/{run_id}/events endpoint."""

    def test_get_run_events_with_persistence(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Returns stored events list for a completed run."""
        mock_persistence.get_run.return_value = {"id": "run-events-1"}
        mock_persistence.get_events.return_value = [
            {"id": "e1", "type": "RUN_STARTED", "data": {"threadId": "t1"}},
            {"id": "e2", "type": "TEXT_MESSAGE_CONTENT", "data": {"delta": "result"}},
            {"id": "e3", "type": "RUN_FINISHED", "data": {}},
        ]
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)

        resp = client.get("/runs/run-events-1/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runId"] == "run-events-1"
        assert data["count"] == 3
        assert len(data["events"]) == 3
        assert data["events"][0]["type"] == "RUN_STARTED"
        assert data["events"][2]["type"] == "RUN_FINISHED"

    def test_get_run_events_filtered_by_event_type(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Supports filtering stored events by event_type query parameter."""
        mock_persistence.get_run.return_value = {"id": "run-events-filter"}
        mock_persistence.get_events.return_value = [
            {"id": "e2", "type": "TEXT_MESSAGE_CONTENT", "data": {"delta": "result"}},
        ]
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)

        resp = client.get(
            "/runs/run-events-filter/events?event_type=TEXT_MESSAGE_CONTENT"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["eventType"] == "TEXT_MESSAGE_CONTENT"
        assert data["count"] == 1
        mock_persistence.get_events.assert_called_once_with(
            run_id="run-events-filter",
            event_type="TEXT_MESSAGE_CONTENT",
            limit=DEFAULT_EVENT_LIMIT,
            offset=0,
        )

    def test_get_run_events_without_persistence_returns_fallback(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When persistence is disabled, returns fallback response with empty events list."""
        monkeypatch.setattr(ag_ui_app, "persistence", None)

        resp = client.get("/runs/run-no-persistence/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runId"] == "run-no-persistence"
        assert data["events"] == []

    def test_get_run_events_not_found_returns_404(
        self,
        client: TestClient,
        mock_persistence: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When persistence is enabled but run does not exist, returns HTTP 404."""
        mock_persistence.get_run.return_value = None
        monkeypatch.setattr(ag_ui_app, "persistence", mock_persistence)

        resp = client.get("/runs/run-missing-99/events")
        assert resp.status_code == 404
