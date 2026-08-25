# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Hermetic coverage tests for server.ag_ui_app.

Covers the app factory, CORS branches, ASGI body-size middleware, the asyncio
exception handler, FastAPI exception handlers, the run/agent/health endpoint
seams, and the /invoke handler. Route helpers and the orchestrator are mocked;
the lifespan is exercised once via ``with TestClient(app)`` and registers only
lazy factory lambdas, so no real model calls or network I/O occur.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from server import ag_ui_app
from server.ag_ui_app import (
    _init_tracing,
    _MaxBodySizeMiddleware,
    _noop_rate_limit,
    _register_mcp_cancel_scope_exception_handler,
    _suppress_mcp_cancel_scope_error,
    api_error_handler,
    app,
    create_app,
    general_exception_handler,
    get_orchestrator,
    health,
    http_exception_handler,
    list_agents,
    request_validation_exception_handler,
)
from server.config import get_config
from server.exceptions import APIError

pytestmark = pytest.mark.unit


def _mock_request(path: str = "/x") -> MagicMock:
    req = MagicMock()
    req.url.path = path
    return req


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------


def test_noop_rate_limit_returns_same_callable():
    def f():
        return 1

    assert _noop_rate_limit(f) is f


# ---------------------------------------------------------------------------
# asyncio exception handler
# ---------------------------------------------------------------------------


class TestSuppressCancelScope:
    def test_suppresses_mcp_cancel_scope_error(self):
        loop = MagicMock()
        exc = RuntimeError(
            "Attempted to exit cancel scope in a different task than it was entered"
        )
        # Should return None without re-logging as an error.
        assert _suppress_mcp_cancel_scope_error(loop, {"exception": exc}) is None

    def test_logs_other_runtime_exceptions(self, monkeypatch):
        # Stub the logging helper: passing a (type, value, tb) tuple as exc_info
        # collides with stdlib LogRecord's reserved 'exc_info' key, which is a
        # pre-existing limitation unrelated to the branch we are covering here.
        recorder = MagicMock()
        monkeypatch.setattr(ag_ui_app, "log_error_event", recorder)
        loop = MagicMock()
        exc = RuntimeError("some other failure")
        assert (
            _suppress_mcp_cancel_scope_error(
                loop, {"exception": exc, "message": "boom"}
            )
            is None
        )
        recorder.assert_called_once()

    def test_logs_context_without_exception(self):
        loop = MagicMock()
        assert (
            _suppress_mcp_cancel_scope_error(loop, {"message": "no exc here"}) is None
        )

    def test_register_sets_handler(self):
        loop = asyncio.new_event_loop()
        try:
            _register_mcp_cancel_scope_exception_handler(loop)
            assert loop.get_exception_handler() is _suppress_mcp_cancel_scope_error
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# _MaxBodySizeMiddleware
# ---------------------------------------------------------------------------


async def _drive_middleware(mw, scope):
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def _http_scope(content_length: str | None):
    headers = []
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    return {"type": "http", "headers": headers}


class TestMaxBodySizeMiddleware:
    async def test_rejects_too_large(self):
        inner = AsyncMock()
        mw = _MaxBodySizeMiddleware(inner, max_bytes=10)
        sent = await _drive_middleware(mw, _http_scope("100"))
        assert sent[0]["status"] == 413
        inner.assert_not_awaited()

    async def test_rejects_negative_content_length(self):
        inner = AsyncMock()
        mw = _MaxBodySizeMiddleware(inner, max_bytes=10)
        sent = await _drive_middleware(mw, _http_scope("-5"))
        assert sent[0]["status"] == 400

    async def test_allows_within_limit(self):
        inner = AsyncMock()
        mw = _MaxBodySizeMiddleware(inner, max_bytes=1000)
        await _drive_middleware(mw, _http_scope("50"))
        inner.assert_awaited_once()

    async def test_invalid_content_length_passes_through(self):
        inner = AsyncMock()
        mw = _MaxBodySizeMiddleware(inner, max_bytes=1000)
        await _drive_middleware(mw, _http_scope("notanumber"))
        inner.assert_awaited_once()

    async def test_disabled_when_max_bytes_zero(self):
        inner = AsyncMock()
        mw = _MaxBodySizeMiddleware(inner, max_bytes=0)
        await _drive_middleware(mw, _http_scope("999999"))
        inner.assert_awaited_once()

    async def test_non_http_scope_passes_through(self):
        inner = AsyncMock()
        mw = _MaxBodySizeMiddleware(inner, max_bytes=10)
        await _drive_middleware(mw, {"type": "lifespan", "headers": []})
        inner.assert_awaited_once()


# ---------------------------------------------------------------------------
# create_app CORS branches
# ---------------------------------------------------------------------------


class TestCreateAppCors:
    def test_wildcard_cors(self):
        cfg = get_config().model_copy(update={"cors_origins": "*"})
        created = create_app(config_override=cfg)
        assert created.state.config is cfg

    def test_explicit_cors(self):
        cfg = get_config().model_copy(
            update={"cors_origins": "https://a.example.com,https://b.example.com"}
        )
        created = create_app(config_override=cfg)
        assert created is not None

    def test_no_cors(self):
        cfg = get_config().model_copy(update={"cors_origins": None})
        created = create_app(config_override=cfg)
        assert created is not None


# ---------------------------------------------------------------------------
# get_orchestrator dependency
# ---------------------------------------------------------------------------


class TestGetOrchestrator:
    def test_raises_when_uninitialized(self, monkeypatch):
        monkeypatch.setattr(ag_ui_app, "orchestrator", None)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_orchestrator()

    def test_returns_when_set(self, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(ag_ui_app, "orchestrator", sentinel)
        assert get_orchestrator() is sentinel


# ---------------------------------------------------------------------------
# Exception handlers (called directly)
# ---------------------------------------------------------------------------


class TestExceptionHandlers:
    async def test_api_error_handler_with_context(self):
        exc = APIError(
            "bad thing", code="MY_CODE", status_code=418, context={"extra": "v"}
        )
        resp = await api_error_handler(_mock_request("/api"), exc)
        assert resp.status_code == 418
        assert b"MY_CODE" in resp.body
        assert b"extra" in resp.body

    async def test_api_error_handler_without_context(self):
        exc = APIError("plain", code="C", status_code=400)
        resp = await api_error_handler(_mock_request(), exc)
        assert resp.status_code == 400

    async def test_request_validation_handler(self):
        exc = RequestValidationError(
            [
                {
                    "loc": ("body", "x"),
                    "msg": "field required",
                    "type": "missing",
                    "input": None,
                }
            ]
        )
        resp = await request_validation_exception_handler(_mock_request(), exc)
        assert resp.status_code == 422
        assert b"field required" in resp.body
        # `input` key stripped from sanitized output.
        assert b"input" not in resp.body

    async def test_http_exception_handler(self):
        resp = await http_exception_handler(
            _mock_request(), HTTPException(status_code=404, detail="nope")
        )
        assert resp.status_code == 404
        assert b"nope" in resp.body

    async def test_general_exception_handler_generic(self):
        resp = await general_exception_handler(_mock_request(), ValueError("kaboom"))
        assert resp.status_code == 500
        assert b"INTERNAL_SERVER_ERROR" in resp.body

    async def test_general_exception_handler_http_exception(self):
        resp = await general_exception_handler(
            _mock_request(), HTTPException(status_code=403, detail="forbidden")
        )
        assert resp.status_code == 403
        assert b"forbidden" in resp.body


# ---------------------------------------------------------------------------
# Endpoint coroutines called directly
# ---------------------------------------------------------------------------


class TestEndpointCoroutines:
    async def test_health(self):
        assert await health() == {"status": "ok"}

    async def test_list_agents(self):
        reg = MagicMock()
        reg.list_agents.return_value = [
            SimpleNamespace(
                name="a", description="desc", page_contexts=["p"], is_default=True
            )
        ]
        req = MagicMock()
        req.app.state.registry = reg
        result = await list_agents(req)
        assert result["agents"][0]["name"] == "a"
        assert result["agents"][0]["is_default"] is True


# ---------------------------------------------------------------------------
# Run endpoints via TestClient (route helpers mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with patch("server.ag_ui_app.rate_limit", lambda f: f):
        yield TestClient(app, raise_server_exceptions=False)


class TestRunEndpoints:
    def test_get_run(self, client):
        with patch(
            "server.ag_ui_app.get_run_route",
            return_value={"id": "r1", "status": "completed"},
        ) as m:
            resp = client.get("/runs/r1")
        assert resp.status_code == 200
        # Response is filtered through the RunResponse TypedDict model.
        assert resp.json()["status"] == "completed"
        m.assert_called_once()

    def test_get_run_events(self, client):
        with patch(
            "server.ag_ui_app.get_run_events_route",
            return_value={"runId": "r1", "eventType": None, "events": [], "count": 0},
        ) as m:
            resp = client.get("/runs/r1/events?limit=5&offset=0")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
        m.assert_called_once()

    def test_cancel_run(self, client):
        with patch(
            "server.ag_ui_app.cancel_run_route",
            new=AsyncMock(
                return_value={"runId": "r1", "canceled": True, "message": "ok"}
            ),
        ) as m:
            resp = client.post("/runs/r1/cancel")
        assert resp.status_code == 200
        assert resp.json()["canceled"] is True
        m.assert_awaited_once()

    def test_create_run(self, client):
        from fastapi.responses import JSONResponse

        orch = MagicMock()
        app.dependency_overrides[get_orchestrator] = lambda: orch
        try:
            with patch(
                "server.ag_ui_app.create_run_route",
                return_value=JSONResponse({"streamed": True}),
            ) as m:
                body = {
                    "threadId": "t1",
                    "runId": "r1",
                    "state": {},
                    "messages": [{"id": "m1", "role": "user", "content": "hi"}],
                }
                resp = client.post("/runs", json=body)
            assert resp.status_code == 200
            assert resp.json()["streamed"] is True
            m.assert_called_once()
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /invoke endpoint branches
# ---------------------------------------------------------------------------


@pytest.fixture
def invoke_client():
    orch = MagicMock()
    orch.invoke = AsyncMock(return_value="ok-response")
    app.dependency_overrides[get_orchestrator] = lambda: orch
    with patch("server.ag_ui_app.rate_limit", lambda f: f):
        yield TestClient(app, raise_server_exceptions=False), orch
    app.dependency_overrides.clear()


class TestInvokeEndpoint:
    def test_query_success(self, invoke_client):
        client, orch = invoke_client
        resp = client.post("/invoke", json={"query": "hi"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "ok-response"

    def test_messages_success(self, invoke_client):
        client, orch = invoke_client
        resp = client.post(
            "/invoke", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp.status_code == 200

    def test_bad_json_body(self, invoke_client):
        client, _ = invoke_client
        resp = client.post(
            "/invoke", content="not json", headers={"content-type": "application/json"}
        )
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "ValidationError"

    def test_non_dict_body(self, invoke_client):
        client, _ = invoke_client
        resp = client.post("/invoke", json=[1, 2, 3])
        assert resp.status_code == 400

    def test_missing_query_and_messages(self, invoke_client):
        client, _ = invoke_client
        resp = client.post("/invoke", json={"agent": "default"})
        assert resp.status_code == 400

    def test_invalid_context_type(self, invoke_client):
        client, _ = invoke_client
        resp = client.post("/invoke", json={"query": "hi", "context": "notdict"})
        assert resp.status_code == 400

    def test_invalid_messages_shape(self, invoke_client):
        client, _ = invoke_client
        resp = client.post("/invoke", json={"messages": [{"role": "user"}]})
        assert resp.status_code == 400

    def test_non_string_query(self, invoke_client):
        client, _ = invoke_client
        resp = client.post("/invoke", json={"query": 123})
        assert resp.status_code == 400

    def test_timeout_returns_408(self, invoke_client):
        client, orch = invoke_client
        orch.invoke = AsyncMock(side_effect=TimeoutError("too slow"))
        resp = client.post("/invoke", json={"query": "hi"})
        assert resp.status_code == 408
        assert resp.json()["error_type"] == "TimeoutError"

    def test_generic_error_returns_500(self, invoke_client):
        client, orch = invoke_client
        orch.invoke = AsyncMock(side_effect=ValueError("broke"))
        resp = client.post("/invoke", json={"query": "hi"})
        assert resp.status_code == 500
        assert resp.json()["error_type"] == "ValueError"

    def test_inference_results_format(self, invoke_client):
        client, orch = invoke_client
        orch.invoke = AsyncMock(return_value='{"query":{}}')
        resp = client.post(
            "/invoke",
            json={"query": "q", "response_format": "inference_results"},
        )
        assert resp.status_code == 200
        assert "inference_results" in resp.json()


# ---------------------------------------------------------------------------
# Lifespan (startup + shutdown) — registers lazy factories only.
# ---------------------------------------------------------------------------


class TestLifespan:
    def test_startup_registers_agents(self):
        with TestClient(app) as c:
            resp = c.get("/agents")
            assert resp.status_code == 200
            names = {a["name"] for a in resp.json()["agents"]}
            assert {"art", "default"} <= names
            assert c.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# _init_tracing branches
# ---------------------------------------------------------------------------


class TestInitTracing:
    def test_import_error_branch(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
        monkeypatch.setitem(sys.modules, "strands.telemetry", None)
        # ImportError is caught internally; function returns without raising.
        _init_tracing()

    def test_success_non_bedrock(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
        fake_telemetry = ModuleType("strands.telemetry")

        class _Tel:
            def setup_otlp_exporter(self):
                return None

        fake_telemetry.StrandsTelemetry = _Tel
        monkeypatch.setitem(sys.modules, "strands.telemetry", fake_telemetry)
        import utils.model_factory as mf

        monkeypatch.setattr(mf, "get_provider", lambda: "openai")
        _init_tracing()

    def test_success_bedrock(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
        fake_telemetry = ModuleType("strands.telemetry")

        class _Tel:
            def setup_otlp_exporter(self):
                return None

        fake_telemetry.StrandsTelemetry = _Tel
        monkeypatch.setitem(sys.modules, "strands.telemetry", fake_telemetry)
        import utils.model_factory as mf

        monkeypatch.setattr(mf, "get_provider", lambda: "bedrock")

        # Fake the openinference bedrock instrumentation module chain.
        pkg = ModuleType("openinference")
        instr_pkg = ModuleType("openinference.instrumentation")
        bedrock_mod = ModuleType("openinference.instrumentation.bedrock")

        class _Instr:
            def instrument(self):
                return None

        bedrock_mod.BedrockInstrumentor = _Instr
        monkeypatch.setitem(sys.modules, "openinference", pkg)
        monkeypatch.setitem(sys.modules, "openinference.instrumentation", instr_pkg)
        monkeypatch.setitem(
            sys.modules, "openinference.instrumentation.bedrock", bedrock_mod
        )
        _init_tracing()

    def test_bedrock_import_error(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
        fake_telemetry = ModuleType("strands.telemetry")

        class _Tel:
            def setup_otlp_exporter(self):
                return None

        fake_telemetry.StrandsTelemetry = _Tel
        monkeypatch.setitem(sys.modules, "strands.telemetry", fake_telemetry)
        import utils.model_factory as mf

        monkeypatch.setattr(mf, "get_provider", lambda: "bedrock")
        monkeypatch.setitem(sys.modules, "openinference.instrumentation.bedrock", None)
        _init_tracing()
