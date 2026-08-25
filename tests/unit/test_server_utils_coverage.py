# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.utils helper functions.

Exercises event-type normalization, request identity/auth helpers, the
persistence wrappers (sync + async retry), and JSON parsing fallbacks.
All I/O and retry timing is mocked; no network, boto3, or real sleeps.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from ag_ui.core import EventType, RunErrorEvent

from server.retry import RetryResult
from server.utils import (
    create_error_event,
    get_event_type_from_object,
    get_event_type_name,
    get_user_id_from_request,
    handle_persistence_read_operation,
    is_authenticated,
    is_event_type,
    log_security_event,
    parse_json_with_fallback,
    require_authenticated_if_auth_enabled,
    safe_persistence_operation,
    safe_persistence_operation_async,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake FastAPI Request
# ---------------------------------------------------------------------------


def _make_request(
    *,
    state: dict | None = None,
    headers: dict | None = None,
    client_host: str | None = "1.2.3.4",
    has_client: bool = True,
    path: str = "/runs",
    config=None,
):
    state_ns = SimpleNamespace(**(state or {}))
    client_ns = SimpleNamespace(host=client_host) if has_client else None
    app_ns = SimpleNamespace(state=SimpleNamespace(config=config))
    return SimpleNamespace(
        state=state_ns,
        headers=headers or {},
        client=client_ns,
        url=SimpleNamespace(path=path),
        app=app_ns,
    )


# ---------------------------------------------------------------------------
# get_event_type_from_object / get_event_type_name / is_event_type
# ---------------------------------------------------------------------------


class TestEventTypeHelpers:
    def test_returns_enum_when_already_enum(self):
        obj = SimpleNamespace(type=EventType.TOOL_CALL_START)
        assert get_event_type_from_object(obj) is EventType.TOOL_CALL_START

    def test_plain_string_normalizes(self):
        obj = SimpleNamespace(type="TEXT_MESSAGE_START")
        assert get_event_type_from_object(obj) is EventType.TEXT_MESSAGE_START

    def test_dotted_string_normalizes(self):
        obj = SimpleNamespace(type="EventType.TOOL_CALL_END")
        assert get_event_type_from_object(obj) is EventType.TOOL_CALL_END

    def test_falls_back_to_event_type_attr(self):
        obj = SimpleNamespace(type=None, eventType="RUN_STARTED")
        assert get_event_type_from_object(obj) is EventType.RUN_STARTED

    def test_none_returns_none(self):
        obj = SimpleNamespace(type=None, eventType=None)
        assert get_event_type_from_object(obj) is None

    def test_invalid_string_returns_none(self):
        obj = SimpleNamespace(type="NOT_A_REAL_EVENT")
        assert get_event_type_from_object(obj) is None

    def test_unknown_type_returns_none(self):
        obj = SimpleNamespace(type=12345)
        assert get_event_type_from_object(obj) is None

    def test_get_event_type_name(self):
        assert get_event_type_name(EventType.RUN_ERROR) == "RUN_ERROR"

    def test_is_event_type_match(self):
        obj = SimpleNamespace(type="TEXT_MESSAGE_START")
        assert is_event_type(obj, EventType.TEXT_MESSAGE_START) is True

    def test_is_event_type_no_match(self):
        obj = SimpleNamespace(type="TEXT_MESSAGE_START")
        assert is_event_type(obj, EventType.RUN_ERROR) is False

    def test_is_event_type_none(self):
        obj = SimpleNamespace(type=None, eventType=None)
        assert is_event_type(obj, EventType.RUN_ERROR) is False


class TestCreateErrorEvent:
    def test_create_error_event(self):
        ev = create_error_event("boom", "RUN_ERROR")
        assert isinstance(ev, RunErrorEvent)
        assert ev.message == "boom"
        assert ev.code == "RUN_ERROR"
        assert ev.type == EventType.RUN_ERROR


# ---------------------------------------------------------------------------
# get_user_id_from_request
# ---------------------------------------------------------------------------


class TestGetUserIdFromRequest:
    def test_state_user_id_wins(self):
        req = _make_request(state={"user_id": "u-from-state"})
        assert get_user_id_from_request(req) == "u-from-state"

    def test_x_user_id_header(self):
        req = _make_request(headers={"X-User-Id": "header-user"})
        assert get_user_id_from_request(req) == "header-user"

    def test_authorization_header_hashed(self):
        req = _make_request(headers={"Authorization": "Bearer abc"})
        uid = get_user_id_from_request(req)
        assert len(uid) == 8

    def test_client_host_fallback(self):
        req = _make_request(client_host="10.0.0.9")
        assert get_user_id_from_request(req) == "client_10.0.0.9"

    def test_client_none_fallback(self):
        req = _make_request(has_client=False)
        assert get_user_id_from_request(req) == "client_unknown"


class TestIsAuthenticated:
    def test_true(self):
        req = _make_request(state={"authenticated": True})
        assert is_authenticated(req) is True

    def test_false_when_missing(self):
        req = _make_request()
        assert is_authenticated(req) is False


# ---------------------------------------------------------------------------
# log_security_event
# ---------------------------------------------------------------------------


class TestLogSecurityEvent:
    def test_auth_failed_logs_warning(self):
        logger = logging.getLogger("test.sec.warn")
        req = _make_request()
        with patch("server.utils.log_warning_event") as warn:
            log_security_event(logger, "auth_failed", request=req, reason="bad")
        warn.assert_called_once()

    def test_auth_success_logs_info(self):
        logger = logging.getLogger("test.sec.info")
        with patch("server.utils.log_info_event") as info:
            log_security_event(logger, "auth_success", user_id="u1")
        info.assert_called_once()


# ---------------------------------------------------------------------------
# require_authenticated_if_auth_enabled
# ---------------------------------------------------------------------------


class TestRequireAuthenticated:
    def test_none_request_is_noop(self):
        assert require_authenticated_if_auth_enabled(None) is None

    def test_auth_disabled_is_noop(self):
        cfg = SimpleNamespace(auth_enabled=False)
        req = _make_request(config=cfg)
        assert require_authenticated_if_auth_enabled(req) is None

    def test_authenticated_is_noop(self):
        cfg = SimpleNamespace(auth_enabled=True)
        req = _make_request(state={"authenticated": True}, config=cfg)
        assert require_authenticated_if_auth_enabled(req) is None

    def test_unauthenticated_raises(self):
        from server.exceptions import UnauthorizedError

        cfg = SimpleNamespace(auth_enabled=True)
        req = _make_request(config=cfg)
        with pytest.raises(UnauthorizedError):
            require_authenticated_if_auth_enabled(req)


# ---------------------------------------------------------------------------
# safe_persistence_operation (sync)
# ---------------------------------------------------------------------------


class TestSafePersistenceOperationSync:
    def test_success_returns_value(self):
        result = safe_persistence_operation("save", lambda: "ok")
        assert result == "ok"

    def test_failure_with_context_returns_none(self):
        def boom(**kwargs):
            raise RuntimeError("db down")

        result = safe_persistence_operation(
            "save_run", boom, run_id="r1", thread_id="t1"
        )
        assert result is None

    def test_failure_without_context_returns_none(self):
        def boom():
            raise ValueError("nope")

        assert safe_persistence_operation("save", boom) is None


# ---------------------------------------------------------------------------
# safe_persistence_operation_async (retry path mocked)
# ---------------------------------------------------------------------------


class TestSafePersistenceOperationAsync:
    async def test_success_with_async_operation(self):
        async def op(**kwargs):
            return "async-result"

        async def fake_retry(operation, **kwargs):
            val = await operation()
            return RetryResult(success=True, result=val, attempts=1, errors=[])

        with patch("server.retry.retry_with_backoff", side_effect=fake_retry):
            result = await safe_persistence_operation_async("save", op, run_id="r1")
        assert result == "async-result"

    async def test_success_with_sync_operation(self):
        def op(**kwargs):
            return "sync-result"

        async def fake_retry(operation, **kwargs):
            val = await operation()
            return RetryResult(success=True, result=val, attempts=1, errors=[])

        with patch("server.retry.retry_with_backoff", side_effect=fake_retry):
            result = await safe_persistence_operation_async("save", op)
        assert result == "sync-result"

    async def test_failure_with_context(self):
        def op(**kwargs):
            return "x"

        async def fake_retry(operation, **kwargs):
            return RetryResult(
                success=False, result=None, attempts=3, errors=[RuntimeError("boom")]
            )

        with patch("server.retry.retry_with_backoff", side_effect=fake_retry):
            result = await safe_persistence_operation_async(
                "save_run", op, run_id="r1", thread_id="t1"
            )
        assert result is None

    async def test_failure_without_context_or_errors(self):
        def op():
            return "x"

        async def fake_retry(operation, **kwargs):
            return RetryResult(success=False, result=None, attempts=2, errors=[])

        with patch("server.retry.retry_with_backoff", side_effect=fake_retry):
            result = await safe_persistence_operation_async("save", op)
        assert result is None


# ---------------------------------------------------------------------------
# handle_persistence_read_operation
# ---------------------------------------------------------------------------


class TestHandlePersistenceRead:
    def test_success(self):
        result = handle_persistence_read_operation(
            "get_run", lambda: {"id": 1}, "ag_ui.err", {"run_id": "r1"}
        )
        assert result == {"id": 1}

    def test_reraises_not_found(self):
        from server.exceptions import NotFoundError

        def op():
            raise NotFoundError("Run", "r1")

        with pytest.raises(NotFoundError):
            handle_persistence_read_operation("get_run", op, "ag_ui.err", {})

    def test_reraises_persistence_not_enabled(self):
        from server.exceptions import PersistenceNotEnabledError

        def op():
            raise PersistenceNotEnabledError("get_run")

        with pytest.raises(PersistenceNotEnabledError):
            handle_persistence_read_operation("get_run", op, "ag_ui.err", {})

    def test_wraps_unexpected_as_internal_error(self):
        from server.exceptions import InternalServerError

        def op():
            raise RuntimeError("kaboom")

        with pytest.raises(InternalServerError):
            handle_persistence_read_operation(
                "get_run", op, "ag_ui.err", {"run_id": "r1"}
            )


# ---------------------------------------------------------------------------
# parse_json_with_fallback
# ---------------------------------------------------------------------------


class TestParseJsonWithFallback:
    def test_empty_returns_fallback(self):
        assert parse_json_with_fallback("", fallback_value="fb") == "fb"

    def test_valid_json(self):
        assert parse_json_with_fallback('{"k": "v"}') == {"k": "v"}

    def test_single_quote_fixup(self):
        assert parse_json_with_fallback("{'k': 'v'}") == {"k": "v"}

    def test_invalid_returns_fallback_value(self):
        assert parse_json_with_fallback("<<<not json", fallback_value="def") == "def"

    def test_invalid_returns_raw_text_when_no_fallback(self):
        assert parse_json_with_fallback("<<<not json") == "<<<not json"
