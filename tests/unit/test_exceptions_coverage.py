# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.exceptions.

Constructs every exception subtype and exercises message, code, status_code,
context, and subtype-specific attribute paths.
"""

from __future__ import annotations

import pytest

from server.exceptions import (
    APIError,
    ConflictError,
    ForbiddenError,
    InternalServerError,
    NotFoundError,
    PersistenceError,
    PersistenceNotEnabledError,
    UnauthorizedError,
    ValidationError,
)

pytestmark = pytest.mark.unit


def test_api_error_defaults() -> None:
    err = APIError("boom")
    assert err.message == "boom"
    assert err.code == "API_ERROR"
    assert err.status_code == 500
    assert err.context == {}
    # Exception str comes from message
    assert str(err) == "boom"
    assert isinstance(err, Exception)


def test_api_error_all_args() -> None:
    err = APIError("boom", code="X", status_code=418, context={"a": 1})
    assert err.code == "X"
    assert err.status_code == 418
    assert err.context == {"a": 1}


def test_persistence_error_default_status() -> None:
    err = PersistenceError("db down")
    assert err.code == "PERSISTENCE_ERROR"
    assert err.status_code == 503
    assert err.context == {}
    assert isinstance(err, APIError)


def test_persistence_error_custom_status_and_context() -> None:
    err = PersistenceError("db down", status_code=500, context={"k": "v"})
    assert err.status_code == 500
    assert err.context == {"k": "v"}


def test_persistence_not_enabled_with_operation() -> None:
    err = PersistenceNotEnabledError(operation="get_run", context={"runId": "r1"})
    assert err.message == "Persistence not enabled (required for get_run)"
    assert err.code == "PERSISTENCE_ERROR"
    assert err.status_code == 503
    assert err.context == {"runId": "r1"}
    assert isinstance(err, PersistenceError)


def test_persistence_not_enabled_without_operation() -> None:
    err = PersistenceNotEnabledError()
    assert err.message == "Persistence not enabled"
    assert err.status_code == 503


def test_not_found_run_adds_run_id_context() -> None:
    err = NotFoundError("Run", "run-123")
    assert err.message == "Run not found: run-123"
    assert err.code == "NOT_FOUND"
    assert err.status_code == 404
    assert err.context["runId"] == "run-123"
    assert err.resource_type == "Run"
    assert err.resource_id == "run-123"


def test_not_found_thread_adds_thread_id_context() -> None:
    err = NotFoundError("Thread", "thread-9", context={"extra": True})
    assert err.context["threadId"] == "thread-9"
    assert err.context["extra"] is True


def test_not_found_other_resource_no_id_context() -> None:
    err = NotFoundError("Message", "m-1")
    assert "runId" not in err.context
    assert "threadId" not in err.context


def test_not_found_does_not_mutate_caller_context() -> None:
    caller_ctx: dict = {}
    err = NotFoundError("Run", "r-1", context=caller_ctx)
    assert caller_ctx == {}  # copy was made
    assert err.context["runId"] == "r-1"


def test_validation_error_with_field() -> None:
    err = ValidationError("bad thread_id", field="thread_id")
    assert err.code == "VALIDATION_ERROR_THREAD_ID"
    assert err.status_code == 400
    assert err.field == "thread_id"


def test_validation_error_without_field() -> None:
    err = ValidationError("invalid")
    assert err.code == "VALIDATION_ERROR"
    assert err.field is None


def test_internal_server_error_default_and_custom() -> None:
    default = InternalServerError()
    assert default.message == "An internal server error occurred"
    assert default.code == "INTERNAL_SERVER_ERROR"
    assert default.status_code == 500

    custom = InternalServerError("kaboom", context={"c": 1})
    assert custom.message == "kaboom"
    assert custom.context == {"c": 1}


def test_unauthorized_error_default_and_custom() -> None:
    default = UnauthorizedError()
    assert default.message == "Authentication required"
    assert default.code == "UNAUTHORIZED"
    assert default.status_code == 401
    assert default.context == {}

    custom = UnauthorizedError("no token", context={"ip": "x"})
    assert custom.message == "no token"
    assert custom.context == {"ip": "x"}


def test_forbidden_error_default_and_custom() -> None:
    default = ForbiddenError()
    assert default.message == "Access forbidden"
    assert default.code == "FORBIDDEN"
    assert default.status_code == 403
    assert default.context == {}

    custom = ForbiddenError("nope", context={"user": "u"})
    assert custom.message == "nope"
    assert custom.context == {"user": "u"}


def test_conflict_error_default_and_custom() -> None:
    default = ConflictError()
    assert default.message == "Resource conflict"
    assert default.code == "CONFLICT"
    assert default.status_code == 409
    assert default.context == {}

    custom = ConflictError("dup run", context={"runId": "r"})
    assert custom.message == "dup run"
    assert custom.context == {"runId": "r"}
