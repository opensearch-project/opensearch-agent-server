# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.error_recovery.

Covers PartialSuccessResult properties, sync/async partial-success execution,
sync/async fallback execution (including async-guard ValueErrors), all six
fallback response generators, and handle_read_operation_with_fallback across
its success / re-raise / persistence-fallback / generic-fallback branches.
"""

from __future__ import annotations

import pytest

from server.error_recovery import (
    PartialSuccessResult,
    create_fallback_events_response,
    create_fallback_messages_response,
    create_fallback_run_response,
    create_fallback_runs_response,
    create_fallback_thread_response,
    create_fallback_threads_response,
    execute_with_fallback,
    execute_with_fallback_async,
    execute_with_partial_success,
    execute_with_partial_success_async,
    handle_read_operation_with_fallback,
)
from server.exceptions import NotFoundError, PersistenceNotEnabledError

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# PartialSuccessResult
# --------------------------------------------------------------------------


def test_partial_success_result_properties() -> None:
    r = PartialSuccessResult(total=4)
    r.successful.append(("a", 1))
    r.successful.append(("b", 2))
    r.failed.append(("c", Exception("x")))
    assert r.success_count == 2
    assert r.failure_count == 1
    assert r.success_rate == pytest.approx(0.5)


def test_partial_success_rate_zero_total() -> None:
    r = PartialSuccessResult(total=0)
    assert r.success_rate == 0.0


# --------------------------------------------------------------------------
# execute_with_partial_success (sync)
# --------------------------------------------------------------------------


def test_partial_success_all_succeed() -> None:
    result = execute_with_partial_success(
        items=[1, 2, 3], operation=lambda x: x * 2, operation_name="double"
    )
    assert result.success_count == 3
    assert result.failure_count == 0
    assert result.partial is False
    assert result.successful == [(1, 2), (2, 4), (3, 6)]


def test_partial_success_mixed_triggers_partial() -> None:
    def op(x: int) -> int:
        if x == 2:
            raise ValueError("bad")
        return x

    result = execute_with_partial_success(
        items=[1, 2, 3], operation=op, context={"batch": "b1"}
    )
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.partial is True


def test_partial_success_stop_on_error() -> None:
    def op(x: int) -> int:
        raise ValueError("always")

    result = execute_with_partial_success(
        items=[1, 2, 3], operation=op, continue_on_error=False
    )
    assert result.failure_count == 1  # stopped after first
    assert result.success_count == 0


def test_partial_success_rejects_async_operation() -> None:
    async def aop(x: int) -> int:
        return x

    result = execute_with_partial_success(items=[1, 2], operation=aop)
    # Each item raises ValueError internally -> all recorded as failures
    assert result.failure_count == 2
    assert result.success_count == 0
    assert all(isinstance(err, ValueError) for _, err in result.failed)


# --------------------------------------------------------------------------
# execute_with_partial_success_async
# --------------------------------------------------------------------------


async def test_partial_success_async_all_succeed() -> None:
    async def op(x: int) -> int:
        return x + 1

    result = await execute_with_partial_success_async(items=[1, 2], operation=op)
    assert result.success_count == 2
    assert result.partial is False


async def test_partial_success_async_mixed() -> None:
    async def op(x: int) -> int:
        if x == 2:
            raise RuntimeError("bad")
        return x

    result = await execute_with_partial_success_async(
        items=[1, 2, 3], operation=op, context={"k": "v"}
    )
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.partial is True


async def test_partial_success_async_stop_on_error() -> None:
    async def op(x: int) -> int:
        raise RuntimeError("always")

    result = await execute_with_partial_success_async(
        items=[1, 2, 3], operation=op, continue_on_error=False
    )
    assert result.failure_count == 1
    assert result.success_count == 0


# --------------------------------------------------------------------------
# execute_with_fallback (sync)
# --------------------------------------------------------------------------


def test_fallback_primary_succeeds() -> None:
    result = execute_with_fallback(
        operation=lambda: "primary", fallback=lambda: "fallback"
    )
    assert result == "primary"


def test_fallback_used_on_primary_failure() -> None:
    def op() -> str:
        raise RuntimeError("primary down")

    result = execute_with_fallback(
        operation=op,
        fallback=lambda: "fallback",
        operation_name="get",
        context={"k": 1},
    )
    assert result == "fallback"


def test_fallback_rejects_async_operation() -> None:
    async def aop() -> str:
        return "x"

    with pytest.raises(ValueError, match="execute_with_fallback_async"):
        execute_with_fallback(operation=aop, fallback=lambda: "fb")


def test_fallback_rejects_async_fallback_when_op_sync() -> None:
    def op() -> str:
        raise RuntimeError("down")

    async def afb() -> str:
        return "fb"

    with pytest.raises(ValueError, match="Fallback function cannot be async"):
        execute_with_fallback(operation=op, fallback=afb)


# --------------------------------------------------------------------------
# execute_with_fallback_async
# --------------------------------------------------------------------------


async def test_fallback_async_primary_succeeds() -> None:
    async def op() -> str:
        return "primary"

    async def fb() -> str:
        return "fb"

    result = await execute_with_fallback_async(operation=op, fallback=fb)
    assert result == "primary"


async def test_fallback_async_used_on_failure() -> None:
    async def op() -> str:
        raise RuntimeError("down")

    async def fb() -> str:
        return "fb"

    result = await execute_with_fallback_async(
        operation=op, fallback=fb, operation_name="fetch", context={"k": 1}
    )
    assert result == "fb"


# --------------------------------------------------------------------------
# Fallback response generators
# --------------------------------------------------------------------------


def test_create_fallback_run_response() -> None:
    resp = create_fallback_run_response("run-1")
    assert resp["id"] == "run-1"
    assert resp["status"] == "unknown"
    assert resp["metadata"]["fallback"] is True


def test_create_fallback_threads_response() -> None:
    resp = create_fallback_threads_response()
    assert resp == {"threads": [], "count": 0}


def test_create_fallback_thread_response() -> None:
    resp = create_fallback_thread_response("t-1")
    assert resp["id"] == "t-1"
    assert resp["metadata"]["reason"] == "persistence_unavailable"


def test_create_fallback_runs_response() -> None:
    resp = create_fallback_runs_response("t-1")
    assert resp == {"threadId": "t-1", "runs": [], "count": 0}


def test_create_fallback_messages_response() -> None:
    resp = create_fallback_messages_response("t-1", run_id="r-1")
    assert resp["threadId"] == "t-1"
    assert resp["runId"] == "r-1"
    assert resp["messages"] == []

    resp_no_run = create_fallback_messages_response("t-2")
    assert resp_no_run["runId"] is None


def test_create_fallback_events_response() -> None:
    resp = create_fallback_events_response("r-1", event_type="TEXT")
    assert resp["runId"] == "r-1"
    assert resp["eventType"] == "TEXT"
    assert resp["events"] == []


# --------------------------------------------------------------------------
# handle_read_operation_with_fallback
# --------------------------------------------------------------------------


def test_handle_read_success_passthrough() -> None:
    def op(a: int, b: int = 0) -> int:
        return a + b

    result = handle_read_operation_with_fallback(
        "read",
        op,
        lambda: -1,
        "ag_ui.err",
        {"k": "v"},
        2,
        b=3,
    )
    assert result == 5


def test_handle_read_not_found_reraises() -> None:
    def op() -> None:
        raise NotFoundError("Run", "r-1")

    with pytest.raises(NotFoundError):
        handle_read_operation_with_fallback(
            operation_name="read",
            operation_func=op,
            fallback_func=lambda: "fb",
            error_event_name="ag_ui.err",
            error_context={},
        )


def test_handle_read_persistence_not_enabled_uses_fallback() -> None:
    def op() -> None:
        raise PersistenceNotEnabledError(operation="read")

    result = handle_read_operation_with_fallback(
        operation_name="read",
        operation_func=op,
        fallback_func=lambda: "fallback",
        error_event_name="ag_ui.err",
        error_context={"k": "v"},
    )
    assert result == "fallback"


def test_handle_read_generic_error_uses_fallback() -> None:
    def op() -> None:
        raise RuntimeError("boom")

    result = handle_read_operation_with_fallback(
        operation_name="read",
        operation_func=op,
        fallback_func=lambda: "fallback",
        error_event_name="ag_ui.err",
        error_context={"k": "v"},
    )
    assert result == "fallback"
