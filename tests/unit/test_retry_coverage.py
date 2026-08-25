# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.retry.

Covers calculate_backoff_delay (jitter on/off, max cap), the RetryResult
dataclass, and retry_with_backoff across success/permanent/exhausted paths
for both sync and async operations. asyncio.sleep is patched so no real
waits occur; timing is fully deterministic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from server.constants import (
    DEFAULT_MIN_RETRY_DELAY,
    DEFAULT_RETRY_INITIAL_DELAY,
)
from server.retry import (
    RetryResult,
    calculate_backoff_delay,
    retry_with_backoff,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def no_sleep() -> AsyncMock:
    """Patch asyncio.sleep in the retry module so retries incur no real wait."""
    with patch("server.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        yield mock_sleep


# --------------------------------------------------------------------------
# calculate_backoff_delay
# --------------------------------------------------------------------------


def test_backoff_no_jitter_is_exponential() -> None:
    # initial=1, base=2 -> 1, 2, 4 for attempts 0,1,2
    assert (
        calculate_backoff_delay(
            0, initial_delay=1.0, exponential_base=2.0, jitter=False
        )
        == 1.0
    )
    assert (
        calculate_backoff_delay(
            1, initial_delay=1.0, exponential_base=2.0, jitter=False
        )
        == 2.0
    )
    assert (
        calculate_backoff_delay(
            2, initial_delay=1.0, exponential_base=2.0, jitter=False
        )
        == 4.0
    )


def test_backoff_caps_at_max_delay() -> None:
    delay = calculate_backoff_delay(
        attempt=50, initial_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False
    )
    assert delay == 5.0


def test_backoff_with_jitter_applies_floor() -> None:
    # Force jitter to push the delay negative so the DEFAULT_MIN_RETRY_DELAY floor is hit.
    with patch("random.uniform", return_value=-1000.0):
        delay = calculate_backoff_delay(0, initial_delay=1.0, jitter=True)
    assert delay == DEFAULT_MIN_RETRY_DELAY


def test_backoff_with_jitter_adds_amount() -> None:
    with patch("random.uniform", return_value=0.5):
        delay = calculate_backoff_delay(0, initial_delay=1.0, jitter=True)
    # 1.0 + 0.5 jitter
    assert delay == pytest.approx(1.5)


# --------------------------------------------------------------------------
# RetryResult
# --------------------------------------------------------------------------


def test_retry_result_defaults() -> None:
    r = RetryResult(success=True)
    assert r.success is True
    assert r.result is None
    assert r.attempts == 0
    assert r.errors == []
    assert r.total_time == 0.0


# --------------------------------------------------------------------------
# retry_with_backoff
# --------------------------------------------------------------------------


async def test_success_on_first_attempt_async(no_sleep: AsyncMock) -> None:
    async def op() -> str:
        return "ok"

    result = await retry_with_backoff(op, operation_name="op")
    assert result.success is True
    assert result.result == "ok"
    assert result.attempts == 1
    assert result.errors == []
    no_sleep.assert_not_called()


async def test_success_on_first_attempt_sync(no_sleep: AsyncMock) -> None:
    def op() -> int:
        return 42

    result = await retry_with_backoff(op)
    assert result.success is True
    assert result.result == 42
    assert result.attempts == 1


async def test_success_after_retries_logs_and_counts(no_sleep: AsyncMock) -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"

    result = await retry_with_backoff(
        flaky,
        max_retries=5,
        operation_name="flaky",
        context={"run_id": "r1"},
    )
    assert result.success is True
    assert result.result == "recovered"
    assert result.attempts == 3  # 2 failures + success
    assert len(result.errors) == 2
    # slept between the two failed attempts
    assert no_sleep.await_count == 2


async def test_permanent_error_not_retried(no_sleep: AsyncMock) -> None:
    async def op() -> None:
        raise ValueError("permanent, non-transient")

    result = await retry_with_backoff(op, max_retries=3, operation_name="perm")
    assert result.success is False
    assert result.result is None
    assert result.attempts == 1  # no retries for non-transient
    assert len(result.errors) == 1
    no_sleep.assert_not_called()


async def test_retries_exhausted(no_sleep: AsyncMock) -> None:
    async def always_fails() -> None:
        raise ConnectionError("still down")

    result = await retry_with_backoff(
        always_fails, max_retries=2, operation_name="dead"
    )
    assert result.success is False
    assert result.attempts == 3  # max_retries + 1
    assert len(result.errors) == 3
    # slept before each retry (attempts 0 and 1), not after the final failure
    assert no_sleep.await_count == 2


async def test_custom_retry_on_predicate(no_sleep: AsyncMock) -> None:
    # retry_on returns False -> treated as permanent even though it's a
    # normally-transient ConnectionError.
    async def op() -> None:
        raise ConnectionError("transient but we choose not to retry")

    result = await retry_with_backoff(
        op, max_retries=3, retry_on=lambda e: False, operation_name="custom"
    )
    assert result.success is False
    assert result.attempts == 1
    no_sleep.assert_not_called()


async def test_custom_retry_on_true_then_exhaust(no_sleep: AsyncMock) -> None:
    async def op() -> None:
        raise ValueError("normally permanent, but forced retry")

    result = await retry_with_backoff(
        op, max_retries=1, retry_on=lambda e: True, operation_name="force"
    )
    assert result.success is False
    assert result.attempts == 2
    assert no_sleep.await_count == 1


async def test_negative_max_retries_reaches_final_return(no_sleep: AsyncMock) -> None:
    # max_retries=-1 makes range(0) empty; loop body never runs and the
    # trailing edge-case return is exercised.
    async def op() -> str:
        return "never called"

    result = await retry_with_backoff(op, max_retries=-1, operation_name="edge")
    assert result.success is False
    assert result.result is None
    assert result.attempts == 0
    assert result.errors == []


async def test_default_initial_delay_used_between_retries() -> None:
    # Verify the delay passed to sleep uses the module default when not overridden.
    with (
        patch("server.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep,
        patch(
            "server.retry.calculate_backoff_delay",
            return_value=DEFAULT_RETRY_INITIAL_DELAY,
        ) as mock_calc,
    ):
        calls = {"n": 0}

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise TimeoutError("transient")
            return "ok"

        result = await retry_with_backoff(flaky, max_retries=3)
    assert result.success is True
    mock_calc.assert_called_once()
    mock_sleep.assert_awaited_once_with(DEFAULT_RETRY_INITIAL_DELAY)
