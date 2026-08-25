# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.error_classification.

Exercises every branch of is_transient_error() and classify_error().
"""

from __future__ import annotations

import pytest

from server.error_classification import (
    ErrorCategory,
    classify_error,
    is_transient_error,
)
from server.exceptions import APIError, ValidationError

pytestmark = pytest.mark.unit


class _StatusError(Exception):
    """Plain exception carrying a status_code attribute (not an APIError)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------
# is_transient_error
# --------------------------------------------------------------------------


def test_transient_api_error_is_never_retryable() -> None:
    assert is_transient_error(APIError("x")) is False
    assert is_transient_error(ValidationError("bad")) is False


@pytest.mark.parametrize(
    "msg",
    [
        "Connection refused",
        "operation timeout",
        "network is down",
        "host unreachable",
        "connection refused by peer",
        "connection reset",
        "broken pipe",
    ],
)
def test_transient_network_message_patterns(msg: str) -> None:
    assert is_transient_error(Exception(msg)) is True


@pytest.mark.parametrize("code", [429, 502, 503, 504])
def test_transient_status_codes(code: int) -> None:
    assert is_transient_error(_StatusError("err", code)) is True


def test_non_transient_status_code() -> None:
    # 400 is not in the transient set; message has no transient keywords
    assert is_transient_error(_StatusError("bad request", 400)) is False


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("x"),
        TimeoutError("x"),
        TimeoutError(),
        OSError("x"),
    ],
)
def test_transient_exception_types(exc: Exception) -> None:
    assert is_transient_error(exc) is True


def test_transient_db_pool_pattern() -> None:
    # "pool" matches the db pattern but none of the network patterns / types
    assert is_transient_error(Exception("db pool saturated")) is True


def test_transient_db_unavailable_pattern() -> None:
    assert is_transient_error(Exception("service unavailable")) is True


def test_transient_rate_limit_message() -> None:
    assert is_transient_error(Exception("rate limit reached")) is True


def test_transient_429_in_message() -> None:
    assert is_transient_error(Exception("HTTP 429 Too Many Requests")) is True


def test_non_transient_default_false() -> None:
    assert is_transient_error(Exception("something odd happened")) is False


# --------------------------------------------------------------------------
# classify_error
# --------------------------------------------------------------------------


def test_classify_api_error_permanent() -> None:
    assert classify_error(APIError("x")) is ErrorCategory.PERMANENT


def test_classify_status_429_rate_limit() -> None:
    assert classify_error(_StatusError("slow down", 429)) is ErrorCategory.RATE_LIMIT


@pytest.mark.parametrize("code", [502, 503, 504])
def test_classify_status_5xx_transient(code: int) -> None:
    assert classify_error(_StatusError("gateway", code)) is ErrorCategory.TRANSIENT


def test_classify_timeout_type() -> None:
    assert classify_error(TimeoutError("x")) is ErrorCategory.TIMEOUT
    assert classify_error(TimeoutError()) is ErrorCategory.TIMEOUT


def test_classify_connection_type_network() -> None:
    assert classify_error(ConnectionError("x")) is ErrorCategory.NETWORK


def test_classify_oserror_network() -> None:
    assert classify_error(OSError("x")) is ErrorCategory.NETWORK


def test_classify_rate_limit_string() -> None:
    # transient via message, not a known type -> string branch
    assert classify_error(Exception("rate limit exceeded")) is ErrorCategory.RATE_LIMIT


def test_classify_timeout_string() -> None:
    assert classify_error(Exception("read timeout occurred")) is ErrorCategory.TIMEOUT


def test_classify_network_string() -> None:
    assert classify_error(Exception("host unreachable")) is ErrorCategory.NETWORK


def test_classify_generic_transient_string() -> None:
    # transient via db "pool" pattern, none of rate/timeout/network strings
    assert classify_error(Exception("db pool saturated")) is ErrorCategory.TRANSIENT


def test_classify_permanent_default() -> None:
    assert classify_error(Exception("unknown problem")) is ErrorCategory.PERMANENT
