# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Error classification utilities for determining error types and recovery strategies.

This module provides utilities for classifying errors into categories
to determine appropriate recovery strategies.

**Key Features:**
- Error category classification (transient, permanent, rate limit, etc.)
- Transient error detection for retry logic
- Classification uses exception types (isinstance) and status_code before
  falling back to string matching, so message changes do not affect behavior
  for known error types.
"""

from __future__ import annotations

import asyncio
from enum import Enum

from server.exceptions import APIError


class ErrorCategory(Enum):
    """Categories of errors for recovery strategy selection."""

    TRANSIENT = "transient"  # Temporary errors that may succeed on retry
    PERMANENT = "permanent"  # Errors that won't succeed on retry
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    NETWORK = "network"  # Network-related errors
    TIMEOUT = "timeout"  # Timeout errors


def is_transient_error(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Transient errors are temporary conditions that may resolve on retry:
    - Network errors (connection refused, timeout)
    - Temporary service unavailability (503, 502)
    - Rate limiting (429 or "rate limit" in message) - with backoff
    - Database connection errors
    - Timeout errors

    Args:
        error: The exception to classify

    Returns:
        True if the error is transient and worth retrying, False otherwise

    """
    # Our API errors are never retryable (client/server contract, not transport)
    if isinstance(error, APIError):
        return False

    error_message = str(error).lower()

    # Network-related errors
    network_errors = (
        "connection",
        "timeout",
        "network",
        "unreachable",
        "refused",
        "reset",
        "broken pipe",
    )
    if any(net_err in error_message for net_err in network_errors):
        return True

    # HTTP status codes that indicate transient errors
    if hasattr(error, "status_code"):
        transient_status_codes = {
            429,
            502,
            503,
            504,
        }  # Rate limit, Bad Gateway, Service Unavailable, Gateway Timeout
        if error.status_code in transient_status_codes:
            return True

    # Specific exception types that are typically transient
    transient_exceptions = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,  # Often network-related
    )
    if isinstance(error, transient_exceptions):
        return True

    # Database connection errors (common patterns)
    db_error_patterns = ("connection", "pool", "timeout", "unavailable")
    if any(pattern in error_message for pattern in db_error_patterns):
        return True

    # Rate limiting (message may indicate 429 when status_code not present)
    if "rate limit" in error_message or "429" in str(error):
        return True

    return False


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an error into a category for recovery strategy selection.

    Uses exception types (isinstance) and status_code for known cases before
    falling back to string matching, so message changes do not affect behavior
    for ConnectionError, TimeoutError, asyncio.TimeoutError, OSError, or
    errors with status_code 429/502/503/504.

    Args:
        error: The exception to classify

    Returns:
        ErrorCategory indicating the type of error

    """
    # Our API errors are always permanent (use exc.status_code for HTTP response)
    if isinstance(error, APIError):
        return ErrorCategory.PERMANENT

    # 1. status_code (most reliable when present)
    if hasattr(error, "status_code"):
        if error.status_code == 429:
            return ErrorCategory.RATE_LIMIT
        if error.status_code in {502, 503, 504}:
            return ErrorCategory.TRANSIENT

    if is_transient_error(error):
        # 2. Exception type checks before string matching
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ErrorCategory.TIMEOUT
        if isinstance(error, (ConnectionError, OSError)):
            return ErrorCategory.NETWORK

        # 3. String matching as fallback for unknown exception types
        error_message = str(error).lower()
        if "rate limit" in error_message or "429" in str(error):
            return ErrorCategory.RATE_LIMIT
        if "timeout" in error_message:
            return ErrorCategory.TIMEOUT
        if any(
            net in error_message for net in ["connection", "network", "unreachable"]
        ):
            return ErrorCategory.NETWORK
        return ErrorCategory.TRANSIENT

    return ErrorCategory.PERMANENT
