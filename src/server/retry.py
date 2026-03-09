"""Retry logic with exponential backoff for transient failures.

This module provides retry utilities with configurable backoff strategies
for handling transient errors.

**Key Features:**
- Exponential backoff with jitter
- Configurable retry strategies
- Retry result tracking
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from server.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_RETRY_DELAY,
    DEFAULT_RETRY_EXPONENTIAL_BASE,
    DEFAULT_RETRY_INITIAL_DELAY,
    DEFAULT_RETRY_JITTER_MULTIPLIER,
    DEFAULT_RETRY_MAX_DELAY,
)
from server.error_classification import classify_error, is_transient_error
from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_warning_event,
)

logger = get_logger(__name__)


@dataclass
class RetryResult:
    """Result of a retry operation.

    Attributes:
        success: Whether the operation eventually succeeded
        result: The result value if successful, None otherwise
        attempts: Number of attempts made
        errors: List of errors encountered during retries
        total_time: Total time spent on retries (seconds)
    """

    success: bool
    result: Any | None = None
    attempts: int = 0
    errors: list[Exception] = field(default_factory=list)
    total_time: float = 0.0


def calculate_backoff_delay(
    attempt: int,
    initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    exponential_base: float = DEFAULT_RETRY_EXPONENTIAL_BASE,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter to prevent thundering herd

    Returns:
        Delay in seconds before next retry

    """
    import random

    # Exponential backoff: initial_delay * (base ^ attempt)
    delay = initial_delay * (exponential_base**attempt)
    delay = min(delay, max_delay)

    # Add jitter to prevent synchronized retries
    if jitter:
        jitter_amount = delay * DEFAULT_RETRY_JITTER_MULTIPLIER
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        delay = max(DEFAULT_MIN_RETRY_DELAY, delay)  # Ensure minimum delay

    return delay


async def retry_with_backoff(
    operation: Callable[[], Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    exponential_base: float = DEFAULT_RETRY_EXPONENTIAL_BASE,
    retry_on: Callable[[Exception | None], bool] | None = None,
    operation_name: str | None = None,
    context: dict[str, Any | None] = None,
) -> RetryResult:
    """Retry an operation with exponential backoff on transient errors.

    Args:
        operation: Async or sync callable to retry
        max_retries: Maximum number of retry attempts
                     (default: DEFAULT_MAX_RETRIES = 3)
        initial_delay: Initial delay in seconds before first retry
                      (default: DEFAULT_RETRY_INITIAL_DELAY = 1.0)
        max_delay: Maximum delay in seconds
                   (default: DEFAULT_RETRY_MAX_DELAY = 60.0)
        exponential_base: Base for exponential backoff calculation
                          (default: DEFAULT_RETRY_EXPONENTIAL_BASE = 2.0)
        retry_on: Optional function to determine if error should be retried.
                  If None, uses is_transient_error()
        operation_name: Name of operation for logging (optional)
        context: Additional context for logging (optional)

    Returns:
        RetryResult with success status, result, attempts, and errors

    """
    context = context or {}
    operation_name = operation_name or "operation"
    retry_on_func = retry_on or is_transient_error

    errors: list[Exception] = []
    start_time = time.time()

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Execute operation (handle both sync and async)
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()

            # Success - return result
            total_time = time.time() - start_time
            if attempt > 0:
                log_warning_event(
                    logger,
                    f"Operation '{operation_name}' succeeded after {attempt} retries",
                    "ag_ui.retry_success",
                    operation_name=operation_name,
                    attempts=attempt + 1,
                    total_time=total_time,
                    **context,
                )
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                errors=errors,
                total_time=total_time,
            )

        except Exception as e:
            errors.append(e)
            error_category = classify_error(e)

            # Check if we should retry this error
            if not retry_on_func(e):
                # Permanent error - don't retry
                total_time = time.time() - start_time
                log_error_event(
                    logger,
                    f"Operation '{operation_name}' failed with permanent error after {attempt + 1} attempts.",
                    "ag_ui.retry_permanent_error",
                    operation_name=operation_name,
                    attempts=attempt + 1,
                    error=str(e),
                    error_category=error_category.value,
                    total_time=total_time,
                    **context,
                )
                return RetryResult(
                    success=False,
                    result=None,
                    attempts=attempt + 1,
                    errors=errors,
                    total_time=total_time,
                )

            # Check if we've exhausted retries
            if attempt >= max_retries:
                total_time = time.time() - start_time
                log_error_event(
                    logger,
                    f"Operation '{operation_name}' failed after {max_retries + 1} attempts: {e}",
                    "ag_ui.retry_exhausted",
                    operation_name=operation_name,
                    attempts=max_retries + 1,
                    error=str(e),
                    error_category=error_category.value,
                    total_time=total_time,
                    **context,
                )
                return RetryResult(
                    success=False,
                    result=None,
                    attempts=max_retries + 1,
                    errors=errors,
                    total_time=total_time,
                )

            # Calculate backoff delay
            delay = calculate_backoff_delay(
                attempt=attempt,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
            )

            log_warning_event(
                logger,
                f"Operation '{operation_name}' failed (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.2f}s: {e}",
                "ag_ui.retry_attempt",
                operation_name=operation_name,
                attempt=attempt + 1,
                max_retries=max_retries + 1,
                delay=delay,
                error=str(e),
                error_category=error_category.value,
                **context,
            )

            # Wait before retry
            await asyncio.sleep(delay)

    # Should never reach here, but handle edge case
    total_time = time.time() - start_time
    return RetryResult(
        success=False,
        result=None,
        attempts=len(errors),
        errors=errors,
        total_time=total_time,
    )
