"""Error recovery strategies for transient failures and partial success handling.

This module provides utilities for handling transient errors, implementing
retry logic and partial success handling.

**Key Features:**
- Retry logic with exponential backoff for transient errors
- Partial success handling for batch operations
- Configurable retry strategies and error classification
- Fallback response generation for API endpoints

**Usage Examples:**

Retry with exponential backoff (import from ``server.retry`` and ``server.error_classification``):
```python
from server.retry import retry_with_backoff
from server.error_classification import is_transient_error

@retry_with_backoff(max_retries=DEFAULT_MAX_RETRIES, initial_delay=DEFAULT_RETRY_INITIAL_DELAY)
async def save_to_database(data):
    # Operation that might fail transiently
    return db.save(data)
```

Partial success handling:
```python
from server.error_recovery import execute_with_partial_success

results = execute_with_partial_success(
    items=events,
    operation=lambda event: save_event(event),
    operation_name="save_events"
)
# Returns: {"successful": [...], "failed": [...], "partial": True}
```

**Module Structure:**
- `error_classification.py` - Error classification and categorization (use for ``is_transient_error``, ``classify_error``)
- `retry.py` - Retry logic with exponential backoff (use for ``retry_with_backoff``, ``RetryResult``)
- `error_recovery.py` - Fallback functions and partial success handling (this module)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from server.exceptions import NotFoundError, PersistenceNotEnabledError
from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_warning_event,
)

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")

__all__ = [
    # Partial success
    "PartialSuccessResult",
    "execute_with_partial_success",
    "execute_with_partial_success_async",
    # Fallback operations
    "execute_with_fallback",
    "execute_with_fallback_async",
    # Fallback response generators
    "create_fallback_run_response",
    "create_fallback_threads_response",
    "create_fallback_thread_response",
    "create_fallback_runs_response",
    "create_fallback_messages_response",
    "create_fallback_events_response",
    "handle_read_operation_with_fallback",
]


@dataclass
class PartialSuccessResult:
    """Result of a partial success operation.

    Attributes:
        successful: List of successful results
        failed: List of failed items with error information
        partial: Whether some items succeeded and some failed
        total: Total number of items processed
    """

    successful: list[tuple[Any, Any]] = field(default_factory=list)
    failed: list[tuple[Any, Exception]] = field(default_factory=list)
    partial: bool = False
    total: int = 0

    @property
    def success_count(self) -> int:
        """Number of successful operations."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed operations."""
        return len(self.failed)

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return self.success_count / self.total


def execute_with_partial_success(
    items: list[T],
    operation: Callable[[T], R],
    operation_name: str = "batch_operation",
    continue_on_error: bool = True,
    context: dict[str, Any | None] = None,
) -> PartialSuccessResult:
    """Execute an operation on multiple items, handling partial success.

    Processes items individually and collects both successful and failed results.
    Useful for batch operations where some items may fail while others succeed.

    Args:
        items: List of items to process
        operation: Function to apply to each item (must be synchronous).
                  Async operations are rejected with ValueError.
                  Use execute_with_partial_success_async() for async operations.
        operation_name: Name of operation for logging
        continue_on_error: If True, continue processing after errors (default: True)
        context: Additional context for logging

    Returns:
        PartialSuccessResult with successful and failed items, including
        success_count, failure_count, and success_rate

    Example:
        ```python
        def save_event(event):
            # Save event to database
            return db.save(event)

        events = [event1, event2, event3]
        result = execute_with_partial_success(
            items=events,
            operation=save_event,
            operation_name="save_events",
            continue_on_error=True,
        )
        # result.successful contains [(event1, saved_event1), (event2, saved_event2)]
        # result.failed contains [(event3, Exception(...))]
        # result.success_rate is 0.67 (2/3 succeeded)
        ```

    """
    context = context or {}
    result = PartialSuccessResult(total=len(items))

    for item in items:
        try:
            # Handle both sync and async operations
            if asyncio.iscoroutinefunction(operation):
                # For async, we'd need to run in event loop
                # This is a simplified version - in practice, callers should use async version
                raise ValueError(
                    "Async operations should use execute_with_partial_success_async"
                )
            else:
                op_result = operation(item)
                result.successful.append((item, op_result))

        except Exception as e:
            result.failed.append((item, e))
            log_warning_event(
                logger,
                f"Item failed in '{operation_name}': {e}",
                "ag_ui.partial_success_item_failed",
                operation_name=operation_name,
                error=str(e),
                **context,
            )

            if not continue_on_error:
                # Stop processing on first error
                break

    result.partial = len(result.successful) > 0 and len(result.failed) > 0

    if result.partial:
        log_warning_event(
            logger,
            f"Operation '{operation_name}' completed with partial success: "
            f"{result.success_count}/{result.total} succeeded",
            "ag_ui.partial_success",
            operation_name=operation_name,
            success_count=result.success_count,
            failure_count=result.failure_count,
            total=result.total,
            success_rate=result.success_rate,
            **context,
        )

    return result


async def execute_with_partial_success_async(
    items: list[T],
    operation: Callable[[T], Any],
    operation_name: str = "batch_operation",
    continue_on_error: bool = True,
    context: dict[str, Any | None] = None,
) -> PartialSuccessResult:
    """Execute an async operation on multiple items, handling partial success.

    Processes items individually and collects both successful and failed results.
    Useful for batch async operations where some items may fail while others succeed.

    Args:
        items: List of items to process
        operation: Async function to apply to each item
        operation_name: Name of operation for logging
        continue_on_error: If True, continue processing after errors (default: True)
        context: Additional context for logging

    Returns:
        PartialSuccessResult with successful and failed items, including
        success_count, failure_count, and success_rate

    Example:
        ```python
        async def save_event_async(event):
            # Save event to database asynchronously
            return await db.save_async(event)

        events = [event1, event2, event3]
        result = await execute_with_partial_success_async(
            items=events,
            operation=save_event_async,
            operation_name="save_events_async",
            continue_on_error=True,
            context={"run_id": "run-123"},
        )
        # result.successful contains [(event1, saved_event1), (event2, saved_event2)]
        # result.failed contains [(event3, Exception(...))]
        # result.success_rate is 0.67 (2/3 succeeded)
        ```

    """
    context = context or {}
    result = PartialSuccessResult(total=len(items))

    for item in items:
        try:
            op_result = await operation(item)
            result.successful.append((item, op_result))
        except Exception as e:
            result.failed.append((item, e))
            log_warning_event(
                logger,
                f"Item failed in '{operation_name}': {e}",
                "ag_ui.partial_success_item_failed",
                operation_name=operation_name,
                error=str(e),
                **context,
            )

            if not continue_on_error:
                break

    result.partial = len(result.successful) > 0 and len(result.failed) > 0

    if result.partial:
        log_warning_event(
            logger,
            f"Operation '{operation_name}' completed with partial success: "
            f"{result.success_count}/{result.total} succeeded",
            "ag_ui.partial_success",
            operation_name=operation_name,
            success_count=result.success_count,
            failure_count=result.failure_count,
            total=result.total,
            success_rate=result.success_rate,
            **context,
        )

    return result


def execute_with_fallback(
    operation: Callable[[], T],
    fallback: Callable[[], T],
    operation_name: str | None = None,
    context: dict[str, Any | None] = None,
) -> T:
    """Execute an operation with a fallback response on failure.

    Attempts to execute the primary operation. If it fails, executes
    the fallback function and returns its result instead of raising an exception.

    This is useful for providing default/cached responses when primary
    operations fail, allowing the API to continue functioning with degraded
    capabilities.

    Args:
        operation: Primary operation to execute (must be synchronous).
                  Async operations are rejected with ValueError.
                  Use execute_with_fallback_async() for async operations.
        fallback: Fallback operation to execute if primary fails (must be synchronous)
        operation_name: Name of operation for logging (optional)
        context: Additional context for logging (optional)

    Returns:
        Result from primary operation if successful, or fallback result if primary fails

    Example:
        ```python
        def get_cached_data():
            return cache.get("key")

        def get_default_data():
            return {"default": "value"}

        result = execute_with_fallback(
            operation=get_cached_data,
            fallback=get_default_data,
            operation_name="get_cached_data",
            context={"key": "my_key"},
        )
        # Returns cached data if available, otherwise default data
        ```

    """
    context = context or {}
    operation_name = operation_name or "operation"

    # Check for async operations before attempting execution
    # This ValueError should not be caught and should propagate to caller
    if asyncio.iscoroutinefunction(operation):
        raise ValueError("Async operations should use execute_with_fallback_async")

    try:
        return operation()
    except Exception as e:
        log_warning_event(
            logger,
            f"Operation '{operation_name}' failed, using fallback: {e}",
            "ag_ui.fallback_used",
            operation_name=operation_name,
            error=str(e),
            **context,
        )
        # Execute fallback
        if asyncio.iscoroutinefunction(fallback):
            raise ValueError("Fallback function cannot be async when operation is sync")
        return fallback()


async def execute_with_fallback_async(
    operation: Callable[[], Any],
    fallback: Callable[[], Any],
    operation_name: str | None = None,
    context: dict[str, Any | None] = None,
) -> Any:
    """Execute an async operation with a fallback response on failure.

    Attempts to execute the primary async operation. If it fails, executes
    the fallback async function and returns its result instead of raising an exception.

    This is useful for providing default/cached responses when primary
    async operations fail, allowing the API to continue functioning with degraded
    capabilities.

    Args:
        operation: Primary async operation to execute (must be a coroutine function).
        fallback: Fallback async operation to execute if primary fails (must be a coroutine function).
        operation_name: Name of operation for logging (optional)
        context: Additional context for logging (optional)

    Returns:
        Result from primary operation if successful, or fallback result if primary fails

    Example:
        ```python
        async def fetch_user_data(user_id: str):
            return await api.get_user(user_id)

        async def get_default_user_data(user_id: str):
            return {"id": user_id, "name": "Unknown"}

        result = await execute_with_fallback_async(
            operation=lambda: fetch_user_data("123"),
            fallback=lambda: get_default_user_data("123"),
            operation_name="fetch_user_data",
            context={"user_id": "123"},
        )
        # Returns user data if API call succeeds, otherwise default data
        ```

    """
    context = context or {}
    operation_name = operation_name or "operation"

    try:
        result = await operation()
        return result
    except Exception as e:
        log_warning_event(
            logger,
            f"Operation '{operation_name}' failed, using fallback: {e}",
            "ag_ui.fallback_used",
            operation_name=operation_name,
            error=str(e),
            **context,
        )
        # Execute fallback
        return await fallback()


# Fallback response generators for API endpoints


def create_fallback_run_response(run_id: str) -> dict[str, Any]:
    """Create a fallback run response when persistence fails.

    Provides a default response structure indicating the run status
    is unknown due to persistence unavailability.

    Args:
        run_id: Run identifier

    Returns:
        Dictionary with fallback run data

    """
    return {
        "id": run_id,
        "thread_id": "unknown",
        "created_at": None,
        "finished_at": None,
        "status": "unknown",
        "error_message": "Run data unavailable - persistence service unavailable",
        "metadata": {
            "fallback": True,
            "reason": "persistence_unavailable",
        },
    }


def create_fallback_threads_response() -> dict[str, Any]:
    """Create a fallback threads response when persistence fails.

    Returns:
        Dictionary with empty threads list

    """
    return {
        "threads": [],
        "count": 0,
    }


def create_fallback_thread_response(thread_id: str) -> dict[str, Any]:
    """Create a fallback thread response when persistence fails.

    Args:
        thread_id: Thread identifier

    Returns:
        Dictionary with fallback thread data

    """
    return {
        "id": thread_id,
        "user_id": None,
        "created_at": None,
        "updated_at": None,
        "metadata": {
            "fallback": True,
            "reason": "persistence_unavailable",
        },
    }


def create_fallback_runs_response(thread_id: str) -> dict[str, Any]:
    """Create a fallback runs response when persistence fails.

    Args:
        thread_id: Thread identifier

    Returns:
        Dictionary with empty runs list

    """
    return {
        "threadId": thread_id,
        "runs": [],
        "count": 0,
    }


def create_fallback_messages_response(
    thread_id: str, run_id: str | None = None
) -> dict[str, Any]:
    """Create a fallback messages response when persistence fails.

    Args:
        thread_id: Thread identifier
        run_id: Optional run identifier

    Returns:
        Dictionary with empty messages list

    """
    return {
        "threadId": thread_id,
        "runId": run_id,
        "messages": [],
        "count": 0,
    }


def create_fallback_events_response(
    run_id: str, event_type: str | None = None
) -> dict[str, Any]:
    """Create a fallback events response when persistence fails.

    Args:
        run_id: Run identifier
        event_type: Optional event type filter

    Returns:
        Dictionary with empty events list

    """
    return {
        "runId": run_id,
        "eventType": event_type,
        "events": [],
        "count": 0,
    }


def handle_read_operation_with_fallback(
    operation_name: str,
    operation_func: Callable[..., T],
    fallback_func: Callable[[], T],
    error_event_name: str,
    error_context: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a read operation with fallback response on failure.

    Attempts to execute the read operation. If it fails with a persistence
    error or unexpected error, returns a fallback response instead of raising
    an exception. This allows the API to continue functioning with degraded
    capabilities.

    **FALLBACK RESPONSE PATTERN:**
    This function implements graceful degradation for read operations. When
    persistence fails, instead of returning an error, it provides a fallback
    response indicating the data is unavailable.

    **Rationale:**
    - Read operations can provide fallback responses without breaking API contract
    - Fallback responses indicate degraded service but allow clients to continue
    - Better user experience than error responses for non-critical data
    - Allows monitoring of persistence failures without breaking client workflows

    **Behavior:**
    - Attempts primary operation first
    - On PersistenceNotEnabledError: Returns fallback response
    - On NotFoundError: Re-raises (not found is a valid response, not a failure)
    - On other errors: Returns fallback response and logs error
    - Includes fallback indicator in response metadata

    Args:
        operation_name: Name of operation for logging
        operation_func: The persistence function to call
        fallback_func: Function that returns fallback response
        error_event_name: Event name for logging
        error_context: Dictionary with context for error logging
        *args: Positional arguments to pass to operation function
        **kwargs: Keyword arguments to pass to operation function

    Returns:
        Result from operation function, or fallback response if operation fails

    """
    try:
        return operation_func(*args, **kwargs)
    except NotFoundError:
        # NotFoundError is a valid response - re-raise it
        raise
    except PersistenceNotEnabledError:
        # Persistence not enabled - return fallback
        log_warning_event(
            logger,
            f"Persistence not enabled for '{operation_name}', using fallback response",
            "ag_ui.fallback_persistence_not_enabled",
            operation_name=operation_name,
            **error_context,
        )
        return fallback_func()
    except Exception as e:
        # Unexpected error - return fallback and log error
        log_error_event(
            logger,
            f"Error in '{operation_name}', using fallback response.",
            error_event_name,
            error=str(e),
            exc_info=True,
            operation_name=operation_name,
            fallback_used=True,
            **error_context,
        )
        return fallback_func()
