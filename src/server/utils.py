"""Utility functions for AG-UI server.

This module contains helper functions to reduce code duplication and improve
maintainability across the server codebase.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from ag_ui.core import EventType, RunErrorEvent
from fastapi import Request

from server.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_INITIAL_DELAY,
    DEFAULT_RETRY_MAX_DELAY_SHORT,
)
from server.types import AGUIEvent
from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)

logger = get_logger(__name__)

# TypeVar for generic operation results
T = TypeVar("T")


class EventLike(Protocol):
    """Protocol for objects that have an event type attribute.

    This protocol is used for structural typing - any object that has
    a 'type' or 'eventType' attribute (even if None) will match.
    """

    type: EventType | str | None
    eventType: EventType | str | None  # noqa: N815  # camelCase to match AG-UI protocol


def get_event_type_from_object(
    event: EventLike | dict[str, Any] | AGUIEvent | object,
) -> EventType | None:
    """Extract event type from event object, normalizing to EventType enum.

    Handles cases where event objects may use either 'type' or 'eventType' as the attribute name,
    and normalizes string values to EventType enum. This ensures internal code always works with
    EventType enum values.

    Args:
        event: Event object (can be Pydantic model, dict, or any object with type/eventType attribute)

    Returns:
        EventType enum value, or None if neither attribute exists or value cannot be normalized

    """
    # Use explicit None check to avoid falling back when first attribute exists but is falsy
    event_type = getattr(event, "type", None)
    if event_type is None:
        event_type = getattr(event, "eventType", None)

    # If already an EventType enum, return as-is
    if isinstance(event_type, EventType):
        return event_type

    # If None, return None
    if event_type is None:
        return None

    # If string, try to convert to EventType enum
    if isinstance(event_type, str):
        # Handle cases like "EventType.TOOL_CALL_START" -> "TOOL_CALL_START"
        type_name = event_type.split(".")[-1] if "." in event_type else event_type
        try:
            return EventType[type_name]
        except (KeyError, AttributeError):
            # Invalid event type string - log warning and return None
            log_warning_event(
                logger,
                f"Invalid event type string: {event_type}",
                "ag_ui.invalid_event_type_string",
                event_type=event_type,
            )
            return None

    # Unknown type - return None
    return None


def create_error_event(message: str, code: str) -> RunErrorEvent:
    """Create a RunErrorEvent with consistent structure.

    Helper function to create error events with standardized format,
    reducing code duplication across the codebase.

    Args:
        message: Error message (can be plain text or formatted markdown)
        code: Error code (e.g., "ENCODING_ERROR", "RUN_ERROR", "INITIALIZATION_ERROR")

    Returns:
        RunErrorEvent instance with the specified message and code

    """
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=message,
        code=code,
    )


def get_event_type_name(event_type: EventType) -> str:
    """Extract event type name from EventType enum.

    Converts EventType enum to its string name. This is used at API boundaries
    (e.g., persistence, logging) where string representation is needed.

    Args:
        event_type: The EventType enum value

    Returns:
        The event type name as a string (e.g., "TOOL_CALL_START")

    """
    return event_type.name


def is_event_type(event: AGUIEvent | dict[str, Any], target_type: EventType) -> bool:
    """Check if an event object matches a target event type.

    Convenience function that extracts event type from an event object and compares
    it to a target EventType enum. This is the most common pattern when checking
    event types in handlers. The function normalizes string types to EventType enum
    internally, ensuring consistent enum-based comparison.

    Args:
        event: Event object (can be Pydantic model, dict, or any object with type/eventType attribute)
        target_type: The target EventType enum to compare against

    Returns:
        True if the event's type matches the target_type, False otherwise

    Example:
        if is_event_type(event, EventType.TEXT_MESSAGE_START):
            # Handle text message start

    """
    event_type = get_event_type_from_object(event)
    if event_type is None:
        return False
    return event_type == target_type


def get_user_id_from_request(request: Request) -> str:
    """Extract user ID from a FastAPI request.

    Attempts to get user ID from:
    1. Request state (set by authentication middleware if authenticated)
    2. X-User-Id header
    3. Authorization header (hash of the header as a partitioning identifier when
       no proper user ID is available; see Note below)
    4. Request client host (as fallback)

    Args:
        request: FastAPI Request object

    Returns:
        User ID string extracted using the fallback chain described above.
        Always returns a non-empty string.

    Note:
        The hashed Authorization-header fallback (step 3) is a
        **non-cryptographic, partitioning-only identifier**. It is used solely
        to partition data (e.g., per-request or per-token scoping) when a
        real identity is not available. It is **not** a security identity and
        must not be relied upon for authentication or authorization decisions.

        The X-User-Id header (step 2) is **trusted** when the header auth
        strategy is used. That is only safe when the server is behind a
        trusted proxy or frontend (e.g. OpenSearch Dashboards) that validates
        and sets X-User-Id. When not behind such a frontend, anyone can
        spoof X-User-Id. See auth middleware and AG-UI auth documentation.

    """
    # Try request state first (set by authentication middleware)
    if hasattr(request.state, "user_id") and request.state.user_id:
        return request.state.user_id

    # Try X-User-Id header
    user_id = request.headers.get("X-User-Id")
    if user_id:
        return user_id  # Early return - exit function immediately

    # Try to extract from Authorization header (partitioning identifier when
    # no proper user ID is available; see get_user_id_from_request docstring)
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # `[:8]` slices string to get first 8 characters
        return hashlib.sha256(auth_header.encode()).hexdigest()[:8]

    # Fallback to client host
    # Ternary operator: `value if condition else other_value`
    # Equivalent to: `if request.client: client_host = getattr(...) else: client_host = "unknown"`
    client_host = (
        getattr(request.client, "host", "unknown") if request.client else "unknown"
    )
    return f"client_{client_host}"


def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated.

    Args:
        request: FastAPI Request object

    Returns:
        True if request is authenticated, False otherwise

    """
    return hasattr(request.state, "authenticated") and request.state.authenticated


def log_security_event(
    logger: logging.Logger,
    event_type: str,  # "auth_failed", "access_denied", "auth_success"
    request: Request | None = None,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    reason: str | None = None,
    **kwargs,
) -> None:
    """Log security events for audit trail.

    Logs security-related events (authentication failures, access denials, etc.)
    with structured context for security monitoring, incident investigation,
    and compliance auditing.

    Args:
        logger: Logger instance
        event_type: Type of security event ("auth_failed", "access_denied", "auth_success")
        request: Optional FastAPI Request object (for IP address and path extraction)
        user_id: Optional user ID associated with the event
        resource_type: Optional type of resource (e.g., "thread", "run")
        resource_id: Optional resource identifier
        reason: Optional reason for the security event
        **kwargs: Additional context fields to include in the log

    Example:
        ```python
        # Log authentication failure
        log_security_event(
            logger,
            "auth_failed",
            request=request,
            reason="Missing or invalid credentials"
        )

        # Log access denial
        log_security_event(
            logger,
            "access_denied",
            request=request,
            user_id=user_id,
            resource_type="thread",
            resource_id=thread_id,
            owner_id=thread.get("user_id"),
            reason="User does not own resource"
        )
        ```
    """
    context = {
        "event_type": event_type,
        "user_id": user_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "reason": reason,
        "ip_address": request.client.host if request and request.client else None,
        "path": request.url.path if request else None,
        **kwargs,
    }

    if event_type in ("auth_failed", "access_denied"):
        log_warning_event(
            logger,
            f"Security event: {event_type}",
            f"ag_ui.security.{event_type}",
            **context,
        )
    else:
        log_info_event(
            logger,
            f"Security event: {event_type}",
            f"ag_ui.security.{event_type}",
            **context,
        )


def require_authenticated_if_auth_enabled(request: Request | None) -> None:
    """Require authentication when auth is enabled; raise if not authenticated.

    When authentication is enabled (config.auth_enabled), unauthenticated
    requests must be rejected to prevent bypassing auth. This helper runs
    that check at the route level as a defense in depth (in addition to
    auth middleware).

    - If ``request`` is None: no-op (e.g. internal or test call without request).
    - If auth is disabled: no-op.
    - If authenticated: no-op.
    - If auth enabled and not authenticated: raises UnauthorizedError (401).

    Args:
        request: FastAPI Request or None. When None, the check is skipped.

    Raises:
        UnauthorizedError: When auth is enabled and the request is not authenticated.

    """
    if request is None:
        return
    from server.config import get_config
    from server.exceptions import UnauthorizedError

    # Prefer app.state.config (set by create_app) so tests can inject without reset_config
    config = getattr(request.app.state, "config", None) or get_config()
    if not config.auth_enabled:
        return
    if is_authenticated(request):
        return

    # Log security event before raising exception
    log_security_event(
        logger, "auth_failed", request=request, reason="Missing or invalid credentials"
    )
    raise UnauthorizedError("Authentication required")


async def safe_persistence_operation_async(
    operation_name: str, operation_func: Callable[..., T], *args: Any, **kwargs: Any
) -> T | None:
    """Safely execute a persistence write operation with retry logic for transient errors.

    This is the async version that includes retry logic. For sync operations,
    use safe_persistence_operation().

    **INTENTIONAL SILENT FAILURE PATTERN:**
    This function implements intentional silent failure for write operations with
    automatic retry for transient errors. Failures are logged as warnings but do
    not raise exceptions, allowing the application to continue operating even when
    persistence is unavailable or failing.

    **Retry Logic:**
    - Automatically retries transient errors (network, timeout, connection errors)
    - Uses exponential backoff with jitter
    - Maximum 3 retries by default
    - Permanent errors are not retried

    **Rationale:**
    - Write operations (save operations) are non-critical for core functionality
    - The application can continue serving requests even if persistence fails
    - Retry logic improves success rate for transient failures
    - Failures are logged with full context for monitoring and debugging
    - This pattern prevents persistence issues from cascading to API failures

    **Monitoring:**
    Structured logs use event ``ag_ui.persistence_operation_failed`` with
    ``operation_name``, ``run_id``, ``thread_id``, ``error``, and ``attempts``.
    Operators should monitor and alert on this event (e.g. when failure rate or
    count exceeds a threshold); see AG_UI_LOGGING.md and AG_UI_PERSISTENCE.md.

    Args:
        operation_name: Name of the operation for logging (e.g., "save_run_start")
        operation_func: The persistence function to call (can be sync or async)
        *args: Positional arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Result of the operation function, or None if an error occurred after retries.

    """
    from server.error_classification import is_transient_error
    from server.retry import retry_with_backoff

    # Extract context for logging
    context = {
        "run_id": kwargs.get("run_id"),
        "thread_id": kwargs.get("thread_id"),
    }

    # Wrap operation for retry logic
    async def operation_with_retry() -> T:
        if asyncio.iscoroutinefunction(operation_func):
            return cast(T, await operation_func(*args, **kwargs))
        else:
            return cast(T, operation_func(*args, **kwargs))

    # Attempt operation with retry logic
    retry_result = await retry_with_backoff(
        operation=operation_with_retry,
        max_retries=DEFAULT_MAX_RETRIES,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY_SHORT,
        retry_on=is_transient_error,
        operation_name=operation_name,
        context=context,
    )

    if retry_result.success:
        return retry_result.result
    else:
        # All retries exhausted - log final failure
        context_parts = []
        if "run_id" in kwargs:
            context_parts.append(f"run_id={kwargs['run_id']}")
        if "thread_id" in kwargs:
            context_parts.append(f"thread_id={kwargs['thread_id']}")

        context_str = ", ".join(context_parts)
        if context_str:
            error_msg = f"Failed to {operation_name} after {retry_result.attempts} attempts: {context_str}"
        else:
            error_msg = (
                f"Failed to {operation_name} after {retry_result.attempts} attempts"
            )

        log_warning_event(
            logger,
            error_msg,
            "ag_ui.persistence_operation_failed",
            exc_info=True,
            operation_name=operation_name,
            run_id=kwargs.get("run_id"),
            thread_id=kwargs.get("thread_id"),
            error=str(retry_result.errors[-1])
            if retry_result.errors
            else "Unknown error",
            attempts=retry_result.attempts,
        )
        return None


def safe_persistence_operation(
    operation_name: str, operation_func: Callable[..., T], *args: Any, **kwargs: Any
) -> T | None:
    """Safely execute a persistence write operation with error handling.

    **INTENTIONAL SILENT FAILURE PATTERN:**
    This function implements intentional silent failure for write operations. Failures
    are logged as warnings but do not raise exceptions, allowing the application to
    continue operating even when persistence is unavailable or failing.

    **Note:** For async operations or when retry logic is needed, use
    `safe_persistence_operation_async()` instead.

    **Rationale:**
    - Write operations (save operations) are non-critical for core functionality
    - The application can continue serving requests even if persistence fails
    - Failures are logged with full context for monitoring and debugging
    - This pattern prevents persistence issues from cascading to API failures

    **Behavior:**
    - Logs warnings on failure (WARNING level, event: "ag_ui.persistence_operation_failed")
    - Returns None if operation fails (does not raise exceptions)
    - Includes full context (run_id, thread_id, error details) in logs
    - Uses exc_info=True for stack traces in logs

    **When to Use:**
    Use this wrapper for all persistence write operations:
    - save_thread()
    - save_run_start()
    - save_run_finish()
    - save_message()
    - save_event()

    **Monitoring:**
    Structured logs use event ``ag_ui.persistence_operation_failed`` with
    ``operation_name``, ``run_id``, ``thread_id``, and ``error``. Operators
    should monitor and alert on this event (e.g. when failure rate or count
    exceeds a threshold); see AG_UI_LOGGING.md and AG_UI_PERSISTENCE.md.

    Args:
        operation_name: Name of the operation for logging (e.g., "save_run_start")
        operation_func: The persistence function to call
        *args: Positional arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Result of the operation function, or None if an error occurred.
        Callers should not rely on the return value for write operations.

    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        # Extract context from kwargs for better error messages
        # `**kwargs` is a dict of keyword arguments passed to the function
        context_parts = []
        # `"key" in dict` checks if key exists
        if "run_id" in kwargs:
            context_parts.append(f"run_id={kwargs['run_id']}")
        if "thread_id" in kwargs:
            context_parts.append(f"thread_id={kwargs['thread_id']}")

        context_str = ", ".join(context_parts)
        if context_str:
            error_msg = f"Failed to {operation_name}: {context_str}, error={e}"
        else:
            error_msg = f"Failed to {operation_name}: error={e}"

        log_warning_event(
            logger,
            error_msg,
            "ag_ui.persistence_operation_failed",
            exc_info=True,
            operation_name=operation_name,
            run_id=kwargs.get("run_id"),
            thread_id=kwargs.get("thread_id"),
            error=str(e),
        )
        return None


def handle_persistence_read_operation(
    operation_name: str,
    operation_func: Callable[..., T],
    error_event_name: str,
    error_context: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a persistence read operation with standardized error handling.

    **EXPLICIT ERROR RAISING PATTERN:**
    This function implements explicit error raising for read operations. Failures
    are logged as errors and exceptions are raised, ensuring API consumers receive
    proper error responses.

    **Rationale:**
    - Read operations are part of the API contract - clients expect data or errors
    - Failures must be communicated to API consumers via HTTP error responses
    - This pattern ensures consistent error handling across all read endpoints
    - Errors are logged with full context before being raised

    **Behavior:**
    - Logs errors on failure (ERROR level, custom event name)
    - Re-raises expected exceptions (PersistenceNotEnabledError, NotFoundError)
    - Converts unexpected exceptions to InternalServerError
    - Includes full context in error logs and exception context
    - Uses exc_info=True for stack traces in logs

    **When to Use:**
    Use this wrapper for all persistence read operations:
    - get_thread()
    - get_threads()
    - get_run()
    - get_runs()
    - get_messages()
    - get_events()

    **Error Handling:**
    Expected exceptions (PersistenceNotEnabledError, NotFoundError) are re-raised
    as-is. Unexpected exceptions are wrapped in InternalServerError with context.

    Args:
        operation_name: Name of the operation for logging (e.g., "get_run")
        operation_func: The persistence function to call
        error_event_name: Event name for logging (e.g., "ag_ui.run_retrieval_error")
        error_context: Dictionary with context for error logging (e.g., {"run_id": "123"})
        *args: Positional arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Result of the operation function

    Raises:
        PersistenceNotEnabledError: If persistence is required but not enabled (re-raised)
        NotFoundError: If resource is not found (re-raised)
        InternalServerError: If an unexpected error occurs (wraps original exception)

    """
    from server.exceptions import (
        InternalServerError,
        NotFoundError,
        PersistenceNotEnabledError,
    )

    try:
        return operation_func(*args, **kwargs)
    except (PersistenceNotEnabledError, NotFoundError):
        # Re-raise expected exceptions
        raise
    except Exception as e:
        # Log unexpected errors with standardized format
        context_str = ", ".join(f"{k}={v}" for k, v in error_context.items())
        log_error_event(
            logger,
            f"Error retrieving {operation_name}: {context_str}, error={e}",
            error_event_name,
            error=str(e),
            exc_info=True,
            operation_name=operation_name,
            **error_context,
        )
        # Raise InternalServerError with context
        raise InternalServerError(
            f"Failed to retrieve {operation_name}: {str(e)}", context=error_context
        ) from e


def parse_json_with_fallback(
    text_content: str, fallback_value: Any | None = None
) -> Any | None:
    """Parse JSON string with fallback attempts for common formatting issues.

    Attempts to parse JSON with multiple strategies:
    1. Direct JSON parsing
    2. Replace single quotes with double quotes (common mistake)
    3. Return fallback value or raw text if all parsing fails

    This function reduces code duplication for JSON parsing patterns that appear
    throughout the codebase, particularly in tool result parsing.

    Args:
        text_content: String content to parse as JSON
        fallback_value: Value to return if all parsing attempts fail (default: None).
                       If None, returns the raw text_content.

    Returns:
        Parsed JSON data (dict/list), fallback_value, or raw text_content if parsing fails

    Example:
        result = parse_json_with_fallback('{"key": "value"}')  # Returns dict
        result = parse_json_with_fallback("{'key': 'value'}")  # Returns dict (after fix)
        result = parse_json_with_fallback("invalid", fallback="default")  # Returns "default"

    """
    if not text_content:
        return fallback_value

    try:
        # First attempt: direct JSON parsing
        parsed = json.loads(text_content)
        # Return parsed value (can be dict, list, str, int, float, bool, None)
        return parsed
    except json.JSONDecodeError:
        try:
            # Second attempt: replace single quotes with double quotes
            json_text = text_content.replace("'", '"')
            parsed = json.loads(json_text)
            # Return parsed value (can be dict, list, str, int, float, bool, None)
            return parsed
        except Exception as e:
            # All parsing attempts failed
            log_warning_event(
                logger,
                f"Failed to parse JSON after fallback attempts: {str(e)}",
                "ag_ui.json_parse_failed",
                error=str(e),
                exc_info=True,
            )
            return fallback_value if fallback_value is not None else text_content


