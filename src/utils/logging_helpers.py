# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Logging helper functions for standardized logging patterns.

Provides helper functions for consistent logging across the codebase with:
- Standardized log levels
- Consistent extra dict key names
- Common logging patterns (events, errors, debug info)

**Structured events:** Prefer `log_*_event` for operational and lifecycle events.
Use raw `logger.*` with `extra=` and %-style formatting for one-off debugging.
Avoid expensive f-strings in log messages when the level may be disabled; use
static messages and pass variable context via kwargs (extra).

**Key Functions:**
- `get_logger()` - Get logger instance for module
- `log_info_event()` - Log info-level events
- `log_warning_event()` - Log warning-level events
- `log_error_event()` - Log error-level events
- `log_debug_event()` - Log debug-level events
- `log_critical_event()` - Log critical-level events

**Error Message Style Guide:**
- Use sentence case (capitalize first letter)
- End with a period
- Keep messages concise; put context in kwargs for structured logging
- Prefer static messages; put variable context in kwargs (avoid f-strings in log
  messages so formatting is not evaluated when the level is disabled)
- Examples:
  - "Failed to encode event."
  - "Event generator error."
  - "Agent initialization failed."

**Usage Example:**
```python
from utils.logging_helpers import get_logger, log_info_event, log_error_event

logger = get_logger(__name__)

# Log info event
log_info_event(
    logger,
    "Starting run.",
    "ag_ui.run_starting",
    run_id="run-123",
    thread_id="thread-456"
)

# Log error event
try:
    result = some_operation()
except Exception as e:
    log_error_event(
        logger,
        "Failed to encode event.",
        "ag_ui.encoding_error",
        error=str(e),
        exc_info=True,
        run_id="run-123"
    )
```
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "STANDARD_KEYS",
    "get_logger",
    "log_critical_event",
    "log_debug_event",
    "log_error_event",
    "log_event",
    "log_info_event",
    "log_warning_event",
]

# Standardized extra dict key names
# Use snake_case consistently, avoid conflicts with LogRecord attributes
STANDARD_KEYS = {
    "request_id": "request_id",  # Injected by RequestIdFilter; "-" when no HTTP request
    "run_id": "run_id",
    "thread_id": "thread_id",
    "user_id": "user_id",
    "message_id": "message_id",
    "tool_call_id": "tool_call_id",
    "tool_name": "tool_name",
    "event_type": "event_type",
    "operation_name": "operation_name",
    "error": "error",
    "file_name": "file_name",  # Use file_name to avoid conflict with LogRecord.filename
    "mime_type": "mime_type",
    "node_id": "node_id",
    "node_type": "node_type",
    "message_count": "message_count",
    "mapping_count": "mapping_count",
    "count": "count",
    "keys": "keys",
    "original_keys": "original_keys",
    "state_type": "state_type",
    "return_type": "return_type",
    "result_content": "result_content",
    "tool_result": "tool_result",
}


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger, level: int, message: str, event_name: str, **kwargs: Any
) -> None:
    """Log an event with standardized extra dict.

    Args:
        logger: Logger instance
        level: Log level (logging.DEBUG, logging.INFO, etc.)
        message: Log message
        event_name: Event name (e.g., "ag_ui.run_starting")
        **kwargs: Additional context fields to include in extra dict
    """
    extra: dict[str, Any] = {"event": event_name}
    extra.update(kwargs)
    logger.log(level, message, extra=extra)


def log_debug_event(
    logger: logging.Logger, message: str, event_name: str, **kwargs: Any
) -> None:
    """Log a debug-level event (detailed diagnostic information).

    Use for:
    - Event flow tracking (e.g., "Received Strands event", "Emitted state snapshot")
    - Detailed state information (e.g., "Reusing cached agent", "No handler matched event")
    - Verbose operation details that are only needed during development/debugging

    Args:
        logger: Logger instance
        message: Log message
        event_name: Event name (e.g., "ag_ui.emitted_state_snapshot")
        **kwargs: Additional context fields to include in extra dict
    """
    log_event(logger, logging.DEBUG, message, event_name, **kwargs)


def log_info_event(
    logger: logging.Logger, message: str, event_name: str, **kwargs: Any
) -> None:
    """Log an info-level event (important operational events).

    Use for:
    - Lifecycle events (e.g., "Starting AG-UI run", "Run completed")
    - Configuration status (e.g., "Persistence enabled", "Phoenix initialized")
    - Important state changes that operators should be aware of

    Args:
        logger: Logger instance
        message: Log message
        event_name: Event name (e.g., "ag_ui.run_starting")
        **kwargs: Additional context fields to include in extra dict
    """
    log_event(logger, logging.INFO, message, event_name, **kwargs)


def log_warning_event(
    logger: logging.Logger,
    message: str,
    event_name: str,
    exc_info: bool = False,
    **kwargs: Any,
) -> None:
    """Log a warning-level event (recoverable issues).

    Use for:
    - Missing optional data (e.g., "EventEncoder not available, using fallback")
    - Recoverable errors (e.g., "Failed to save event to persistence" - operation continues)
    - Edge cases handled gracefully (e.g., "Completing remaining active tool calls")
    - Deprecated feature usage

    Args:
        logger: Logger instance
        message: Log message
        event_name: Event name (e.g., "ag_ui.persistence_operation_failed")
        exc_info: If True, include exception info
        **kwargs: Additional context fields to include in extra dict
    """
    extra: dict[str, Any] = {"event": event_name}
    extra.update(kwargs)
    logger.warning(message, extra=extra, exc_info=exc_info)


def log_error_event(
    logger: logging.Logger,
    message: str,
    event_name: str,
    error: Exception | str | None = None,
    exc_info: bool | tuple[type[BaseException], BaseException, object] = True,
    **kwargs: Any,
) -> None:
    """Log an error-level event (errors that require attention).

    Use for:
    - Operation failures that are handled (e.g., "Encoding error", "Run error")
    - Missing required data that indicates a bug (e.g., "Missing tool_call_id")
    - Always use exc_info=True for exceptions

    Args:
        logger: Logger instance
        message: Log message
        event_name: Event name (e.g., "ag_ui.encoding_error")
        error: Exception instance or error message string
        exc_info: If True, use sys.exc_info(); if a (type, value, tb) tuple, use it
        **kwargs: Additional context fields to include in extra dict
    """
    extra: dict[str, Any] = {"event": event_name}
    if error is not None:
        extra["error"] = str(error)
    extra.update(kwargs)
    # When exc_info is a tuple (e.g. from asyncio exception handler), put it in
    # extra so tests and log processors can access it; Logger.error accepts it.
    if isinstance(exc_info, tuple):
        extra["exc_info"] = exc_info
    logger.error(message, extra=extra, exc_info=exc_info)


def log_critical_event(
    logger: logging.Logger,
    message: str,
    event_name: str,
    error: Exception | str | None = None,
    exc_info: bool = True,
    **kwargs: Any,
) -> None:
    """Log a critical-level event (system-level failures).

    Use for:
    - System-level failures (e.g., "Failed to initialize agent system")
    - Unrecoverable errors that prevent the service from functioning

    Args:
        logger: Logger instance
        message: Log message
        event_name: Event name (e.g., "ag_ui.system_initialization_failed")
        error: Exception instance or error message string
        exc_info: If True, include exception info (default: True for critical errors)
        **kwargs: Additional context fields to include in extra dict
    """
    extra: dict[str, Any] = {"event": event_name}
    if error is not None:
        extra["error"] = str(error)
    extra.update(kwargs)
    logger.critical(message, extra=extra, exc_info=exc_info)
