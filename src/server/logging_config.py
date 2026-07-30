# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Logging configuration utilities for AG-UI server.

Provides JSON formatter for structured logging, request ID injection, and
configuration helpers.

**Key Components:**
- `JSONFormatter` - Formats log records as JSON for log aggregation systems
- `RequestIdFilter` - Injects request_id from context into each log record
- `request_id_contextvar` - Context variable set by RequestIdMiddleware per request
- `configure_logging()` - Configures logging with JSON or human-readable format
- `get_logging_config()` - Reads from get_config() (for AG-UI server)
- `get_logging_config_from_env()` - Reads only AG_UI_LOG_* from os.environ (for Chainlit, tests, scripts; no get_config dependency)

**Usage Example:**
```python
from server.logging_config import configure_logging, get_logging_config, get_logging_config_from_env

# AG-UI server: from get_config()
use_json, log_level = get_logging_config()
configure_logging(use_json=use_json, log_level=log_level)

# Chainlit, tests, scripts: from env only (no get_config)
use_json, log_level = get_logging_config_from_env()
configure_logging(use_json=use_json, log_level=log_level, force=True)

# Or configure programmatically
configure_logging(use_json=True, log_level="DEBUG", force=True)
```

**Environment Variables:**
- `AG_UI_LOG_FORMAT` - "json" for JSON format, anything else for human-readable
- `AG_UI_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from typing import Any

from server.config import get_config

# Context variable for request-scoped request ID. Set by RequestIdMiddleware for
# HTTP requests; unset for non-HTTP entry points (Chainlit, tests, scripts).
# RequestIdFilter injects this into each log record, using "-" when unset.
request_id_contextvar: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Formats log records as JSON with all extra fields included,
    making logs easily parseable by log aggregation systems.

    Example output:
    {
        "timestamp": "2025-12-09T10:00:00.000Z",
        "level": "INFO",
        "logger": "server.ag_ui_app",
        "message": "Starting AG-UI run: run_id=123, thread_id=456",
        "event": "ag_ui.run_starting",
        "run_id": "123",
        "thread_id": "456",
        "user_id": "user1",
        "message_count": 1,
        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Converts a LogRecord into a JSON string with all relevant fields,
        including extra fields passed via the `extra` parameter in logging calls.
        Standard LogRecord attributes are excluded to avoid duplication.

        Args:
            record: LogRecord to format

        Returns:
            JSON string representation of the log record with all extra fields included

        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add all extra fields from the record
        if hasattr(record, "__dict__"):
            # Filter out standard LogRecord attributes that we don't want in JSON
            excluded_attrs = {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            }

            for key, value in record.__dict__.items():
                if key not in excluded_attrs and not key.startswith("_"):
                    # Convert non-serializable objects to strings
                    try:
                        json.dumps(value)  # Test if serializable
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class RequestIdFilter(logging.Filter):
    """Injects request_id from context into each log record.

    Reads request_id_contextvar (set by RequestIdMiddleware for HTTP requests).
    When unset (e.g. Chainlit, tests, startup), uses "-" so the human-readable
    formatter and JSON output have a consistent field.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_contextvar.get(None)
        record.request_id = request_id if request_id is not None else "-"
        return True


def configure_logging(
    use_json: bool = False, log_level: str = "INFO", force: bool = False
) -> None:
    """Configure logging for the AG-UI server.

    Sets up logging with either JSON format (for production/log aggregation)
    or human-readable format (for development).

    Args:
        use_json: If True, use JSON formatter. If False, use human-readable format.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        force: If True, replace existing handlers. If False, only configure if no handlers exist.

    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Only clear handlers if forced or if no handlers exist
    # This prevents breaking existing logging configuration
    if force or not root_logger.handlers:
        root_logger.handlers.clear()

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.addFilter(RequestIdFilter())

        # Set formatter based on format preference
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
            )

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_logging_config_from_env() -> tuple[bool, str]:
    """Get logging format and level from AG_UI_LOG_* env vars only.

    Use for entry points that run without the AG-UI server (Chainlit, tests,
    scripts). Does not call get_config(), so it avoids server config and
    works when AG-UI server env is not set.

    Uses the same env var names (AG_UI_LOG_FORMAT, AG_UI_LOG_LEVEL) so a
    shared .env keeps logging in sync when both Chainlit and the server run.

    Returns:
        Tuple of (use_json: bool, log_level: str). Defaults: (False, "INFO").
    """
    fmt = (os.environ.get("AG_UI_LOG_FORMAT") or "human").strip().lower()
    lvl = (os.environ.get("AG_UI_LOG_LEVEL") or "INFO").strip().upper()
    return (fmt == "json", lvl)


def get_logging_config() -> tuple[bool, str]:
    """Get logging configuration from centralized config.

    Returns:
        Tuple of (use_json: bool, log_level: str)

    """
    config = get_config()
    use_json = config.log_format == "json"
    log_level = config.log_level

    return use_json, log_level
