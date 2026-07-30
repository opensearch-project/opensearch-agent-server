# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Utility functions for tool error formatting.

This module provides standardized error response formatting for all tool functions
to ensure consistent JSON error responses across the codebase. Use log_tool_error
in except blocks to both log the exception (with traceback) and return a formatted
string for the caller.
"""

from __future__ import annotations

import json
import logging


def format_tool_error(
    error_message: str,
    error_type: str = "error",
) -> str:
    """Format tool error as standardized JSON response.

    Args:
        error_message: Human-readable error message
        error_type: Type of error (default: "error")

    Returns:
        JSON-formatted error response string
    """
    return json.dumps({error_type: error_message}, indent=2)


def log_tool_error(
    logger: logging.Logger,
    message: str,
    error_type: str = "error",
) -> str:
    """Log the current exception with traceback and return a formatted tool error.

    Call from within an except block so the exception context is available.
    Logs for operators (traceback, timestamps); return value is for the agent/user.

    Args:
        logger: Module or component logger (e.g. from get_logger(__name__)).
        message: Human-readable error message (used for both log and return).
        error_type: Key used in the JSON response (default: "error").

    Returns:
        JSON-formatted error string suitable for tool return (format_tool_error).
    """
    logger.exception(message)
    return format_tool_error(message, error_type=error_type)
