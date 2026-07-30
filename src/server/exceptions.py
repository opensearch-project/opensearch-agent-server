# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Standardized exception classes for AG-UI server error handling.

This module provides a consistent exception hierarchy for API errors,
ensuring uniform error responses across all endpoints.

**Exception Hierarchy:**
- `APIError` - Base exception for all API errors
  - `PersistenceError` - Persistence-related errors (503)
    - `PersistenceNotEnabledError` - Persistence required but not enabled
  - `NotFoundError` - Resource not found (404)
  - `UnauthorizedError` - Authentication required (401)
  - `ValidationError` - Request validation errors (400)
  - `InternalServerError` - Unexpected errors (500)
  - `ForbiddenError` - Access forbidden (403)
  - `ConflictError` - Resource conflict, e.g. duplicate run (409)

**Usage Example:**
```python
from server.exceptions import ValidationError, NotFoundError

# Raise validation error
if not thread_id:
    raise ValidationError("thread_id is required", field="thread_id")

# Raise not found error
if not run:
    raise NotFoundError("Run", run_id, context={"runId": run_id})
```
"""

from __future__ import annotations

__all__ = [
    "APIError",
    "ConflictError",
    "ForbiddenError",
    "InternalServerError",
    "NotFoundError",
    "PersistenceError",
    "PersistenceNotEnabledError",
    "UnauthorizedError",
    "ValidationError",
]


class APIError(Exception):
    """Base exception for API errors.

    All API errors should inherit from this class to ensure consistent
    error handling and response formatting.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        status_code: HTTP status code for the error
        context: Optional dictionary with contextual fields (e.g., runId, threadId)

    """

    def __init__(
        self,
        message: str,
        code: str = "API_ERROR",
        status_code: int = 500,
        context: dict | None = None,
    ) -> None:
        """Initialize an API error.

        Args:
            message: Human-readable error message
            code: Machine-readable error code (default: "API_ERROR")
            status_code: HTTP status code (default: 500)
            context: Optional dictionary with contextual fields for error response

        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.context = context or {}
        super().__init__(self.message)


class PersistenceError(APIError):
    """Persistence-related errors.

    Raised when persistence operations fail or when persistence
    is required but not enabled.

    Attributes:
        message: Human-readable error message
        code: Always "PERSISTENCE_ERROR"
        status_code: HTTP status code (default: 503 Service Unavailable)
        context: Optional dictionary with contextual fields for error response

    """

    def __init__(
        self, message: str, status_code: int = 503, context: dict | None = None
    ) -> None:
        """Initialize a persistence error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code (default: 503 Service Unavailable)
            context: Optional dictionary with contextual fields for error response

        """
        super().__init__(
            message, code="PERSISTENCE_ERROR", status_code=status_code, context=context
        )


class PersistenceNotEnabledError(PersistenceError):
    """Error raised when persistence is required but not enabled.

    This is a specific case of PersistenceError with a 503 status code
    and a standard message format. Used when an endpoint requires persistence
    but the persistence layer is not configured or enabled.

    Attributes:
        message: Human-readable error message indicating persistence is not enabled
        code: Always "PERSISTENCE_ERROR"
        status_code: Always 503 (Service Unavailable)
        context: Optional dictionary with contextual fields for error response

    """

    def __init__(
        self, operation: str | None = None, context: dict | None = None
    ) -> None:
        """Initialize a persistence not enabled error.

        Args:
            operation: Optional operation name for context
            context: Optional dictionary with contextual fields for error response

        """
        if operation:
            message = f"Persistence not enabled (required for {operation})"
        else:
            message = "Persistence not enabled"
        super().__init__(message, status_code=503, context=context)


class NotFoundError(APIError):
    """Resource not found errors.

    Raised when a requested resource (run, thread, message, etc.)
    cannot be found in the persistence layer.

    Attributes:
        message: Human-readable error message with resource type and ID
        code: Always "NOT_FOUND"
        status_code: Always 404 (Not Found)
        context: Dictionary with contextual fields (includes runId or threadId if applicable)
        resource_type: Type of resource that was not found (e.g., "Run", "Thread")
        resource_id: Identifier of the resource that was not found

    """

    def __init__(
        self, resource_type: str, resource_id: str, context: dict | None = None
    ) -> None:
        """Initialize a not found error.

        Args:
            resource_type: Type of resource (e.g., "Run", "Thread")
            resource_id: Identifier of the resource that was not found
            context: Optional dictionary with contextual fields for error response

        """
        message = f"{resource_type} not found: {resource_id}"
        # Add resource identifier to context (copy to avoid mutating caller's dict)
        context = dict(context or {})
        if resource_type.lower() == "run":
            context["runId"] = resource_id
        elif resource_type.lower() == "thread":
            context["threadId"] = resource_id
        super().__init__(message, code="NOT_FOUND", status_code=404, context=context)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ValidationError(APIError):
    """Request validation errors.

    Raised when request data fails validation (e.g., missing required fields,
    invalid format, out-of-range values).

    Attributes:
        message: Human-readable error message describing the validation failure
        code: Machine-readable error code (e.g., "VALIDATION_ERROR_FIELD_NAME" or "VALIDATION_ERROR")
        status_code: Always 400 (Bad Request)
        context: Empty dictionary (no additional context)
        field: Optional name of the field that failed validation

    """

    def __init__(self, message: str, field: str | None = None) -> None:
        """Initialize a validation error.

        Args:
            message: Human-readable error message
            field: Optional field name that failed validation

        """
        if field:
            code = f"VALIDATION_ERROR_{field.upper()}"
        else:
            code = "VALIDATION_ERROR"
        super().__init__(message, code=code, status_code=400)
        self.field = field


class InternalServerError(APIError):
    """Internal server errors.

    Raised when an unexpected error occurs during request processing.
    This is a catch-all for errors that are not expected and indicate
    a bug or system issue rather than a client error.

    Attributes:
        message: Human-readable error message (default: generic message)
        code: Always "INTERNAL_SERVER_ERROR"
        status_code: Always 500 (Internal Server Error)
        context: Optional dictionary with contextual fields for error response

    """

    def __init__(
        self,
        message: str = "An internal server error occurred",
        context: dict | None = None,
    ) -> None:
        """Initialize an internal server error.

        Args:
            message: Human-readable error message (default: generic message)
            context: Optional dictionary with contextual fields for error response

        """
        super().__init__(
            message, code="INTERNAL_SERVER_ERROR", status_code=500, context=context
        )


class UnauthorizedError(APIError):
    """Unauthorized (authentication required) errors.

    Raised when a user must be authenticated to access a resource but the
    request is not authenticated (e.g., when auth is enabled and the
    request has no valid credentials).

    Attributes:
        message: Human-readable error message
        code: Always "UNAUTHORIZED"
        status_code: Always 401 (Unauthorized)
        context: Optional dictionary with contextual fields for error response

    """

    def __init__(
        self,
        message: str = "Authentication required",
        context: dict | None = None,
    ) -> None:
        """Initialize an unauthorized error.

        Args:
            message: Human-readable error message
            context: Optional dictionary with contextual fields for error response

        """
        super().__init__(
            message, code="UNAUTHORIZED", status_code=401, context=context or {}
        )


class ForbiddenError(APIError):
    """Forbidden access errors.

    Raised when a user attempts to access a resource they don't have permission
    to access (e.g., accessing another user's thread).

    Attributes:
        message: Human-readable error message
        code: Always "FORBIDDEN"
        status_code: Always 403 (Forbidden)
        context: Optional dictionary with contextual fields for error response

    """

    def __init__(
        self, message: str = "Access forbidden", context: dict | None = None
    ) -> None:
        """Initialize a forbidden error.

        Args:
            message: Human-readable error message
            context: Optional dictionary with contextual fields for error response

        """
        super().__init__(
            message, code="FORBIDDEN", status_code=403, context=context or {}
        )


class ConflictError(APIError):
    """Conflict (duplicate or incompatible state) errors.

    Raised when a request conflicts with current server state (e.g., starting
    a run with a run_id that already has an active run).

    Attributes:
        message: Human-readable error message
        code: Always "CONFLICT"
        status_code: Always 409 (Conflict)
        context: Optional dictionary with contextual fields for error response

    """

    def __init__(
        self, message: str = "Resource conflict", context: dict | None = None
    ) -> None:
        """Initialize a conflict error.

        Args:
            message: Human-readable error message
            context: Optional dictionary with contextual fields for error response

        """
        super().__init__(
            message, code="CONFLICT", status_code=409, context=context or {}
        )
