"""Standardized authorization helpers for AG-UI route handlers.

This module provides consistent authorization patterns across all routes,
ensuring uniform access control checks and reducing maintenance burden.

**Authorization Pattern:**
- All authorization checks follow the same pattern:
  1. Check if auth is enabled (if disabled, skip authorization)
  2. If auth enabled, require request and authentication (already verified by require_authenticated_if_auth_enabled)
  3. If auth enabled, require persistence (can't verify ownership without it)
  4. Verify ownership of the resource
  5. Raise ForbiddenError if ownership check fails

**Usage Example:**
```python
from server.authorization import require_thread_ownership, require_run_ownership

# In a route handler (after require_authenticated_if_auth_enabled)
def get_thread_route(persistence, thread_id, request):
    require_authenticated_if_auth_enabled(request)  # Enforces authentication
    require_thread_ownership(persistence, thread_id, request)  # Enforces authorization
    # Ownership verified - safe to proceed
    thread = persistence.get_thread(thread_id)
    return thread

# For run routes
def get_run_route(persistence, run_id, request):
    require_authenticated_if_auth_enabled(request)  # Enforces authentication
    require_run_ownership(persistence, run_id, request)  # Enforces authorization
    # Ownership verified - safe to proceed
    run = persistence.get_run(run_id)
    return run
```

**Decorator Usage:**
```python
from server.authorization import require_ownership

@require_ownership("thread", "thread_id")
def get_thread_route(persistence, thread_id, request, _cached_thread=None):
    # Ownership already checked by decorator
    # Use cached thread to avoid duplicate query
    if _cached_thread is not None:
        return _cached_thread
    # Fallback if cache not available (e.g., auth disabled)
    return persistence.get_thread(thread_id)

@require_ownership("run", "run_id")
def get_run_route(persistence, run_id, request, _cached_run=None):
    # Ownership already checked by decorator
    # Use cached run to avoid duplicate query
    if _cached_run is not None:
        return _cached_run
    # Fallback if cache not available (e.g., auth disabled)
    return persistence.get_run(run_id)
```

**Performance Note:**
The decorator caches the fetched resource and passes it via `_cached_{resource_type}`
to eliminate duplicate database queries. Always check for the cached resource first
before fetching from persistence.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from server.config import ServerConfig

from fastapi import Request

from server.exceptions import (
    ForbiddenError,
    NotFoundError,
    PersistenceNotEnabledError,
    UnauthorizedError,
)
from server.types import PersistenceProtocol
from server.utils import get_user_id_from_request, is_authenticated, log_security_event
from utils.logging_helpers import get_logger

logger = get_logger(__name__)

# Type variables for decorator type hints
# P: ParamSpec capturing the parameter signature of the function being decorated
# R: TypeVar representing the return type of the function being decorated
# These allow the decorator to preserve the original function's type signature
P = ParamSpec("P")
R = TypeVar("R")


def _get_config_from_request(request: Request | None) -> ServerConfig:
    """Get config from request.app.state or fallback to get_config().

    This helper function standardizes config retrieval across authorization
    functions. It prefers the config from request.app.state (set by create_app)
    to allow tests to inject config without reset_config(), but falls back to
    get_config() if request is None or doesn't have app.state.config.

    Args:
        request: Optional FastAPI Request object

    Returns:
        ServerConfig instance
    """
    from server.config import get_config

    app = getattr(request, "app", None) if request else None
    if app is not None:
        config = getattr(app.state, "config", None)
        if config is not None:
            return config
    return get_config()


def _require_auth_request_and_persistence(
    request: Request | None,
    persistence: PersistenceProtocol | None,
    context: dict | None,
    operation: str,
    resource_context_key: str,
    resource_context_value: str,
) -> bool:
    """Require auth enabled, request, authentication, and persistence; otherwise raise or return False.

    Shared pre-check for ownership functions. When auth is disabled, returns False so the
    caller can skip authorization. When auth is enabled, raises if request is missing,
    user is not authenticated, or persistence is missing; otherwise returns True.

    Args:
        request: Optional FastAPI request (required when auth enabled).
        persistence: Optional persistence (required when auth enabled).
        context: Optional dict to merge into error context.
        operation: Description for PersistenceNotEnabledError (e.g. "thread ownership check").
        resource_context_key: Key to set in error context (e.g. "threadId", "runId").
        resource_context_value: Value for that key (e.g. thread_id, run_id).

    Returns:
        False if auth is disabled; True if auth enabled and all pre-checks passed.

    Raises:
        ForbiddenError: Auth enabled but request is None.
        UnauthorizedError: Auth enabled but user not authenticated.
        PersistenceNotEnabledError: Auth enabled but persistence is None.
    """
    config = _get_config_from_request(request)
    if not config.auth_enabled:
        return False

    if not request:
        raise ForbiddenError(
            "Request required for authorization when auth is enabled",
            context=context or {},
        )
    if not is_authenticated(request):
        raise UnauthorizedError(
            "Authentication required for authorization",
            context=context or {},
        )
    if not persistence:
        error_context = dict(context or {})
        error_context[resource_context_key] = resource_context_value
        raise PersistenceNotEnabledError(
            operation=operation,
            context=error_context,
        )
    return True


def require_thread_ownership(
    persistence: PersistenceProtocol | None,
    thread_id: str,
    request: Request | None,
    context: dict | None = None,
) -> dict | None:
    """Require that the authenticated user owns the specified thread.

    This function standardizes thread ownership checks across all routes.
    It verifies that:
    1. Auth is enabled (if disabled, authorization is skipped)
    2. If auth enabled, request is provided and user is authenticated
    3. If auth enabled, persistence is enabled (required for ownership verification)
    4. The thread exists and belongs to the authenticated user

    Args:
        persistence: Optional persistence instance (must be enabled when auth is enabled)
        thread_id: Thread identifier to check ownership for
        request: Optional FastAPI request object (for user authentication)
        context: Optional dictionary to include in error context

    Returns:
        The fetched thread dictionary if ownership check passes; None if auth is disabled
        or if the thread does not exist (no ownership to enforce). This allows the caller
        to reuse the fetched resource and avoid duplicate queries.

    Raises:
        ForbiddenError: If the user is authenticated, persistence is enabled,
            and the user doesn't own the thread.
        PersistenceNotEnabledError: If auth is enabled but persistence is not enabled
            (can't verify ownership without persistence).

    Note:
        This function only enforces authorization when auth is enabled. When auth
        is disabled, authorization is skipped to allow development/testing scenarios.
        However, when auth is enabled, all requirements (request, authentication,
        persistence) must be met.
    """
    if not _require_auth_request_and_persistence(
        request, persistence, context, "thread ownership check", "threadId", thread_id
    ):
        return None

    # Get user ID and verify ownership
    user_id = get_user_id_from_request(request)
    thread = persistence.get_thread(thread_id)

    if thread and thread.get("user_id") != user_id:
        # Log security event before raising exception
        log_security_event(
            logger,
            "access_denied",
            request=request,
            user_id=user_id,
            resource_type="thread",
            resource_id=thread_id,
            owner_id=thread.get("user_id"),
            reason="User does not own resource",
        )
        error_context = dict(context or {})
        error_context["threadId"] = thread_id
        raise ForbiddenError(
            f"Access denied to thread {thread_id}",
            context=error_context,
        )

    # Return the fetched thread for reuse by the caller
    return thread


def require_run_ownership(
    persistence: PersistenceProtocol | None,
    run_id: str,
    request: Request | None,
    context: dict | None = None,
) -> dict | None:
    """Require that the authenticated user owns the thread for the specified run.

    This function standardizes run ownership checks across all routes.
    It verifies that:
    1. Auth is enabled (if disabled, authorization is skipped)
    2. If auth enabled, request is provided and user is authenticated
    3. If auth enabled, persistence is enabled (required for ownership verification)
    4. The run exists (raises NotFoundError if not found)
    5. The run belongs to a thread owned by the authenticated user (raises ForbiddenError if not owned)

    **Order of Operations:**
    This function checks existence BEFORE ownership to ensure proper error semantics:
    - Non-existent runs → 404 NotFoundError
    - Existent but unauthorized runs → 403 ForbiddenError

    Args:
        persistence: Optional persistence instance (must be enabled when auth is enabled)
        run_id: Run identifier to check ownership for
        request: Optional FastAPI request object (for user authentication)
        context: Optional dictionary to include in error context

    Returns:
        The fetched run dictionary if ownership check passes, None if auth is disabled.
        This allows the caller to reuse the fetched resource and avoid duplicate queries.

    Raises:
        NotFoundError: If the run does not exist (checked before ownership).
        ForbiddenError: If the user is authenticated, persistence is enabled,
            and the user doesn't own the run's thread.
        PersistenceNotEnabledError: If auth is enabled but persistence is not enabled
            (can't verify ownership without persistence).

    Note:
        This function only enforces authorization when auth is enabled. When auth
        is disabled, authorization is skipped to allow development/testing scenarios.
        However, when auth is enabled, all requirements (request, authentication,
        persistence) must be met.

        **Performance Optimization:**
        The function uses `get_run_with_ownership_check` which combines run retrieval
        with ownership verification in a single database query (join). This eliminates
        the N+1 query pattern in the common case. Error paths may still require
        additional queries to distinguish between 404 (not found) and 403 (forbidden)
        errors, but the happy path is optimized to a single query.
    """
    if not _require_auth_request_and_persistence(
        request, persistence, context, "run ownership check", "runId", run_id
    ):
        return None

    # Get user ID and verify ownership via thread
    user_id = get_user_id_from_request(request)

    # Use optimized single-query method that combines run retrieval with ownership check
    # This eliminates the N+1 query pattern (get_run + get_thread) in the common case
    run = persistence.get_run_with_ownership_check(run_id, user_id)

    # If we got a run, ownership is verified - return it (happy path: 1 query)
    if run:
        return run

    # get_run_with_ownership_check returned None, which could mean:
    # 1. Run doesn't exist → 404 NotFoundError
    # 2. Run exists but user doesn't own it → 403 ForbiddenError
    # 3. Run exists but has no thread_id → Allow access (preserve old behavior)
    # We need to distinguish between these cases for proper error semantics
    # Check if run exists (without ownership check) - this is the error path, so 2 queries is acceptable
    existing_run = persistence.get_run(run_id)

    if not existing_run:
        # Run doesn't exist
        error_context = dict(context or {})
        error_context["runId"] = run_id
        raise NotFoundError("Run", run_id, context=error_context)

    # Check if run has no thread_id - allow access (preserve old behavior)
    # This handles edge cases where runs might not have a thread_id
    thread_id = existing_run.get("thread_id") or existing_run.get("threadId")
    if not thread_id:
        # No thread_id means no ownership check needed - return the run
        return existing_run

    # Run exists with thread_id - get thread and verify ownership
    thread = persistence.get_thread(thread_id)
    if thread and thread.get("user_id") == user_id:
        # User owns the thread - allow access
        return existing_run

    # User doesn't own the thread - log and raise ForbiddenError
    log_security_event(
        logger,
        "access_denied",
        request=request,
        user_id=user_id,
        resource_type="run",
        resource_id=run_id,
        owner_id=thread.get("user_id") if thread else None,
        reason="User does not own resource",
    )
    error_context = dict(context or {})
    error_context["runId"] = run_id
    if thread_id:
        error_context["threadId"] = thread_id
    raise ForbiddenError(
        f"Access denied to run {run_id}",
        context=error_context,
    )


def require_ownership(
    resource_type: str,
    id_param: str | None = None,
) -> Callable[P, R]:
    """Decorator to require resource ownership.

    This decorator automatically performs ownership checks before the route handler
    executes, reducing boilerplate and ensuring consistent authorization patterns.

    **Performance Optimization:**
    The decorator caches the fetched resource (thread or run) and passes it to the
    route handler via a `_cached_{resource_type}` keyword argument. This eliminates
    duplicate database queries when the route handler needs to access the same resource.

    Args:
        resource_type: Type of resource ("thread" or "run")
        id_param: Name of parameter containing resource ID (default: f"{resource_type}_id")

    Returns:
        Decorator function that wraps the route handler

    Example:
        ```python
        @require_ownership("run", "run_id")
        def get_run_route(persistence, run_id, request, _cached_run=None):
            # Ownership already checked by decorator
            # Use cached run to avoid duplicate query
            if _cached_run is not None:
                return _cached_run
            # Fallback if cache not available (e.g., auth disabled)
            return persistence.get_run(run_id)
        ```

    Note:
        The decorator extracts `request` and `persistence` from the function arguments
        (either positional or keyword). It also extracts the resource ID using the
        `id_param` parameter. The decorator works with both positional and keyword
        arguments, making it compatible with various route handler signatures.

        The decorator calls the appropriate ownership function (`require_thread_ownership`
        or `require_run_ownership`) which handles all the authorization logic including
        checking if auth is enabled, verifying authentication, and checking ownership.

        The cached resource is only available when auth is enabled and the ownership
        check was performed. When auth is disabled, the cached resource will be None
        and the route handler should fetch the resource normally.
    """
    # Default id_param if not provided
    if id_param is None:
        id_param = f"{resource_type}_id"

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map positional args to parameter names
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            # Extract request and persistence from args/kwargs
            request = None
            persistence = None
            resource_id = None

            # First, try to get from kwargs
            request = kwargs.get("request")
            persistence = kwargs.get("persistence")
            resource_id = kwargs.get(id_param)

            # If not in kwargs, try to get from positional args using parameter names
            if request is None or persistence is None or resource_id is None:
                for i, arg in enumerate(args):
                    param_name = param_names[i] if i < len(param_names) else None
                    if param_name == "request" and request is None:
                        request = arg
                    elif param_name == "persistence" and persistence is None:
                        persistence = arg
                    elif param_name == id_param and resource_id is None:
                        resource_id = arg

            # If we have the required parameters, perform ownership check
            if (
                request is not None
                and persistence is not None
                and resource_id is not None
            ):
                # Build context for error reporting (new dict, not mutating caller data)
                context = {f"{resource_type}Id": resource_id}

                # Call the appropriate ownership function and cache the fetched resource
                cached_resource = None
                if resource_type == "thread":
                    cached_resource = require_thread_ownership(
                        persistence=persistence,
                        thread_id=resource_id,
                        request=request,
                        context=context,
                    )
                elif resource_type == "run":
                    cached_resource = require_run_ownership(
                        persistence=persistence,
                        run_id=resource_id,
                        request=request,
                        context=context,
                    )
                # If resource_type is not recognized, skip the check
                # (allows for future extensibility)

                # Pass the cached resource to the function via kwargs to avoid duplicate queries
                # Use a special parameter name that won't conflict with normal function parameters
                # Only pass if the function signature accepts it (to avoid TypeError)
                cache_param_name = f"_cached_{resource_type}"
                if cached_resource is not None and cache_param_name in param_names:
                    kwargs[cache_param_name] = cached_resource

            # Call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
