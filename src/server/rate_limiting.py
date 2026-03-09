"""Rate limiting middleware for AG-UI FastAPI server.

This module provides rate limiting functionality to protect the API from
abuse and ensure fair resource usage across all clients.

Rate limiting is configurable via environment variables:
- AG_UI_RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
- AG_UI_RATE_LIMIT_PER_MINUTE: Requests per minute per client (default: 60)
- AG_UI_RATE_LIMIT_PER_HOUR: Requests per hour per client (default: 1000)

The rate limiter uses the client's IP address or user ID (from X-User-Id header)
as the key for rate limiting. If both are available, user ID takes precedence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import FastAPI, Request
from slowapi import (  # type: ignore[import-untyped]
    Limiter,
    _rate_limit_exceeded_handler,
)
from slowapi.errors import RateLimitExceeded  # type: ignore[import-untyped]
from slowapi.util import get_remote_address  # type: ignore[import-untyped]

from server.config import ServerConfig, get_config
from server.utils import get_user_id_from_request
from utils.logging_helpers import get_logger, log_info_event

F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)


def get_rate_limit_key(request: Request) -> str:
    """Get the key to use for rate limiting.

    Prefers user ID from X-User-Id header if available, otherwise falls back
    to client IP address. This allows per-user rate limiting when authentication
    is enabled, and per-IP rate limiting otherwise.

    Args:
        request: FastAPI request object

    Returns:
        String key for rate limiting (user_id or IP address)
    """
    # Try to get user ID from request (checks X-User-Id header, Authorization header, or client host)
    user_id = get_user_id_from_request(request)
    if user_id and user_id != get_remote_address(request):
        # Use user ID if it's different from IP (i.e., actual user ID, not fallback)
        return f"user:{user_id}"
    # Fall back to IP address
    return get_remote_address(request)


# Initialize rate limiter (must be after get_rate_limit_key is defined)
limiter = Limiter(key_func=get_rate_limit_key)


def create_rate_limiter(config: ServerConfig | None = None) -> Limiter | None:
    """Create and configure rate limiter based on configuration.

    Args:
        config: Optional ServerConfig instance. When None, uses get_config().
                Callers can inject config for testing without reset_config.

    Returns:
        Configured Limiter instance if rate limiting is enabled, None otherwise
    """
    cfg = config if config is not None else get_config()

    if not cfg.rate_limit_enabled:
        log_info_event(
            logger,
            "Rate limiting disabled (set AG_UI_RATE_LIMIT_ENABLED=true to enable)",
            "ag_ui.rate_limiting_disabled",
            enabled=False,
        )
        return None

    # Get rate limit configuration
    per_minute = cfg.rate_limit_per_minute
    per_hour = cfg.rate_limit_per_hour

    log_info_event(
        logger,
        f"Rate limiting enabled: {per_minute} requests/minute, {per_hour} requests/hour",
        "ag_ui.rate_limiting_enabled",
        enabled=True,
        per_minute=per_minute,
        per_hour=per_hour,
    )

    return limiter


def setup_rate_limiting(app: FastAPI, limiter_instance: Limiter | None) -> None:
    """Set up rate limiting middleware and exception handler.

    Args:
        app: FastAPI application instance
        limiter_instance: Limiter instance (or None if rate limiting is disabled)
    """
    if limiter_instance is None:
        return

    # Attach limiter to app
    app.state.limiter = limiter_instance
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    log_info_event(
        logger,
        "Rate limiting middleware configured",
        "ag_ui.rate_limiting_middleware_configured",
        enabled=True,
    )


def get_rate_limit_decorator(
    limiter_instance: Limiter | None,
    per_minute: int | None = None,
    per_hour: int | None = None,
    config: ServerConfig | None = None,
) -> Callable[[F], F]:
    """Get rate limit decorator for use on route handlers.

    Args:
        limiter_instance: Limiter instance (or None if rate limiting is disabled)
        per_minute: Optional override for requests per minute
        per_hour: Optional override for requests per hour
        config: Optional ServerConfig for defaults when per_minute/per_hour not set.
                When None, uses get_config(). Callers can inject for testing.

    Returns:
        Decorator function (no-op if rate limiting is disabled)
    """
    if limiter_instance is None:
        # Return no-op decorator if rate limiting is disabled
        def noop_decorator(func: F) -> F:
            return func

        return noop_decorator

    # Build rate limit string
    rate_limit_str = ""
    if per_minute:
        rate_limit_str = f"{per_minute}/minute"
    if per_hour:
        if rate_limit_str:
            rate_limit_str += f", {per_hour}/hour"
        else:
            rate_limit_str = f"{per_hour}/hour"

    if not rate_limit_str:
        # Use defaults from configuration
        cfg = config if config is not None else get_config()
        per_minute = cfg.rate_limit_per_minute
        per_hour = cfg.rate_limit_per_hour
        rate_limit_str = f"{per_minute}/minute, {per_hour}/hour"

    return limiter_instance.limit(rate_limit_str)
