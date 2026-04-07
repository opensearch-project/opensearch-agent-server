"""Per-request OBO token injection via contextvars.

Provides concurrency-safe token propagation: each async task carries its
own token through Python's ContextVar, and the httpx Auth class reads it
at request time.  No shared mutable state.

Usage::

    # In the request handler (once per incoming request):
    set_obo_token("eyJhbG...")

    # The shared httpx.AsyncClient is created with ``auth=OboAuth()``.
    # Every outgoing HTTP request automatically gets the correct token
    # for the current async context — concurrent users never interfere.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Generator

import httpx

from utils.logging_helpers import get_logger, log_debug_event

logger = get_logger(__name__)

# Each async task (i.e. each Dashboards request) sets its own value.
# Concurrent requests never see each other's tokens.
_current_obo_token: ContextVar[str | None] = ContextVar(
    "obo_token", default=None
)


def set_obo_token(token: str | None) -> None:
    """Set the OBO token for the current async context."""
    _current_obo_token.set(token)
    log_debug_event(
        logger,
        f"OBO token set for current context (present={token is not None})",
        "obo_context.token_set",
    )


def get_obo_token() -> str | None:
    """Get the OBO token for the current async context."""
    return _current_obo_token.get()


class OboAuth(httpx.Auth):
    """httpx Auth that injects the OBO token from the current async context.

    Attached to the shared httpx.AsyncClient at creation time.  On every
    outgoing HTTP request, it reads the ContextVar — so each concurrent
    agent run automatically gets its own user's token.
    """

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        token = _current_obo_token.get()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):  # type: ignore[override]
        token = _current_obo_token.get()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request
