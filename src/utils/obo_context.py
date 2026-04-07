"""Per-request OBO token injection for httpx clients.

Provides concurrency-safe token propagation that works across async contexts
AND background threads (as used by the Strands MCP client).

The Strands SDK's MCPClient executes tool calls on a background thread via
``asyncio.run_coroutine_threadsafe()``.  Python's ``ContextVar`` does NOT
propagate across threads, so a pure-ContextVar approach would lose the token.

Instead we use a **thread-safe dict** keyed by ``OboAuth`` instance identity.
Each agent's httpx client gets its own ``OboAuth`` instance at creation time.
Before each run, the orchestrator calls ``auth_instance.set_token(jwt)`` which
stores the token in a ``threading.Lock``-protected dict.  When httpx fires a
request — even on the MCP background thread — ``OboAuth.async_auth_flow()``
reads from the same dict.

Concurrent users are safe because each request's ``set_token()`` call is
atomic and the agent run is sequential per-agent (the orchestrator awaits the
full event stream before starting the next run for the same agent).

Usage::

    # At agent creation time:
    auth = OboAuth()
    http_client = httpx.AsyncClient(auth=auth)

    # Before each agent run (in the request handler):
    auth.set_token("eyJhbG...")

    # Every outgoing httpx request — including those on the MCP background
    # thread — automatically gets the correct token.
"""

from __future__ import annotations

import threading
from typing import Generator

import httpx

from utils.logging_helpers import get_logger, log_debug_event

logger = get_logger(__name__)


class OboAuth(httpx.Auth):
    """httpx Auth that injects an OBO token into every outgoing request.

    Thread-safe: the token is stored behind a lock so it can be read from
    the MCP client's background thread and written from the main async
    context.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None

    def set_token(self, token: str | None) -> None:
        """Set the OBO token.  Called by the orchestrator before each run."""
        with self._lock:
            self._token = token
        log_debug_event(
            logger,
            f"OBO token set (present={token is not None})",
            "obo_auth.token_set",
        )

    def get_token(self) -> str | None:
        """Get the current OBO token (thread-safe)."""
        with self._lock:
            return self._token

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        token = self.get_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):  # type: ignore[override]
        token = self.get_token()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request
