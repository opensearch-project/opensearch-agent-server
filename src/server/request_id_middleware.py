# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Request ID middleware for AG-UI server.

Assigns a unique request_id to each HTTP request, stores it in request.state and
in a context variable for log correlation, and sets the X-Request-Id response
header so clients can correlate requests with server logs.

**Behavior:**
- Generates a UUID v4 for each request
- Sets request_id_contextvar so RequestIdFilter can inject it into all logs
  during the request (including in agents, tools, and agent code)
- Sets request.state.request_id for use in route handlers
- Adds X-Request-Id to the response for client-side correlation
- Resets the context variable in a finally block so it does not leak across requests

**When request_id is not set (e.g. "-" in logs):**
- Non-HTTP entry points: Chainlit, tests, scripts, lifespan, asyncio exception handler
- RequestIdFilter uses "-" when the context variable is unset
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from server.logging_config import request_id_contextvar


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a request_id to each HTTP request.

    Sets the context variable for logging and adds X-Request-Id to the response.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[..., Response]
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        token = request_id_contextvar.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            request_id_contextvar.reset(token)
