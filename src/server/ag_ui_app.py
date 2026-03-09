"""AG-UI Protocol FastAPI Server.

Exposes the multi-agent system via AG-UI protocol for frontend integration.
create_app() builds the FastAPI app; the lifespan context manager handles
startup (config validation, persistence, rate limiting, AgentOrchestrator)
and shutdown (orchestrator cleanup).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from server.constants import (
    DEFAULT_EVENT_LIMIT,
)
from server.exceptions import APIError
from server.types import (
    CancelRunResponse,
    RunEventsResponse,
    RunResponse,
)
from server.validators import ValidatedRunAgentInput
from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_info_event,
    log_warning_event,
)

# Configure logging (will be configured by ag_ui_server.py or can be configured here)
# If not already configured, use default human-readable format
if not logging.root.handlers:
    from server.logging_config import (  # noqa: E402  # conditional import for config
        configure_logging,
        get_logging_config,
    )

    use_json, log_level = get_logging_config()
    configure_logging(use_json=use_json, log_level=log_level)

logger = get_logger(__name__)

# Imports below run after logging config so that:
# - Phoenix and route modules see configured logging if we configured it here.
# - No need to reorder if new modules depend on config or logging.
from server.agent_orchestrator import AgentOrchestrator  # noqa: E402
from server.auth_middleware import (  # noqa: E402
    AuthenticationMiddleware,
    create_auth_middleware,
)
from server.config import ServerConfig, get_config  # noqa: E402
from server.rate_limiting import (  # noqa: E402
    create_rate_limiter,
    get_rate_limit_decorator,
    setup_rate_limiting,
)
from server.request_id_middleware import RequestIdMiddleware  # noqa: E402
from server.run_routes import (  # noqa: E402
    cancel_run_route,
    create_run_route,
    get_run_events_route,
    get_run_route,
)

# Set by lifespan at startup; used by routes at request time.
persistence: Any | None = None
orchestrator: AgentOrchestrator | None = None


def _suppress_mcp_cancel_scope_error(
    loop: asyncio.AbstractEventLoop, context: dict[str, Any]
) -> None:
    """Suppress RuntimeError about cancel scopes from MCP connection cleanup.

    This error occurs when MCP stdio_client async generators are garbage collected
    in a different task than they were created in. Since we manually close streams
    in cleanup(), this error is harmless and can be safely suppressed.

    Args:
        loop: The asyncio event loop
        context: Exception context dict with 'exception' and 'message' keys
    """
    exception = context.get("exception")
    if isinstance(exception, RuntimeError):
        error_msg = str(exception).lower()
        if "cancel scope" in error_msg and "different task" in error_msg:
            # Suppress this specific error - it's harmless since streams are already closed
            log_debug_event(
                logger,
                "Suppressed MCP cancel scope error during cleanup (expected during GC).",
                "ag_ui.mcp_cancel_scope_error_suppressed",
                error=str(exception),
            )
            return

    # For all other exceptions, log them using the standard event helper
    message = context.get("message", "Unhandled exception in event loop")
    if exception:
        exc_info_tuple = (
            type(exception),
            exception,
            getattr(exception, "__traceback__", None),
        )
        log_error_event(
            logger,
            f"✗ {message}: {exception}",
            "ag_ui.asyncio_exception",
            error=exception,
            exception_type=type(exception).__name__,
            exc_info=exc_info_tuple,
        )
    else:
        log_error_event(
            logger,
            f"✗ {message}",
            "ag_ui.asyncio_exception",
            context=context,
            exc_info=False,
        )


def _register_mcp_cancel_scope_exception_handler(
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Register the MCP cancel-scope exception handler on the event loop."""
    loop.set_exception_handler(_suppress_mcp_cancel_scope_error)


def _noop_rate_limit(f: Any) -> Any:
    """Placeholder no-op so route handlers can use @rate_limit before the app exists."""
    return f


rate_limit: Any = _noop_rate_limit


class _MaxBodySizeMiddleware:
    """ASGI middleware that returns 413 when Content-Length exceeds max_bytes."""

    def __init__(self, app: Any, max_bytes: int = 0) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if self.max_bytes > 0 and scope.get("type") == "http":
            content_length = None
            for key, value in scope.get("headers") or []:
                if key.lower() == b"content-length":
                    try:
                        content_length = int(value.decode())
                    except (ValueError, TypeError, UnicodeDecodeError):
                        pass
                    break
            if content_length is not None:
                if content_length < 0:
                    body = json.dumps(
                        {"detail": "Invalid Content-Length: must be non-negative."}
                    ).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 400,
                            "headers": [[b"content-type", b"application/json"]],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
                if content_length > self.max_bytes:
                    body = json.dumps(
                        {
                            "detail": (
                                f"Request body too large. "
                                f"Maximum size is {self.max_bytes} bytes."
                            )
                        }
                    ).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [[b"content-type", b"application/json"]],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
        await self.app(scope, receive, send)


def create_app(config_override: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_override: Optional ServerConfig. When None, uses get_config().

    Returns:
        Configured FastAPI application instance.
    """
    config_resolved = config_override if config_override is not None else get_config()
    rate_limiter = create_rate_limiter(config_resolved)
    rate_limit_deco = get_rate_limit_decorator(rate_limiter, config=config_resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        """FastAPI lifespan: startup then yield, then shutdown."""
        loop = asyncio.get_running_loop()
        _register_mcp_cancel_scope_exception_handler(loop)

        from server.config import validate_config_on_startup

        validate_config_on_startup(config_resolved)

        global persistence, orchestrator
        persistence = None
        if config_resolved.enable_persistence:
            try:
                from utils.persistence import AGUIPersistence

                persistence = AGUIPersistence(db_path=config_resolved.db_path)
                log_info_event(
                    logger,
                    "✓ AG-UI data persistence enabled",
                    "ag_ui.persistence_enabled",
                    enabled=True,
                )
            except Exception as e:
                log_warning_event(
                    logger,
                    f"✗ Failed to initialize AG-UI persistence: {e}. "
                    "Continuing without persistence.",
                    "ag_ui.persistence_initialization_failed",
                    exc_info=True,
                    error=str(e),
                    enabled=False,
                )
                persistence = None
        else:
            log_info_event(
                logger,
                "AG-UI data persistence disabled (set AG_UI_ENABLE_PERSISTENCE=true to enable)",
                "ag_ui.persistence_disabled",
                enabled=False,
            )

        setup_rate_limiting(app, rate_limiter)

        # --- Orchestrator setup: create agents eagerly ---
        from agents.art_agent import create_art_agent
        from agents.fallback_agent import create_fallback_agent
        from orchestrator.registry import AgentRegistration, AgentRegistry
        from orchestrator.router import PageContextRouter

        registry = AgentRegistry()
        opensearch_url = config_resolved.opensearch_url

        # Register ART agent (search relevance page)
        registry.register(AgentRegistration(
            name="art",
            description="Search Relevance Testing agent (ART) — hypothesis generation, "
            "evaluation, UBI analysis, and online A/B testing",
            page_contexts=["search-relevance", "searchRelevance"],
            is_fallback=False,
        ))

        # Register fallback agent (handles all unmatched page contexts)
        registry.register(AgentRegistration(
            name="fallback",
            description="General OpenSearch assistant with MCP tools",
            page_contexts=[],
            is_fallback=True,
        ))

        log_info_event(
            logger,
            f"Registered {len(registry.list_agents())} agent(s): "
            + ", ".join(a.name for a in registry.list_agents()),
            "ag_ui.agents_registered",
            agent_count=len(registry.list_agents()),
        )

        # Store registry on app for the /agents endpoint
        app.state.registry = registry

        router = PageContextRouter(registry)

        # Create orchestrator and eagerly initialize agents
        orchestrator = AgentOrchestrator(router)

        # Create fallback agent eagerly
        try:
            fallback_agent = await asyncio.to_thread(
                create_fallback_agent, opensearch_url
            )
            orchestrator.register_agent(
                name="fallback",
                strands_agent=fallback_agent,
                description="General OpenSearch assistant with MCP tools",
            )
            log_info_event(
                logger,
                "✓ Fallback agent created and registered",
                "ag_ui.fallback_agent_ready",
            )
        except Exception as e:
            log_error_event(
                logger,
                f"✗ Failed to create fallback agent: {e}",
                "ag_ui.fallback_agent_creation_failed",
                error=str(e),
                exc_info=True,
            )

        # Create ART agent eagerly
        try:
            art_agent = await asyncio.to_thread(
                create_art_agent, opensearch_url
            )
            orchestrator.register_agent(
                name="art",
                strands_agent=art_agent,
                description="Search Relevance Testing agent (ART)",
            )
            log_info_event(
                logger,
                "✓ ART agent created and registered",
                "ag_ui.art_agent_ready",
            )
        except Exception as e:
            log_warning_event(
                logger,
                f"✗ ART agent not available (os_art not installed?): {e}",
                "ag_ui.art_agent_creation_failed",
                error=str(e),
                exc_info=True,
            )

        yield

    app = FastAPI(
        title="OpenSearch Agent Server for OpenSearch Dashboards",
        description=(
            "Multi-agent orchestrator for OpenSearch Dashboards. Routes requests "
            "by page context to specialized sub-agents via the AG-UI protocol (SSE)."
        ),
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "runs", "description": "Agent run management endpoints."},
            {"name": "agents", "description": "Agent discovery and registry."},
            {
                "name": "health",
                "description": "Health check and system status endpoints.",
            },
        ],
    )
    app.state.config = config_resolved

    cors_origins = config_resolved.get_cors_origins_list()
    if cors_origins == ["*"]:
        log_warning_event(
            logger,
            "CORS configured to allow all origins (*). This is insecure and should only be used for development. "
            "For production, specify exact origins in AG_UI_CORS_ORIGINS environment variable.",
            "ag_ui.cors_wildcard_warning",
            cors_origins_env=config_resolved.cors_origins or "",
        )
    elif not cors_origins:
        log_info_event(
            logger,
            "CORS disabled (no AG_UI_CORS_ORIGINS set). To enable CORS, set AG_UI_CORS_ORIGINS environment variable.",
            "ag_ui.cors_disabled",
        )
    cors_methods = config_resolved.get_cors_methods_list()
    cors_headers = config_resolved.get_cors_headers_list()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=cors_methods,
            allow_headers=cors_headers,
        )
        log_info_event(
            logger,
            f"CORS enabled: origins={cors_origins}, methods={cors_methods}, headers={cors_headers}",
            "ag_ui.cors_enabled",
            origins=cors_origins,
            methods=cors_methods,
            headers=cors_headers,
        )

    auth_middleware_config = create_auth_middleware(app, config_resolved)
    if auth_middleware_config:
        app.add_middleware(
            AuthenticationMiddleware,
            enabled=auth_middleware_config["enabled"],
            mode=auth_middleware_config["mode"],
            strategies=auth_middleware_config["strategies"],
            config=auth_middleware_config["config"],
        )

    app.add_middleware(RequestIdMiddleware)

    global rate_limit
    rate_limit = rate_limit_deco

    app.add_middleware(
        _MaxBodySizeMiddleware, max_bytes=config_resolved.max_request_body_bytes
    )

    return app


app = create_app()


def get_orchestrator() -> AgentOrchestrator:
    """Dependency function to provide AgentOrchestrator instance.

    Returns:
        AgentOrchestrator instance configured for the application.

    Raises:
        RuntimeError: If called before app lifespan has run.
    """
    if orchestrator is None:
        raise RuntimeError(
            "AgentOrchestrator not initialized. Ensure app lifespan has run (e.g. use "
            "TestClient with lifespan context or start the server)."
        )
    return orchestrator


# Exception handlers for consistent error responses
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle API errors with consistent response format."""
    log_error_event(
        logger,
        f"✗ API error: code={exc.code}, message={exc.message}, path={request.url.path}",
        "ag_ui.api_error",
        exc_info=True,
        error_code=exc.code,
        error_message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
    )
    response_content = {
        "error": exc.message,
        "code": exc.code,
        "detail": exc.message,
    }
    if exc.context:
        response_content.update(exc.context)

    return JSONResponse(status_code=exc.status_code, content=response_content)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors (422) with a sanitized detail."""
    sanitized = [
        {k: v for k, v in e.items() if k in ("loc", "msg", "type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": sanitized})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException with proper status code."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with consistent error response."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    log_error_event(
        logger,
        f"✗ Unexpected error: {exc}, path={request.url.path}",
        "ag_ui.unexpected_error",
        error=str(exc),
        exc_info=True,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal server error occurred",
            "code": "INTERNAL_SERVER_ERROR",
        },
    )


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/agents", tags=["agents"])
async def list_agents(request: Request) -> dict:
    """List registered agents and their page contexts."""
    registry = request.app.state.registry
    return {
        "agents": [
            {
                "name": reg.name,
                "description": reg.description,
                "page_contexts": reg.page_contexts,
                "is_fallback": reg.is_fallback,
            }
            for reg in registry.list_agents()
        ]
    }


@app.post("/runs", tags=["runs"])
@rate_limit
async def create_run(
    input_data: ValidatedRunAgentInput,
    request: Request,
    orch: AgentOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    """Start a new agent run and stream AG-UI events via SSE."""
    return create_run_route(
        orchestrator=orch,
        persistence=persistence,
        input_data=input_data,
        request=request,
    )


@app.get("/runs/{run_id}", tags=["runs"])
async def get_run(run_id: str, request: Request) -> RunResponse:
    """Get run details including status and metadata."""
    return get_run_route(persistence=persistence, run_id=run_id, request=request)


@app.get("/runs/{run_id}/events", tags=["runs"])
async def get_run_events(
    run_id: str,
    request: Request,
    event_type: str | None = None,
    limit: int = DEFAULT_EVENT_LIMIT,
    offset: int = 0,
) -> RunEventsResponse:
    """Get events for a run, optionally filtered by event_type."""
    return get_run_events_route(
        persistence=persistence,
        run_id=run_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
        request=request,
    )


@app.post("/runs/{run_id}/cancel", tags=["runs"])
async def cancel_run(run_id: str, request: Request) -> CancelRunResponse:
    """Cancel a running agent."""
    return await cancel_run_route(
        persistence=persistence, run_id=run_id, request=request
    )


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(app, host=config.server_host, port=config.server_port)
