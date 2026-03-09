"""Route handlers for run-related AG-UI endpoints.

This module contains route handlers for creating runs, getting run details,
canceling runs, and retrieving run events.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import Request
from fastapi.responses import StreamingResponse

from server.ag_ui_event_processor import AGUIEventProcessor, generate_events
from server.agent_orchestrator import AgentOrchestrator
from server.authorization import require_ownership
from server.config import get_config
from server.constants import DEFAULT_EVENT_LIMIT
from server.error_recovery import (
    create_fallback_events_response,
    create_fallback_run_response,
    handle_read_operation_with_fallback,
)
from server.exceptions import ConflictError, NotFoundError
from server.route_helpers import (
    create_encoder,
    ensure_thread_has_title,
    save_initial_messages,
)
from server.run_manager import get_run_manager
from server.run_route_helpers import (
    consume_event_generator_with_cancellation,
    create_event_queue,
    yield_events_from_queue,
)
from server.types import (
    CancelRunResponse,
    PersistenceProtocol,
    RunEventsResponse,
    RunResponse,
)
from server.utils import (
    get_user_id_from_request,
    require_authenticated_if_auth_enabled,
    safe_persistence_operation,
)
from server.validators import ValidatedRunAgentInput
from utils.activity_monitor import AGUIActivityMonitor
from utils.logging_helpers import get_logger, log_info_event, log_warning_event

logger = get_logger(__name__)


def create_run_route(
    orchestrator: AgentOrchestrator,
    persistence: PersistenceProtocol | None,
    input_data: ValidatedRunAgentInput,
    request: Request,
) -> StreamingResponse:
    """Start a new agent run and stream AG-UI events via SSE.

    Args:
        orchestrator: AgentOrchestrator instance.
        persistence: Optional AGUIPersistence instance.
        input_data: ValidatedRunAgentInput with thread_id, run_id, and messages.
        request: FastAPI request object (for Accept header).

    Returns:
        SSE stream of AG-UI events.

    Raises:
        UnauthorizedError: If authentication is required and the request is not authenticated.
        ConflictError: If a run with this run_id is already in progress (409).
    """
    require_authenticated_if_auth_enabled(request)
    accept_header = request.headers.get("accept", "text/event-stream")
    encoder = create_encoder(accept_header)

    thread_id = input_data.thread_id
    run_id = input_data.run_id

    run_agent_input = input_data.to_run_agent_input()

    user_id = get_user_id_from_request(request)

    if persistence:
        existing_run = persistence.get_run(run_id)
        if existing_run and existing_run.get("status") == "running":
            log_warning_event(
                logger,
                f"Rejected duplicate run: run_id={run_id} already has an active run",
                "ag_ui.duplicate_run_rejected",
                run_id=run_id,
                thread_id=thread_id,
            )
            raise ConflictError(
                f"A run with run_id {run_id} is already in progress. "
                "Use a unique run_id or wait for the current run to finish.",
                context={"runId": run_id, "threadId": thread_id},
            )

    if persistence:
        safe_persistence_operation(
            "save_thread", persistence.save_thread, thread_id=thread_id, user_id=user_id
        )
        safe_persistence_operation(
            "save_run_start",
            persistence.save_run_start,
            run_id=run_id,
            thread_id=thread_id,
        )
        save_initial_messages(persistence, run_agent_input, thread_id, run_id)
        ensure_thread_has_title(persistence, thread_id, run_agent_input)

    message_count = len(input_data.messages)
    log_info_event(
        logger,
        f"Starting AG-UI run: run_id={run_id}, thread_id={thread_id}, "
        f"user_id={user_id}, message_count={message_count}",
        "ag_ui.run_starting",
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        message_count=message_count,
    )
    start_time = datetime.now()

    config = getattr(request.app.state, "config", None) or get_config()

    activity_monitor = AGUIActivityMonitor(run_id=run_id, thread_id=thread_id)

    event_processor = AGUIEventProcessor(
        encoder=encoder,
        persistence=persistence,
        activity_monitor=activity_monitor,
    )

    async def cancellable_event_stream() -> AsyncIterator[str]:
        """Event stream wrapper that checks for cancellation."""
        run_manager = get_run_manager()
        event_queue = create_event_queue()
        generator_error: Exception | None = None

        event_generator = generate_events(
            orchestrator=orchestrator,
            input_data=run_agent_input,
            event_processor=event_processor,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            start_time=start_time,
            config=config,
        )

        async def consume_event_generator() -> None:
            nonlocal generator_error
            generator_error = await consume_event_generator_with_cancellation(
                event_generator,
                run_id,
                thread_id,
                encoder,
                event_queue,
            )

        task = asyncio.create_task(consume_event_generator())
        await run_manager.register_run(run_id, task)

        try:
            async for event in yield_events_from_queue(
                event_queue, task, generator_error, run_id, thread_id
            ):
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await run_manager.unregister_run(run_id)

    return StreamingResponse(
        cancellable_event_stream(),
        media_type=encoder.get_content_type(),
    )


@require_ownership("run", "run_id")
def get_run_route(
    persistence: PersistenceProtocol | None,
    run_id: str,
    request: Request | None = None,
    _cached_run: dict | None = None,
) -> RunResponse:
    """Get run details including status and metadata."""
    require_authenticated_if_auth_enabled(request)
    if not persistence:
        return create_fallback_run_response(run_id)

    if _cached_run is not None:
        return _cached_run

    def fallback() -> RunResponse:
        return create_fallback_run_response(run_id)

    run = handle_read_operation_with_fallback(
        operation_name="run",
        operation_func=persistence.get_run,
        fallback_func=fallback,
        error_event_name="ag_ui.run_retrieval_error",
        error_context={"run_id": run_id, "runId": run_id},
        run_id=run_id,
    )

    if run:
        return run
    raise NotFoundError("Run", run_id, context={"runId": run_id, "status": "unknown"})


@require_ownership("run", "run_id")
def get_run_events_route(
    persistence: PersistenceProtocol | None,
    run_id: str,
    event_type: str | None = None,
    limit: int = DEFAULT_EVENT_LIMIT,
    offset: int = 0,
    request: Request | None = None,
    _cached_run: dict | None = None,
) -> RunEventsResponse:
    """Get events for a run, optionally filtered by event_type."""
    require_authenticated_if_auth_enabled(request)
    if not persistence:
        return create_fallback_events_response(run_id, event_type)

    run = _cached_run
    if run is None:
        try:
            run = persistence.get_run(run_id)
        except Exception as e:
            log_warning_event(
                logger,
                f"Could not check run existence for events, using fallback: {e}",
                "ag_ui.run_events_run_check_error",
                run_id=run_id,
                runId=run_id,
            )
            return create_fallback_events_response(run_id, event_type)
    if run is None:
        raise NotFoundError("Run", run_id, context={"runId": run_id})

    def fallback() -> RunEventsResponse:
        return create_fallback_events_response(run_id, event_type)

    result = handle_read_operation_with_fallback(
        operation_name="events for run",
        operation_func=persistence.get_events,
        fallback_func=fallback,
        error_event_name="ag_ui.run_events_retrieval_error",
        error_context={
            "run_id": run_id,
            "runId": run_id,
            "event_type": event_type,
            "limit": limit,
            "offset": offset,
            "events": [],
        },
        run_id=run_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )
    if isinstance(result, dict):
        return result
    return {
        "runId": run_id,
        "eventType": event_type,
        "events": result,
        "count": len(result),
    }


@require_ownership("run", "run_id")
async def cancel_run_route(
    persistence: PersistenceProtocol | None,
    run_id: str,
    request: Request | None = None,
) -> CancelRunResponse:
    """Cancel a running agent."""
    require_authenticated_if_auth_enabled(request)

    run_manager = get_run_manager()

    is_active = await run_manager.is_run_active(run_id)

    if not is_active:
        was_canceled = await run_manager.is_run_canceled(run_id)
        if was_canceled:
            log_info_event(
                logger,
                f"Run was already canceled: run_id={run_id}",
                "ag_ui.run_already_canceled",
                run_id=run_id,
            )
            return {
                "runId": run_id,
                "canceled": True,
                "message": "Run was already canceled",
            }
        else:
            log_warning_event(
                logger,
                f"Run not found or already completed: run_id={run_id}",
                "ag_ui.run_not_found_for_cancel",
                run_id=run_id,
            )
            return {
                "runId": run_id,
                "canceled": False,
                "message": "Run not found or already completed",
            }

    canceled = await run_manager.cancel_run(
        run_id, reason="User requested cancellation"
    )

    if canceled:
        log_info_event(
            logger,
            f"Successfully canceled run: run_id={run_id}",
            "ag_ui.run_cancellation_success",
            run_id=run_id,
        )
        return {
            "runId": run_id,
            "canceled": True,
            "message": "Run cancellation requested successfully",
        }
    else:
        log_warning_event(
            logger,
            f"Run cancellation failed (may have completed): run_id={run_id}",
            "ag_ui.run_cancellation_failed",
            run_id=run_id,
        )
        return {
            "runId": run_id,
            "canceled": False,
            "message": "Run may have already completed",
        }
