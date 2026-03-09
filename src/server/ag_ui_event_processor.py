"""Event Processor for AG-UI Protocol.

Handles event generation, persistence, and activity monitoring for AG-UI events.
The ``ag_ui_strands.StrandsAgent`` wrapper now handles all Strands→AG-UI event
conversion; this module only concerns itself with:

* SSE encoding
* Optional persistence
* Optional activity monitoring
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from ag_ui.core import RunAgentInput

from server.ag_ui_event_strategy import (
    AGUIEventContext,
    create_agui_event_handler_chain,
)
from server.agent_orchestrator import AgentOrchestrator
from server.config import ServerConfig
from server.types import (
    ActivityMonitorProtocol,
    AGUIEvent,
    EventEncoderProtocol,
    PersistenceProtocol,
)
from server.utils import (
    create_error_event,
    get_event_type_from_object,
    get_event_type_name,
    safe_persistence_operation,
)
from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)

logger = get_logger(__name__)


class AGUIEventProcessor:
    """Processes AG-UI events for persistence, activity monitoring, and encoding."""

    def __init__(
        self,
        encoder: EventEncoderProtocol,
        persistence: PersistenceProtocol | None = None,
        activity_monitor: ActivityMonitorProtocol | None = None,
    ) -> None:
        self.persistence = persistence
        self.activity_monitor = activity_monitor
        self.encoder = encoder
        self._handler_chain = create_agui_event_handler_chain(
            persistence=persistence,
            activity_monitor=activity_monitor,
        )

    def process_event(
        self,
        event: AGUIEvent,
        run_id: str,
        thread_id: str,
        current_message_id: str | None,
        current_message_content: list,
    ) -> tuple[str | None, list, str]:
        """Process a single event for persistence and activity monitoring.

        Returns:
            Tuple of (updated message_id, updated message_content, encoded_event_string)
        """
        if self.persistence:
            self._save_event_to_persistence(event, run_id, thread_id)

        context = AGUIEventContext(
            event=event,
            run_id=run_id,
            thread_id=thread_id,
            current_message_id=current_message_id,
            current_message_content=current_message_content,
            persistence=self.persistence,
            activity_monitor=self.activity_monitor,
        )
        updated_message_id, updated_message_content = self._handler_chain.process_event(
            context
        )

        try:
            encoded_event = self.encoder.encode(event)
        except Exception as e:
            log_error_event(
                logger,
                "Failed to encode event.",
                "ag_ui.encoding_error",
                error=str(e),
                exc_info=True,
                run_id=run_id,
                thread_id=thread_id,
            )
            try:
                error_event = create_error_event(
                    message=f"Encoding error: {str(e)}",
                    code="ENCODING_ERROR",
                )
                encoded_event = self.encoder.encode(error_event)
            except Exception as fallback_error:
                log_error_event(
                    logger,
                    "Failed to encode error event.",
                    "ag_ui.encoding_error_fallback_failed",
                    error=str(fallback_error),
                    exc_info=True,
                    run_id=run_id,
                    thread_id=thread_id,
                    original_error=str(e),
                )
                encoded_event = f"data: {json.dumps({'error': f'Encoding failed: {str(e)}', 'code': 'ENCODING_ERROR'})}\n\n"

        return updated_message_id, updated_message_content, encoded_event

    def _save_event_to_persistence(
        self, event: AGUIEvent, run_id: str, thread_id: str
    ) -> None:
        event_id = str(uuid.uuid4())
        event_type = get_event_type_from_object(event)
        event_type_str = (
            get_event_type_name(event_type) if event_type is not None else "UNKNOWN"
        )

        if hasattr(event, "model_dump"):
            event_data = event.model_dump(exclude_none=True)
        elif hasattr(event, "dict"):
            event_data = event.dict(exclude_none=True)
        else:
            event_data = {"type": event_type_str}

        safe_persistence_operation(
            "save_event",
            self.persistence.save_event,
            event_id=event_id,
            run_id=run_id,
            event_type=event_type_str,
            event_data=event_data,
        )


async def _process_event_stream(
    orchestrator: AgentOrchestrator,
    input_data: RunAgentInput,
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Process the AG-UI event stream from the orchestrator.

    The ``ag_ui_strands.StrandsAgent`` inside the orchestrator handles all
    Strands→AG-UI conversion.  This function simply iterates the AG-UI events
    and passes each one through ``event_processor.process_event()`` for
    persistence, activity monitoring, and SSE encoding.
    """
    current_message_id = None
    current_message_content: list = []

    async for event in orchestrator.run(input_data):
        current_message_id, current_message_content, encoded_event = (
            event_processor.process_event(
                event,
                run_id,
                thread_id,
                current_message_id,
                current_message_content,
            )
        )
        yield encoded_event


def _handle_run_error(
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    user_id: str,
    error: Exception,
) -> str:
    """Handle run error by emitting error event."""
    log_error_event(
        logger,
        "Run error.",
        "ag_ui.run_error",
        error=str(error),
        exc_info=True,
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
    )
    try:
        error_event = create_error_event(
            message=str(error),
            code="RUN_ERROR",
        )
        return event_processor.encoder.encode(error_event)
    except Exception as encoding_error:
        log_error_event(
            logger,
            "Failed to encode run error event.",
            "ag_ui.run_error_encoding_failed",
            error=str(encoding_error),
            exc_info=True,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            original_error=str(error),
        )
        return f"data: {json.dumps({'error': str(error), 'code': 'RUN_ERROR'})}\n\n"


def _complete_run(
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    user_id: str,
    event_count: int,
    start_time: datetime,
) -> None:
    """Complete run by handling cleanup, persistence, and logging."""
    if event_processor.activity_monitor:
        remaining_tool_calls = (
            event_processor.activity_monitor.get_remaining_tool_calls()
        )
        if remaining_tool_calls:
            log_warning_event(
                logger,
                f"Completing remaining active tool calls at run end: "
                f"run_id={run_id}, thread_id={thread_id}, count={len(remaining_tool_calls)}",
                "ag_ui.completing_remaining_tool_calls",
                run_id=run_id,
                thread_id=thread_id,
                count=len(remaining_tool_calls),
            )
            event_processor.activity_monitor.complete_remaining_tool_calls(
                error="Run completed before tool call finished"
            )

    if event_processor.persistence:
        safe_persistence_operation(
            "save_run_finish",
            event_processor.persistence.save_run_finish,
            run_id=run_id,
            status="completed",
        )

    duration = (datetime.now() - start_time).total_seconds()
    log_info_event(
        logger,
        f"Completed AG-UI run: run_id={run_id}, thread_id={thread_id}, "
        f"user_id={user_id}, events={event_count}, duration={duration:.2f}s",
        "ag_ui.run_completed",
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        event_count=event_count,
        duration_seconds=duration,
    )

    if event_processor.activity_monitor:
        event_processor.activity_monitor.log_summary()


async def generate_events(
    orchestrator: AgentOrchestrator,
    input_data: RunAgentInput,
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    user_id: str,
    start_time: datetime,
    config: ServerConfig | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events from the orchestrator with processing.

    Args:
        orchestrator: AgentOrchestrator instance.
        input_data: RunAgentInput with thread_id, run_id, and messages.
        event_processor: AGUIEventProcessor instance.
        run_id: Run identifier.
        thread_id: Thread identifier.
        user_id: User identifier.
        start_time: Start time for duration calculation.
        config: Optional ServerConfig (unused, kept for API compatibility).

    Yields:
        Encoded SSE event strings.
    """
    event_count = 0

    try:
        async for encoded_event in _process_event_stream(
            orchestrator,
            input_data,
            event_processor,
            run_id,
            thread_id,
        ):
            event_count += 1
            yield encoded_event
    except Exception as e:
        error_event_str = _handle_run_error(
            event_processor, run_id, thread_id, user_id, e
        )
        yield error_event_str
    finally:
        _complete_run(
            event_processor, run_id, thread_id, user_id, event_count, start_time
        )
