"""Strategy Pattern Event Handlers for AG-UI Event Processor.

Implements the strategy pattern for handling different AG-UI event types in the
event processor, replacing complex nested conditionals with a clean handler-based approach.

ARCHITECTURE NOTE:
This module is part of Stage 2 of the event processing pipeline:
- Stage 1 (strands_event_strategy.py): Converts raw Strands SDK events → AG-UI events
- Stage 2 (this module): Processes AG-UI events for persistence/activity monitoring

This module operates on already-converted AG-UI protocol events (Pydantic models)
and handles persistence, activity tracking, and message state management.
It uses synchronous handlers that return state updates, and allows multiple handlers
to process the same event (e.g., both message tracking and activity tracking).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from ag_ui.core import EventType

from server.types import ActivityMonitorProtocol, AGUIEvent, PersistenceProtocol
from server.utils import is_event_type, safe_persistence_operation
from utils.logging_helpers import get_logger, log_error_event, log_warning_event

logger = get_logger(__name__)


class AGUIEventContext:
    """Context object passed to AG-UI event handlers containing all necessary state."""

    def __init__(
        self,
        event: AGUIEvent,
        run_id: str,
        thread_id: str,
        current_message_id: str | None,
        current_message_content: list[str],
        persistence: PersistenceProtocol | None,
        activity_monitor: ActivityMonitorProtocol | None,
    ) -> None:
        """Initialize AG-UI event context.

        Args:
            event: AG-UI event object
            run_id: Run identifier
            thread_id: Thread identifier
            current_message_id: Current message ID being tracked
            current_message_content: Current message content chunks
            persistence: Optional persistence implementation (PersistenceProtocol)
            activity_monitor: Optional activity monitor implementation (ActivityMonitorProtocol)

        """
        self.event = event
        self.run_id = run_id
        self.thread_id = thread_id
        self.current_message_id = current_message_id
        self.current_message_content = current_message_content
        self.persistence = persistence
        self.activity_monitor = activity_monitor


def _log_missing_tool_call_id_error(
    context: AGUIEventContext,
    event_type: str,
    event_name: str,
) -> None:
    """Log error for missing tool_call_id in tool call events.

    Helper function to ensure consistent error handling when tool_call_id is missing
    from tool call events. Used by both ToolCallStartActivityHandler and
    ToolCallEndActivityHandler.

    Args:
        context: AGUIEventContext containing event and state
        event_type: Event type string (e.g., "TOOL_CALL_START", "TOOL_CALL_END")
        event_name: Event name for logging (e.g., "ag_ui.missing_tool_call_id_start")
    """
    log_error_event(
        logger,
        f"Missing tool_call_id in {event_type} event - "
        f"skipping activity tracking: run_id={context.run_id}, thread_id={context.thread_id}. "
        f"This indicates a bug in event generation.",
        event_name,
        exc_info=False,
        run_id=context.run_id,
        thread_id=context.thread_id,
        event_type=event_type,
    )


def _is_tool_call_error(event: AGUIEvent) -> bool:
    """Check if a tool call event indicates an error.

    An error is indicated by either:
    1. The `is_error` attribute being True, OR
    2. The `error` attribute being present and non-empty (not None and not empty string)

    Args:
        event: Tool call event to check

    Returns:
        True if the event indicates an error, False otherwise
    """
    # Check explicit error flag
    is_error_flag = getattr(event, "is_error", False)
    if is_error_flag:
        return True

    # Check error message/attribute
    error_value = getattr(event, "error", None)
    # Treat both None and empty string as "no error"
    if error_value is not None and error_value != "":
        return True

    return False


def _get_tool_call_error_message(event: AGUIEvent) -> str | None:
    """Extract error message from a tool call event.

    Prefers the `error` attribute if present and non-empty, otherwise falls back
    to the `message` attribute. Returns None if neither is available.

    Args:
        event: Tool call event to extract error message from

    Returns:
        Error message string, or None if no error message is available
    """
    error_value = getattr(event, "error", None)
    if error_value:
        return error_value

    # Fall back to message attribute if error is not available
    message = getattr(event, "message", None)
    return message


class AGUIEventHandler(ABC):
    """Base class for AG-UI event handlers using the strategy pattern."""

    @abstractmethod
    def can_handle(self, event: AGUIEvent) -> bool:
        """Check if this handler can process the given event.

        Args:
            event: AG-UI event object

        Returns:
            True if this handler can process the event, False otherwise

        """
        raise NotImplementedError

    @abstractmethod
    def handle(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Handle the event and update context state.

        Args:
            context: AGUIEventContext containing event and state

        Returns:
            Tuple of (updated message_id, updated message_content)

        """
        raise NotImplementedError


class TextMessageStartHandler(AGUIEventHandler):
    """Handler for TEXT_MESSAGE_START events."""

    def can_handle(self, event: AGUIEvent) -> bool:
        """Check if event is TEXT_MESSAGE_START."""
        return is_event_type(event, EventType.TEXT_MESSAGE_START)

    def handle(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Handle text message start - initialize message tracking."""
        context.current_message_id = getattr(context.event, "message_id", None) or str(
            uuid.uuid4()
        )
        context.current_message_content = []
        return context.current_message_id, context.current_message_content


class TextMessageContentHandler(AGUIEventHandler):
    """Handler for TEXT_MESSAGE_CONTENT events."""

    def can_handle(self, event: AGUIEvent) -> bool:
        """Check if event is TEXT_MESSAGE_CONTENT."""
        return is_event_type(event, EventType.TEXT_MESSAGE_CONTENT)

    def handle(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Handle text message content - append delta to content."""
        # Use explicit None check to avoid falling back when delta exists but is falsy (e.g., empty string)
        delta = getattr(context.event, "delta", None)
        if delta is None:
            delta = getattr(context.event, "content", "")
        if delta:
            context.current_message_content.append(delta)
        return context.current_message_id, context.current_message_content


class TextMessageEndHandler(AGUIEventHandler):
    """Handler for TEXT_MESSAGE_END events."""

    def can_handle(self, event: AGUIEvent) -> bool:
        """Check if event is TEXT_MESSAGE_END."""
        return is_event_type(event, EventType.TEXT_MESSAGE_END)

    def handle(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Handle text message end - save message to persistence."""
        if not context.persistence:
            log_warning_event(
                logger,
                "TEXT_MESSAGE_END received but persistence is not enabled - message will not be saved.",
                "ag_ui.persistence_disabled_message_not_saved",
                thread_id=context.thread_id,
                run_id=context.run_id,
            )
        if context.persistence:
            # Get message_id from context or event; do not generate - skip save if missing
            message_id = context.current_message_id or getattr(
                context.event, "message_id", None
            )
            if not message_id:
                # No message_id from context or event - skip persistence to avoid
                # saving with an arbitrary generated id (test expectation: no save)
                context.current_message_id = None
                context.current_message_content = []
                return None, []

            # Get message content from accumulated context content
            # Check if list exists and has items (not just truthy check, since [] is falsy)
            if (
                context.current_message_content is not None
                and len(context.current_message_content) > 0
            ):
                message_content = "".join(context.current_message_content)
            else:
                # If no content in context, try to get from event (though TEXT_MESSAGE_END typically doesn't have content)
                message_content = getattr(context.event, "message", "") or ""

            try:
                safe_persistence_operation(
                    "save_message",
                    context.persistence.save_message,
                    message_id=message_id,
                    thread_id=context.thread_id,
                    role="assistant",
                    content=message_content,  # Save even if empty
                    run_id=context.run_id,
                )
            except Exception as e:
                log_error_event(
                    logger,
                    "Failed to save assistant message.",
                    "ag_ui.save_message_failed",
                    error=e,
                    thread_id=context.thread_id,
                    run_id=context.run_id,
                    message_id=message_id,
                    content_length=len(message_content),
                    error_type=type(e).__name__,
                )
        context.current_message_id = None
        context.current_message_content = []
        return None, []


class ToolCallStartActivityHandler(AGUIEventHandler):
    """Handler for tracking TOOL_CALL_START events in activity monitor."""

    def can_handle(self, event: AGUIEvent) -> bool:
        """Check if event is TOOL_CALL_START."""
        return is_event_type(event, EventType.TOOL_CALL_START)

    def handle(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Handle tool call start - track in activity monitor."""
        if not context.activity_monitor:
            return context.current_message_id, context.current_message_content

        tool_call_id = getattr(context.event, "tool_call_id", None)
        if not tool_call_id:
            _log_missing_tool_call_id_error(
                context, "TOOL_CALL_START", "ag_ui.missing_tool_call_id_start"
            )
        else:
            tool_name = getattr(context.event, "tool_call_name", "unknown")
            arguments = getattr(context.event, "arguments", None)
            context.activity_monitor.track_tool_call_start(
                tool_call_id=tool_call_id, tool_name=tool_name, arguments=arguments
            )

        return context.current_message_id, context.current_message_content


class ToolCallEndActivityHandler(AGUIEventHandler):
    """Handler for tracking TOOL_CALL_END events in activity monitor."""

    def can_handle(self, event: AGUIEvent) -> bool:
        """Check if event is TOOL_CALL_END."""
        return is_event_type(event, EventType.TOOL_CALL_END)

    def handle(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Handle tool call end - track in activity monitor."""
        if not context.activity_monitor:
            return context.current_message_id, context.current_message_content

        tool_call_id = getattr(context.event, "tool_call_id", None)
        if not tool_call_id:
            _log_missing_tool_call_id_error(
                context, "TOOL_CALL_END", "ag_ui.missing_tool_call_id_end"
            )
        else:
            # Check if there's an error and extract error message
            is_error = _is_tool_call_error(context.event)
            error_message = _get_tool_call_error_message(context.event)
            context.activity_monitor.track_tool_call_end(
                tool_call_id=tool_call_id, success=not is_error, error=error_message
            )

        return context.current_message_id, context.current_message_content


class AGUIEventHandlerChain:
    """Chain of event handlers that processes events using the strategy pattern.

    Handlers are checked in order, and all matching handlers process the event.
    This allows multiple handlers to process the same event (e.g., both message
    tracking and activity tracking).
    """

    def __init__(self, handlers: list[AGUIEventHandler]) -> None:
        """Initialize handler chain.

        Args:
            handlers: List of event handlers in priority order

        """
        self.handlers = handlers

    def process_event(self, context: AGUIEventContext) -> tuple[str | None, list[str]]:
        """Process an event using the handler chain.

        All matching handlers process the event, allowing multiple handlers
        to handle the same event type (e.g., message tracking + activity tracking).

        Args:
            context: AGUIEventContext containing event and state

        Returns:
            Tuple of (updated message_id, updated message_content)

        """
        for handler in self.handlers:
            if handler.can_handle(context.event):
                context.current_message_id, context.current_message_content = (
                    handler.handle(context)
                )

        return context.current_message_id, context.current_message_content


def create_agui_event_handler_chain(
    persistence: PersistenceProtocol | None,
    activity_monitor: ActivityMonitorProtocol | None,
) -> AGUIEventHandlerChain:
    """Create a handler chain with default handlers.

    Args:
        persistence: Optional persistence implementation (PersistenceProtocol)
        activity_monitor: Optional activity monitor implementation (ActivityMonitorProtocol)

    Returns:
        AGUIEventHandlerChain configured with appropriate handlers

    """
    handlers: list[AGUIEventHandler] = []

    # Add message tracking handlers if persistence is enabled
    if persistence:
        handlers.extend(
            [
                TextMessageStartHandler(),
                TextMessageContentHandler(),
                TextMessageEndHandler(),
            ]
        )

    # Add activity tracking handlers if activity monitor is enabled
    if activity_monitor:
        handlers.extend(
            [
                ToolCallStartActivityHandler(),
                ToolCallEndActivityHandler(),
            ]
        )

    return AGUIEventHandlerChain(handlers)
