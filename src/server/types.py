"""Type definitions for AG-UI server.

This module provides Protocol classes and type aliases for better type safety
throughout the codebase.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

# Import AG-UI event types for type aliases
try:
    from ag_ui.core import (
        CustomEvent,
        MessagesSnapshotEvent,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        StateSnapshotEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
    )

    # Type alias for all AG-UI events (PEP 604 union)
    AGUIEvent = (
        TextMessageStartEvent
        | TextMessageContentEvent
        | TextMessageEndEvent
        | ToolCallStartEvent
        | ToolCallArgsEvent
        | ToolCallResultEvent
        | ToolCallEndEvent
        | MessagesSnapshotEvent
        | StateSnapshotEvent
        | CustomEvent
        | RunStartedEvent
        | RunFinishedEvent
        | RunErrorEvent
    )
except ImportError:
    # Fallback if ag_ui.core is not available
    AGUIEvent = dict[str, Any] | Any  # type: ignore[misc]


class PersistenceProtocol(Protocol):
    """Protocol defining the interface for AG-UI persistence services.

    This protocol allows any object that implements these methods to be used
    as a persistence layer, enabling better type checking and flexibility.
    """

    def save_thread(
        self,
        thread_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save or update a thread.

        Args:
            thread_id: Unique identifier for the thread
            user_id: Optional user identifier who owns the thread
            metadata: Optional dictionary with additional thread metadata

        """
        ...

    def save_run_start(
        self, run_id: str, thread_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Save a run start event.

        Args:
            run_id: Unique identifier for the run
            thread_id: Thread identifier this run belongs to
            metadata: Optional dictionary with additional run metadata

        """
        ...

    def save_run_finish(
        self,
        run_id: str,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """Save a run finish event.

        Args:
            run_id: Unique identifier for the run
            status: Run status (e.g., "completed", "failed")
            error_message: Optional error message if run failed

        """
        ...

    def save_message(
        self,
        message_id: str,
        thread_id: str,
        role: str,
        content: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a message.

        Args:
            message_id: Unique identifier for the message
            thread_id: Thread identifier this message belongs to
            role: Message role (e.g., "user", "assistant", "tool")
            content: Message content text
            run_id: Optional run identifier this message belongs to
            metadata: Optional dictionary with additional message metadata

        """
        ...

    def save_event(
        self, event_id: str, run_id: str, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Save an AG-UI event.

        Args:
            event_id: Unique identifier for the event
            run_id: Run identifier this event belongs to
            event_type: Event type (e.g., "TEXT_MESSAGE_START", "TOOL_CALL_END")
            event_data: dict[str, Any]ionary containing event data

        """
        ...

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get a thread by ID.

        Args:
            thread_id: Thread identifier to retrieve

        Returns:
            Thread dictionary if found, None otherwise

        """
        ...

    def get_threads(
        self, user_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get threads, optionally filtered by user_id.

        Args:
            user_id: Optional user identifier to filter threads
            limit: Maximum number of threads to return
            offset: Offset for pagination

        Returns:
            List of thread dictionaries

        """
        ...

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a run by ID.

        Args:
            run_id: Run identifier to retrieve

        Returns:
            Run dictionary if found, None otherwise

        """
        ...

    def get_run_with_ownership_check(
        self, run_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Get a run by ID if the user owns its thread. Single query when possible.

        This method combines run retrieval with ownership verification in a single
        database query, eliminating the N+1 query pattern where we would first fetch
        the run, then fetch the thread to check ownership.

        Args:
            run_id: Run identifier to retrieve
            user_id: User identifier to verify ownership

        Returns:
            Run dictionary if found and user owns the thread, None otherwise

        """
        ...

    def get_runs(
        self, thread_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get runs for a thread.

        Args:
            thread_id: Thread identifier to get runs for
            limit: Maximum number of runs to return
            offset: Offset for pagination

        Returns:
            List of run dictionaries

        """
        ...

    def get_messages(
        self,
        thread_id: str,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get messages for a thread or run.

        Args:
            thread_id: Thread identifier to get messages for
            run_id: Optional run identifier to filter messages
            limit: Maximum number of messages to return
            offset: Offset for pagination

        Returns:
            List of message dictionaries

        """
        ...

    def get_events(
        self,
        run_id: str,
        event_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get events for a run.

        Args:
            run_id: Run identifier to get events for
            event_type: Optional event type to filter by
            limit: Maximum number of events to return
            offset: Offset for pagination

        Returns:
            List of event dictionaries

        """
        ...

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread and all associated data.

        Args:
            thread_id: Thread identifier to delete

        """
        ...


class EventEncoderProtocol(Protocol):
    """Protocol defining the interface for AG-UI event encoders.

    This protocol allows any object that implements these methods to be used
    as an event encoder, enabling better type checking and flexibility.
    """

    def encode(self, event: AGUIEvent | dict[str, Any]) -> str:
        r"""Encode an AG-UI event to SSE format.

        Args:
            event: AG-UI event object (Pydantic model or dict)

        Returns:
            SSE-formatted string (data: {...}\n\n)

        """
        ...

    def get_content_type(self) -> str:
        """Get the content type for SSE responses.

        Returns:
            Content type string (typically "text/event-stream")

        """
        ...


class ActivityMonitorProtocol(Protocol):
    """Protocol defining the interface for AG-UI activity monitors.

    This protocol allows any object that implements these methods to be used
    as an activity monitor, enabling better type checking and flexibility.
    """

    def track_tool_call_start(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any] | Any | None = None,
    ) -> None:
        """Track the start of a tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
            arguments: Optional dictionary with tool call arguments

        """
        ...

    def track_tool_call_end(
        self, tool_call_id: str, success: bool = True, error: str | None = None
    ) -> None:
        """Track the end of a tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            success: Whether the tool call succeeded
            error: Optional error message if the tool call failed

        """
        ...

    def get_remaining_tool_calls(self) -> list[str]:
        """Get list of tool call IDs that are still active.

        Returns:
            List of tool_call_id strings for active tool calls

        """
        ...

    def complete_remaining_tool_calls(
        self, error: str = "Run completed before tool call finished"
    ) -> None:
        """Complete all remaining active tool calls as failed.

        Args:
            error: Error message to associate with the completed tool calls

        """
        ...

    def log_summary(self) -> None:
        """Log the activity summary.

        Logs a summary of all tool calls tracked during the run, including
        success/failure counts and timing information.
        """
        ...


# API Response TypedDicts for better type safety


class RunResponse(TypedDict, total=False):
    """Response type for GET /runs/{run_id} endpoint.

    Attributes:
        id: Run identifier
        thread_id: Thread identifier this run belongs to
        created_at: ISO format datetime string when run was created
        finished_at: ISO format datetime string when run finished (None if still running)
        status: Run status (e.g., "completed", "running", "failed")
        error_message: Error message if run failed (None if successful)
        metadata: Optional dictionary with additional run metadata

    """

    id: str
    thread_id: str
    created_at: str | None  # ISO format datetime string
    finished_at: str | None  # ISO format datetime string
    status: str
    error_message: str | None
    metadata: dict[str, Any | None]


class ThreadsResponse(TypedDict):
    """Response type for GET /threads endpoint.

    Attributes:
        threads: List of thread dictionaries
        count: Number of threads returned (may be less than total if paginated)

    """

    threads: list[dict[str, Any]]
    count: int


class ThreadResponse(TypedDict, total=False):
    """Response type for GET /threads/{thread_id} endpoint.

    Attributes:
        id: Thread identifier
        user_id: User identifier who owns this thread (None if not specified)
        created_at: ISO format datetime string when thread was created
        updated_at: ISO format datetime string when thread was last updated
        metadata: Optional dictionary with additional thread metadata

    """

    id: str
    user_id: str | None
    created_at: str | None  # ISO format datetime string
    updated_at: str | None  # ISO format datetime string
    metadata: dict[str, Any | None]


class ThreadRunsResponse(TypedDict):
    """Response type for GET /threads/{thread_id}/runs endpoint.

    Attributes:
        threadId: Thread identifier
        runs: List of run dictionaries for this thread
        count: Number of runs returned (may be less than total if paginated)

    """

    threadId: str
    runs: list[dict[str, Any]]
    count: int


class ThreadMessagesResponse(TypedDict):
    """Response type for GET /threads/{thread_id}/messages endpoint.

    Attributes:
        threadId: Thread identifier
        runId: Optional run identifier if messages are filtered by run
        messages: List of message dictionaries
        count: Number of messages returned (may be less than total if paginated)

    """

    threadId: str
    runId: str | None
    messages: list[dict[str, Any]]
    count: int


class RunEventsResponse(TypedDict):
    """Response type for GET /runs/{run_id}/events endpoint.

    Attributes:
        runId: Run identifier
        eventType: Optional event type filter if events are filtered by type
        events: List of event dictionaries
        count: Number of events returned (may be less than total if paginated)

    """

    runId: str
    eventType: str | None
    events: list[dict[str, Any]]
    count: int


class CancelRunResponse(TypedDict):
    """Response type for POST /runs/{run_id}/cancel endpoint.

    Attributes:
        runId: Run identifier
        canceled: Whether the cancellation was successful
        message: Human-readable message about the cancellation status

    """

    runId: str
    canceled: bool
    message: str
