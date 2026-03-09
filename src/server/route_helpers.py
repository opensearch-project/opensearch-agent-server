"""Helper functions for AG-UI route handlers.

This module provides utility functions used across route handlers,
including encoder creation and message saving.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ag_ui.core import RunAgentInput

from server.constants import ROLE_USER
from server.types import AGUIEvent, EventEncoderProtocol, PersistenceProtocol
from server.utils import safe_persistence_operation
from utils.logging_helpers import get_logger, log_debug_event, log_warning_event

logger = get_logger(__name__)

# Try to import EventEncoder first (preferred for encoding)
try:
    from ag_ui.encoder import EventEncoder

    HAS_EVENT_ENCODER = True
except ImportError:
    HAS_EVENT_ENCODER = False
    EventEncoder = None  # type: ignore[assignment]


def create_encoder(accept_header: str) -> EventEncoderProtocol:
    """Create an EventEncoder instance, with fallback if not available.

    Args:
        accept_header: Accept header from request
    Returns:
        EventEncoder instance (or fallback SimpleEncoder) conforming to EventEncoderProtocol

    """
    # Check if EventEncoder is available (imported at module level)
    if HAS_EVENT_ENCODER:
        return EventEncoder(accept=accept_header)

    # Fallback encoder only if EventEncoder is not available
    log_warning_event(
        logger,
        "EventEncoder not available, using fallback encoder. "
        "Install 'ag-ui' package for full encoding format support.",
        "ag_ui.event_encoder_unavailable",
    )

    # Nested class definition - class defined inside function
    # This keeps SimpleEncoder local to this function
    class SimpleEncoder:
        """Fallback encoder for AG-UI events when EventEncoder is not available."""

        def encode(self, event: AGUIEvent | dict[str, Any]) -> str:  # type: ignore[override]
            r"""Encode an AG-UI event to SSE format.

            Args:
                event: AG-UI event object (Pydantic model or dict)


            Returns:
                SSE-formatted string (data: {...}\n\n)

            """
            # Check if event has Pydantic model methods
            if hasattr(event, "model_dump_json"):
                # Pydantic v2 method - converts model to JSON string
                return f"data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
            elif hasattr(event, "model_dump"):
                # Pydantic v2 method - converts model to dict, then JSON string
                # Use by_alias=True to convert snake_case field names to camelCase (AG-UI protocol standard)
                return f"data: {json.dumps(event.model_dump(by_alias=True, exclude_none=True))}\n\n"
            else:
                # Fallback: treat as plain dict
                return f"data: {json.dumps(event)}\n\n"

        def get_content_type(self) -> str:
            """Get the content type for SSE responses.

            Returns:
                Content type string ("text/event-stream")

            """
            return "text/event-stream"

    return SimpleEncoder()


def save_initial_messages(
    persistence: PersistenceProtocol | None,
    input_data: RunAgentInput,
    thread_id: str,
    run_id: str,
) -> None:
    """Save initial user messages to persistence.

    Only saves messages that don't already exist in persistence to avoid duplicates.
    This prevents re-saving the full conversation history when a new message is sent.

    Args:
        persistence: AGUIPersistence instance or None
        input_data: RunAgentInput with messages
        thread_id: Thread identifier
        run_id: Run identifier

    """
    if (
        not persistence
        or not hasattr(input_data, "messages")
        or not input_data.messages
    ):
        return

    # Get existing messages from persistence to check for duplicates
    # Only check last 100 messages for duplicates (most recent are most likely to be duplicates)
    # This limits memory usage for threads with thousands of messages
    existing_messages = []
    try:
        existing_messages = (
            persistence.get_messages(thread_id=thread_id, limit=100) or []
        )
    except Exception as e:
        # If we can't retrieve existing messages, log and continue
        # We'll save all messages to be safe (better than missing messages)
        log_warning_event(
            logger,
            "Failed to retrieve existing messages to check for duplicates.",
            "ag_ui.message_duplicate_check_failed",
            thread_id=thread_id,
            error=str(e),
        )

    # Create a set of existing message contents for fast lookup
    # We compare by role and content to identify duplicates
    existing_content_set = {
        (msg.get("role", ""), msg.get("content", "")) for msg in existing_messages
    }

    # Only save messages that don't already exist
    for msg in input_data.messages:
        if isinstance(msg, dict):
            role = msg.get("role", ROLE_USER)
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", ROLE_USER)
            # Use explicit None check to avoid falling back when content exists but is falsy (e.g., empty string)
            content = getattr(msg, "content", None)
            if content is None:
                content = getattr(msg, "text", "")

        if role == ROLE_USER and content:
            # Check if this message already exists
            message_key = (role, content)
            if message_key not in existing_content_set:
                message_id = str(uuid.uuid4())
                safe_persistence_operation(
                    "save_message",
                    persistence.save_message,
                    message_id=message_id,
                    thread_id=thread_id,
                    role=role,
                    content=content,
                    run_id=run_id,
                )
                # Add to set to avoid saving duplicates within the same batch
                existing_content_set.add(message_key)


def generate_thread_title_from_message(
    message_content: str, max_length: int = 60
) -> str:
    """Generate a human-readable thread title from a message.

    Extracts a title from the first user message by:
    - Truncating to max_length characters
    - Stripping whitespace
    - Removing newlines and extra spaces
    - Adding ellipsis if truncated

    Args:
        message_content: The message content to generate title from
        max_length: Maximum length for the title (default: 60)

    Returns:
        A human-readable title string
    """
    if not message_content:
        return "New Chat"

    # Handle different content types
    if isinstance(message_content, list):
        # Extract text from content list (AG-UI format)
        text_parts = []
        for item in message_content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
                elif "type" in item and item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        content = " ".join(text_parts) if text_parts else str(message_content)
    elif isinstance(message_content, dict):
        # Extract text from dict
        content = str(
            message_content.get(
                "text", message_content.get("content", str(message_content))
            )
        )
    else:
        content = str(message_content)

    # Clean up the content
    # Remove newlines and extra whitespace
    title = " ".join(content.split())

    # Truncate to max_length (result length at most max_length)
    if len(title) > max_length:
        # Try to truncate at a word boundary
        truncated = title[:max_length].rsplit(" ", 1)[0]
        if (
            len(truncated) < max_length * 0.7
        ):  # If truncation removed too much, use simple truncation
            truncated = title[: max_length - 3]
        # Cap truncated length so result never exceeds max_length
        truncated = truncated[: max_length - 3]
        title = truncated + "..."

    return title.strip() or "New Chat"


def ensure_thread_has_title(
    persistence: PersistenceProtocol | None,
    thread_id: str,
    input_data: RunAgentInput,
) -> None:
    """Ensure a thread has a human-readable title, generating one from the first user message if needed.

    Checks if the thread already has a title in its metadata. If not, generates
    a title from the first user message and updates the thread metadata.

    Args:
        persistence: AGUIPersistence instance or None
        thread_id: Thread identifier
        input_data: RunAgentInput with messages to extract title from
    """
    if not persistence:
        return

    try:
        # Check if thread exists and has a title
        thread = persistence.get_thread(thread_id)
        if thread:
            metadata = thread.get("metadata")
            if metadata:
                # Parse metadata if it's a JSON string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                # Check if title already exists
                if isinstance(metadata, dict) and metadata.get("title"):
                    return  # Thread already has a title, no need to generate one
            elif metadata is None:
                # Thread exists but has no metadata - we'll generate title below
                pass

        # Find first user message to generate title from
        first_user_message = None
        if hasattr(input_data, "messages") and input_data.messages:
            for msg in input_data.messages:
                if isinstance(msg, dict):
                    role = msg.get("role", ROLE_USER)
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "role", ROLE_USER)
                    content = getattr(msg, "content", None)
                    if content is None:
                        content = getattr(msg, "text", "")

                if role == ROLE_USER and content:
                    first_user_message = content
                    break

        # Generate title from first user message
        if first_user_message:
            title = generate_thread_title_from_message(first_user_message)
            # Update thread metadata with title
            existing_metadata = {}
            if thread:
                thread_metadata = thread.get("metadata")
                if thread_metadata:
                    if isinstance(thread_metadata, str):
                        try:
                            existing_metadata = json.loads(thread_metadata)
                        except (json.JSONDecodeError, TypeError):
                            existing_metadata = {}
                    elif isinstance(thread_metadata, dict):
                        existing_metadata = thread_metadata

            # Merge title into existing metadata
            updated_metadata = {**existing_metadata, "title": title}
            safe_persistence_operation(
                "save_thread",
                persistence.save_thread,
                thread_id=thread_id,
                metadata=updated_metadata,
            )
            log_debug_event(
                logger,
                "Generated thread title.",
                "ag_ui.thread_title_generated",
                thread_id=thread_id,
                title=title,
            )
    except Exception as e:
        # Log but don't fail - title generation is a nice-to-have feature
        log_warning_event(
            logger,
            "Failed to generate thread title.",
            "ag_ui.thread_title_generation_error",
            thread_id=thread_id,
            error=str(e),
        )
