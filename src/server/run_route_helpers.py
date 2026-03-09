"""Helper functions for create_run_route to reduce complexity.

This module extracts cancellation handling and event stream management logic
from the main route handler to improve maintainability and testability.

**Key Functions:**
- `emit_cancellation_events()` - Emit cancellation error and finish events (with retry logic)
- `check_and_handle_cancellation()` - Check cancellation status and handle if canceled
- `consume_event_generator_with_cancellation()` - Consume event generator with cancellation checks
- `yield_events_from_queue()` - Yield events from queue with timeout handling
- `create_event_queue()` - Create bounded event queue with backpressure handling
- `put_critical_event_with_retry()` - Queue critical events with exponential backoff retry

**Usage Example:**
```python
from server.run_route_helpers import (
    consume_event_generator_with_cancellation,
    yield_events_from_queue,
    create_event_queue,
)

async def cancellable_event_stream():
    event_queue = create_event_queue()
    generator_done = False
    generator_error = None

    async def consume_event_generator():
        nonlocal generator_done, generator_error
        generator_error = await consume_event_generator_with_cancellation(
            event_generator, run_id, thread_id, encoder, event_queue
        )
        generator_done = True

    task = asyncio.create_task(consume_event_generator())  # noqa: RUF100  # task ref passed to consumer; intentionally not stored
    async for event in yield_events_from_queue(
        event_queue, task, generator_error, run_id, thread_id
    ):
        yield event
```
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ag_ui.core import EventType, RunFinishedEvent

from server.config import get_config
from server.constants import (
    DEFAULT_EVENT_STREAM_CHECK_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_INITIAL_TIMEOUT,
    DEFAULT_RETRY_MAX_TIMEOUT,
)
from server.run_manager import get_run_manager
from server.types import EventEncoderProtocol
from server.utils import create_error_event
from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,  # noqa: F401 (used in event-dropped path)
)

if TYPE_CHECKING:
    from server.run_manager import RunManager

logger = get_logger(__name__)


def create_event_queue() -> asyncio.Queue[str]:
    """Create a bounded event queue with backpressure handling.

    Creates an asyncio.Queue with a maximum size to prevent unbounded memory growth.
    When the queue is full, put() operations will block until space is available,
    applying backpressure to the producer.

    Returns:
        Bounded asyncio.Queue[str] for event streaming

    Configuration:
        Queue size is controlled by AG_UI_MAX_EVENT_QUEUE_SIZE environment variable.
        Default: 1000 events.
    """
    config = get_config()
    return asyncio.Queue(maxsize=config.max_event_queue_size)


async def put_event_with_backpressure(
    event_queue: asyncio.Queue[str],
    event: str,
    run_id: str,
    thread_id: str,
    timeout: float = DEFAULT_RETRY_INITIAL_TIMEOUT,
) -> bool:
    """Put an event into the queue with backpressure handling and timeout.

    If the queue is full, this will block until space is available (backpressure).
    If the timeout is exceeded, the event is dropped (not queued) and an error is logged.

    **Behavior:**
    - If queue has space: Event is queued immediately, returns True
    - If queue is full: Blocks waiting for space (backpressure)
    - If timeout exceeded: Event is dropped, error logged, returns False

    **Caller Responsibilities:**
    - Check return value: If False, event was dropped and should be handled appropriately
    - For critical events: Use `put_critical_event_with_retry()` instead, which retries with exponential backoff
    - For non-critical events: Logging failure is sufficient; caller may continue processing

    Args:
        event_queue: Queue to put event into
        event: Encoded event string to queue
        run_id: Run identifier for logging
        thread_id: Thread identifier for logging
        timeout: Maximum time to wait for queue space (seconds).
                 Default: DEFAULT_RETRY_INITIAL_TIMEOUT (5.0 seconds)

    Returns:
        True if event was successfully queued, False if timeout exceeded (event was dropped)

    Example:
        ```python
        success = await put_event_with_backpressure(event_queue, event, run_id, thread_id)
        if not success:
            # Event was dropped - handle appropriately
            log_warning_event(logger, "✗ Event dropped.", "ag_ui.event_dropped", run_id=run_id)
        ```
    """
    try:
        await asyncio.wait_for(event_queue.put(event), timeout=timeout)
        return True
    except TimeoutError:
        # Queue is full and timeout exceeded - log error (event is lost)
        queue_size = event_queue.qsize()
        log_error_event(
            logger,
            f"✗ Event queue full, timeout exceeded: run_id={run_id}, "
            f"queue_size={queue_size}, timeout={timeout}s. Event was dropped.",
            "ag_ui.event_queue_timeout",
            run_id=run_id,
            thread_id=thread_id,
            queue_size=queue_size,
            timeout=timeout,
        )
        return False


async def put_critical_event_with_retry(
    event_queue: asyncio.Queue[str],
    event: str,
    run_id: str,
    thread_id: str,
    event_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_timeout: float = DEFAULT_RETRY_INITIAL_TIMEOUT,
    max_timeout: float = DEFAULT_RETRY_MAX_TIMEOUT,
) -> bool:
    """Put a critical event into the queue with retry logic and exponential backoff.

    Critical events (like cancellation, finish) are retried with exponential backoff
    to ensure they are delivered even under high load. Each retry uses a longer timeout.

    **Behavior:**
    - Attempts to queue the event with exponential backoff (5s → 10s → 20s → 30s max)
    - Retries up to `max_retries` times if queue is full
    - Adds jitter to prevent synchronized retries
    - If all retries exhausted: Event is dropped, error logged, returns False

    **When to Use:**
    - Use for critical events that must be delivered (cancellation, finish, errors)
    - Use `put_event_with_backpressure()` for non-critical events (regular streaming)

    **Caller Responsibilities:**
    - Check return value: If False, all retries exhausted and event was dropped
    - Logging failure is handled internally, but caller may want to take additional action

    Args:
        event_queue: Queue to put event into
        event: Encoded event string to queue
        run_id: Run identifier for logging
        thread_id: Thread identifier for logging
        event_name: Name of event type for logging (e.g., "cancellation error event")
        max_retries: Maximum number of retry attempts (default: 3, total attempts = 4)
        initial_timeout: Initial timeout in seconds
                         (default: DEFAULT_RETRY_INITIAL_TIMEOUT = 5.0)
        max_timeout: Maximum timeout in seconds
                     (default: DEFAULT_RETRY_MAX_TIMEOUT = 30.0)

    Returns:
        True if event was successfully queued, False if all retries exhausted (event was dropped)

    Example:
        ```python
        success = await put_critical_event_with_retry(
            event_queue, event, run_id, thread_id, "cancellation event"
        )
        if not success:
            # All retries exhausted - critical event was dropped
            # This is logged internally, but caller may want to take additional action
        ```
    """
    for attempt in range(max_retries + 1):
        # Calculate timeout with exponential backoff: initial * 2^attempt, capped at max
        timeout = min(initial_timeout * (2.0**attempt), max_timeout)

        try:
            await asyncio.wait_for(event_queue.put(event), timeout=timeout)
            # Success - event queued
            if attempt > 0:
                log_info_event(
                    logger,
                    f"✓ Critical event '{event_name}' queued after {attempt} retries: "
                    f"run_id={run_id}",
                    "ag_ui.critical_event_queued_after_retry",
                    run_id=run_id,
                    thread_id=thread_id,
                    event_name=event_name,
                    attempts=attempt + 1,
                )
            return True
        except TimeoutError:
            queue_size = event_queue.qsize()

            if attempt < max_retries:
                # Calculate delay before next retry (exponential backoff with jitter)
                delay = min(initial_timeout * (2.0**attempt), max_timeout / 2)
                # Add jitter (±25%) to prevent synchronized retries
                jitter = delay * 0.25
                delay = delay + random.uniform(-jitter, jitter)
                delay = max(
                    DEFAULT_EVENT_STREAM_CHECK_TIMEOUT, delay
                )  # Ensure minimum delay

                log_error_event(
                    logger,
                    f"✗ Critical event '{event_name}' queue timeout (attempt {attempt + 1}/{max_retries + 1}): "
                    f"run_id={run_id}, queue_size={queue_size}, retrying in {delay:.2f}s",
                    "ag_ui.critical_event_queue_retry",
                    run_id=run_id,
                    thread_id=thread_id,
                    event_name=event_name,
                    attempt=attempt + 1,
                    max_retries=max_retries + 1,
                    queue_size=queue_size,
                    delay=delay,
                )
                # Wait before retry
                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                log_error_event(
                    logger,
                    f"✗ Critical event '{event_name}' failed to queue after {max_retries + 1} attempts: "
                    f"run_id={run_id}, queue_size={queue_size}. Event was dropped.",
                    "ag_ui.critical_event_queue_failed",
                    run_id=run_id,
                    thread_id=thread_id,
                    event_name=event_name,
                    attempts=max_retries + 1,
                    queue_size=queue_size,
                )
                return False

    # Should never reach here, but handle edge case
    return False


async def emit_cancellation_events(
    encoder: EventEncoderProtocol,
    run_id: str,
    thread_id: str,
    event_queue: asyncio.Queue[str],
    log_event_name: str,
) -> None:
    """Emit cancellation error event and RunFinishedEvent to the event queue.

    Uses retry logic with exponential backoff for critical cancellation events
    to ensure they are delivered even under high load. Retries up to 3 times
    with increasing timeouts (5s, 10s, 20s, capped at 30s).

    Args:
        encoder: EventEncoder instance
        run_id: Run identifier
        thread_id: Thread identifier
        event_queue: Queue to put events into
        log_event_name: Event name for logging
    """
    log_info_event(
        logger,
        f"Run canceled: run_id={run_id}, thread_id={thread_id}",
        log_event_name,
        run_id=run_id,
        thread_id=thread_id,
    )
    # Emit cancellation error event with retry logic for critical events
    error_event = create_error_event(
        message="Run was canceled by user",
        code="RUN_CANCELED",
    )
    await put_critical_event_with_retry(
        event_queue,
        encoder.encode(error_event),
        run_id,
        thread_id,
        event_name="cancellation error event",
        max_retries=DEFAULT_MAX_RETRIES,
        initial_timeout=DEFAULT_RETRY_INITIAL_TIMEOUT,
        max_timeout=DEFAULT_RETRY_MAX_TIMEOUT,
    )

    # Emit RunFinishedEvent with retry logic for critical events
    await put_critical_event_with_retry(
        event_queue,
        encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id
            )
        ),
        run_id,
        thread_id,
        event_name="RunFinishedEvent",
        max_retries=DEFAULT_MAX_RETRIES,
        initial_timeout=DEFAULT_RETRY_INITIAL_TIMEOUT,
        max_timeout=DEFAULT_RETRY_MAX_TIMEOUT,
    )


async def check_and_handle_cancellation(
    run_manager: RunManager,
    run_id: str,
    thread_id: str,
    encoder: EventEncoderProtocol,
    event_queue: asyncio.Queue[str],
) -> bool:
    """Check if run was canceled and handle it if so.

    Args:
        run_manager: RunManager instance
        run_id: Run identifier
        thread_id: Thread identifier
        encoder: EventEncoder instance
        event_queue: Queue to put events into

    Returns:
        True if run was canceled, False otherwise
    """
    if await run_manager.is_run_canceled(run_id):
        await emit_cancellation_events(
            encoder,
            run_id,
            thread_id,
            event_queue,
            "ag_ui.run_canceled_during_generation",
        )
        return True
    return False


async def consume_event_generator_with_cancellation(
    event_generator: AsyncIterator[str],
    run_id: str,
    thread_id: str,
    encoder: EventEncoderProtocol,
    event_queue: asyncio.Queue[str],
) -> Exception | None:
    """Consume event generator with cancellation checks and error handling.

    Args:
        event_generator: Async generator yielding encoded events
        run_id: Run identifier
        thread_id: Thread identifier
        encoder: EventEncoder instance
        event_queue: Queue to put events into

    Returns:
        Exception if one occurred, None otherwise
    """
    run_manager = get_run_manager()
    generator_error: Exception | None = None

    try:
        async for event in event_generator:
            # Check if run was canceled
            if await check_and_handle_cancellation(
                run_manager, run_id, thread_id, encoder, event_queue
            ):
                break
            # Put event with backpressure handling
            if not await put_event_with_backpressure(
                event_queue, event, run_id, thread_id
            ):
                # Queue timeout - log error and break to prevent infinite blocking
                log_error_event(
                    logger,
                    f"✗ Event queue timeout during generation, stopping: run_id={run_id}",
                    "ag_ui.event_generator_queue_timeout",
                    run_id=run_id,
                    thread_id=thread_id,
                )
                break
    except asyncio.CancelledError:
        # Handle cancellation gracefully
        await emit_cancellation_events(
            encoder, run_id, thread_id, event_queue, "ag_ui.run_task_canceled"
        )
    except Exception as e:
        generator_error = e
        log_error_event(
            logger,
            "✗ Event generator error.",
            "ag_ui.event_generator_error",
            error=str(e),
            exc_info=True,
            run_id=run_id,
            thread_id=thread_id,
        )

    return generator_error


async def yield_events_from_queue(
    event_queue: asyncio.Queue[str],
    generator_task: asyncio.Task,
    generator_error: Exception | None,
    run_id: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Yield events from queue with proper timeout and error handling.

    Args:
        event_queue: Queue containing events
        generator_task: Task that consumes the event generator (used to check completion)
        generator_error: Exception if generator failed
        run_id: Run identifier
        thread_id: Thread identifier

    Yields:
        Encoded SSE event strings

    Raises:
        Exception: If generator_error is set
        asyncio.CancelledError: If stream is canceled externally
    """
    import time

    # Get timeout configuration from centralized config
    config = get_config()
    MAX_GENERATOR_WAIT_TIME = config.max_generator_wait_time
    MAX_CONSECUTIVE_TIMEOUTS = config.max_consecutive_timeouts
    start_time = time.time()
    consecutive_timeouts = 0

    try:
        # Check task.done() directly instead of generator_done flag to avoid closure issue
        while not generator_task.done() or not event_queue.empty():
            # Check if we've been waiting too long for the generator to complete
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_GENERATOR_WAIT_TIME:
                log_info_event(
                    logger,
                    f"✗ Event stream timeout: generator not done after {elapsed_time:.1f}s, run_id={run_id}",
                    "ag_ui.event_stream_timeout",
                    run_id=run_id,
                    thread_id=thread_id,
                    elapsed_time=elapsed_time,
                )
                break

            try:
                # Wait for event with timeout to check if generator is done
                event = await asyncio.wait_for(
                    event_queue.get(), timeout=DEFAULT_EVENT_STREAM_CHECK_TIMEOUT
                )
                consecutive_timeouts = 0  # Reset timeout counter on successful event
                yield event
            except TimeoutError:
                consecutive_timeouts += 1
                # Check if generator is done and queue is empty
                if generator_task.done() and event_queue.empty():
                    break
                # If we've had too many consecutive timeouts and no events, break
                # This handles the case where TestClient doesn't consume the stream fully
                if (
                    consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS
                    and event_queue.empty()
                ):
                    log_info_event(
                        logger,
                        f"✗ Event stream timeout: no events for {consecutive_timeouts * DEFAULT_EVENT_STREAM_CHECK_TIMEOUT:.1f}s, run_id={run_id}",
                        "ag_ui.event_stream_timeout",
                        run_id=run_id,
                        thread_id=thread_id,
                        consecutive_timeouts=consecutive_timeouts,
                    )
                    break
                continue

        # If there was an error, raise it
        if generator_error:
            raise generator_error

    except asyncio.CancelledError:
        # Task was canceled externally (e.g., client disconnect)
        log_info_event(
            logger,
            f"✗ Event stream canceled: run_id={run_id}",
            "ag_ui.event_stream_canceled",
            run_id=run_id,
            thread_id=thread_id,
        )
        raise
