"""Constants for AG-UI server configuration.

This module contains all magic numbers and hardcoded values used throughout
the server codebase, extracted to named constants for better maintainability.
"""

from __future__ import annotations

# Default pagination limits
DEFAULT_THREAD_LIMIT: int = 50
"""Default maximum number of threads to return in paginated responses."""

DEFAULT_MESSAGE_LIMIT: int = 100
"""Default maximum number of messages to return in paginated responses."""

DEFAULT_EVENT_LIMIT: int = 1000
"""Default maximum number of events to return in paginated responses."""

# Default server configuration
DEFAULT_SERVER_PORT: int = 8001
"""Default port number for the AG-UI server."""

DEFAULT_SERVER_HOST: str = "0.0.0.0"
"""Default host address for the AG-UI server (0.0.0.0 means listen on all interfaces)."""

DEFAULT_MAX_REQUEST_BODY_BYTES: int = 5 * 1024 * 1024
"""Default maximum request body size in bytes (5 MiB). 0 to disable. Oversized requests get 413."""

# Default CORS configuration
DEFAULT_CORS_METHODS: list[str] = ["GET", "POST", "OPTIONS"]
"""Default CORS allowed HTTP methods."""

DEFAULT_CORS_HEADERS: list[str] = [
    "Content-Type",
    "Accept",
    "Authorization",
    "X-User-Id",
]
"""Default CORS allowed headers."""

# Default rate limiting configuration
DEFAULT_RATE_LIMIT_ENABLED: bool = True
"""Default rate limiting enabled status.

Set via AG_UI_RATE_LIMIT_ENABLED environment variable:
- "true": Rate limiting enabled (default)
- "false": Rate limiting disabled
"""

DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60
"""Default maximum requests per minute per client.

Set via AG_UI_RATE_LIMIT_PER_MINUTE environment variable.
Default: 60 requests per minute.
"""

DEFAULT_RATE_LIMIT_PER_HOUR: int = 1000
"""Default maximum requests per hour per client.

Set via AG_UI_RATE_LIMIT_PER_HOUR environment variable.
Default: 1000 requests per hour.
"""

# Default authentication configuration
DEFAULT_AUTH_ENABLED: bool = True
"""Default authentication enabled status.

Set via AG_UI_AUTH_ENABLED environment variable:
- "true": Authentication enabled (default)
- "false": Authentication disabled (for development only)
"""

DEFAULT_AUTH_MODE: str = "strict"
"""Default authentication mode.

Set via AG_UI_AUTH_MODE environment variable:
- "strict": Reject unauthenticated requests (401 Unauthorized) - default
- "permissive": Allow unauthenticated requests but log warnings
"""

DEFAULT_AUTH_STRATEGIES: str = "header"
"""Default authentication strategies (comma-separated).

Set via AG_UI_AUTH_STRATEGIES environment variable:
- "header": Header-based authentication (X-User-Id header) - default
- "token": Token-based authentication (JWT/Bearer tokens) - future
- "apikey": API key authentication - future

Multiple strategies can be specified: "header,token"
"""

# Default event queue configuration
DEFAULT_MAX_EVENT_QUEUE_SIZE: int = 1000
"""Default maximum size for event queues to prevent unbounded memory growth.

Set via AG_UI_MAX_EVENT_QUEUE_SIZE environment variable.
Default: 1000 events per queue.

When a queue reaches this size, backpressure is applied:
- Producer will block until space is available (default behavior)
- This prevents memory exhaustion under high load
- Consider increasing if you have high event throughput and sufficient memory
"""

# Default retry configuration
DEFAULT_MAX_RETRIES: int = 3
"""Default maximum number of retry attempts for critical operations.

This value is used for:
- Retrying transient errors in persistence operations
- Retrying critical event queue operations
- Retrying operations with exponential backoff

Set via AG_UI_MAX_RETRIES environment variable in ServerConfig.
Default: 3 retries (4 total attempts including initial attempt).
"""

# Default timeout configuration
DEFAULT_EVENT_QUEUE_TIMEOUT: float = 5.0
"""Default timeout for event queue put operations (seconds).

If exceeded, event is dropped and error is logged.
Used for non-critical events that can be dropped under load.

Set via AG_UI_EVENT_QUEUE_TIMEOUT environment variable in ServerConfig.
Default: 5.0 seconds.
"""

DEFAULT_EVENT_STREAM_CHECK_TIMEOUT: float = 0.1
"""Default timeout for event stream check operations (seconds).

Small timeout used to periodically check event queues without busy-waiting.
100ms provides real-time feel while avoiding excessive CPU usage.

Used in:
- Event stream polling loops
- Python tool event queue checks

Default: 0.1 seconds (100ms).
"""

DEFAULT_RETRY_INITIAL_TIMEOUT: float = 5.0
"""Default initial timeout for retry operations with exponential backoff (seconds).

Used as the starting timeout for critical event retries.
Subsequent retries use exponential backoff: initial * (2^attempt), capped at max_timeout.

Set via AG_UI_RETRY_INITIAL_TIMEOUT environment variable in ServerConfig.
Default: 5.0 seconds.
"""

DEFAULT_RETRY_MAX_TIMEOUT: float = 30.0
"""Default maximum timeout for retry operations (seconds).

Caps the exponential backoff timeout for critical event retries.
Prevents excessively long waits while still allowing reasonable retry attempts.

Set via AG_UI_RETRY_MAX_TIMEOUT environment variable in ServerConfig.
Default: 30.0 seconds.
"""

# Default retry delay configuration
DEFAULT_RETRY_INITIAL_DELAY: float = 1.0
"""Default initial delay before first retry attempt (seconds).

Used in exponential backoff calculations for retry operations.
Subsequent delays increase exponentially: initial * (base^attempt), capped at max_delay.

Default: 1.0 seconds.
"""

DEFAULT_RETRY_MAX_DELAY: float = 60.0
"""Default maximum delay between retry attempts (seconds).

Caps the exponential backoff delay to prevent excessively long waits.
Used in general retry operations with exponential backoff.

Default: 60.0 seconds.
"""

DEFAULT_RETRY_MAX_DELAY_SHORT: float = 10.0
"""Default maximum delay for short-duration retry operations (seconds).

Used for operations that should retry more quickly (e.g., persistence operations).
Shorter than DEFAULT_RETRY_MAX_DELAY to provide faster feedback.

Default: 10.0 seconds.
"""

DEFAULT_RETRY_EXPONENTIAL_BASE: float = 2.0
"""Default exponential base for backoff delay calculations.

Used to calculate exponential backoff: delay = initial * (base^attempt).
Base of 2.0 means delays double with each retry: 1s, 2s, 4s, 8s, etc.

Default: 2.0.
"""

DEFAULT_RETRY_JITTER_MULTIPLIER: float = 0.2
"""Default jitter multiplier for retry delays (±20%).

Jitter prevents synchronized retries from multiple clients.
Applied as: delay ± (delay * multiplier).

Default: 0.2 (20% jitter).
"""

DEFAULT_MIN_RETRY_DELAY: float = 0.1
"""Default minimum delay for retry operations (seconds).

Ensures retry delays never go below this value, even with jitter.
Prevents negative or zero delays that could cause busy-waiting.

Default: 0.1 seconds (100ms).
"""

# Text-based MIME types for file content extraction
TEXT_MIME_TYPES: list[str] = [
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-sh",
    "application/yaml",
    "application/x-yaml",
]
"""List of MIME type prefixes that indicate text-based file content.

Files with these MIME types will have their content decoded and included
in the user message text. Other file types will be included as metadata only.
"""

# Default MIME type for binary files
DEFAULT_MIME_TYPE: str = "application/octet-stream"
"""Default MIME type used when file MIME type cannot be determined."""

# Default user message when no message content is available
DEFAULT_USER_MESSAGE: str = "Hello"
"""Default message text used when no user message content is available."""

# Message role constants
ROLE_USER: str = "user"
"""Role constant for user messages in the AG-UI protocol."""

ROLE_ASSISTANT: str = "assistant"
"""Role constant for assistant messages in the AG-UI protocol."""

# Server log messages (use with f-strings)
SERVER_MESSAGES: dict[str, str] = {
    "STARTING": "🚀 Starting AG-UI Server on",
    "ENDPOINT": "📡 Endpoint: http://",
    "HEALTH": "❤️ Health check: http://",
}
"""Dictionary of formatted log messages for server startup and status.

These messages are used with f-strings to create formatted log output.
Keys correspond to different log message types.
"""
