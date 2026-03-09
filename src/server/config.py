"""Centralized AG-UI server configuration using Pydantic settings.

This module provides centralized configuration management using Pydantic settings
for validation and type safety. All server configuration should go through
this module to ensure consistency and proper validation.

**Usage:**
    >>> from server.config import get_config
    >>> config = get_config()
    >>> print(f"Server: {config.server_host}:{config.server_port}")
    >>> print(f"OpenSearch: {config.opensearch_url}")

**Testability:** Prefer injecting a ``ServerConfig`` instance (e.g. into
``create_app(config_override=...)``, ``create_rate_limiter(config=...)``, or
``StrandsAgent(..., cache_max_size=...)``) so tests can override without
calling ``reset_config()``. Routes and helpers use ``request.app.state.config``
when set by ``create_app``, falling back to ``get_config()``.
"""

from __future__ import annotations

import os

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from server.constants import (
    DEFAULT_AUTH_ENABLED,
    DEFAULT_AUTH_MODE,
    DEFAULT_AUTH_STRATEGIES,
    DEFAULT_CORS_HEADERS,
    DEFAULT_CORS_METHODS,
    DEFAULT_EVENT_LIMIT,
    DEFAULT_EVENT_QUEUE_TIMEOUT,
    DEFAULT_MAX_EVENT_QUEUE_SIZE,
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MESSAGE_LIMIT,
    DEFAULT_RATE_LIMIT_ENABLED,
    DEFAULT_RATE_LIMIT_PER_HOUR,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DEFAULT_RETRY_INITIAL_TIMEOUT,
    DEFAULT_RETRY_MAX_TIMEOUT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_THREAD_LIMIT,
)
from utils.logging_helpers import get_logger, log_error_event, log_warning_event


def _inject_env_var(data: dict, key: str, env_var_name: str) -> None:
    """Inject a non-prefixed env var into settings data if not already set.

    Used for env vars that don't follow the AG_UI_ prefix (e.g. OPENSEARCH_URL).
    Mutates ``data`` in place. Only sets the key if neither the lowercase key
    nor the uppercase env name is present in data, and the env var is set.

    Args:
        data: Raw settings dict (from Pydantic) to mutate.
        key: Field key (lowercase with underscores, e.g. ``opensearch_url``).
        env_var_name: Environment variable name (e.g. ``OPENSEARCH_URL``).
    """
    if key not in data and env_var_name not in data:
        value = os.getenv(env_var_name)
        if value:
            data[key] = value


class ServerConfig(BaseSettings):
    """Centralized server configuration using Pydantic settings.

    This class centralizes all server configuration options with validation
    and type safety. All configuration values can be set via environment variables
    (with automatic type conversion and validation) or via direct instantiation.

    **Environment Variable Naming:**
    - Environment variables use uppercase with underscores (e.g., `AG_UI_SERVER_HOST`)
    - Pydantic automatically converts these to lowercase with underscores for field names
    - Field aliases map environment variables to Python field names

    **Examples:**
        >>> # Using environment variables (recommended)
        >>> import os
        >>> os.environ["AG_UI_SERVER_HOST"] = "0.0.0.0"
        >>> os.environ["AG_UI_SERVER_PORT"] = "8001"
        >>> config = ServerConfig()
        >>> print(config.server_host)  # "0.0.0.0"
        >>> print(config.server_port)  # 8001

        >>> # Direct instantiation
        >>> config = ServerConfig(server_host="127.0.0.1", server_port=9000)
        >>> print(config.server_host)  # "127.0.0.1"
    """

    model_config = SettingsConfigDict(
        env_prefix="AG_UI_",  # All env vars prefixed with AG_UI_
        case_sensitive=False,  # Case-insensitive env var matching
        extra="ignore",  # Ignore extra env vars not defined here
    )

    # Server Configuration
    server_host: str = Field(
        default=DEFAULT_SERVER_HOST,
        description="Server host address (default: 0.0.0.0 means listen on all interfaces)",
    )
    server_port: int = Field(
        default=DEFAULT_SERVER_PORT,
        ge=1,
        le=65535,
        description="Server port number (1-65535)",
    )
    max_request_body_bytes: int = Field(
        default=DEFAULT_MAX_REQUEST_BODY_BYTES,
        ge=0,
        description="Maximum request body size in bytes (0 to disable). Oversized requests get 413. Default 5 MiB.",
    )

    # OpenSearch Configuration (no AG_UI_ prefix)
    opensearch_url: str = Field(
        default="http://localhost:9200",
        description="OpenSearch cluster URL (from OPENSEARCH_URL env var)",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_env_vars_without_prefix(cls, data: dict) -> dict:
        """Handle env vars which don't have AG_UI_ prefix (OPENSEARCH_URL, PHOENIX_URL)."""
        if isinstance(data, dict):
            _inject_env_var(data, "opensearch_url", "OPENSEARCH_URL")
            _inject_env_var(data, "phoenix_url", "PHOENIX_URL")
            _inject_env_var(data, "phoenix_public_url", "PHOENIX_PUBLIC_URL")
        return data

    # CORS Configuration
    cors_origins: str | None = Field(
        default=None,
        description="CORS allowed origins. Set via AG_UI_CORS_ORIGINS. "
        "Empty/unset: CORS disabled (most secure). "
        "Comma-separated list: specific origins (e.g. http://localhost:3000,https://example.com). "
        "'*': allow all origins (insecure, use only for development).",
    )
    cors_methods: str | None = Field(
        default=None,
        description="CORS allowed HTTP methods (comma-separated). "
        f"Defaults to: {','.join(DEFAULT_CORS_METHODS)}",
    )
    cors_headers: str | None = Field(
        default=None,
        description="CORS allowed headers (comma-separated). "
        f"Defaults to: {','.join(DEFAULT_CORS_HEADERS)}",
    )

    # Persistence Configuration
    enable_persistence: bool = Field(
        default=False,
        description="Enable AG-UI data persistence (threads, runs, messages, events)",
    )
    db_path: str = Field(
        default=".ag-ui/chat_history.db",
        description="SQLite database path for persistence",
    )

    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(
        default=DEFAULT_RATE_LIMIT_ENABLED,
        description="Enable/disable rate limiting",
    )
    rate_limit_per_minute: int = Field(
        default=DEFAULT_RATE_LIMIT_PER_MINUTE,
        ge=1,
        description="Maximum requests per minute per client",
    )
    rate_limit_per_hour: int = Field(
        default=DEFAULT_RATE_LIMIT_PER_HOUR,
        ge=1,
        description="Maximum requests per hour per client",
    )

    # Event Queue Configuration
    max_event_queue_size: int = Field(
        default=DEFAULT_MAX_EVENT_QUEUE_SIZE,
        ge=1,
        description="Maximum size for event queues (prevents unbounded memory growth)",
    )

    # Pagination Limits
    default_thread_limit: int = Field(
        default=DEFAULT_THREAD_LIMIT,
        ge=1,
        description="Default maximum number of threads in paginated responses",
    )
    default_message_limit: int = Field(
        default=DEFAULT_MESSAGE_LIMIT,
        ge=1,
        description="Default maximum number of messages in paginated responses",
    )
    default_event_limit: int = Field(
        default=DEFAULT_EVENT_LIMIT,
        ge=1,
        description="Default maximum number of events in paginated responses",
    )

    # Observability Configuration
    phoenix_url: str = Field(
        default="http://phoenix:6006",
        description="Phoenix observability URL (internal)",
    )
    phoenix_public_url: str | None = Field(
        default=None,
        description="Phoenix public URL (for browser access)",
    )

    # Logging Configuration
    log_format: str = Field(
        default="human",
        description="Log format: 'human' (readable) or 'json' (structured)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    # Event Stream Timeout Configuration
    max_generator_wait_time: float = Field(
        default=1800.0,
        ge=1.0,
        description="Maximum time to wait for event generator to complete (seconds). "
        "Increased to match longest agent timeout (orchestrator/evaluation: 1800s)",
    )
    max_consecutive_timeouts: int = Field(
        default=600,
        ge=1,
        description="Maximum consecutive timeouts before considering stream stuck. "
        "600 * 0.1s = 60 seconds (1 minute) of no events",
    )

    # Event Queue Timeout Configuration
    event_queue_timeout: float = Field(
        default=DEFAULT_EVENT_QUEUE_TIMEOUT,
        ge=0.1,
        description="Timeout for queue put operations (seconds). "
        "If exceeded, event is dropped and error is logged",
    )

    # Retry Configuration
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description="Maximum number of retry attempts for critical operations",
    )
    retry_initial_timeout: float = Field(
        default=DEFAULT_RETRY_INITIAL_TIMEOUT,
        ge=0.1,
        description="Initial timeout for retry operations (seconds)",
    )
    retry_max_timeout: float = Field(
        default=DEFAULT_RETRY_MAX_TIMEOUT,
        ge=1.0,
        description="Maximum timeout for retry operations (seconds)",
    )

    # Authentication Configuration
    auth_enabled: bool = Field(
        default=DEFAULT_AUTH_ENABLED,
        description="Enable/disable authentication middleware (default: true). "
        "Set to false for development/testing only.",
    )
    auth_mode: str = Field(
        default=DEFAULT_AUTH_MODE,
        description="Authentication mode: 'strict' (reject unauthenticated) or "
        "'permissive' (allow with warnings). Default: strict",
    )
    auth_strategies: str = Field(
        default=DEFAULT_AUTH_STRATEGIES,
        description="Comma-separated list of authentication strategies: "
        "'header' (X-User-Id header), 'token' (JWT/Bearer), 'apikey' (API key). "
        "Default: header",
    )

    # JWT Configuration (optional, only required if 'token' strategy is enabled)
    jwt_secret: str | None = Field(
        default=None,
        description="JWT secret key for HS256 algorithm (symmetric key). "
        "Required if using JWT authentication with HS256.",
    )
    jwt_public_key: str | None = Field(
        default=None,
        description="JWT public key for RS256 algorithm (asymmetric key). "
        "Required if using JWT authentication with RS256.",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm to use: 'HS256' (symmetric) or 'RS256' (asymmetric). "
        "Default: HS256",
    )
    jwt_user_id_claim: str = Field(
        default="sub",
        description="JWT claim name containing user ID. Default: 'sub'. "
        "Falls back to 'user_id' if 'sub' is not present.",
    )

    # API Key Configuration (optional, only required if 'apikey' strategy is enabled)
    api_keys: str | None = Field(
        default=None,
        description="JSON string mapping API keys to user IDs. "
        'Format: \'{"key1": "user1", "key2": "user2"}\'. '
        "Required if using API key authentication.",
    )

    # Trusted Proxy Configuration (required when using header authentication)
    trusted_proxy_enabled: bool = Field(
        default=False,
        description="Enable trusted proxy mode. Required when using header authentication. "
        "Set to true when server is behind a trusted proxy/frontend (e.g., OpenSearch Dashboards) "
        "that validates users and sets X-User-Id header. Default: false",
    )

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is either 'human' or 'json'."""
        v_lower = v.lower()
        if v_lower not in ("human", "json"):
            raise ValueError("log_format must be 'human' or 'json'")
        return v_lower

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid logging level."""
        v_upper = v.upper()
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator("auth_mode")
    @classmethod
    def validate_auth_mode(cls, v: str) -> str:
        """Validate auth mode is either 'strict' or 'permissive'."""
        v_lower = v.lower()
        if v_lower not in ("strict", "permissive"):
            raise ValueError("auth_mode must be 'strict' or 'permissive'")
        return v_lower

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm is either 'HS256' or 'RS256'."""
        v_upper = v.upper()
        if v_upper not in ("HS256", "RS256"):
            raise ValueError("jwt_algorithm must be 'HS256' or 'RS256'")
        return v_upper

    @model_validator(mode="after")
    def validate_auth_config(self) -> ServerConfig:
        """Validate authentication configuration when strategies are enabled."""
        strategies = [s.strip().lower() for s in self.auth_strategies.split(",")]

        # Validate JWT config if token strategy is enabled
        if "token" in strategies:
            if self.jwt_algorithm.upper() == "HS256" and not self.jwt_secret:
                raise ValueError(
                    "jwt_secret is required when using JWT authentication with HS256 algorithm"
                )
            if self.jwt_algorithm.upper() == "RS256" and not self.jwt_public_key:
                raise ValueError(
                    "jwt_public_key is required when using JWT authentication with RS256 algorithm"
                )

        # Validate API key config if apikey strategy is enabled
        if "apikey" in strategies and not self.api_keys:
            raise ValueError("api_keys is required when using API key authentication")

        return self

    def get_cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list, handling wildcard and empty cases.

        Returns:
            List of allowed origins, or empty list if CORS disabled.
            If wildcard ('*') is set, returns ['*'].
        """
        if not self.cors_origins:
            return []
        origins = [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]
        if "*" in origins:
            return ["*"]
        return origins

    def get_cors_methods_list(self) -> list[str]:
        """Get CORS methods as a list, with defaults.

        Returns:
            List of allowed HTTP methods.
        """
        if not self.cors_methods:
            return DEFAULT_CORS_METHODS.copy()
        return [
            method.strip() for method in self.cors_methods.split(",") if method.strip()
        ]

    def get_cors_headers_list(self) -> list[str]:
        """Get CORS headers as a list, with defaults.

        Returns:
            List of allowed headers.
        """
        if not self.cors_headers:
            return DEFAULT_CORS_HEADERS.copy()
        return [
            header.strip() for header in self.cors_headers.split(",") if header.strip()
        ]


# Global singleton instance
_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    """Get the global server configuration instance.

    This function returns a singleton instance of ServerConfig, ensuring
    that configuration is loaded once and reused throughout the application.

    Returns:
        ServerConfig instance with current configuration values

    Examples:
        >>> config = get_config()
        >>> print(f"Server: {config.server_host}:{config.server_port}")
        >>> print(f"OpenSearch: {config.opensearch_url}")
    """
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config


def reset_config() -> None:
    """Reset the global server configuration singleton.

    This function clears the cached configuration instance, forcing
    `get_config()` to create a new instance on the next call. This is
    primarily useful for testing when environment variables are modified
    and you need to ensure the configuration is re-read.

    **Warning:** This should only be used in tests. Resetting the config
    in production code can cause unexpected behavior.

    Examples:
        >>> # In tests, reset config before modifying environment variables
        >>> import os
        >>> from server.config import reset_config, get_config
        >>> reset_config()
        >>> os.environ["AG_UI_SERVER_PORT"] = "9000"
        >>> config = get_config()  # Will read new port value
        >>> assert config.server_port == 9000
    """
    global _config
    _config = None


def validate_config(config: ServerConfig) -> list[tuple[str, str]]:
    """Validate configuration and return list of (severity, message) tuples.

    This function performs security and operational validation of configuration
    combinations to prevent insecure deployments and catch configuration errors
    at startup rather than during incidents.

    Args:
        config: ServerConfig instance to validate

    Returns:
        List of (severity, message) tuples where severity is "error" or "warning".
        Errors should prevent startup, warnings should be logged but allow startup.

    Examples:
        >>> config = ServerConfig(auth_enabled=True, auth_strategies="header")
        >>> issues = validate_config(config)
        >>> for severity, message in issues:
        ...     print(f"{severity}: {message}")
    """
    issues = []

    # Critical: Header auth without trusted proxy
    # In production, this is an error. In dev environments, it's a warning.
    strategies = [s.strip().lower() for s in config.auth_strategies.split(",")]
    if config.auth_enabled and "header" in strategies:
        if not config.trusted_proxy_enabled:
            environment = (os.getenv("ENVIRONMENT") or "").lower()
            is_production = environment == "production"

            if is_production:
                # Production: Error - prevent insecure deployment
                issues.append(
                    (
                        "error",
                        "Header authentication requires trusted proxy in production. "
                        "Set AG_UI_TRUSTED_PROXY_ENABLED=true or use token/apikey strategies.",
                    )
                )
            else:
                # Development: Warning - allow but warn about security implications
                issues.append(
                    (
                        "warning",
                        "Header authentication without trusted proxy is insecure. "
                        "Anyone can send X-User-Id header to impersonate users. "
                        "For development, consider: (1) Set AG_UI_TRUSTED_PROXY_ENABLED=true if behind a proxy, "
                        "(2) Use AG_UI_AUTH_STRATEGIES=token or apikey, or (3) Set AG_UI_AUTH_ENABLED=false for local testing.",
                    )
                )

    # Warning: Auth enabled but persistence disabled
    if config.auth_enabled and not config.enable_persistence:
        issues.append(
            (
                "warning",
                "Authentication enabled but persistence disabled. "
                "Authorization checks will be skipped. This is insecure for production.",
            )
        )

    # Warning: Permissive mode in production
    if config.auth_mode == "permissive" and os.getenv("ENVIRONMENT") == "production":
        issues.append(
            (
                "warning",
                "Permissive auth mode in production. Consider using strict mode.",
            )
        )

    return issues


def validate_config_on_startup(config: ServerConfig | None = None) -> None:
    """Validate configuration on application startup.

    This function should be called during application startup (e.g., in create_app's
    lifespan) to catch configuration errors before the server starts accepting requests.
    Errors will raise ValueError to prevent startup, warnings will be logged.

    Args:
        config: Optional ServerConfig instance. If None, uses get_config().

    Raises:
        ValueError: If configuration validation finds critical errors that should
            prevent startup.

    Examples:
        >>> # In create_app lifespan
        >>> validate_config_on_startup(resolved)
    """
    logger = get_logger(__name__)
    resolved_config = config if config is not None else get_config()

    issues = validate_config(resolved_config)
    for severity, message in issues:
        if severity == "error":
            log_error_event(
                logger,
                "Configuration error.",
                "config.validation_error",
                error=message,
            )
            raise ValueError(f"Invalid configuration: {message}")
        else:
            log_warning_event(
                logger,
                "Configuration warning.",
                "config.validation_warning",
                details=message,
            )
