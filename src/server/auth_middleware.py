"""Authentication middleware for AG-UI server.

This module provides configurable authentication middleware that supports multiple
authentication strategies. It validates user authentication and extracts user IDs
from requests for use in route handlers.

**Authentication Strategies:**
- Header-based: Extracts user ID from `X-User-Id` header (for OpenSearch Dashboards)
- Token-based: Validates JWT/Bearer tokens (HS256 or RS256)
- API key: Validates API keys from `X-API-Key` header or `Authorization: ApiKey <key>`

**X-User-Id (header) trust model:**
The header strategy **trusts** `X-User-Id` when present. Use it **only** when the AG-UI
server is behind a trusted proxy or frontend (e.g. OpenSearch Dashboards) that has
already authenticated the user and sets `X-User-Id`. When the server is **not** behind
such a frontend, anyone can send `X-User-Id: arbitrary` and be treated as that user.
For standalone or development use, prefer ``auth_enabled=false`` or the ``token`` or
``apikey`` strategies, and understand the associated risks (see env docs).

**Authentication Modes:**
- `strict`: Reject unauthenticated requests (401 Unauthorized)
- `permissive`: Allow unauthenticated requests but log warnings

**Access Control Behavior:**

Access control is enforced at the route handler level when:
1. Request object is provided
2. User is authenticated (`is_authenticated(request)` returns `True`)
3. Persistence is enabled (for routes that need to fetch thread data)

**When Persistence is Disabled:**
- Access control checks are skipped entirely
- All requests are allowed (backward compatibility)
- This allows development/testing without persistence layer

**When Request is Not Provided:**
- Access control checks are skipped
- Allows backward compatibility with code that doesn't pass request
- Used for internal calls or testing scenarios

**Unauthenticated Requests:**
- In strict mode: Rejected at middleware level (401) before reaching route handlers.
- In permissive mode: Middleware allows through with a fallback user_id, but route handlers
  call `require_authenticated_if_auth_enabled(request)` and reject with 401 when auth is
  enabled and the request is not authenticated. This prevents bypassing authentication.

**Usage:**
    The middleware is automatically applied to all routes when registered in ag_ui_app.py.
    Route handlers can access authenticated user ID via `get_user_id_from_request(request)`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import TypedDict

import jwt
from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from server.config import ServerConfig, get_config
from server.utils import get_user_id_from_request
from utils.logging_helpers import get_logger, log_info_event, log_warning_event

logger = get_logger(__name__)


class AuthMode(str, Enum):
    """Authentication mode enumeration."""

    STRICT = "strict"
    PERMISSIVE = "permissive"


class AuthStrategy(str, Enum):
    """Authentication strategy enumeration."""

    HEADER = "header"
    TOKEN = "token"
    API_KEY = "apikey"


class AuthMiddlewareConfig(TypedDict):
    """Configuration dict returned by create_auth_middleware when auth is enabled."""

    enabled: bool
    mode: AuthMode
    strategies: list[AuthStrategy]
    config: ServerConfig


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for AG-UI server.

    Validates user authentication based on configured strategies and mode.
    Extracts user ID from requests and stores it in request state for use by route handlers.

    Attributes:
        enabled: Whether authentication is enabled
        mode: Authentication mode (strict or permissive)
        strategies: List of enabled authentication strategies

    """

    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        mode: AuthMode = AuthMode.STRICT,
        strategies: list[AuthStrategy] | None = None,
        config: ServerConfig | None = None,
    ) -> None:
        """Initialize authentication middleware.

        Args:
            app: FastAPI/Starlette application instance
            enabled: Whether authentication is enabled (default: True)
            mode: Authentication mode - strict (reject) or permissive (allow with warnings)
            strategies: List of authentication strategies to use (default: [HEADER])
            config: Optional ServerConfig instance (defaults to get_config())

        """
        super().__init__(app)
        self.enabled = enabled
        self.mode = mode
        # Filter None so iteration never sees invalid entries
        raw = strategies or [AuthStrategy.HEADER]
        self.strategies = [s for s in raw if s is not None]
        if not self.strategies:
            self.strategies = [AuthStrategy.HEADER]
        self.config = config or get_config()
        self._api_keys_map: dict[str, str | None] | None = None

    async def dispatch(
        self, request: Request, call_next: Callable[..., Response]
    ) -> Response:
        """Process request through authentication middleware.

        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler in the chain

        Returns:
            Response from next handler, or 401 Unauthorized if authentication fails in strict mode

        """
        # Skip authentication for health check endpoint
        if request.url.path == "/health":
            return await call_next(request)

        # If authentication is disabled, allow all requests
        if not self.enabled:
            return await call_next(request)

        # Try to authenticate using configured strategies
        user_id = None
        authenticated = False

        for strategy in self.strategies:
            if strategy == AuthStrategy.HEADER:
                user_id = self._authenticate_header(request)
                if user_id:
                    authenticated = True
                    break
            elif strategy == AuthStrategy.TOKEN:
                user_id = self._authenticate_token(request)
                if user_id:
                    authenticated = True
                    break
            elif strategy == AuthStrategy.API_KEY:
                user_id = self._authenticate_api_key(request)
                if user_id:
                    authenticated = True
                    break

        # Handle authentication result based on mode
        if not authenticated:
            if self.mode == AuthMode.STRICT:
                log_warning_event(
                    logger,
                    f"Unauthenticated request rejected: {request.method} {request.url.path}",
                    "ag_ui.auth_unauthenticated_rejected",
                    method=request.method,
                    path=request.url.path,
                )
                # Return JSONResponse directly instead of raising HTTPException.
                # Raising from BaseHTTPMiddleware causes Starlette ASGI errors
                # because the exception bypasses FastAPI's exception handlers.
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:  # PERMISSIVE mode
                log_warning_event(
                    logger,
                    f"Unauthenticated request allowed (permissive mode): {request.method} {request.url.path}",
                    "ag_ui.auth_unauthenticated_allowed",
                    method=request.method,
                    path=request.url.path,
                )
                # Fallback to client host for user ID
                user_id = get_user_id_from_request(request)

        # Store user ID in request state for route handlers
        if user_id:
            request.state.user_id = user_id
            request.state.authenticated = authenticated

        return await call_next(request)

    def _authenticate_header(self, request: Request) -> str | None:
        """Authenticate using X-User-Id header.

        This strategy trusts the X-User-Id header when present. Use it **only** when
        the AG-UI server is behind a trusted proxy or frontend (e.g. OpenSearch
        Dashboards) that has already authenticated the user and sets ``X-User-Id``.
        When the server is **not** behind such a frontend, anyone can send
        ``X-User-Id: arbitrary`` and be treated as that user. For standalone or dev
        use, use ``auth_enabled=false`` or the ``token``/``apikey`` strategies instead.

        Args:
            request: FastAPI request object

        Returns:
            User ID if header is present, None otherwise

        """
        user_id = request.headers.get("X-User-Id")
        if user_id:
            log_info_event(
                logger,
                f"Authenticated via header: user_id={user_id}",
                "ag_ui.auth_header_success",
                user_id=user_id,
            )
            return user_id
        return None

    def _authenticate_token(self, request: Request) -> str | None:
        """Authenticate using Bearer token (JWT).

        Validates JWT token signature and expiration, then extracts user ID from claims.

        Args:
            request: FastAPI request object

        Returns:
            User ID if token is valid, None otherwise

        """
        # Extract Bearer token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        try:
            # Get JWT configuration
            algorithm = self.config.jwt_algorithm.upper()
            user_id_claim = self.config.jwt_user_id_claim

            # Prepare key based on algorithm
            if algorithm == "HS256":
                if not self.config.jwt_secret:
                    log_warning_event(
                        logger,
                        "JWT secret not configured for HS256 algorithm",
                        "ag_ui.auth_jwt_missing_secret",
                    )
                    return None
                key = self.config.jwt_secret
            elif algorithm == "RS256":
                if not self.config.jwt_public_key:
                    log_warning_event(
                        logger,
                        "JWT public key not configured for RS256 algorithm",
                        "ag_ui.auth_jwt_missing_public_key",
                    )
                    return None
                key = self.config.jwt_public_key
            else:
                log_warning_event(
                    logger,
                    f"Unsupported JWT algorithm: {algorithm}",
                    "ag_ui.auth_jwt_unsupported_algorithm",
                    algorithm=algorithm,
                )
                return None

            # Decode and validate token
            decoded = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                options={"verify_exp": True, "verify_signature": True},
            )

            # Extract user ID from claims
            user_id = (
                decoded.get(user_id_claim)
                or decoded.get("user_id")
                or decoded.get("sub")
            )
            if not user_id:
                log_warning_event(
                    logger,
                    f"JWT token missing user ID claim: {user_id_claim}",
                    "ag_ui.auth_jwt_missing_user_id",
                    claim=user_id_claim,
                )
                return None

            log_info_event(
                logger,
                f"Authenticated via JWT: user_id={user_id}",
                "ag_ui.auth_jwt_success",
                user_id=user_id,
            )
            return str(user_id)

        except jwt.ExpiredSignatureError:
            log_warning_event(
                logger,
                "JWT token expired",
                "ag_ui.auth_jwt_expired",
            )
            return None
        except jwt.InvalidSignatureError:
            log_warning_event(
                logger,
                "JWT token has invalid signature",
                "ag_ui.auth_jwt_invalid_signature",
            )
            return None
        except jwt.DecodeError as e:
            log_warning_event(
                logger,
                f"JWT token decode error: {e}",
                "ag_ui.auth_jwt_decode_error",
                error=str(e),
            )
            return None
        # Broad catch intentional: avoid leaking JWT internals; treat any failure as unauthenticated.
        except Exception as e:
            log_warning_event(
                logger,
                f"JWT authentication error: {e}",
                "ag_ui.auth_jwt_error",
                error=str(e),
            )
            return None

    def _authenticate_api_key(self, request: Request) -> str | None:
        """Authenticate using API key.

        Validates API key against configured mapping and returns associated user ID.

        Args:
            request: FastAPI request object

        Returns:
            User ID if API key is valid, None otherwise

        """
        # Load API keys mapping if not already loaded
        if self._api_keys_map is None:
            if not self.config.api_keys:
                log_warning_event(
                    logger,
                    "API keys not configured",
                    "ag_ui.auth_apikey_missing_config",
                )
                return None

            try:
                self._api_keys_map = json.loads(self.config.api_keys)
                if not isinstance(self._api_keys_map, dict):
                    log_warning_event(
                        logger,
                        "API keys configuration must be a JSON object",
                        "ag_ui.auth_apikey_invalid_format",
                    )
                    self._api_keys_map = {}
                    return None
            except json.JSONDecodeError as e:
                log_warning_event(
                    logger,
                    f"Failed to parse API keys JSON: {e}",
                    "ag_ui.auth_apikey_parse_error",
                    error=str(e),
                )
                self._api_keys_map = {}
                return None

        # Extract API key from headers
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Try Authorization: ApiKey <key> format
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("ApiKey "):
                api_key = auth_header[7:]  # Remove "ApiKey " prefix

        if not api_key:
            return None

        # Lookup user ID
        user_id = self._api_keys_map.get(api_key)
        if user_id:
            # Log first 8 characters only for security
            key_prefix = api_key[:8] + "..." if len(api_key) > 8 else api_key[:8]
            log_info_event(
                logger,
                f"Authenticated via API key: user_id={user_id}",
                "ag_ui.auth_apikey_success",
                user_id=user_id,
                key_prefix=key_prefix,
            )
            return str(user_id)

        # Invalid API key - log attempt (first 8 chars only)
        key_prefix = api_key[:8] + "..." if len(api_key) > 8 else api_key[:8]
        log_warning_event(
            logger,
            f"Invalid API key attempted: {key_prefix}",
            "ag_ui.auth_apikey_invalid",
            key_prefix=key_prefix,
        )
        return None


def create_auth_middleware(
    app: ASGIApp, config: ServerConfig | None = None
) -> AuthMiddlewareConfig | None:
    """Create authentication middleware configuration from config.

    Args:
        app: FastAPI/Starlette application instance
        config: Optional ServerConfig instance (defaults to get_config())

    Returns:
        AuthMiddlewareConfig with keys 'enabled', 'mode', 'strategies', 'config'
        if auth is enabled, None otherwise.

    """
    if config is None:
        config = get_config()

    enabled = config.auth_enabled
    mode_str = config.auth_mode.lower()
    strategies_str = config.auth_strategies

    # Parse mode
    try:
        mode = AuthMode(mode_str)
    except ValueError:
        log_warning_event(
            logger,
            f"Invalid auth_mode '{mode_str}', defaulting to 'strict'",
            "ag_ui.auth_invalid_mode",
            mode=mode_str,
        )
        mode = AuthMode.STRICT

    # Parse strategies
    strategies = []
    for strategy_str in strategies_str.split(","):
        strategy_str = strategy_str.strip().lower()
        try:
            strategy = AuthStrategy(strategy_str)
            strategies.append(strategy)
        except ValueError:
            log_warning_event(
                logger,
                f"Invalid auth_strategy '{strategy_str}', skipping",
                "ag_ui.auth_invalid_strategy",
                strategy=strategy_str,
            )

    if not strategies:
        log_warning_event(
            logger,
            "No valid auth strategies configured, defaulting to 'header'",
            "ag_ui.auth_no_strategies",
        )
        strategies = [AuthStrategy.HEADER]

    if enabled:
        log_info_event(
            logger,
            f"Authentication middleware enabled: mode={mode.value}, strategies={[s.value for s in strategies]}",
            "ag_ui.auth_middleware_enabled",
            mode=mode.value,
            strategies=[s.value for s in strategies],
        )
        return {
            "enabled": enabled,
            "mode": mode,
            "strategies": strategies,
            "config": config,
        }
    else:
        log_info_event(
            logger,
            "Authentication middleware disabled",
            "ag_ui.auth_middleware_disabled",
        )
        return None
