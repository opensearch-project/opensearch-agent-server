"""
Unit tests for the OboAuth httpx auth handler.

These cover the scheme-aware header injection that lets the OBO pipeline carry
credentials forwarded from the incoming /runs request down to the MCP server:

- A raw token (e.g. a bare JWT) is wrapped as ``Bearer <token>``.
- A token that already carries a scheme (``Bearer ``/``Basic ``/``ApiKey ``,
  case-insensitive) is forwarded verbatim, so a Basic credential is NOT
  corrupted into ``Bearer Basic <b64>`` (the bug that denied access to the UBI
  indices during evals).
- When no token is set, no Authorization header is added.
- The sync and async flows behave identically.
"""

import base64

import httpx
import pytest

from utils.obo_context import OboAuth

pytestmark = pytest.mark.unit


def _run_sync_flow(auth: OboAuth, request: httpx.Request) -> httpx.Request:
    """Drive the sync auth flow and return the (mutated) request."""
    flow = auth.sync_auth_flow(request)
    return next(flow)


async def _run_async_flow(auth: OboAuth, request: httpx.Request) -> httpx.Request:
    """Drive the async auth flow and return the (mutated) request."""
    flow = auth.async_auth_flow(request)
    return await flow.__anext__()


def _request() -> httpx.Request:
    return httpx.Request("GET", "http://localhost:3001/mcp")


class TestSetGetToken:
    """Token storage is thread-safe and round-trips."""

    def test_default_token_is_none(self):
        assert OboAuth().get_token() is None

    def test_set_then_get(self):
        auth = OboAuth()
        auth.set_token("eyJhbGc")
        assert auth.get_token() == "eyJhbGc"

    def test_set_none_clears_token(self):
        auth = OboAuth()
        auth.set_token("eyJhbGc")
        auth.set_token(None)
        assert auth.get_token() is None


class TestSyncAuthFlow:
    """Header injection for the synchronous flow."""

    def test_no_token_adds_no_header(self):
        auth = OboAuth()
        request = _run_sync_flow(auth, _request())
        assert "Authorization" not in request.headers

    def test_raw_token_is_wrapped_as_bearer(self):
        auth = OboAuth()
        auth.set_token("eyJhbGc.payload.sig")
        request = _run_sync_flow(auth, _request())
        assert request.headers["Authorization"] == "Bearer eyJhbGc.payload.sig"

    def test_existing_bearer_scheme_is_preserved(self):
        auth = OboAuth()
        auth.set_token("Bearer eyJhbGc.payload.sig")
        request = _run_sync_flow(auth, _request())
        # Not double-wrapped into "Bearer Bearer ...".
        assert request.headers["Authorization"] == "Bearer eyJhbGc.payload.sig"

    def test_basic_scheme_is_forwarded_verbatim(self):
        """Regression: a Basic credential must not become 'Bearer Basic <b64>'."""
        b64 = base64.b64encode(b"admin:secret").decode()
        token = f"Basic {b64}"
        auth = OboAuth()
        auth.set_token(token)
        request = _run_sync_flow(auth, _request())
        assert request.headers["Authorization"] == token

    def test_apikey_scheme_is_forwarded_verbatim(self):
        auth = OboAuth()
        auth.set_token("ApiKey abc123")
        request = _run_sync_flow(auth, _request())
        assert request.headers["Authorization"] == "ApiKey abc123"

    @pytest.mark.parametrize(
        "token",
        ["basic dXNlcjpwdw==", "BEARER tok", "apikey k", "BaSiC dXNlcjpwdw=="],
    )
    def test_scheme_detection_is_case_insensitive(self, token):
        auth = OboAuth()
        auth.set_token(token)
        request = _run_sync_flow(auth, _request())
        assert request.headers["Authorization"] == token


class TestAsyncAuthFlow:
    """The async flow mirrors the sync flow."""

    async def test_no_token_adds_no_header(self):
        auth = OboAuth()
        request = await _run_async_flow(auth, _request())
        assert "Authorization" not in request.headers

    async def test_raw_token_is_wrapped_as_bearer(self):
        auth = OboAuth()
        auth.set_token("eyJhbGc.payload.sig")
        request = await _run_async_flow(auth, _request())
        assert request.headers["Authorization"] == "Bearer eyJhbGc.payload.sig"

    async def test_basic_scheme_is_forwarded_verbatim(self):
        b64 = base64.b64encode(b"admin:secret").decode()
        token = f"Basic {b64}"
        auth = OboAuth()
        auth.set_token(token)
        request = await _run_async_flow(auth, _request())
        assert request.headers["Authorization"] == token
